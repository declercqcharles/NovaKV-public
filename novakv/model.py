"""Inference and measurement library for an already-compressed NovaKV checkpoint.

Every attention layer of a NovaKV checkpoint (except a few uncompressed "skip" layers) stores
its KV cache as a low-rank latent instead of full-width K/V: `VS` projects hidden_states down to
a rank-r latent (that latent IS what lives in the cache), and `U` expands it back to full
head_dim only when attention needs it. This module contains everything needed to LOAD such a
checkpoint and RUN it; it contains no code that produces one.

Two K layouts exist in the wild and both are supported here. "joint": one rank-r latent shared by
every KV head, expanded by a single nn.Linear U -- this is the validated NovaKV champion.
"grouped": the KV heads are split into g groups, each group carries its OWN latent and its OWN
reconstruction block, so U is a raw (num_groups, group_size*head_dim, rank) tensor instead of an
nn.Linear, plus an `inv_perm` index undoing the head reordering the grouped decomposition applied.
V is always joint, in both cases.

Provides:
  - load_truncated_model: rebuild a dense HF model's attention layers at the checkpoint's own
    per-layer ranks and load it -- the only entry point for getting a compressed model
  - required_checkpoint_keys: the exact state_dict keys that load is allowed to depend on
    (everything else comes from the dense base model)
  - VProjInferenceWrapper / KProjInferenceWrapper: the (.VS, .U) pair LlamaCustomAttention talks
    to, for the joint and grouped layouts respectively
  - _patched_attn_forward / set_model_mode: unified prefill+decode forward on a growing cache
  - CompressedStaticCache + _static_cache_attn_forward(_graphable) +
    install_static_cache_attn / build_static_cache_for_model / run_compressed_layers_step:
    fixed-shape KV cache and decode step, for CUDA-graph-capturable low-latency inference
  - get_rank / true_full_kv_params / collect_*_parameter_size: read the real deployed K/V
    projection sizes back off a loaded compressed model, for compression accounting
"""

import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache

from novakv.attention import LlamaCustomAttention
from novakv.attention import apply_rotary_pos_emb_custom as _rope
from novakv.attention import rope_full_range


# ---------------------------------------------------------------------------
# Inference wrappers (satisfy LlamaCustomAttention's .U and .VS interface)
# ---------------------------------------------------------------------------

class VProjInferenceWrapper(nn.Module):
    """Wraps a (VS, U) pair so LlamaCustomAttention can access .VS and .U. .VS maps
    hidden_states to the rank-r latent stored in the cache; .U is a plain nn.Linear
    reconstructing all heads at once from that one combined latent.

    Used for v_proj always, and for k_proj in the "joint" layout. The forwards below detect the
    layout with isinstance(.U, nn.Linear)."""

    def __init__(self, VS_linear: nn.Linear, U_linear: nn.Linear):
        super().__init__()
        self.VS = VS_linear
        self.U = U_linear


class KProjInferenceWrapper(nn.Module):
    """Wraps a grouped k_proj so LlamaCustomAttention can access .VS and .U.

    .U is a raw (num_groups, group_size*head_dim, rank) buffer tensor -- one independent
    reconstruction slice per group, not an nn.Linear. .VS is still a plain nn.Linear, mapping
    hidden_states to the concatenation of the per-group latents.

    .inv_perm is a LongTensor undoing the head reordering the grouped decomposition applied (it
    groups heads by similarity, so the group axis is in permuted head order), or None when no
    reordering exists. _expand_grouped_key applies it; see its docstring for why that has to
    happen before anything treats the result as being in original head order.
    """

    def __init__(self, VS_linear: nn.Linear, U_tensor: torch.Tensor,
                 inv_perm: Optional[torch.Tensor] = None):
        super().__init__()
        self.VS = VS_linear
        self.register_buffer("U", U_tensor)
        if inv_perm is not None:
            self.register_buffer("inv_perm", inv_perm)
        else:
            self.inv_perm = None


# Both wrappers expose the same (.VS, .U) interface; the accounting helpers below accept either.
_KV_WRAPPERS = (VProjInferenceWrapper, KProjInferenceWrapper)


def _u_out_features(module) -> int:
    """Full reconstructed width of a compressed projection (num_kv_heads * head_dim), whichever
    layout its U is in: joint's nn.Linear states it directly as its output width; grouped's raw
    (num_groups, group_size*head_dim, rank) buffer spreads it over the group axis."""
    U = module.U
    if isinstance(U, nn.Linear):
        return int(U.weight.shape[0])
    return int(U.shape[0]) * int(U.shape[1])


def _u_numel(module) -> int:
    """Number of parameters actually deployed in U, for either layout."""
    U = module.U
    return int(U.weight.numel() if isinstance(U, nn.Linear) else U.numel())


# ---------------------------------------------------------------------------
# Rank / parameter-size accounting on a loaded compressed model
# ---------------------------------------------------------------------------

def get_rank(module) -> int:
    """Real deployed rank of a compressed projection: the total width of the latent actually
    stored in the KV cache, read straight off the loaded VS matrix. For grouped K that total is
    the concatenation of the per-group latents, which is exactly what the cache holds."""
    return int(module.VS.weight.shape[0])


def _param_size(module, equal: bool) -> int:
    rank = get_rank(module)
    in_features = module.VS.weight.shape[1]
    out_features = _u_out_features(module)
    n = _u_numel(module) + in_features * rank
    if equal:
        return min(n, in_features * out_features)
    return n


def _kv_wrappers(model, which: str):
    """Yield (layer_idx, wrapper) for the compressed k_proj/v_proj of each layer. `which` is
    "k_proj", "v_proj", or "kv". Layers left uncompressed (skip layers) still hold a plain
    nn.Linear and are silently ignored."""
    names = ("k_proj", "v_proj") if which == "kv" else (which,)
    for i, block in enumerate(model.model.layers):
        for name in names:
            mod = getattr(block.self_attn, name, None)
            if isinstance(mod, _KV_WRAPPERS):
                yield i, mod


def true_full_kv_params(model, skip_layers: tuple) -> Tuple[int, int]:
    """Uncompressed (in_features * out_features) reference for k_proj/v_proj, summed over
    non-skip layers -- the denominator the compressed sizes below are measured against. Works on
    a compressed model: the original dense shapes are still recoverable from VS's input width
    and U's output width."""
    full_k = full_v = 0
    skip_set = set(skip_layers)
    for i, block in enumerate(model.model.layers):
        if i in skip_set:
            continue
        for name, acc in (("k_proj", "k"), ("v_proj", "v")):
            mod = getattr(block.self_attn, name)
            if isinstance(mod, _KV_WRAPPERS):
                n = mod.VS.weight.shape[1] * _u_out_features(mod)
            else:
                n = mod.in_features * mod.out_features
            if acc == "k":
                full_k += n
            else:
                full_v += n
    return full_k, full_v


def collect_K_parameter_size(model, equal: bool = False) -> int:
    return sum(_param_size(m, equal) for _, m in _kv_wrappers(model, "k_proj"))


def collect_V_parameter_size(model, equal: bool = False) -> int:
    return sum(_param_size(m, equal) for _, m in _kv_wrappers(model, "v_proj"))


def collect_KV_parameter_size(model, equal: bool = False) -> int:
    return sum(_param_size(m, equal) for _, m in _kv_wrappers(model, "kv"))


# ---------------------------------------------------------------------------
# Unified prefill+decode attention forward (dynamic/growing cache)
# ---------------------------------------------------------------------------

def _expand_grouped_key(ks: torch.Tensor, k_u: torch.Tensor, inv_perm: Optional[torch.Tensor],
                        head_dim: int) -> torch.Tensor:
    """Reconstruct full per-head K from an already-per-group-split grouped latent.

    `ks`: [..., Hg, T, min_r] -- already split into the group axis at position -3, matching the
    layout both cache paths use. `k_u`: the (Hg, block_out, min_r) buffer tensor (block_out ==
    group_size * head_dim). `inv_perm`: None, or the static LongTensor undoing the grouped
    decomposition's head reordering. Returns [..., num_original_heads, T, head_dim], in ORIGINAL
    head order, same axis convention as the joint reconstruction -- safe to feed straight into
    RoPE/repeat_interleave/SDPA as-is, no further transpose needed.

    The permutation matters: the grouped decomposition reorders heads by similarity before
    slicing them into groups, so the group axis comes out in PERMUTED head order. Every consumer
    downstream (RoPE, repeat_interleave for GQA, SDPA) assumes original head order, so inv_perm
    has to be applied here, before the result leaves this function -- applying it later, or not
    at all, silently swaps heads with no error anywhere.

    Note for callers: reshape the flat latent using `k_u.shape[0]` (the real number of groups),
    never num_key_value_heads -- the two are equal only for the joint layout, and grouped's
    num_groups is smaller."""
    Hg, block_out, min_r = k_u.shape
    out_g = torch.matmul(ks, k_u.transpose(-2, -1).to(ks.dtype))  # [..., Hg, T, block_out]
    if inv_perm is None:
        return out_g
    group_size = block_out // head_dim
    T = out_g.shape[-2]
    out_g = out_g.view(*out_g.shape[:-1], group_size, head_dim)      # [..., Hg, T, group_size, Hd]
    out_g = out_g.movedim(-2, -3)                                    # [..., Hg, group_size, T, Hd]
    out_g = out_g.reshape(*out_g.shape[:-4], Hg * group_size, T, head_dim)  # perm-order heads
    return out_g.index_select(-3, inv_perm)  # original head order


def _patched_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Prefill uses SDPA; decode uses a plain matmul QK^T against the re-expanded K."""
    input_shape = hidden_states.shape[:-1]
    query_len = input_shape[-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    k_u, k_vs = self.k_proj.U, self.k_proj.VS
    v_u, v_vs = self.v_proj.U, self.v_proj.VS
    H, Hd = self.num_key_value_heads, self.head_dim
    # Joint K (.U an nn.Linear) reconstructs all heads at once from one shared latent. Grouped K
    # (.U a raw (num_groups, group_size*head_dim, rank) buffer) has one independent latent and
    # one independent reconstruction slice per group, so its latent is kept split on a group axis
    # all the way through the cache. V is always joint.
    k_joint = isinstance(k_u, nn.Linear)

    def _expand_key(ks):
        if k_joint:
            return (
                torch.matmul(ks, k_u.weight.T.to(ks.dtype))
                .view(*ks.shape[:-1], H, Hd)
                .transpose(-3, -2)
            )
        # Grouped: `ks` arrives already split as [B, Hg, T, min_r] (see k_inter below), and
        # _expand_grouped_key returns [B, H, T, Hd] in original head order -- same axis
        # convention as the joint branch above, no further transpose needed.
        return _expand_grouped_key(ks, k_u, self.k_proj.inv_perm, Hd)

    if k_joint:
        k_inter = torch.matmul(hidden_states, k_vs.weight.T)
    else:
        # k_u.shape[0] is the real number of groups -- read it off the buffer, never assume
        # num_key_value_heads (correct only for the joint layout, too large for grouped).
        k_inter = (
            torch.matmul(hidden_states, k_vs.weight.T)
            .view(*input_shape, k_u.shape[0], -1)
            .transpose(1, 2)
        )
    v_inter = torch.matmul(hidden_states, v_vs.weight.T)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(k_inter, v_inter, self.layer_idx)
    else:
        key_states, value_states = k_inter, v_inter

    cos, sin = position_embeddings
    query_states = _rope(query_states, cos, sin)

    if query_len > 1:
        # Prefill: materialise full K/V and use SDPA (Flash Attention path).
        key_full = _rope(_expand_key(key_states), cos, sin)
        # GQA: key_full has num_key_value_heads; query has num_attention_heads. Expand K to
        # match Q so SDPA head dims line up (no-op if num_key_value_groups == 1).
        key_full = key_full.repeat_interleave(self.num_key_value_groups, dim=1)
        value_full = (
            torch.matmul(value_states, v_u.weight.T)
            .view(*value_states.shape[:-1], H, Hd)
            .transpose(-3, -2)
        )
        value_full = value_full.repeat_interleave(self.num_key_value_groups, dim=1)
        causal_mask = (
            attention_mask[:, :, :, : key_states.shape[-2]]
            if attention_mask is not None else None
        )
        # SDPA needs is_causal=True explicitly when attention_mask is None (plain causal
        # prefill, no padding) -- without it, a None mask silently produces full bidirectional
        # attention instead of causal.
        attn_output = F.scaled_dot_product_attention(
            query_states, key_full, value_full, attn_mask=causal_mask,
            is_causal=(causal_mask is None), scale=self.scaling,
        )
    else:
        # Decode: key_full spans the whole cached history (many positions), but `cos`/`sin`
        # from position_embeddings only cover the current decode token (1 position) --
        # apply_rotary_pos_emb_custom's last-position-broadcast fallback would rotate every
        # cached key by the CURRENT token's angle regardless of its true position. Recompute
        # the correct per-position cos/sin for the full range instead, anchored on
        # position_ids (this row's true current position, already correctly offset for
        # left-padding by standard HF code upstream) -- a shared arange(0, seq_len) would
        # silently assume every row's cache starts at position 0, wrong for a padded batch.
        key_full = _expand_key(key_states)
        current_pos = position_ids[:, -1] if position_ids is not None else None
        cos_k, sin_k = rope_full_range(
            self.rotary_emb, key_full, key_full.shape[-2], current_position=current_pos,
        )
        key_full = _rope(key_full, cos_k.to(key_full.dtype), sin_k.to(key_full.dtype))
        key_full = key_full.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query_states, key_full.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            mask_slice = attention_mask[:, :, :, : key_states.shape[-2]]
            # This mask is boolean (True=attend, False=masked) in the HF versions this project
            # runs against -- adding it directly to attn_weights would silently cast True/False
            # to +1.0/+0.0 (a negligible logit nudge instead of a real exclusion). SDPA (prefill
            # branch above) accepts bool masks natively; this manual matmul+softmax path does
            # not, so it needs an explicit conversion. Only matters when a batch actually
            # contains left-padding.
            if mask_slice.dtype == torch.bool:
                mask_slice = torch.zeros_like(attn_weights).masked_fill(
                    ~mask_slice, torch.finfo(attn_weights.dtype).min
                )
            attn_weights = attn_weights + mask_slice

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        # Aggregate compact V then expand with U (avoids full V materialisation). attn_weights
        # spans all query heads (Hq); value_states is the shared global compact V. prob_v is
        # per query head, but the U expansion block is per KV head, so repeat each KV head's U
        # across its query-head group (GQA; no-op for MHA).
        rank_v = value_states.shape[-1]
        prob_v = attn_weights.squeeze(2) @ value_states  # [B, Hq, rank_v]
        v_u_head = v_u.weight.view(H, Hd, rank_v).repeat_interleave(self.num_key_value_groups, dim=0)
        attn_output = prob_v.unsqueeze(-2) @ v_u_head.transpose(-1, -2)  # [B, Hq, 1, Hd]

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


# ---------------------------------------------------------------------------
# Fixed-shape KV cache, for CUDA-graph-capturable low-latency decode
# ---------------------------------------------------------------------------

class CompressedStaticCache:
    """Fixed-shape KV cache: pre-allocated buffers of a fixed max_len, written in place at
    absolute position instead of growing via torch.cat. Required for CUDA graph capture (which
    needs identical tensor shapes across replays); a normal growing cache can't provide that.

    Buffer position == absolute token position, always (simple left-to-right fill, no eviction/
    sliding window) -- this sidesteps rope_full_range's "count backwards from current position"
    logic entirely (that convention assumes the buffer's LAST slot is always the newest token,
    which does not hold here).
    """

    def __init__(self, num_layers, batch_size, max_len, k_ranks, v_ranks, dtype, device):
        # k_ranks[i]/v_ranks[i] == None marks a skip layer (uncompressed, standard attention) --
        # those get no static buffer and fall back to a plain growing cache inside update(),
        # matching DynamicCache's own behavior for a standard LlamaAttention.forward call.
        self.max_len = max_len
        self.k_buf, self.v_buf = {}, {}
        self.dyn_k, self.dyn_v = {}, {}
        self._seen_tokens = 0
        for i in range(num_layers):
            if k_ranks[i] is None:
                continue
            self.k_buf[i] = torch.zeros(batch_size, max_len, k_ranks[i], dtype=dtype, device=device)
            self.v_buf[i] = torch.zeros(batch_size, max_len, v_ranks[i], dtype=dtype, device=device)

    def update(self, key_states, value_states, layer_idx, cache_position=None, track_seen=True):
        """key_states/value_states: [B, n_new, rank] for a compressed (static-buffer) layer, or
        [B, H, n_new, head_dim] for a skip layer. cache_position: 1D LongTensor of absolute
        positions to write (length n_new) -- required for compressed layers. track_seen=False
        skips the _seen_tokens update (needs .item(), illegal during CUDA graph capture) -- used
        by the manual-capture driver, which never calls get_seq_length()/get_mask_sizes()."""
        if layer_idx not in self.k_buf:
            k, v = self.dyn_k.get(layer_idx), self.dyn_v.get(layer_idx)
            self.dyn_k[layer_idx] = key_states if k is None else torch.cat([k, key_states], dim=-2)
            self.dyn_v[layer_idx] = value_states if v is None else torch.cat([v, value_states], dim=-2)
            return self.dyn_k[layer_idx], self.dyn_v[layer_idx]
        assert cache_position is not None, f"cache_position required for compressed layer {layer_idx}"
        self.k_buf[layer_idx][:, cache_position, :] = key_states
        self.v_buf[layer_idx][:, cache_position, :] = value_states
        if track_seen:
            # HF's LlamaModel.forward calls past_key_values.get_seq_length() itself, needs this
            # tracked explicitly since there's no single tensor whose length can be queried the
            # way DynamicCache's growing tensors allow.
            self._seen_tokens = max(self._seen_tokens, int(cache_position.max().item()) + 1)
        return self.k_buf[layer_idx], self.v_buf[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple:
        # HF's create_causal_mask calls this unconditionally to build its own attention_mask
        # before calling into attention forward. _static_cache_attn_forward ignores that
        # auto-built mask in its decode branch (builds its own fixed-max_len pad mask instead)
        # -- this just needs to return the semantically correct (kv_length, kv_offset) so HF's
        # own construction doesn't error out. No sliding window here, so kv_offset is always 0.
        return self._seen_tokens + query_length, 0


def _static_cache_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[CompressedStaticCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Attention forward for CompressedStaticCache. Prefill (query_len > 1) mirrors
    _patched_attn_forward's SDPA path, additionally writing into the static buffer. Decode
    (query_len == 1) uses the fixed-max_len-shape path -- see CompressedStaticCache's docstring
    for why this doesn't reuse rope_full_range. `cache_position` (required here, unlike
    _patched_attn_forward where it's optional) must be the absolute position(s) being written."""
    assert cache_position is not None, "_static_cache_attn_forward requires cache_position"

    input_shape = hidden_states.shape[:-1]
    query_len = input_shape[-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    k_u, k_vs = self.k_proj.U, self.k_proj.VS
    v_u, v_vs = self.v_proj.U, self.v_proj.VS
    H, Hd = self.num_key_value_heads, self.head_dim
    k_joint = isinstance(k_u, nn.Linear)
    max_len = past_key_values.max_len

    def _expand_k(flat_ks):
        """flat_ks: [..., T, k_rank], as stored flat in the static-cache buffer (unlike the
        growing-cache path, the buffer stays flat for both layouts and grouped K is split onto
        its group axis here, at expansion time). Returns [..., H, T, Hd]."""
        if k_joint:
            return (
                torch.matmul(flat_ks, k_u.weight.T).view(*flat_ks.shape[:-1], H, Hd).transpose(-3, -2)
            )
        Hg, _block_out, min_r = k_u.shape  # Hg == num groups, NOT num_key_value_heads
        ks = flat_ks.view(*flat_ks.shape[:-1], Hg, min_r).transpose(-3, -2)  # [..., Hg, T, min_r]
        return _expand_grouped_key(ks, k_u, self.k_proj.inv_perm, Hd)

    k_inter = torch.matmul(hidden_states, k_vs.weight.T)
    v_inter = torch.matmul(hidden_states, v_vs.weight.T)
    key_buf, value_buf = past_key_values.update(k_inter, v_inter, self.layer_idx, cache_position)
    valid_len = int(cache_position[-1].item()) + 1

    if query_len > 1:
        key_slice, value_slice = key_buf[:, :valid_len, :], value_buf[:, :valid_len, :]
        key_full = _expand_k(key_slice)
        cos, sin = position_embeddings
        query_states = _rope(query_states, cos, sin)
        key_full = _rope(key_full, cos, sin).repeat_interleave(self.num_key_value_groups, dim=1)
        value_full = (
            torch.matmul(value_slice, v_u.weight.T).view(*value_slice.shape[:-1], H, Hd).transpose(-3, -2)
        )
        value_full = value_full.repeat_interleave(self.num_key_value_groups, dim=1)
        causal_mask = attention_mask[:, :, :, :valid_len] if attention_mask is not None else None
        attn_output = F.scaled_dot_product_attention(
            query_states, key_full, value_full, attn_mask=causal_mask,
            is_causal=(causal_mask is None), scale=self.scaling,
        )
    else:
        # Fixed max_len shape throughout, plain arange positions (buffer idx == absolute
        # position), explicit valid-length mask instead of relying on tensor length.
        key_full = _expand_k(key_buf)
        position_ids_full = torch.arange(max_len, device=key_full.device).unsqueeze(0)
        cos_full, sin_full = self.rotary_emb(key_full, position_ids_full)
        q_idx = valid_len - 1
        query_states = _rope(query_states, cos_full[:, q_idx:q_idx + 1, :], sin_full[:, q_idx:q_idx + 1, :])
        key_full = _rope(key_full, cos_full, sin_full).repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_full.transpose(2, 3)) * self.scaling
        pos_range = torch.arange(max_len, device=query_states.device).view(1, 1, 1, max_len)
        pad_mask = torch.where(
            pos_range < valid_len,
            torch.zeros((), device=query_states.device, dtype=attn_weights.dtype),
            torch.full((), float("-inf"), device=query_states.device, dtype=attn_weights.dtype),
        )
        attn_weights = F.softmax(attn_weights + pad_mask, dim=-1, dtype=torch.float32).to(query_states.dtype)

        rank_v = value_buf.shape[-1]
        prob_v = attn_weights.squeeze(2) @ value_buf
        v_u_head = v_u.weight.view(H, Hd, rank_v).repeat_interleave(self.num_key_value_groups, dim=0)
        attn_output = prob_v.unsqueeze(-2) @ v_u_head.transpose(-1, -2)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def _static_cache_attn_forward_graphable(
    self,
    hidden_states: torch.Tensor,
    past_key_values: CompressedStaticCache,
    cache_position: torch.LongTensor,
):
    """Decode-ONLY variant of _static_cache_attn_forward, with every `.item()`/Python-int
    control-flow point removed -- required for manual torch.cuda.CUDAGraph() capture (automatic
    torch.compile cudagraph mode was tried first and rejected: it refuses to capture the
    in-place cache-buffer write at all ("mutated inputs"), and separately hits its
    recompile_limit from per-layer non-uniform K/V ranks being treated as distinct specialized
    graphs). Manual capture sidesteps both: buffer mutation is the standard way to feed new data
    into a replayed graph, and there's no dynamo tracing/shape-guard step to trip on differing
    per-layer ranks.

    Only the decode branch (query_len == 1) -- prefill and the graph-capturable warmup/replay
    driver stay outside this function. `cache_position` must be a 1-element LongTensor (a
    persistent GPU buffer the caller updates in place before each replay, never reassigned)."""
    B = hidden_states.shape[0]
    hidden_shape = (B, 1, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    k_u, k_vs = self.k_proj.U, self.k_proj.VS
    v_u, v_vs = self.v_proj.U, self.v_proj.VS
    H, Hd = self.num_key_value_heads, self.head_dim
    k_joint = isinstance(k_u, nn.Linear)
    max_len = past_key_values.max_len

    k_inter = torch.matmul(hidden_states, k_vs.weight.T)
    v_inter = torch.matmul(hidden_states, v_vs.weight.T)
    key_buf, value_buf = past_key_values.update(
        k_inter, v_inter, self.layer_idx, cache_position, track_seen=False,
    )
    valid_len = cache_position[-1:] + 1  # 1-element tensor, never a Python int

    # The layout branch is a Python-level isinstance on a module attribute, fixed for the whole
    # lifetime of the layer -- it is resolved once when the graph is traced, and introduces no
    # data-dependent control flow into the captured region.
    if k_joint:
        key_full = torch.matmul(key_buf, k_u.weight.T).view(*key_buf.shape[:-1], H, Hd).transpose(-3, -2)
    else:
        Hg, _block_out, min_r = k_u.shape  # Hg == num groups, NOT num_key_value_heads
        ks = key_buf.view(*key_buf.shape[:-1], Hg, min_r).transpose(-3, -2)  # [..., Hg, T, min_r]
        key_full = _expand_grouped_key(ks, k_u, self.k_proj.inv_perm, Hd)  # [..., H, T, Hd]
    position_ids_full = torch.arange(max_len, device=key_full.device).unsqueeze(0)
    cos_full, sin_full = self.rotary_emb(key_full, position_ids_full)
    q_idx = valid_len - 1  # 1-element tensor
    cos_q = cos_full.index_select(1, q_idx)  # advanced indexing, not Python-int slicing
    sin_q = sin_full.index_select(1, q_idx)
    query_states = _rope(query_states, cos_q, sin_q)
    key_full = _rope(key_full, cos_full, sin_full).repeat_interleave(self.num_key_value_groups, dim=1)

    attn_weights = torch.matmul(query_states, key_full.transpose(2, 3)) * self.scaling
    pos_range = torch.arange(max_len, device=query_states.device).view(1, 1, 1, max_len)
    pad_mask = torch.where(
        pos_range < valid_len,
        torch.zeros((), device=query_states.device, dtype=attn_weights.dtype),
        torch.full((), float("-inf"), device=query_states.device, dtype=attn_weights.dtype),
    )
    attn_weights = F.softmax(attn_weights + pad_mask, dim=-1, dtype=torch.float32).to(query_states.dtype)

    rank_v = value_buf.shape[-1]
    prob_v = attn_weights.squeeze(2) @ value_buf
    v_u_head = v_u.weight.view(H, Hd, rank_v).repeat_interleave(self.num_key_value_groups, dim=0)
    attn_output = prob_v.unsqueeze(-2) @ v_u_head.transpose(-1, -2)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, 1, -1).contiguous()
    return self.o_proj(attn_output)


def run_compressed_layers_step(model, hidden_states, static_cache, cache_position, layer_indices):
    """The exact op sequence to CUDA-graph-capture: one decode step through the given
    (compressed) layers only, using _static_cache_attn_forward_graphable directly (not via
    block.self_attn.forward/install_static_cache_attn's monkeypatch, so this never touches skip
    layers or anything HF-forward-loop-shaped). Caller runs skip layers, embed_tokens, final
    norm, and lm_head outside the captured region."""
    for i in layer_indices:
        block = model.model.layers[i]
        residual = hidden_states
        hs = block.input_layernorm(hidden_states)
        hs = _static_cache_attn_forward_graphable(block.self_attn, hs, static_cache, cache_position)
        hidden_states = residual + hs
        residual = hidden_states
        hs = block.mlp(block.post_attention_layernorm(hidden_states))
        hidden_states = residual + hs
    return hidden_states


def install_static_cache_attn(model, skip_layers: tuple = (0, 1, 31)):
    """Monkeypatches non-skip layers' self_attn.forward to _static_cache_attn_forward. Must be
    called AFTER load_truncated_model (this only swaps .forward, it doesn't build the U/VS
    modules). Skip layers keep their original attention forward -- CompressedStaticCache.update()
    detects them automatically (no static buffer allocated) and falls back to a plain growing
    cache."""
    for i, block in enumerate(model.model.layers):
        if i not in set(skip_layers):
            block.self_attn.forward = types.MethodType(_static_cache_attn_forward, block.self_attn)


def build_static_cache_for_model(model, skip_layers, batch_size, max_len, dtype, device):
    """Inspects the already-loaded compressed model to read each non-skip layer's real rank
    directly from its k_proj.VS/v_proj.VS shape, and allocates a correctly-sized
    CompressedStaticCache."""
    num_layers = len(model.model.layers)
    k_ranks, v_ranks = [None] * num_layers, [None] * num_layers
    for i, block in enumerate(model.model.layers):
        if i in set(skip_layers):
            continue
        k_ranks[i] = block.self_attn.k_proj.VS.weight.shape[0]
        v_ranks[i] = block.self_attn.v_proj.VS.weight.shape[0]
    return CompressedStaticCache(num_layers, batch_size, max_len, k_ranks, v_ranks, dtype, device)


# ---------------------------------------------------------------------------
# Loading a compressed checkpoint
# ---------------------------------------------------------------------------

def _k_layout(state_dict, layer_idx: int) -> str:
    """Which K layout layer `layer_idx` uses in `state_dict`, decided by the key that is actually
    there: `k_proj.U.weight` (an nn.Linear) means joint, a bare `k_proj.U` (a raw per-group
    tensor) means grouped. Defaults to "joint" when neither is present, so the caller asks for
    the joint key name and reports it as missing -- never silently treats an incomplete layer as
    a different layout."""
    base = f"model.layers.{layer_idx}.self_attn.k_proj"
    if f"{base}.U.weight" in state_dict:
        return "joint"
    if f"{base}.U" in state_dict:
        return "grouped"
    return "joint"


def required_checkpoint_keys(state_dict, num_layers: int, skip_layers: tuple) -> list:
    """The exact state_dict keys load_truncated_model cannot do without, for THIS checkpoint: the
    compressed K/V factors of every non-skip layer. Everything else in the model (embeddings,
    MLPs, norms, lm_head, q_proj/o_proj, and the skip layers' own attention) comes from the dense
    base checkpoint loaded by from_pretrained, so a checkpoint carrying only these keys -- a
    "delta" -- is enough. scripts/export_delta.py is built on top of this list.

    The K key names depend on the layer's layout, which only the checkpoint itself can tell us
    (see _k_layout), so the state_dict has to be passed in: joint needs `k_proj.U.weight`, grouped
    needs the raw `k_proj.U` plus `k_proj.inv_perm` when that layer carries one. `num_layers` is
    still supplied by the caller and still enumerated in full, so a layer missing from the
    checkpoint entirely is reported as missing rather than quietly skipped. V is joint in both
    layouts and always contributes the same two keys."""
    skip_set = set(skip_layers)
    keys = []
    for i in range(num_layers):
        if i in skip_set:
            continue
        k_base = f"model.layers.{i}.self_attn.k_proj"
        keys.append(f"{k_base}.VS.weight")
        if _k_layout(state_dict, i) == "grouped":
            keys.append(f"{k_base}.U")
            # inv_perm is optional (a grouped export without head reordering has none), so it is
            # only required when the checkpoint actually carries it -- but then it MUST survive
            # into a delta, since dropping it silently permutes the heads.
            if f"{k_base}.inv_perm" in state_dict:
                keys.append(f"{k_base}.inv_perm")
        else:
            keys.append(f"{k_base}.U.weight")
        keys.append(f"model.layers.{i}.self_attn.v_proj.VS.weight")
        keys.append(f"model.layers.{i}.self_attn.v_proj.U.weight")
    return keys


def _copy_linear_(dst: nn.Linear, src: nn.Linear):
    dst.weight.data.copy_(src.weight.data.to(dst.weight.dtype))
    if getattr(src, "bias", None) is not None and getattr(dst, "bias", None) is not None:
        dst.bias.data.copy_(src.bias.data.to(dst.bias.dtype))


def load_truncated_model(model, config, truncated_path: str, skip_layers: tuple = (0, 1, 31),
                          dtype: torch.dtype = torch.bfloat16):
    """Load a physically-truncated NovaKV checkpoint into an already-loaded dense model. Reads
    each non-skip layer's real rank straight from the saved k_proj.VS/v_proj.VS tensor shapes,
    allocates LlamaCustomAttention + the matching inference wrapper at exactly those sizes, then
    loads the checkpoint's state_dict in one pass.

    K's layout is detected PER LAYER from the keys the checkpoint actually contains (see
    _k_layout): joint gives a VProjInferenceWrapper with an nn.Linear U, grouped gives a
    KProjInferenceWrapper with a raw per-group U buffer plus the checkpoint's inv_perm (identity
    if it carries none). V is joint in both cases.

    Nothing is derived or recomputed here: the checkpoint's own on-disk size already IS the
    deployed size, so loading is just "allocate at the right shapes and copy". `model` should
    already be loaded via AutoModelForCausalLM.from_pretrained (any dtype; skip layers are used
    as-is, non-skip layers' self_attn is replaced entirely before loading).

    Works with a FULL checkpoint or with a KV-only DELTA (see scripts/export_delta.py). The
    rebuilt LlamaCustomAttention starts as a fresh module, so its q_proj/o_proj are copied over
    from the dense layer being replaced BEFORE loading -- without that, a delta (which carries
    no q_proj/o_proj) would leave those two projections randomly initialised. If the checkpoint
    does carry them, load_state_dict below simply overwrites the copies.

    The keys that must be present are checked up front (required_checkpoint_keys): a genuinely
    missing compressed factor raises instead of being swallowed by strict=False. Keys reported
    missing afterwards are base-model weights already supplied by from_pretrained -- expected,
    and the normal case for a delta.
    """
    sd = torch.load(truncated_path, map_location="cpu", weights_only=False)
    skip_set = set(skip_layers)
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    required = required_checkpoint_keys(sd, len(model.model.layers), skip_layers)
    absent = [k for k in required if k not in sd]
    if absent:
        raise KeyError(
            f"{truncated_path} is missing {len(absent)} compressed K/V tensor(s) that "
            f"load_truncated_model needs, e.g. {absent[:5]}. Check --skip-layers matches the "
            f"checkpoint."
        )

    for i, block in enumerate(model.model.layers):
        if i in skip_set:
            continue
        device = next(block.parameters()).device
        orig = block.self_attn
        k_base = f"model.layers.{i}.self_attn.k_proj"
        k_rank = sd[f"{k_base}.VS.weight"].shape[0]
        v_rank = sd[f"model.layers.{i}.self_attn.v_proj.VS.weight"].shape[0]

        new_attn = LlamaCustomAttention(config, layer_idx=i).to(device=device, dtype=dtype)
        with torch.no_grad():
            _copy_linear_(new_attn.q_proj, orig.q_proj)
            _copy_linear_(new_attn.o_proj, orig.o_proj)
        k_VS = nn.Linear(config.hidden_size, k_rank, bias=False).to(device=device, dtype=dtype)
        v_VS = nn.Linear(config.hidden_size, v_rank, bias=False).to(device=device, dtype=dtype)
        v_U = nn.Linear(v_rank, num_kv_heads * head_dim, bias=False).to(device=device, dtype=dtype)
        if _k_layout(sd, i) == "grouped":
            # Every grouped dimension comes off the saved U tensor -- num_groups and the width of
            # one reconstruction block are not derivable from the config. What IS checked against
            # the config is that the blocks together cover the dense K width, and that the saved
            # latent really does split evenly across the groups: getting either wrong would give
            # a silently mis-shaped reconstruction rather than a load error.
            u_saved = sd[f"{k_base}.U"]
            if u_saved.dim() != 3:
                raise ValueError(
                    f"{truncated_path}: {k_base}.U must be a 3-D (num_groups, block_out, rank) "
                    f"tensor for the grouped layout, got shape {tuple(u_saved.shape)}"
                )
            num_groups, block_out, block_rank = (int(d) for d in u_saved.shape)
            if num_groups * block_out != num_kv_heads * head_dim:
                raise ValueError(
                    f"{truncated_path}: {k_base}.U reconstructs {num_groups} * {block_out} = "
                    f"{num_groups * block_out} K features, but the model needs "
                    f"{num_kv_heads} * {head_dim} = {num_kv_heads * head_dim}"
                )
            if block_rank * num_groups != k_rank:
                raise ValueError(
                    f"{truncated_path}: {k_base}.VS holds a latent of width {k_rank}, which does "
                    f"not split into {num_groups} group latents of width {block_rank}"
                )
            k_U = torch.empty(num_groups, block_out, block_rank, device=device, dtype=dtype)
            if f"{k_base}.inv_perm" in sd:
                perm_saved = sd[f"{k_base}.inv_perm"]
                if perm_saved.numel() != num_kv_heads:
                    raise ValueError(
                        f"{truncated_path}: {k_base}.inv_perm has {perm_saved.numel()} entries, "
                        f"expected one per KV head ({num_kv_heads})"
                    )
                # Allocated empty here, filled by load_state_dict below like every other tensor.
                inv_perm = torch.empty(num_kv_heads, device=device, dtype=torch.long)
            else:
                # A grouped export that did no head reordering carries no inv_perm. The identity
                # permutation reproduces exactly that, and keeps _expand_grouped_key on its
                # per-head-reshaping path -- which a group of more than one head needs regardless
                # of whether the heads were reordered. Said out loud because a checkpoint that WAS
                # reordered and then lost this key would load fine and silently attend with
                # permuted heads.
                print(f"  load_truncated_model: layer {i} is grouped with no inv_perm -- "
                      f"assuming its groups are in original head order")
                inv_perm = torch.arange(num_kv_heads, device=device, dtype=torch.long)
            new_attn.k_proj = KProjInferenceWrapper(k_VS, k_U, inv_perm)
        else:
            k_U = nn.Linear(k_rank, num_kv_heads * head_dim, bias=False).to(device=device, dtype=dtype)
            new_attn.k_proj = VProjInferenceWrapper(k_VS, k_U)
        new_attn.v_proj = VProjInferenceWrapper(v_VS, v_U)
        new_attn.forward = types.MethodType(_patched_attn_forward, new_attn)
        block.self_attn = new_attn

    result = model.load_state_dict(sd, strict=False)
    # A missing key among `required` would mean load_state_dict silently declined a tensor that
    # IS in sd (shape/name mismatch) -- that is a real failure, not a delta artefact.
    required_missing = [k for k in result.missing_keys if k in set(required)]
    if required_missing:
        raise RuntimeError(
            f"load_state_dict did not consume {len(required_missing)} required compressed "
            f"tensor(s), e.g. {required_missing[:5]}"
        )
    print(f"  load_truncated_model: loaded {len(required)} compressed K/V tensors; "
          f"missing={len(result.missing_keys)} (base-model weights from from_pretrained) "
          f"unexpected={len(result.unexpected_keys)}")
    if result.unexpected_keys:
        print(f"    first few unexpected: {result.unexpected_keys[:5]}")
    return model


def set_model_mode(model, skip_layers: tuple = (0, 1, 31)):
    """Ensure all non-skip attention layers use _patched_attn_forward. Kept as a function (not
    inlined into load_truncated_model) since some callers re-apply it after other
    monkeypatching -- e.g. to switch back from install_static_cache_attn's static-cache forward
    to the growing-cache one."""
    for i, block in enumerate(model.model.layers):
        if i not in set(skip_layers):
            block.self_attn.forward = types.MethodType(_patched_attn_forward, block.self_attn)
