"""The attention module structure and RoPE application used by the compressed model.
`LlamaCustomAttention` only provides parameter/buffer structure (q/k/v/o_proj, rotary_emb) --
its actual forward passes are installed by model.py (_patched_attn_forward /
_static_cache_attn_forward), which replace `.forward` after construction.

Derived from Hugging Face transformers (Apache 2.0): `LlamaCustomAttention.__init__` follows
`LlamaAttention`'s parameter layout, and `apply_rotary_pos_emb_custom` is a single-tensor
adaptation of `apply_rotary_pos_emb` with a last-position broadcast for the decode path.
`rope_full_range` is specific to this project."""
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaConfig, rotate_half, LlamaRotaryEmbedding,
)


def apply_rotary_pos_emb_custom(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if x.shape[-2] != cos.shape[-2]:
        cos = cos[:, :, -1, :].unsqueeze(2)
        sin = sin[:, :, -1, :].unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


def rope_full_range(rotary_emb, x, seq_len, current_position=None):
    """Correct per-position cos/sin for the WHOLE re-expanded cache range, to use instead of
    apply_rotary_pos_emb_custom's last-position broadcast (which is only valid when x's
    sequence dimension already matches cos/sin's).

    `current_position`, if given, is the true absolute position of the newest (current) token,
    per row of the batch (shape (B,) or (B,1), taken from the caller's own position_ids/
    cache_position -- already correctly padding-aware, never recomputed here). The full range
    is built by counting backwards from this anchor, so every row gets its own correctly offset
    positions instead of a single shared arange(0, seq_len), which would silently assume every
    row's real content starts at position 0 -- wrong for any left-padded row in a batch. If
    `current_position` is None (prefill; shapes already match, so this path isn't exercised by
    the fallback this function replaces), falls back to the plain 0..seq_len-1 range.
    """
    if current_position is None:
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
    else:
        current_position = current_position.reshape(-1, 1).to(x.device)
        offsets = torch.arange(seq_len - 1, -1, -1, device=x.device).unsqueeze(0)
        position_ids = current_position - offsets
    return rotary_emb(x, position_ids)


class LlamaCustomAttention(nn.Module):
    """Llama attention with low-rank compressed K/V projections: k_proj/v_proj are replaced
    (by model.py's load_truncated_model) with VS/U inference wrappers -- VS projects
    hidden_states to the compressed rank-r latent stored in the KV cache, U expands back to
    full head_dim when needed. Only holds parameter structure; see model.py for the forward
    passes installed onto instances of this class."""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        # Own copy of the model's rotary embedding, so this module can recompute correct
        # per-position cos/sin for the full re-expanded cache range at decode time (see
        # rope_full_range) instead of relying on the caller's cos/sin, which only ever covers
        # the current input tokens.
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        # Mistral has no attention_bias config field at all (always no-bias, no option) --
        # Llama has it. getattr keeps Llama's real per-checkpoint value, defaults False for
        # any config lacking the field.
        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias)
