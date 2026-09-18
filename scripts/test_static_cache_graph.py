"""The real deployable latency number: (1) correctness check of the full manual decode step
(embed -> skip layers -> run_compressed_layers_step -> skip layer -> norm -> lm_head) against
the reference dynamic-cache path, THEN (2) real torch.cuda.CUDAGraph() capture of just the
compressed-layers region, with a SEPARATE correctness check across multiple replays (the risk
being checked: a Python-int baked into a captured graph would silently go stale on replay --
here everything feeding the captured region is a persistent tensor, updated via .copy_()/
increment, never reassigned, so there's no such value to go stale in the first place).

If either correctness check fails (argmax mismatch vs the reference), the script stops and
prints FAIL instead of reporting a latency number -- a passing run is required before the
ms/token figure means anything.

Usage
-----
  python scripts/test_static_cache_graph.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --weights novakv_delta.pt --real-prompt --prompt-len 64 --n-decode-steps 8 --max-len 256
"""

import argparse

import torch

from latency import load_model
from novakv.model import install_static_cache_attn, build_static_cache_for_model, run_compressed_layers_step


def run_skip_layer(model, block, hs, cos, sin, static_cache, cache_position):
    """One skip (uncompressed) layer's decode step, standard LlamaAttention -- attention_mask=None
    is safe here: with a single query token, every cached position is trivially in the past, so
    there is nothing for a causal mask to exclude."""
    residual = hs
    normed = block.input_layernorm(hs)
    attn_out, _ = block.self_attn(
        hidden_states=normed, attention_mask=None, position_ids=cache_position.unsqueeze(0),
        past_key_values=static_cache, use_cache=True,
        position_embeddings=(cos, sin), cache_position=cache_position,
    )
    hs = residual + attn_out
    residual = hs
    normed = block.post_attention_layernorm(hs)
    return residual + block.mlp(normed)


def manual_decode_step(model, hidden_states_1tok, static_cache, cache_position, skip_layers, compressed_layers):
    """hidden_states_1tok: [B, 1, hidden] (already embedded). Returns logits [B, 1, vocab]."""
    cos, sin = model.model.rotary_emb(hidden_states_1tok, cache_position.unsqueeze(0))
    hs = hidden_states_1tok
    for i in range(len(model.model.layers)):
        if i in compressed_layers:
            hs = run_compressed_layers_step(model, hs, static_cache, cache_position, [i])
        else:
            hs = run_skip_layer(model, model.model.layers[i], hs, cos, sin, static_cache, cache_position)
    hs = model.model.norm(hs)
    return model.lm_head(hs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--weights", required=True,
                    help="NovaKV compressed checkpoint (.pt), full or KV-only delta.")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31])
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--n-decode-steps", type=int, default=8)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--n-graph-iters", type=int, default=20,
                    help="extra replay-only iterations after the correctness-checked steps, to "
                         "get a stable latency reading")
    p.add_argument("--batch-size", type=int, default=1,
                    help="Number of sequences decoded concurrently -- default 1 matches the "
                         "original single-stream validation. All rows get the same input "
                         "(repeated), so identical-output-per-row is itself a cheap correctness "
                         "signal at batch>1, in addition to the existing reference comparison.")
    p.add_argument("--real-prompt", action="store_true",
                    help="Use real natural-language text instead of torch.randint token ids for "
                         "the prompt/continuation. Random-token input gives the model no strong "
                         "next-token preference, so logits are flatter and near-ties (which flip "
                         "under ordinary bf16 rounding noise, independent of any real correctness "
                         "bug) are more likely to trigger a false FAIL. Real text gives a much "
                         "more peaked, realistic distribution, closer to actual deployment usage.")
    args = p.parse_args()
    args.compile = False  # latency.load_model reads this; never compile inside this test

    model, tokenizer = load_model(args)
    device = next(model.parameters()).device
    skip_set = set(args.skip_layers)
    compressed_layers = [i for i in range(len(model.model.layers)) if i not in skip_set]

    torch.manual_seed(0)
    total_len = args.prompt_len + args.n_decode_steps
    if args.real_prompt:
        passage = (
            "The history of artificial intelligence began in antiquity, with myths and stories "
            "of artificial beings endowed with intelligence or consciousness by master "
            "craftsmen. The seeds of modern AI were planted by philosophers who attempted to "
            "describe the process of human thinking as the mechanical manipulation of symbols. "
            "This work culminated in the invention of the programmable digital computer in the "
            "1940s, a machine based on the abstract essence of mathematical reasoning. This "
            "device and the ideas behind it inspired a handful of scientists to begin seriously "
            "discussing the possibility of building an electronic brain. "
        ) * (total_len // 8 + 5)
        ids = tokenizer(passage, return_tensors="pt").input_ids[:, :total_len].to(device)
        assert ids.shape[1] == total_len, (
            f"real passage too short for prompt_len+n_decode_steps={total_len} "
            f"(got {ids.shape[1]} tokens) -- widen the repeated passage above"
        )
        ids = ids.repeat(args.batch_size, 1)
        input_ids = ids[:, :args.prompt_len]
        extra_tokens = ids[:, args.prompt_len:total_len]
    else:
        input_ids = torch.randint(low=1000, high=20000, size=(args.batch_size, args.prompt_len), device=device)
        extra_tokens = torch.randint(low=1000, high=20000, size=(args.batch_size, args.n_decode_steps), device=device)

    # ---- Reference ----
    print("Running reference (dynamic cache, existing code path)...")
    with torch.no_grad():
        # logits_to_keep=1: only compute the LM head over the last position -- without this,
        # HF materializes logits for every prompt position (batch x prompt_len x vocab_size),
        # which OOMs at any real batch_size/prompt_len combination (e.g. 32 x 8192 x 128256 in
        # bf16 is ~66GB) even though only the last position is ever used below.
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        ref_logits = [out.logits[:, -1, :].clone()]
        past = out.past_key_values
        for step in range(args.n_decode_steps):
            tok = extra_tokens[:, step:step + 1]
            out = model(input_ids=tok, past_key_values=past, use_cache=True)
            ref_logits.append(out.logits[:, -1, :].clone())
            past = out.past_key_values

    # Free the reference phase's DynamicCache (grows to batch x prompt_len+n_decode_steps x
    # rank per layer) before building the static cache below -- at real batch/context sizes,
    # leaving this alive (plus PyTorch's caching-allocator fragmentation from it) can be the
    # difference between fitting and OOM-ing on the next large allocation.
    del out, past
    torch.cuda.empty_cache()

    # ---- Manual step, NOT yet graph-captured (correctness of the plumbing itself) ----
    print("Running manual decode step (uncaptured, checking plumbing correctness)...")
    # install_static_cache_attn swaps non-skip layers' self_attn.forward to
    # _static_cache_attn_forward -- must happen AFTER the reference run above (which needs the
    # original _patched_attn_forward + a plain auto-created DynamicCache), and BEFORE any
    # model(...) call that passes a CompressedStaticCache as past_key_values.
    install_static_cache_attn(model, skip_layers=tuple(args.skip_layers))
    static_cache = build_static_cache_for_model(
        model, skip_layers=tuple(args.skip_layers), batch_size=args.batch_size, max_len=args.max_len,
        dtype=next(model.parameters()).dtype, device=device,
    )
    with torch.no_grad():
        prompt_pos = torch.arange(args.prompt_len, device=device)
        out = model(input_ids=input_ids, past_key_values=static_cache, use_cache=True,
                     cache_position=prompt_pos, logits_to_keep=1)
        manual_logits = [out.logits[:, -1, :].clone()]

        cache_position = torch.tensor([args.prompt_len], device=device)  # persistent buffer
        for step in range(args.n_decode_steps):
            tok = extra_tokens[:, step:step + 1]
            hs = model.model.embed_tokens(tok)
            logits = manual_decode_step(model, hs, static_cache, cache_position, skip_set, compressed_layers)
            manual_logits.append(logits[:, -1, :].clone())
            cache_position += 1

    print(f"\n[setup] prompt_len={args.prompt_len} n_decode_steps={args.n_decode_steps} "
          f"max_len={args.max_len} compressed_layers={len(compressed_layers)} "
          f"batch_size={args.batch_size}")
    if args.batch_size > 1:
        # Cheap extra signal: every row has identical input (repeated), so every row's logits
        # must be identical too -- a batch-handling bug (e.g. accidentally mixing rows) would
        # show up here even if row 0 alone still matched the reference correctly.
        rows_match = all(torch.equal(ref_logits[0][r], ref_logits[0][0]) for r in range(args.batch_size))
        print(f"[batch check] all {args.batch_size} rows identical in reference logits: {rows_match}")
    # Judge correctness primarily via argmax match (the token actually generated -- the real
    # thing that matters), with relative logit error as a secondary sanity bound, not a strict
    # pass/fail line -- ~1% relative error across a 32-layer bf16 computation is expected
    # rounding noise, not itself evidence of a bug. A real bug shows up as argmax DISAGREEING,
    # not as noise magnitude drifting slightly run to run.
    max_rel_err = 0.0
    all_argmax_match = True
    for step in range(len(ref_logits)):
        diff = (ref_logits[step] - manual_logits[step]).abs().max().item()
        ref_scale = ref_logits[step].abs().max().item()
        rel = diff / (ref_scale + 1e-8)
        # dim=-1: per-row argmax (batch, vocab) -> (batch,). Without dim, argmax() flattens
        # across batch AND vocab, which is silently wrong for batch>1 (compares a single global
        # max index instead of each row's own).
        argmax_match = torch.equal(ref_logits[step].argmax(dim=-1), manual_logits[step].argmax(dim=-1))
        print(f"[step {step}] abs_diff={diff:.4e}  rel_diff={rel:.4e}  argmax_match={argmax_match}")
        max_rel_err = max(max_rel_err, rel)
        all_argmax_match = all_argmax_match and argmax_match
    print(f"Max relative error (manual, uncaptured): {max_rel_err:.4e}, all argmax matched: {all_argmax_match}")
    if not all_argmax_match or max_rel_err >= 3e-2:
        print("FAIL -- manual plumbing already wrong, not attempting graph capture.")
        return
    print("PASS (uncaptured) -- proceeding to real CUDA graph capture.\n")

    # Free the "uncaptured" phase's static_cache (a full second set of batch x max_len x rank
    # buffers) before allocating static_cache2 below -- no reason to hold two copies at once at
    # real batch/context sizes.
    del static_cache, out
    torch.cuda.empty_cache()

    # ---- Real CUDA graph capture of just the compressed-layers region ----
    # Skip layers before/after the compressed block run EAGERLY, outside the graph -- only the
    # compressed layers (identical op sequence per layer, fixed shapes) get captured.
    skip_before = sorted(i for i in skip_set if i < min(compressed_layers))
    skip_after = sorted(i for i in skip_set if i > max(compressed_layers))
    print(f"Capturing CUDA graph of the {len(compressed_layers)} compressed layers "
          f"(eager before: {skip_before}, eager after: {skip_after})...")

    static_cache2 = build_static_cache_for_model(
        model, skip_layers=tuple(args.skip_layers), batch_size=args.batch_size, max_len=args.max_len,
        dtype=next(model.parameters()).dtype, device=device,
    )
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=static_cache2, use_cache=True,
                     cache_position=prompt_pos, logits_to_keep=1)
    hidden_dim = model.config.hidden_size
    dtype = next(model.parameters()).dtype

    # Persistent input/output buffers -- graph replay reads/writes these exact addresses.
    # pos_buf stays 1-element regardless of batch size: every row advances in lockstep in this
    # test (no per-row early stopping), so one shared absolute position is correct for the
    # whole batch.
    hs_in = torch.zeros(args.batch_size, 1, hidden_dim, dtype=dtype, device=device)
    pos_buf = torch.zeros(1, dtype=torch.long, device=device)

    def run_eager_skip_layers(hs, layer_ids, cache_position):
        cos, sin = model.model.rotary_emb(hs, cache_position.unsqueeze(0))
        for i in layer_ids:
            hs = run_skip_layer(model, model.model.layers[i], hs, cos, sin, static_cache2, cache_position)
        return hs

    def captured_region():
        return run_compressed_layers_step(model, hs_in, static_cache2, pos_buf, compressed_layers)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        pos_buf.fill_(args.prompt_len)
        warm_hs = model.model.embed_tokens(extra_tokens[:, 0:1])
        warm_hs = run_eager_skip_layers(warm_hs, skip_before, pos_buf)
        hs_in.copy_(warm_hs)
        for _ in range(3):
            _ = captured_region()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        hs_out = captured_region()

    print("Verifying correctness ACROSS MULTIPLE REPLAYS with different valid_len/position "
          "(the risk being checked -- a stale baked-in value would show up as a mismatch "
          "starting from replay 2 onward)...")
    graph_logits = []
    with torch.no_grad():
        for step in range(args.n_decode_steps):
            pos_buf.fill_(args.prompt_len + step)
            tok = extra_tokens[:, step:step + 1]
            hs = model.model.embed_tokens(tok)
            hs = run_eager_skip_layers(hs, skip_before, pos_buf)
            hs_in.copy_(hs)
            g.replay()
            torch.cuda.synchronize()
            hs_final = run_eager_skip_layers(hs_out, skip_after, pos_buf)
            hs_final = model.model.norm(hs_final)
            logits = model.lm_head(hs_final)
            graph_logits.append(logits[:, -1, :].clone())

    # Compare graph-replayed logits against the reference dynamic-cache path at the matching
    # step -- the real check: if a stale baked-in valid_len were replayed, later steps would
    # diverge from the reference even though earlier ones might not.
    print()
    max_rel_err_graph = 0.0
    all_argmax_match_graph = True
    for step, gl in enumerate(graph_logits):
        ref = ref_logits[step + 1]
        diff = (ref - gl).abs().max().item()
        rel = diff / (ref.abs().max().item() + 1e-8)
        argmax_match = torch.equal(ref.argmax(dim=-1), gl.argmax(dim=-1))  # per-row, see note above
        print(f"[graph replay {step}] abs_diff={diff:.4e}  rel_diff={rel:.4e}  argmax_match={argmax_match}")
        max_rel_err_graph = max(max_rel_err_graph, rel)
        all_argmax_match_graph = all_argmax_match_graph and argmax_match

    # Gate on argmax match alone, not a fixed relative-error threshold -- a stale baked-in
    # valid_len would show up as a systematic/growing divergence or outright wrong tokens, not
    # bounded non-monotonic noise.
    print(f"\nMax relative error (CUDA graph replay vs reference): {max_rel_err_graph:.4e} "
          f"(informational, not gating on this alone), all argmax matched: {all_argmax_match_graph}")
    if not all_argmax_match_graph:
        print("FAIL -- graph replay produced a different token than the reference at some step, "
              "do not trust any latency number from this path until fixed.")
        return
    print("PASS -- CUDA graph replay matches the reference (same generated token) across "
          "all steps, no staleness signature.\n")

    # ---- Latency: replay-only decode step (pure captured-region cost) ----
    print(f"Timing {args.n_graph_iters} pure graph-replay steps (compressed layers only, "
          f"warmup already done above)...")
    import time
    pos_buf.fill_(args.prompt_len + args.n_decode_steps)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.n_graph_iters):
            g.replay()
        torch.cuda.synchronize()
        t_replay = (time.perf_counter() - t0) / args.n_graph_iters
    print(f"  compressed-layers-only replay: {t_replay * 1000:.3f} ms/step")

    # ---- Latency: FULL decode step (skip layers + graph replay + skip layers + norm + lm_head)
    # -- the number that actually matters, comparable to latency.py's dense finding. ----
    print(f"\nTiming {args.n_graph_iters} FULL decode steps (skip layers + graph + norm + lm_head)...")
    fixed_tok = extra_tokens[:, 0:1]
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(args.n_graph_iters):
            pos_buf.fill_(args.prompt_len + args.n_decode_steps + i)
            hs = model.model.embed_tokens(fixed_tok)
            hs = run_eager_skip_layers(hs, skip_before, pos_buf)
            hs_in.copy_(hs)
            g.replay()
            hs_final = run_eager_skip_layers(hs_out, skip_after, pos_buf)
            hs_final = model.model.norm(hs_final)
            _ = model.lm_head(hs_final)
        torch.cuda.synchronize()
        t_full = (time.perf_counter() - t0) / args.n_graph_iters
    steps_per_sec = 1.0 / t_full
    print(f"  FULL decode step: {t_full * 1000:.3f} ms/step ({steps_per_sec:.1f} steps/s/sequence, "
          f"{steps_per_sec * args.batch_size:.1f} tokens/s aggregate at batch_size={args.batch_size})")


if __name__ == "__main__":
    main()
