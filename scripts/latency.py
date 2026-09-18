"""Measures pure per-token DECODE latency (not prefill) at a given context length, for either
the dense baseline or a NovaKV compressed checkpoint. This is the naive/unoptimized path
(standard growing cache, eager attention) -- see test_static_cache_graph.py for the real
deployable number (fixed-shape cache + CUDA graph capture).

Methodology: time two generate() calls at the same prompt/context length with different
--min-new-tokens (a short one and a longer one); min_new_tokens forces exactly that many decode
steps (no early EOS stop). The marginal time (t_long - t_short) / (n_long - n_short) cancels out
the (roughly constant, sizeable) prefill cost and isolates the true steady-state per-token decode
cost.

One model per invocation -- run once with no --weights (dense baseline), once with --weights
(the compressed checkpoint, loaded directly from its own shapes). Compare the printed tables.
Single-stream (batch=1) by default; --batch-size raises it.

Usage
-----
  # Dense baseline
  python scripts/latency.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --context-lengths 2048 8192 16384 --output latency_dense.json

  # NovaKV compressed checkpoint
  python scripts/latency.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --weights novakv_delta.pt \\
      --context-lengths 2048 8192 16384 --output latency_novakv.json
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from novakv.model import load_truncated_model, set_model_mode


def load_model(args):
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", use_cache=False, use_safetensors=True,
    )
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.bfloat16()
    if args.weights:
        print(f"Loading NovaKV checkpoint: {args.weights}")
        model = load_truncated_model(model, config, args.weights,
                                      skip_layers=tuple(args.skip_layers), dtype=torch.bfloat16)
        set_model_mode(model, skip_layers=tuple(args.skip_layers))
    else:
        print("No --weights given: benchmarking raw uncompressed baseline.")

    model.eval()
    model.config.use_cache = True

    if args.compile:
        # Profiling showed the decode slowdown is CPU/dispatch-bound (many small unfused ops),
        # not raw CUDA compute, so operator fusion is the targeted fix. --compile-mode
        # reduce-overhead enables CUDA graphs but needs fixed tensor shapes across replays --
        # this path's KV cache grows every decode step, so this either degrades gracefully
        # (recompiles per shape) or errors out -- see test_static_cache_graph.py for the
        # fixed-shape alternative that actually works.
        print(f"Compiling model with torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)

    return model, tokenizer


@torch.no_grad()
def timed_generate(model, input_ids, n_new_tokens, pad_token_id):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(
        input_ids=input_ids, min_new_tokens=n_new_tokens, max_new_tokens=n_new_tokens,
        do_sample=False, use_cache=True, pad_token_id=pad_token_id,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure_decode_latency(model, context_len, n_short, n_long, device, pad_token_id, batch_size=1):
    torch.manual_seed(0)
    # Synthetic prompt: random token ids. Latency depends on tensor shapes/compute, not content.
    input_ids = torch.randint(low=1000, high=20000, size=(batch_size, context_len), device=device)

    timed_generate(model, input_ids, 3, pad_token_id)  # warmup: JIT/cache before timing
    t_short = timed_generate(model, input_ids, n_short, pad_token_id)
    t_long = timed_generate(model, input_ids, n_long, pad_token_id)

    per_token_s = (t_long - t_short) / (n_long - n_short)
    # Per-token here means "per decode step" (one step advances every sequence in the batch by
    # one token simultaneously) -- tokens_per_sec_total is the aggregate throughput across the
    # whole batch (what a serving system would actually report), tokens_per_sec is the
    # per-sequence rate (comparable across batch sizes to see the per-request latency cost).
    tokens_per_sec = 1.0 / per_token_s if per_token_s > 0 else float("inf")
    return {
        "context_len": context_len, "n_short": n_short, "n_long": n_long, "batch_size": batch_size,
        "t_short_s": t_short, "t_long_s": t_long,
        "decode_ms_per_step": per_token_s * 1000,
        "steps_per_sec": tokens_per_sec, "tokens_per_sec_total": tokens_per_sec * batch_size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--weights", default=None,
                    help="NovaKV compressed checkpoint (.pt), full or KV-only delta. Omit for "
                         "the dense baseline.")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31])
    p.add_argument("--compile", action="store_true",
                    help="Wrap the model in torch.compile().")
    p.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--context-lengths", type=int, nargs="+", default=[2048, 8192, 16384])
    p.add_argument("--batch-size", type=int, default=1,
                    help="Number of sequences decoded concurrently -- default 1 matches every "
                         "other measurement in this repo (single-stream). Use e.g. 8/32/64 to "
                         "check whether the compute-bound reconstruction cost reasserts itself "
                         "once CPU-dispatch overhead no longer dominates.")
    p.add_argument("--n-short", type=int, default=10,
                    help="short generate() call, isolates prefill-heavy timing (subtracted out)")
    p.add_argument("--n-long", type=int, default=60,
                    help="long generate() call -- (t_long - t_short)/(n_long - n_short) = "
                         "marginal per-token decode cost, prefill cancels out")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    model, tokenizer = load_model(args)
    device = next(model.parameters()).device
    # Llama-3.1's config.eos_token_id is a LIST (several valid stop tokens) -- generate()'s
    # pad_token_id must be a single int, take the tokenizer's own (single-int) eos id instead.
    pad_token_id = tokenizer.eos_token_id

    results = []
    label = args.weights or "dense"
    for ctx in args.context_lengths:
        print(f"\n=== context_len={ctx} batch_size={args.batch_size} ===")
        r = measure_decode_latency(model, ctx, args.n_short, args.n_long, device, pad_token_id,
                                    batch_size=args.batch_size)
        r["label"] = label
        print(f"  decode: {r['decode_ms_per_step']:.2f} ms/step "
              f"({r['steps_per_sec']:.1f} steps/s/sequence, "
              f"{r['tokens_per_sec_total']:.1f} tokens/s aggregate)")
        results.append(r)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
