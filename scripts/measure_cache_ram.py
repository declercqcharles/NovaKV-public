"""Measure REAL GPU memory occupied by the KV cache after a real prefill, dense vs NovaKV
compressed, at matched context lengths. Not a theoretical estimate from weight ranks (see
measure_compression.py) -- this reads the actual allocated bytes torch is holding once the cache
exists, via a before/after torch.cuda.memory_allocated() delta (isolates the cache from
transient prefill activations, since those are freed once the forward call returns and only
`outputs.past_key_values` + `input_ids` are kept alive afterwards).

Pass SEVERAL --context-lengths in one invocation: the first forward pass in a process pays a
one-time CUDA allocation cost that is misattributed to the cache delta (it inflates that first
reading), so only the later lengths of a multi-length run are warmed up and trustworthy.

Usage:
  # dense baseline
  python scripts/measure_cache_ram.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --context-lengths 2048 8192 16384
  # compressed
  python scripts/measure_cache_ram.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --weights novakv_delta.pt --context-lengths 2048 8192 16384
"""
import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from novakv.model import load_truncated_model, set_model_mode


def build_prompt(tokenizer, length, device):
    text = "The quick brown fox jumps over the lazy dog. " * (length // 8 + 50)
    return tokenizer(text, return_tensors="pt").input_ids[:, :length].to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["dense", "novakv"], default=None,
                   help="'dense' measures the untouched base model's KV cache; 'novakv' loads "
                        "the compressed checkpoint given by --weights and measures its cache. "
                        "Inferred from --weights when omitted.")
    p.add_argument("--weights", default=None,
                   help="NovaKV compressed checkpoint (.pt), full or KV-only delta. Omit to "
                        "measure the dense baseline.")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31])
    p.add_argument("--context-lengths", type=int, nargs="+", default=[2048, 8192, 16384],
                   help="Give SEVERAL lengths in one invocation. The first forward pass in a "
                        "process pays a one-time CUDA allocation cost (cuBLAS workspace, kernel "
                        "buffers) that gets misattributed to the cache delta, inflating that "
                        "first reading; only the later lengths are warmed up and trustworthy.")
    args = p.parse_args()
    if args.mode is None:
        args.mode = "novakv" if args.weights else "dense"

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    config = model.config

    model = model.bfloat16()
    if args.mode == "novakv":
        assert args.weights, "--weights required for --mode novakv"
        model = load_truncated_model(model, config, args.weights,
                                      skip_layers=tuple(args.skip_layers), dtype=torch.bfloat16)
        set_model_mode(model, skip_layers=tuple(args.skip_layers))

    model = model.to(device).eval()
    model.config.use_cache = True

    print(f"\n=== mode={args.mode} weights={args.weights} ===")
    for length in args.context_lengths:
        input_ids = build_prompt(tokenizer, length, device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()

        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        cache = out.past_key_values
        del out
        torch.cuda.synchronize()
        cache_bytes = torch.cuda.memory_allocated() - baseline

        print(f"  context_len={length:>6}  cache_delta={cache_bytes / 1e6:9.1f} MB   "
              f"({cache_bytes / length / 1e3:.2f} KB/token)")
        del cache, input_ids
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
