#!/usr/bin/env python3
"""Report the real per-layer K/V cache compression of a NovaKV checkpoint, read directly from
the shapes of the tensors it actually contains. No model load, no GPU: the rank stored per layer
IS the width of the latent that the KV cache holds per token, and it is literally the first
dimension of that layer's VS matrix.

Two numbers are reported per layer:
  - cache%   : reduction in KV-cache bytes per token (rank vs the full head_dim * num_kv_heads
               width the dense model would store). This is the number the RAM measurement in
               measure_cache_ram.py should corroborate.
  - weight%  : reduction in the k_proj/v_proj weight matrices themselves
               (in*rank + rank*out vs in*out). Can be negative at high rank -- a factorisation
               only shrinks the weights when the rank is below in*out/(in+out).

Usage:
  python scripts/measure_compression.py --checkpoint novakv_delta.pt
"""
import argparse

import torch


def analyze(sd, target, non_skip):
    rows = []
    for i in non_skip:
        VS_key = f"model.layers.{i}.self_attn.{target}.VS.weight"
        U_key = f"model.layers.{i}.self_attn.{target}.U.weight"
        raw_U_key = f"model.layers.{i}.self_attn.{target}.U"
        if VS_key not in sd:
            continue
        if U_key in sd:
            # Joint layout: U is an nn.Linear, (num_kv_heads*head_dim, rank).
            U = sd[U_key]
            out_features = U.shape[0]
        elif raw_U_key in sd:
            # Grouped K: U is a raw (num_groups, group_size*head_dim, rank) tensor. The groups
            # together reconstruct the same full width, and VS's first dimension is still the
            # total latent width the cache stores per token -- so cache% is unaffected by the
            # layout, only the weight count has to be read off the 3-D tensor.
            U = sd[raw_U_key]
            out_features = U.shape[0] * U.shape[1]
        else:
            continue
        VS = sd[VS_key]
        rank, in_features = VS.shape
        rows.append(dict(
            layer=i, rank=rank, in_features=in_features, out_features=out_features,
            weight_true=in_features * rank + U.numel(),
            weight_dense=in_features * out_features,
        ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="NovaKV checkpoint (.pt), full or KV-only delta.")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31])
    p.add_argument("--num-layers", type=int, default=32)
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    non_skip = [i for i in range(args.num_layers) if i not in set(args.skip_layers)]

    hdr = (f"{'target':>7}  {'layer':>6}  {'rank':>6}  {'dense width':>12}  "
           f"{'cache%':>8}  {'weight%':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    totals = {}
    for target in ("k_proj", "v_proj"):
        rows = analyze(sd, target, non_skip)
        cache_true = cache_dense = w_true = w_dense = 0
        for r in rows:
            cache_pct = 100 * (1 - r["rank"] / r["out_features"])
            weight_pct = 100 * (1 - r["weight_true"] / r["weight_dense"])
            print(f"{target:>7}  {r['layer']:>6}  {r['rank']:>6}  {r['out_features']:>12}  "
                  f"{cache_pct:>7.2f}%  {weight_pct:>7.2f}%")
            cache_true += r["rank"]
            cache_dense += r["out_features"]
            w_true += r["weight_true"]
            w_dense += r["weight_dense"]
        if rows:
            totals[target] = (cache_true, cache_dense, w_true, w_dense)

    if totals:
        print()
        ct = cd = wt = wd = 0
        for target, (a, b, c, d) in totals.items():
            print(f"  {target}: cache {100 * (1 - a / b):.2f}% smaller over "
                  f"{len(non_skip)} compressed layers, weights {100 * (1 - c / d):.2f}% smaller")
            ct += a; cd += b; wt += c; wd += d
        print(f"  K+V   : cache {100 * (1 - ct / cd):.2f}% smaller, "
              f"weights {100 * (1 - wt / wd):.2f}% smaller "
              f"(compressed layers only -- skip layers {args.skip_layers} are uncompressed and "
              f"still count in the whole-model cache footprint)")


if __name__ == "__main__":
    main()
