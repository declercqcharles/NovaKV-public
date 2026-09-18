#!/usr/bin/env python3
"""Shrink a full NovaKV checkpoint down to the KV-only DELTA that is actually distributed.

A full NovaKV checkpoint is a complete state_dict: it carries every weight of the base model
(embeddings, MLPs, norms, lm_head, q_proj/o_proj, the uncompressed skip layers) plus the
compressed K/V factors. Almost all of that is bit-identical to the Hugging Face base model that
`load_truncated_model` already loads via `from_pretrained` before touching the checkpoint. The
delta keeps ONLY the keys that `load_truncated_model` genuinely consumes -- the `VS`/`U` pair of
`k_proj` and `v_proj` for every non-skip layer, i.e. exactly `required_checkpoint_keys()` -- and
drops the rest. `load_truncated_model` loads it with `strict=False`, so the dropped weights stay
as `from_pretrained` provided them.

`required_checkpoint_keys` is given the input state_dict, so the exact K key names it asks for
follow whichever layout this particular checkpoint uses (joint `k_proj.U.weight`, or grouped
`k_proj.U` plus `k_proj.inv_perm`); nothing here needs to know which one it is holding.

This is the only script in the public tree that reads a full checkpoint, and it exists so that
whoever holds one can verify the published delta against it. It reconstructs nothing: it only
selects a subset of tensors that already exist in its input.

A caveat worth stating plainly: the delta is correct only if every dropped tensor really is
identical to the base model's. Pass --base-model to check that exactly (it loads the base model
and compares each dropped tensor); without it the script assumes it and says so out loud.
`--include-qo` additionally keeps q_proj/o_proj for the non-skip layers, for a checkpoint whose
q/o projections were not left untouched.

Usage:
  python scripts/export_delta.py --input novakv_full.pt --output novakv_delta.pt
  python scripts/export_delta.py --input novakv_full.pt --output novakv_delta.pt \\
      --base-model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import os

import torch

from novakv.model import required_checkpoint_keys


def _qo_keys(num_layers):
    """q_proj/o_proj for EVERY layer, skip layers included. When a checkpoint was trained with
    q/o unfrozen, they drift from the base model everywhere -- the skip layers keep their K/V
    uncompressed but their q/o was still updated by the distillation, so restricting this to the
    non-skip layers would silently drop 6 real tensors."""
    keys = []
    for i in range(num_layers):
        for proj in ("q_proj", "o_proj"):
            keys.append(f"model.layers.{i}.self_attn.{proj}.weight")
    return keys


def main():
    p = argparse.ArgumentParser(description="Export the distributable KV-only NovaKV delta.")
    p.add_argument("--input", required=True, help="Full NovaKV checkpoint (.pt)")
    p.add_argument("--output", required=True, help="Destination for the delta (.pt)")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31],
                   help="Layers left uncompressed in the checkpoint.")
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--include-qo", action="store_true",
                   help="Also keep q_proj/o_proj of EVERY layer, skip layers included. Needed "
                        "whenever the checkpoint was trained with q/o unfrozen, in which case "
                        "--base-model reports them as differing. Adds ~2 GB in bf16.")
    p.add_argument("--base-model", default=None,
                   help="HuggingFace name or local path of the dense base model. If given, every "
                        "DROPPED tensor is compared against the base model's own weight and any "
                        "difference is reported as an error -- the rigorous check that the delta "
                        "loses nothing.")
    args = p.parse_args()

    print(f"Loading full checkpoint: {args.input}")
    sd = torch.load(args.input, map_location="cpu", weights_only=False)
    non_skip = [i for i in range(args.num_layers) if i not in set(args.skip_layers)]

    # The single source of truth for what must be kept -- the same function load_truncated_model
    # checks against at load time, so the two can never drift apart.
    keep_keys = list(required_checkpoint_keys(sd, args.num_layers, tuple(args.skip_layers)))
    if args.include_qo:
        keep_keys += _qo_keys(args.num_layers)

    absent = [k for k in keep_keys if k not in sd]
    if absent:
        raise KeyError(
            f"{args.input} does not contain {len(absent)} key(s) the delta must carry, e.g. "
            f"{absent[:5]}. Check --skip-layers / --num-layers against this checkpoint."
        )

    delta = {k: sd[k] for k in keep_keys}
    dropped = [k for k in sd if k not in set(keep_keys)]

    # Verification 1 (always): no key that load_truncated_model consumes may end up dropped.
    lib_required = set(required_checkpoint_keys(sd, args.num_layers, tuple(args.skip_layers)))
    consumed_but_dropped = sorted(lib_required & set(dropped))
    if consumed_but_dropped:
        raise RuntimeError(
            f"BUG: {len(consumed_but_dropped)} key(s) that load_truncated_model consumes would "
            f"be dropped, e.g. {consumed_but_dropped[:5]}"
        )
    missing_from_delta = sorted(lib_required - set(delta))
    if missing_from_delta:
        raise RuntimeError(
            f"BUG: the delta is missing {len(missing_from_delta)} key(s) that "
            f"load_truncated_model requires, e.g. {missing_from_delta[:5]}"
        )
    print(f"  verified: all {len(lib_required)} keys required by load_truncated_model are kept")

    # Verification 2 (optional): every dropped tensor is identical to the base model's.
    if args.base_model:
        from transformers import AutoModelForCausalLM
        print(f"Loading base model for the dropped-key check: {args.base_model}")
        base = AutoModelForCausalLM.from_pretrained(args.base_model, use_safetensors=True)
        base_sd = base.state_dict()
        differing, unknown = [], []
        for k in dropped:
            if k not in base_sd:
                unknown.append(k)
                continue
            a, b = sd[k], base_sd[k]
            if a.shape != b.shape or not torch.equal(a.to(torch.float32), b.to(torch.float32)):
                differing.append(k)
        if unknown:
            print(f"  WARNING: {len(unknown)} dropped key(s) have no counterpart in the base "
                  f"model and would simply vanish, e.g. {unknown[:5]}")
        if differing:
            print(f"  ERROR: {len(differing)} dropped key(s) DIFFER from the base model and "
                  f"would be silently lost, e.g. {differing[:5]}")
            if any(".q_proj." in k or ".o_proj." in k for k in differing):
                print("         some of them are q_proj/o_proj -- re-run with --include-qo")
            raise SystemExit(1)
        if not unknown:
            print(f"  verified: all {len(dropped)} dropped tensors are identical to "
                  f"{args.base_model}")
    else:
        print("  NOT verified: without --base-model this script assumes every dropped tensor is "
              "identical to the base model's. Run once with --base-model before publishing.")

    torch.save(delta, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nWrote {args.output}")
    print(f"  keys kept    : {len(delta)}  ({len(non_skip)} compressed layers"
          f"{', q_proj/o_proj included' if args.include_qo else ''})")
    print(f"  keys dropped : {len(dropped)}")
    print(f"  file size    : {size_mb:.1f} MB "
          f"(input: {os.path.getsize(args.input) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
