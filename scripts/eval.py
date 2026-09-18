"""Evaluate a NovaKV compressed-KV-cache checkpoint (or the dense baseline, through the exact
same harness, for an apples-to-apples reference).

Supports:
  - Perplexity on WikiText-2, C4, PTB
  - lm-eval tasks: PIQA, WinoGrande, ARC, HellaSwag, OpenBookQA, HumanEval, MBPP, MMLU
  - Long-context: LongBench, RULER

`--weights` points at the released compressed checkpoint (full or KV-only delta -- see
scripts/export_delta.py). It is loaded as-is: the file's own shapes are the deployed shapes,
nothing is recomputed at load time. Omit `--weights` to evaluate the untouched base model.

Example
-------
  # Compressed checkpoint, full evaluation suite
  python scripts/eval.py --model meta-llama/Llama-3.1-8B-Instruct \\
    --weights novakv_delta.pt \\
    --ppl --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag,humaneval_instruct,mbpp_instruct,mmlu \\
    --confirm-unsafe-code --apply-chat-template --longbench --ruler --output eval.json

  # Dense baseline through the same harness
  python scripts/eval.py --model meta-llama/Llama-3.1-8B-Instruct \\
    --ppl --longbench --ruler --output eval_dense.json
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from novakv.model import load_truncated_model, set_model_mode


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def _get_ppl_data(name: str, tokenizer, seqlen: int):
    if "wikitext2" in name:
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    if "c4" in name:
        class _Wrap:
            def __init__(self, ids): self.input_ids = ids
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        enc = tokenizer(" ".join(data[:1100]["text"]), return_tensors="pt")
        return _Wrap(enc.input_ids[:, : 256 * seqlen])
    if "ptb" in name:
        data = load_dataset("ptb_text_only", "penn_treebank", split="test")
        return tokenizer("\n\n".join(data["sentence"]), return_tensors="pt")
    raise ValueError(f"Unknown PPL dataset: {name}")


@torch.no_grad()
def evaluate_ppl(model, tokenizer, datasets: str, seqlen: int = 2048, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    results = {}
    for name in datasets.split(","):
        name = name.strip()
        loader = _get_ppl_data(name, tokenizer, seqlen)
        enc = loader.input_ids
        nsamples = enc.numel() // seqlen
        nlls = []
        for i in tqdm(range(nsamples), desc=f"PPL [{name}]"):
            batch = enc[:, i * seqlen : (i + 1) * seqlen].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = enc[:, i * seqlen : (i + 1) * seqlen][:, 1:].to(device)
            loss = nn.CrossEntropyLoss()(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            )
            nlls.append(loss.float() * seqlen)
        avg_loss = torch.stack(nlls).sum() / (len(nlls) * seqlen)
        ppl = torch.exp(avg_loss).item()
        results[name] = {"loss": avg_loss.item(), "ppl": ppl}
        print(f"  {name:12s}  seqlen={seqlen}  loss={avg_loss.item():.4f}  ppl={ppl:.2f}")
    return results


# ---------------------------------------------------------------------------
# lm-eval evaluation
# ---------------------------------------------------------------------------

def evaluate_lmeval(model, tokenizer, tasks: str, batch_size: int,
                    max_length: int = None, model_name: str = "", limit: int = None,
                    run_label: str = "", confirm_unsafe_code: bool = False,
                    log_samples: bool = False, apply_chat_template: bool = False):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table

    kwargs = {"pretrained": model, "tokenizer": tokenizer, "add_bos_token": False,
              "batch_size": batch_size}
    if max_length is not None:
        kwargs["max_length"] = max_length
        model.config.max_position_embeddings = max_length

    lm_obj = HFLM(**kwargs)
    # RULER's niah tasks need a "pretrained" string (not the live model object) to build their
    # own tokenizer for synthetic needle-in-haystack data generation, independent of the
    # already-instantiated HFLM. simple_evaluate only builds its own metadata-carrying
    # TaskManager when none is passed in -- since an explicit task_manager is passed below,
    # metadata must be given here at construction time instead.
    task_manager = TaskManager(metadata={"pretrained": model_name} if model_name else None)

    task_list = [t.strip() for t in tasks.split(",")]
    print(f"Running lm-eval tasks: {task_list}")
    with torch.no_grad():
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=task_list,
            task_manager=task_manager,
            log_samples=log_samples,
            limit=limit,
            # Code-execution tasks (humaneval, mbpp) refuse to run without this -- lm-eval-
            # harness actually executes the model's generated code to check test cases, so this
            # is an explicit opt-in, not a default.
            confirm_run_unsafe_code=confirm_unsafe_code,
            # Default (False) sends raw, non-chat-wrapped prompts even for an Instruct model --
            # an unwrapped prompt may get a conversational reply instead of a bare completion,
            # which the harness can't parse/execute (suspected cause of HumanEval's outsized
            # collapse under compression). Opt-in, not default: flipping this also changes the
            # already-established zero-shot numbers, which were measured without it.
            apply_chat_template=apply_chat_template,
        )
    print(make_table(results))
    if run_label:
        print(f"[weights: {run_label}]")
    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a NovaKV compressed-KV-cache checkpoint.")
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    p.add_argument("--weights", default=None,
                   help="Path to the released NovaKV checkpoint (.pt), full or KV-only delta. "
                        "Omit to evaluate the raw uncompressed baseline model, for an "
                        "apples-to-apples reference through the same eval harness.")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31],
                   help="Attention layers left uncompressed in the checkpoint. Must match the "
                        "checkpoint being loaded.")

    # Zero-shot accuracy
    p.add_argument("--tasks", default=None,
                   help="Comma-separated lm-eval tasks, e.g. piqa,winogrande,arc_easy")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--confirm-unsafe-code", action="store_true",
                   help="Required by lm-eval-harness for code-execution tasks in --tasks "
                        "(humaneval, mbpp) -- it actually runs the model's generated code to "
                        "check test cases, so this is an explicit opt-in.")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="Wrap prompts in the tokenizer's chat template before sending to "
                        "lm-eval-harness -- only applied to the --tasks path (zero-shot/"
                        "humaneval/mbpp), not --longbench/--ruler. Off by default to keep "
                        "existing zero-shot numbers comparable across configs.")

    # Perplexity
    p.add_argument("--ppl", action="store_true", help="Run perplexity evaluation")
    p.add_argument("--ppl-datasets", default="wikitext2,c4",
                   help="Comma-separated PPL datasets: wikitext2, c4, ptb")
    p.add_argument("--ppl-seqlen", type=int, default=2048)

    # Long-context benchmarks
    p.add_argument("--longbench-tasks", default=None,
                    help="Restrict --longbench to a comma-separated subset (e.g. "
                         "'longbench_hotpotqa,longbench_narrativeqa') for a faster targeted "
                         "check -- reuses the exact same evaluate_lmeval call (long_batch_size, "
                         "max_length) as the full --longbench run, just fewer tasks.")
    p.add_argument("--longbench", action="store_true", help="Run LongBench tasks")
    p.add_argument("--ruler", action="store_true", help="Run RULER tasks")
    p.add_argument("--long-batch-size", type=int, default=4, help="Batch size for long-context tasks")
    p.add_argument("--max-length", type=int, default=31500, help="Max context length for long-context evaluations")
    p.add_argument("--eval-limit", type=int, default=None,
                    help="Cap docs/task (lm_eval's own 'limit' kwarg) for a fast smoke test of "
                         "the RULER/LongBench code path -- not for real numbers.")
    p.add_argument("--log-samples", action="store_true",
                   help="Save raw per-example generations into --output's JSON, under each "
                        "task's 'samples' key -- off by default (bloats the output a lot on a "
                        "full run), meant for small --eval-limit diagnostic runs.")

    p.add_argument("--output", default=None, help="JSON file to write all results (optional)")
    p.add_argument("--cuda-devices", default="0", help="CUDA_VISIBLE_DEVICES string")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print("No --weights given: evaluating raw uncompressed baseline.")

    model.eval()
    model.config.use_cache = True

    all_results = {}

    # -- Perplexity ----------------------------------------------------------
    if args.ppl:
        print("\n=== Perplexity Evaluation ===")
        for seqlen in [1024, 2048]:
            res = evaluate_ppl(model, tokenizer, args.ppl_datasets, seqlen=seqlen, device=get_device())
            all_results[f"ppl_seqlen{seqlen}"] = res
        if args.weights:
            print(f"[weights: {args.weights}]")

    # -- Zero-shot accuracy --------------------------------------------------
    if args.tasks:
        print("\n=== Zero-shot Accuracy ===")
        res = evaluate_lmeval(
            model, tokenizer, tasks=args.tasks, batch_size=args.batch_size,
            model_name=args.model, limit=args.eval_limit, run_label=args.weights,
            confirm_unsafe_code=args.confirm_unsafe_code, log_samples=args.log_samples,
            apply_chat_template=args.apply_chat_template,
        )
        all_results["lmeval"] = res

    # -- LongBench -----------------------------------------------------------
    if args.longbench:
        longbench_tasks = args.longbench_tasks or (
            "longbench_hotpotqa,longbench_qasper,longbench_triviaqa,"
            "longbench_multi_news,longbench_trec,longbench_lcc,"
            "longbench_samsum,longbench_narrativeqa,longbench_qmsum,"
            "longbench_vcsum,longbench_dureader"
        )
        print("\n=== LongBench ===")
        res = evaluate_lmeval(
            model, tokenizer, tasks=longbench_tasks, batch_size=args.long_batch_size,
            max_length=args.max_length, model_name=args.model, limit=args.eval_limit,
            run_label=args.weights, log_samples=args.log_samples,
        )
        all_results["longbench"] = res

    # -- RULER ---------------------------------------------------------------
    if args.ruler:
        print("\n=== RULER ===")
        # The "ruler" group includes ruler_qa_hotpot, which downloads HotpotQA dev data from an
        # external host that's been unreachable -- run the other 12 RULER subtasks explicitly
        # instead of the group name, so one dead external dependency doesn't block the rest.
        ruler_tasks = (
            "niah_single_1,niah_single_2,niah_single_3,"
            "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
            "niah_multiquery,niah_multivalue,"
            "ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad"
        )
        res = evaluate_lmeval(
            model, tokenizer, tasks=ruler_tasks, batch_size=args.long_batch_size,
            max_length=args.max_length, model_name=args.model, limit=args.eval_limit,
            run_label=args.weights, log_samples=args.log_samples,
        )
        all_results["ruler"] = res

    # -- Save results --------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
