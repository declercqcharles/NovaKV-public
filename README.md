# NovaKV

Post-hoc low-rank compression of the KV cache for frozen LLMs. Load a compressed checkpoint, the
cache gets ~45% smaller. No retraining of the base model.

Inference and evaluation code only. Training and calibration code is not included.

## Method and credits

NovaKV treats K and V as the same low-rank problem at two different group sizes, and picks the
right end of that axis for each.

**K** gets a single SVD per layer, shared across all KV heads, with its rank chosen by a trained
differentiable soft threshold and initialised from an activation-whitened basis. The soft-threshold
rank selection is from **STAR-KV** — *STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding
for Adaptive Rank Control*, arXiv [2606.08382](https://arxiv.org/abs/2606.08382), ICML 2026.
Sharing one basis across every head, and whitening its initialisation against real activations,
are not from that work: their released code decomposes each head independently.

**V** gets the closed-form **Offline Value Calibration** of **ReCalKV**, at a fixed rank, weighted
by the Gram matrix of real activations — and is frozen from the first step while K trains.

The asymmetry is the point. Every configuration that applied one mechanism to both projections
lost to this pairing.

Parts of `novakv/attention.py` derive from Hugging Face `transformers` (Apache 2.0). STAR-KV's
repository carries no license file; no code is redistributed from it here.

## Results

Measured on the physically truncated checkpoints, reproducible with the commands below.

### Llama-3.1-8B-Instruct

| | Dense | NovaKV |
|---|---:|---:|
| PPL WikiText-2 @2048 | 7.22 | 7.57 |
| PPL WikiText-2 @1024 | 8.05 | 8.45 |
| PPL C4 @2048 | 11.40 | 12.78 |
| PPL C4 @1024 | 11.55 | 13.24 |
| RULER | 97.60% | 96.87% |
| LongBench | 40.12% | 37.91% |
| MMLU | 63.23% | 59.00% |
| HumanEval-instruct | 69.51% | 60.37% |
| MBPP-instruct | 60.40% | 50.00% |
| Cache reduction | — | 44.5% |

### Mistral-7B-Instruct-v0.3

| | Dense | NovaKV |
|---|---:|---:|
| PPL WikiText-2 @2048 | 5.50 | 5.76 |
| PPL WikiText-2 @1024 | 6.14 | 6.44 |
| PPL C4 @2048 | 8.85 | 9.47 |
| PPL C4 @1024 | 9.37 | 10.01 |
| RULER | 97.12% | 96.35% |
| LongBench | 43.44% | 39.72% |
| MMLU | 59.98% | 57.44% |
| HumanEval-instruct | 42.07% | 32.93% |
| MBPP-instruct | 42.60% | 35.20% |
| Cache reduction | — | 45.4% |

### Cache memory

Real `torch.cuda.memory_allocated()` delta after a prefill, not estimated from ranks.

| | Dense | NovaKV | Reduction |
|---|---:|---:|---:|
| Llama-3.1-8B-Instruct | 131.07 KB/token | 72.70 KB/token | 44.5% |
| Mistral-7B-Instruct-v0.3 | 131.07 KB/token | 71.51 KB/token | 45.4% |

The weight file itself shrinks by only ~6%. K and V projections are a small share of an 8B model;
the saving is in the cache, which grows with context length and batch size.

### Latency

Single-stream decode, ms/token, measured in one session at matched prompt lengths. Dense is the
stock Hugging Face path; NovaKV is the shipped path, a fixed-shape cache with a captured CUDA
graph. Correctness is gated on an argmax match against the dense model at every decode step.

Llama-3.1-8B-Instruct

| context | Dense | NovaKV | |
|---:|---:|---:|---:|
| 256 | 27.00 | 10.51 | 2.57× |
| 2048 | 27.19 | 12.79 | 2.13× |
| 8192 | 27.39 | 18.77 | 1.46× |

Mistral-7B-Instruct-v0.3

| context | Dense | NovaKV | |
|---:|---:|---:|---:|
| 256 | 27.58 | 10.29 | 2.68× |
| 2048 | 27.22 | 12.50 | 2.18× |
| 8192 | 27.23 | 18.43 | 1.48× |

Dense is flat across context: at batch 1 it is bound by CPU dispatch, not by the cache. The
advantage therefore narrows as context grows. The dense baseline is not itself CUDA-graph
captured.

## Usage

Weights are published as a delta: only the tensors that differ from the base model (~2.4 GB). The
base model is fetched from Hugging Face and the delta is loaded on top, giving a model numerically
identical to the full 15.9 GB checkpoint.

| File | Base model | Repo |
|---|---|---|
| `novakv_llama31_8b_joint_delta.pt` | Llama-3.1-8B-Instruct | [Llama-3.1-NovaKV](https://huggingface.co/declercqcharles/Llama-3.1-NovaKV) |
| `novakv_mistral7b_v03_joint_delta.pt` | Mistral-7B-Instruct-v0.3 | [Mistral-7B-NovaKV](https://huggingface.co/declercqcharles/Mistral-7B-NovaKV) |

```sh
pip install -r requirements.txt
hf download declercqcharles/Llama-3.1-NovaKV novakv_llama31_8b_joint_delta.pt --local-dir .
```

```sh
PYTHONPATH=. python scripts/eval.py --model meta-llama/Llama-3.1-8B-Instruct \
  --weights novakv_llama31_8b_joint_delta.pt \
  --ppl --ppl-datasets wikitext2,c4 --ppl-seqlen 2048

PYTHONPATH=. python scripts/eval.py --model meta-llama/Llama-3.1-8B-Instruct \
  --weights novakv_llama31_8b_joint_delta.pt --longbench --ruler

PYTHONPATH=. python scripts/measure_cache_ram.py --model meta-llama/Llama-3.1-8B-Instruct \
  --mode novakv --weights novakv_llama31_8b_joint_delta.pt \
  --context-lengths 2048 8192 16384
```

Swap `--weights` and `--model` for the Mistral checkpoint. The loader reads each layer's rank from
the file.

## Notes

- `measure_cache_ram.py` needs several `--context-lengths` in one call. The first forward of a
  process pays a one-time CUDA allocation cost that inflates a single-length reading by ~5%. Only
  the last lengths are trustworthy.
- RULER is averaged over 12 of its 13 subtasks. `ruler_qa_hotpot` fetches its data from a host
  that has been unreachable since July 2026. The exclusion is applied to the dense baseline and to
  every compressed model alike, so the comparisons here are consistent, but not directly
  comparable to published RULER numbers over all 13.
- The latency table uses the shipped path (`test_static_cache_graph.py`). Run the same model
  through `latency.py` instead, which uses an eager dynamic cache, and it is slower than dense —
  that path re-expands the whole cached history every decode step and is not what this ships.
- Mistral's optional sliding-window attention is not implemented. Harmless with
  `sliding_window: null`, which is the case here.

## License

MIT for the code. Weights are derived from Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 and
remain under their respective upstream licenses.
