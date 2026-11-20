# Agent prefill energy: does prefix caching make it linear?

Final project for **Introduction to Computer Engineering, Fall 2026 (Indiana University)**. Author: Sarrthak Tripathi.

An AI agent calls the same model many times, and each call carries the whole conversation so far. Without prefix (KV) caching, every call recomputes that growing context, so prefill work grows **quadratically** with the number of turns. With caching, only the new tokens are processed, so it grows **linearly**. This repo measures whether **GPU energy** actually follows that token count on a laptop GPU.

It builds on Fernandez et al., *Energy Considerations of Large Language Model Inference and Efficiency Optimizations*, ACL 2025.

## The prediction being tested

A task has `T` model calls. Before each call a tool observation of `Δ` tokens is appended; each call then generates `g` tokens, which are appended too. `C0` is the initial prompt.

| | prefill tokens per task |
|---|---|
| cache off | `T·C0 + Δ·T(T+1)/2 + g·T(T−1)/2` |
| cache on  | `C0 + T·Δ` |

With `C0 = 2000`, `Δ = 500`, `g = 0` the predicted cache-off / cache-on token ratio is 3.9×, 6.8×, 12.1× and 22.3× at T = 5, 10, 20 and 40. The question is whether the **energy** ratio matches. Fixed per-call overheads on a small GPU could flatten it.

> These ratios come from an analytical model derived with an LLM (Claude). They are **predictions**, not measurements, until the runs below confirm or refute them.

## Setup

Needs an NVIDIA GPU with a recent driver and Python 3.10+.

```bash
# 1. PyTorch with CUDA: pick the command for your OS/CUDA at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. everything else
pip install -r requirements.txt
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB in float16), which fits an 8 GB laptop GPU with room for a 23k-token context. It downloads from Hugging Face on first run.

## Run

```bash
# 0. check the harness (CPU, tiny random model, ~10 s, no GPU or download)
pytest -q

# 1. baseline: energy per token should fall with batch size (Fernandez et al. trend)
python run_baseline.py --batch 1 4 16 --repeats 5

# 2. main experiment
python run_agent_energy.py --turns 5 10 20 40 --cache on off --repeats 5

# 3. plots and table
python plot_results.py
```

Add `--smi-check` to step 2 to log an independent energy estimate from `nvidia-smi` alongside NVML. `--gen 0` reproduces the slide formula exactly.

Outputs in `results/`:

| file | contents |
|---|---|
| `agent_runs.csv` | one row per task: prefill tokens (measured and predicted), energy (raw and idle-subtracted), time |
| `agent_runs_calls.csv` | one row per model call |
| `agent_runs.json` | model, GPU, driver-side meter mode, all arguments |
| `baseline.csv` | energy per generated token by batch size |
| `prefill_vs_turns.png`, `ratio_vs_turns.png`, `baseline.png` | figures |
| `summary.csv`, `ratios.csv` | aggregated numbers behind the figures |

## How the measurement works

* **Agent loop** (`agent_energy/loop.py`): synthetic, seeded token ids, so no tokenizer is needed and both modes see identical tool observations. Cache on keeps the Hugging Face KV cache across calls; cache off drops it, like a stateless API call.
* **Energy** (`agent_energy/energy.py`): NVML's cumulative energy counter when the GPU exposes it; otherwise NVML power samples at 50 Hz, integrated. GPU board energy only.
* **Idle subtraction**: idle power is measured for 5 s before the sweep; `*_net` columns subtract `idle_W × duration`.
* **Order**: all (T, cache, repeat) jobs are shuffled so thermal drift does not favour one mode.

## Verification checks

1. **Token counts match the formula exactly.** Every row records `tokens_match`; `tests/test_loop.py` checks it for several settings.
2. **Caching does not change outputs.** With greedy decoding, cache on and cache off generate identical tokens (`test_cache_does_not_change_outputs`), so the modes do the same work and differ only in recomputation.
3. **Idle power subtracted**, raw values kept.
4. **NVML cross-checked** with `nvidia-smi` (`--smi-check`).
5. **Run-to-run spread** reported as error bars over repeats.

## Simplifications

Synthetic tool outputs (no real tools or network); fixed output length per call; batch size 1; GPU energy only (CPU, DRAM and tool-wait idle are excluded, so results are a lower bound); a 0.5B model on a laptop GPU, so compare **trends and ratios**, not absolute joules. Very short phases can fall below the energy counter's update interval; compare per-task totals over several repeats.

## Labelling of results

Every number in the accompanying slides is marked as **measured** (from this code), **paper-reported** (Fernandez, Kim, Luo et al.), or **LLM-predicted** (the analytical model above).

## Use of LLMs

Claude (Anthropic, Claude Opus 5.5) helped write the literature review, the analytical model, the slides and first drafts of this code. The author reviewed the code, defined the verification checks, and ran the experiments.
