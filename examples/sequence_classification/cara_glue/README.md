# CaRA GLUE benchmark

Paper-ready GLUE sweep for **CaRA** (Cayley Rotational Adaptation) on
`microsoft/deberta-v3-base`. Mirrors the PSOFT protocol
(arXiv [2505.11235](https://arxiv.org/abs/2505.11235),
[code](https://github.com/fei407/PSOFT)) so numbers are directly comparable.

Default sweep: `{cola, stsb, rte, mrpc, sst2, qnli} × {42, 1234, 24}` = 18 runs
at `r=8` with CaRA noise defaults (`noise_alpha=0.01`, `noise_step_interval=5`).

## Setup

**For a guided Korean walkthrough, see [`ENV_SETUP_KO.md`](./ENV_SETUP_KO.md).**

Quick setup (RTX 50xx / CUDA 12.8 verified):

```bash
# From the peft repo root
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r examples/sequence_classification/cara_glue/requirements.txt
uv pip install --python .venv/bin/python -e .
```

Important:
- RTX 50xx (sm_120) requires the `cu128` PyTorch build (`+cu128`). The default `pip install torch` ships `cu124` which **does not run on RTX 5090**.
- `microsoft/deberta-v3-base` requires `sentencepiece` (included in `requirements.txt`).
- DeBERTaV3 has a known overflow with `bf16` attention masks — train in **fp32** (default). Do not pass `--bf16` / `--fp16`.

## Verify first (recommended)

Single 1-epoch smoke test (~2 min, sanity-checks trainable params and I/O):

```bash
python examples/sequence_classification/cara_glue/run_glue_cara.py \
  --task mrpc --seed 42 \
  --num_train_epochs 1 \
  --output_dir results/smoke/mrpc_seed42
```

Expected: `print_trainable_parameters()` reports trainable params in the ~0.3M
range and `result.json` is written.

Then reproduce two already-reported cells from the user's table:

```bash
python examples/sequence_classification/cara_glue/run_glue_cara.py \
  --task cola --seed 42 \
  --output_dir results/verify/cola_seed42

python examples/sequence_classification/cara_glue/run_glue_cara.py \
  --task mrpc --seed 42 \
  --output_dir results/verify/mrpc_seed42
```

Acceptance targets: CoLA Matthews ≥ 69.5 (target 70.4) and MRPC F1 ≥ 90.7
(target 91.18). If either misses by > 1.0, check the target-module names
(`query_proj`/`value_proj` for DeBERTaV3) and that the classifier head is in
`modules_to_save` before tuning learning rate.

## Full sweep

```bash
python examples/sequence_classification/cara_glue/sweep.py \
  --base_output_dir results/cara_r8 --bf16
```

The sweep is resumable: runs whose `result.json` already exists are skipped.
Each run writes `train.log` next to `result.json` for later inspection.

Partial sweeps:

```bash
# Only fill MRPC + RTE for seeds 1234 and 24
python examples/sequence_classification/cara_glue/sweep.py \
  --tasks mrpc rte --seeds 1234 24
```

## Aggregate into a paper table

```bash
python examples/sequence_classification/cara_glue/aggregate.py \
  --results_dir results/cara_r8
```

Produces:
- `results/cara_r8/summary.csv` — flat per-run metrics.
- `results/cara_r8/table.md` — markdown table mirroring the screenshot layout
  (rows = seeds, columns = tasks, final column = mean across tasks); includes
  `mean` and `std` rows per task.
- `results/cara_r8/table.tex` — LaTeX `tabular` for direct paper insertion.

## Task-level metric choices

| Task   | Metric reported                                 |
| ------ | ----------------------------------------------- |
| CoLA   | Matthews correlation                            |
| STS-B  | mean(Pearson, Spearman) — `combined_score`      |
| MRPC   | F1                                              |
| RTE    | Accuracy                                        |
| SST-2  | Accuracy                                        |
| QNLI   | Accuracy                                        |

These match the GLUE / LoRA / PSOFT conventions. Best-epoch checkpoint is
selected via `metric_for_best_model` on the validation set
(`load_best_model_at_end=True`).
