"""Walk a cara_glue results directory and produce the paper-ready table.

Emits three artifacts next to (or at the provided --out):
  summary.csv  — flat table of (task, seed, primary_metric, score, trainable_params)
  table.md     — markdown table matching the screenshot layout (rows = seeds, cols = tasks)
  table.tex    — LaTeX tabular for paper insertion

Example:
    python aggregate.py --results_dir results/cara_r8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


TASK_ORDER = ["cola", "stsb", "rte", "mrpc", "sst2", "qnli"]
TASK_LABEL = {
    "cola": "CoLA",
    "stsb": "STS-B",
    "rte": "RTE",
    "mrpc": "MRPC",
    "sst2": "SST2",
    "qnli": "QNLI",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/cara_r8")
    p.add_argument("--method_label", default="CaRA (r=8)")
    p.add_argument("--params_label", default="0.3M")
    return p.parse_args()


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for result_path in sorted(results_dir.glob("*/seed_*/result.json")):
        try:
            runs.append(json.loads(result_path.read_text()))
        except Exception as e:
            print(f"[aggregate] skipped {result_path}: {e}")
    return runs


def fmt_score(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x * 100:.2f}"


def write_csv(runs: list[dict], path: Path) -> None:
    fields = ["task", "seed", "primary_metric", "primary_score", "trainable_params", "lr", "epochs", "bs", "max_len"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            hp = r.get("hyperparams", {})
            w.writerow({
                "task": r["task"],
                "seed": r["seed"],
                "primary_metric": r["primary_metric"],
                "primary_score": r["primary_score"],
                "trainable_params": r.get("trainable_params"),
                "lr": hp.get("lr"),
                "epochs": hp.get("epochs"),
                "bs": hp.get("per_device_train_batch_size"),
                "max_len": hp.get("max_seq_length"),
            })


def pivot(runs: list[dict]) -> tuple[list[int], dict[int, dict[str, float]]]:
    seeds = sorted({r["seed"] for r in runs})
    by_seed: dict[int, dict[str, float]] = {s: {} for s in seeds}
    for r in runs:
        by_seed[r["seed"]][r["task"]] = r["primary_score"]
    return seeds, by_seed


def write_markdown(runs: list[dict], path: Path, method_label: str, params_label: str) -> None:
    seeds, by_seed = pivot(runs)
    header_tasks = [t for t in TASK_ORDER if any(t in by_seed[s] for s in seeds)]

    lines = []
    head = "|  | #params |  | " + " | ".join(TASK_LABEL[t] for t in header_tasks) + " | avg |"
    sep = "| --- | --- | --- | " + " | ".join(["---"] * len(header_tasks)) + " | --- |"
    lines.append(head)
    lines.append(sep)

    per_task_scores: dict[str, list[float]] = {t: [] for t in header_tasks}
    row_avgs: list[float] = []

    for i, seed in enumerate(seeds):
        method_cell = method_label if i == 0 else ""
        params_cell = params_label if i == 0 else ""
        seed_cell = f"seed{i + 1}({seed})"
        row_vals = [by_seed[seed].get(t) for t in header_tasks]
        filled = [v for v in row_vals if v is not None]
        avg = statistics.mean(filled) if filled else None
        if avg is not None:
            row_avgs.append(avg)
        for t, v in zip(header_tasks, row_vals):
            if v is not None:
                per_task_scores[t].append(v)
        row_str = (
            f"| {method_cell} | {params_cell} | {seed_cell} | "
            + " | ".join(fmt_score(v) for v in row_vals)
            + f" | {fmt_score(avg)} |"
        )
        lines.append(row_str)

    mean_row_vals = [statistics.mean(per_task_scores[t]) if per_task_scores[t] else None for t in header_tasks]
    overall = statistics.mean([v for v in mean_row_vals if v is not None]) if any(mean_row_vals) else None
    lines.append(
        "|  |  | **mean** | "
        + " | ".join(fmt_score(v) for v in mean_row_vals)
        + f" | {fmt_score(overall)} |"
    )

    if len(seeds) >= 2:
        std_row_vals = [
            statistics.stdev(per_task_scores[t]) if len(per_task_scores[t]) >= 2 else None
            for t in header_tasks
        ]
        lines.append(
            "|  |  | **std** | "
            + " | ".join(fmt_score(v) if v is not None else "" for v in std_row_vals)
            + " |  |"
        )

    path.write_text("\n".join(lines) + "\n")


def write_results_doc(
    runs: list[dict],
    path: Path,
    method_label: str,
    params_label: str,
    table_md: str,
) -> None:
    seeds_present = sorted({r["seed"] for r in runs})
    tasks_present = sorted({r["task"] for r in runs}, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99)
    first = runs[0]
    base_model = first.get("model", "microsoft/deberta-v3-base")
    rank = first.get("r", 8)
    noise_alpha = first.get("noise_alpha")
    noise_step_interval = first.get("noise_step_interval")
    cara_params = first.get("cara_params")
    trainable_params = first.get("trainable_params")

    hp_rows = []
    seen_hp_keys = set()
    for r in runs:
        key = (r["task"], json.dumps(r.get("hyperparams", {}), sort_keys=True))
        if key in seen_hp_keys:
            continue
        seen_hp_keys.add(key)
        hp = r.get("hyperparams", {})
        hp_rows.append(
            f"| {TASK_LABEL.get(r['task'], r['task'])} | {hp.get('lr')} | "
            f"{hp.get('per_device_train_batch_size')} | {hp.get('epochs')} | "
            f"{hp.get('max_seq_length')} | {hp.get('warmup_ratio')} |"
        )

    metric_rows = []
    seen_metric_keys = set()
    for r in runs:
        if r["task"] in seen_metric_keys:
            continue
        seen_metric_keys.add(r["task"])
        metric_rows.append(f"| {TASK_LABEL.get(r['task'], r['task'])} | `{r['primary_metric']}` |")

    lines = []
    lines.append(f"# {method_label} on GLUE")
    lines.append("")
    lines.append(f"Paper-ready benchmark of **{method_label}** on GLUE using `{base_model}`.")
    lines.append("Fine-tuning protocol mirrors PSOFT (arXiv 2505.11235).")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Base model:** `{base_model}`")
    lines.append(f"- **Adapter:** CaRA (Cayley Rotational Adaptation), rank `r={rank}`")
    lines.append(f"- **CaRA noise:** `alpha={noise_alpha}`, `step_interval={noise_step_interval}`")
    lines.append(f"- **Target modules:** `query_proj`, `value_proj`")
    lines.append(f"- **Modules also trained:** classifier head + pooler (randomly initialized at load time)")
    if cara_params is not None:
        lines.append(f"- **CaRA params:** {cara_params:,} (~{cara_params / 1e6:.2f}M) — this is what appears as `#params` in comparison tables")
    if trainable_params is not None:
        lines.append(f"- **All trainable params (incl. pooler+classifier):** {trainable_params:,} (~{trainable_params / 1e6:.2f}M)")
    lines.append(f"- **Seeds evaluated:** {seeds_present}")
    lines.append(f"- **Tasks evaluated:** {[TASK_LABEL.get(t, t) for t in tasks_present]}")
    lines.append("- **Trainer:** HuggingFace `Trainer`, AdamW, linear LR schedule, warmup 10%, `load_best_model_at_end=True`")
    eval_mode = first.get("eval_mode", "full_val")
    if eval_mode == "psoft_val_test":
        lines.append(
            "- **Evaluation protocol:** PSOFT-style split — the original GLUE validation set is shuffled with "
            "the training seed's numpy RNG and partitioned 50/50 into val (best-checkpoint selection) and "
            "test (final reported score). Matches the protocol in arXiv 2505.11235."
        )
    else:
        lines.append(
            "- **Evaluation protocol:** full validation — trained checkpoint with highest validation score "
            "is loaded at end (`load_best_model_at_end=True`) and the same full validation set is reported. "
            "Slightly optimistic vs. a held-out test split."
        )
    lines.append("")
    lines.append("## Per-task hyperparameters")
    lines.append("")
    lines.append("| Task | lr | batch | epochs | max_len | warmup |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.extend(hp_rows)
    lines.append("")
    lines.append("## Reported metric per task")
    lines.append("")
    lines.append("| Task | Metric |")
    lines.append("| --- | --- |")
    lines.extend(metric_rows)
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(table_md.strip())
    lines.append("")
    lines.append("Values are percent. Best-epoch checkpoint on the validation set, selected via "
                 "`metric_for_best_model`. STS-B uses mean of Pearson and Spearman (`combined_score`).")
    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python examples/sequence_classification/cara_glue/sweep.py --seeds {' '.join(str(s) for s in seeds_present)} \\")
    lines.append(f"  --base_output_dir results/cara_r8")
    lines.append(f"python examples/sequence_classification/cara_glue/aggregate.py --results_dir results/cara_r8")
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines))


def write_latex(runs: list[dict], path: Path, method_label: str, params_label: str) -> None:
    seeds, by_seed = pivot(runs)
    header_tasks = [t for t in TASK_ORDER if any(t in by_seed[s] for s in seeds)]
    n_cols = 3 + len(header_tasks) + 1
    col_spec = "l l l " + " ".join(["c"] * len(header_tasks)) + " c"

    lines = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    lines.append(
        "Method & \\#params & Seed & "
        + " & ".join(TASK_LABEL[t] for t in header_tasks)
        + " & Avg \\\\"
    )
    lines.append("\\midrule")

    per_task_scores: dict[str, list[float]] = {t: [] for t in header_tasks}
    for i, seed in enumerate(seeds):
        method_cell = method_label if i == 0 else ""
        params_cell = params_label if i == 0 else ""
        seed_cell = f"seed{i + 1}({seed})"
        row_vals = [by_seed[seed].get(t) for t in header_tasks]
        filled = [v for v in row_vals if v is not None]
        avg = statistics.mean(filled) if filled else None
        for t, v in zip(header_tasks, row_vals):
            if v is not None:
                per_task_scores[t].append(v)
        cells = [method_cell, params_cell, seed_cell] + [fmt_score(v) for v in row_vals] + [fmt_score(avg)]
        lines.append(" & ".join(cells) + " \\\\")

    mean_row = [statistics.mean(per_task_scores[t]) if per_task_scores[t] else None for t in header_tasks]
    overall = statistics.mean([v for v in mean_row if v is not None]) if any(mean_row) else None
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{3}{l}{Mean} & "
        + " & ".join(fmt_score(v) for v in mean_row)
        + f" & {fmt_score(overall)} \\\\"
    )
    if len(seeds) >= 2:
        std_row = [
            statistics.stdev(per_task_scores[t]) if len(per_task_scores[t]) >= 2 else None
            for t in header_tasks
        ]
        lines.append(
            "\\multicolumn{3}{l}{Std} & "
            + " & ".join(
                (f"$\\pm$ {fmt_score(v)}" if v is not None else "") for v in std_row
            )
            + " &  \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")
    _ = n_cols  # keep col count for future use if layout tweaks needed


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir)
    if not runs:
        print(f"[aggregate] no result.json files under {results_dir}")
        return

    write_csv(runs, results_dir / "summary.csv")
    write_markdown(runs, results_dir / "table.md", args.method_label, args.params_label)
    write_latex(runs, results_dir / "table.tex", args.method_label, args.params_label)
    table_md = (results_dir / "table.md").read_text()
    write_results_doc(runs, results_dir / "RESULTS.md", args.method_label, args.params_label, table_md)

    print(f"[aggregate] {len(runs)} runs loaded")
    print(f"[aggregate] wrote {results_dir / 'summary.csv'}")
    print(f"[aggregate] wrote {results_dir / 'table.md'}")
    print(f"[aggregate] wrote {results_dir / 'table.tex'}")
    print(f"[aggregate] wrote {results_dir / 'RESULTS.md'}")
    print("\n" + table_md)


if __name__ == "__main__":
    main()
