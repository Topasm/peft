"""Run CaRA GLUE (task x seed) sweep as isolated subprocesses.

Each (task, seed) pair runs in its own python subprocess so a crash in one run
doesn't take down the rest of the sweep. Existing result.json files are skipped,
making the sweep resumable.

Example:
    python sweep.py                                       # full default sweep
    python sweep.py --tasks mrpc rte --seeds 42           # partial sweep
    python sweep.py --base_output_dir results/cara_r8_v2  # alternate output root
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = ["cola", "stsb", "rte", "mrpc", "sst2", "qnli"]
DEFAULT_SEEDS = [42, 1234, 24]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--base_output_dir", default="results/cara_r8")
    p.add_argument("--model_name_or_path", default="microsoft/deberta-v3-base")
    p.add_argument("--script", default="run_glue_cara.py",
                   help="Which trainer script to invoke (run_glue_cara.py or run_glue_oft.py).")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--noise_alpha", type=float, default=0.01)
    p.add_argument("--noise_step_interval", type=int, default=5)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="extra args forwarded verbatim to run_glue_cara.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().parent / args.script
    if not script.exists():
        raise SystemExit(f"script not found: {script}")
    base = Path(args.base_output_dir)
    passthrough = args.extra or []
    # Only forward CaRA-specific args (noise_alpha, noise_step_interval) when running run_glue_cara.py
    is_cara = "cara" in args.script.lower()

    n_total = len(args.tasks) * len(args.seeds)
    done, skipped, failed = [], [], []
    idx = 0

    for task in args.tasks:
        for seed in args.seeds:
            idx += 1
            run_dir = base / task / f"seed_{seed}"
            result_path = run_dir / "result.json"
            tag = f"[{idx}/{n_total}] {task} seed={seed}"

            if result_path.exists():
                print(f"{tag}  SKIP (result.json exists at {result_path})")
                skipped.append((task, seed))
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"

            cmd = [
                sys.executable, str(script),
                "--task", task,
                "--seed", str(seed),
                "--model_name_or_path", args.model_name_or_path,
                "--output_dir", str(run_dir),
                "--r", str(args.r),
            ]
            if is_cara:
                cmd.extend([
                    "--noise_alpha", str(args.noise_alpha),
                    "--noise_step_interval", str(args.noise_step_interval),
                ])
            if args.bf16:
                cmd.append("--bf16")
            if args.fp16:
                cmd.append("--fp16")
            cmd.extend(passthrough)

            print(f"{tag}  RUN  -> {run_dir}")
            with log_path.open("w") as log:
                log.write("$ " + " ".join(cmd) + "\n\n")
                log.flush()
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

            if proc.returncode == 0 and result_path.exists():
                done.append((task, seed))
                print(f"{tag}  DONE")
            else:
                failed.append((task, seed))
                print(f"{tag}  FAIL (see {log_path})")

    print("\n=== sweep summary ===")
    print(f"  done:    {len(done)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed:  {len(failed)}")
    if failed:
        print("\nfailed runs:")
        for task, seed in failed:
            print(f"  {task} seed={seed}  log: {base / task / f'seed_{seed}' / 'train.log'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
