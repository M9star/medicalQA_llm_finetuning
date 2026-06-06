"""End-to-end fine-tuning experiment: base vs fine-tuned comparison.

For each dataset (MedMCQA, PubMedQA) this script:
  1. Evaluates the BASE model           -> experiments/<name>/base_eval.csv
  2. Fine-tunes a LoRA adapter          -> experiments/<name>/final_adapter/
  3. Evaluates the FINE-TUNED model     -> experiments/<name>/finetuned_eval.csv

Each phase runs as an isolated subprocess so GPU/MPS memory is fully released
between steps. Results are aggregated into:
  experiments/RESULTS.json    (machine-readable)
  experiments/COMPARISON.md   (human-readable summary table)

Usage:
  python scripts/run_experiment.py
  python scripts/run_experiment.py --datasets medmcqa --max-steps 50 --num-samples 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PREPARED_DIR = PROJECT_ROOT / "prepared_data"
PY = sys.executable  # the venv interpreter running this script

RANDOM_BASELINE = {"medmcqa": 25.0, "pubmedqa": 33.3}


def run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output, and raise on failure."""
    print(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}", flush=True)
    subprocess.run(cmd, check=True)


def accuracy_from_csv(csv_path: Path) -> dict:
    """Compute accuracy and counts from an evaluation CSV."""
    df = pd.read_csv(csv_path)
    total = len(df)
    correct = int(df["is_correct"].sum())
    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": round(100 * correct / total, 2) if total else 0.0,
    }


def run_dataset(name: str, max_steps: int, num_samples: int, resume: bool = False) -> dict:
    """Run the full base -> train -> fine-tuned pipeline for one dataset."""
    out_dir = EXPERIMENTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = PREPARED_DIR / f"{name}_alpaca"
    adapter_path = out_dir / "final_adapter"
    base_csv = out_dir / "base_eval.csv"
    ft_csv = out_dir / "finetuned_eval.csv"

    # 1. Base model evaluation
    run([
        PY, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--dataset", name,
        "--num-samples", str(num_samples),
        "--output-csv", str(base_csv),
    ])

    # 2. Fine-tune LoRA adapter
    train_cmd = [
        PY, str(PROJECT_ROOT / "scripts" / "train.py"),
        "--dataset-path", str(dataset_path),
        "--output-dir", str(out_dir),
        "--max-steps", str(max_steps),
    ]
    if resume:
        train_cmd.append("--resume")
    run(train_cmd)

    # 3. Fine-tuned model evaluation
    run([
        PY, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--dataset", name,
        "--adapter-path", str(adapter_path),
        "--num-samples", str(num_samples),
        "--output-csv", str(ft_csv),
    ])

    base = accuracy_from_csv(base_csv)
    finetuned = accuracy_from_csv(ft_csv)
    return {
        "dataset": name,
        "random_baseline_pct": RANDOM_BASELINE.get(name),
        "base": base,
        "finetuned": finetuned,
        "improvement_pct": round(finetuned["accuracy_pct"] - base["accuracy_pct"], 2),
        "adapter_path": str(adapter_path.relative_to(PROJECT_ROOT)),
        "base_eval_csv": str(base_csv.relative_to(PROJECT_ROOT)),
        "finetuned_eval_csv": str(ft_csv.relative_to(PROJECT_ROOT)),
    }


def write_comparison(results: dict) -> None:
    """Write RESULTS.json and COMPARISON.md."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "RESULTS.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Base vs Fine-Tuned Comparison",
        "",
        f"_Generated: {results['generated_at']}_",
        "",
        f"- Base model: `{results['model_id']}`",
        f"- Fine-tuning: LoRA, {results['max_steps']} steps",
        f"- Eval samples per dataset: {results['num_samples']}",
        "",
        "| Dataset | Random | Base | Fine-tuned | Improvement |",
        "|---------|-------:|-----:|-----------:|------------:|",
    ]
    for r in results["datasets"]:
        lines.append(
            f"| {r['dataset']} | {r['random_baseline_pct']}% "
            f"| {r['base']['accuracy_pct']}% ({r['base']['correct']}/{r['base']['total']}) "
            f"| {r['finetuned']['accuracy_pct']}% ({r['finetuned']['correct']}/{r['finetuned']['total']}) "
            f"| {r['improvement_pct']:+.2f}% |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
    ])
    for r in results["datasets"]:
        lines.append(f"### {r['dataset']}")
        lines.append(f"- Adapter: `{r['adapter_path']}`")
        lines.append(f"- Base eval: `{r['base_eval_csv']}`")
        lines.append(f"- Fine-tuned eval: `{r['finetuned_eval_csv']}`")
        lines.append("")
    (EXPERIMENTS_DIR / "COMPARISON.md").write_text("\n".join(lines))
    print(f"\nWrote {EXPERIMENTS_DIR / 'RESULTS.json'}")
    print(f"Wrote {EXPERIMENTS_DIR / 'COMPARISON.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Base vs fine-tuned experiment runner.")
    parser.add_argument("--datasets", nargs="+", default=["medmcqa", "pubmedqa"],
                        choices=["medmcqa", "pubmedqa"])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--model-id", default="unsloth/Llama-3.2-1B")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint in each dataset's experiment dir.")
    args = parser.parse_args()

    dataset_results = []
    for name in args.datasets:
        print(f"\n########## EXPERIMENT: {name} ##########", flush=True)
        dataset_results.append(run_dataset(name, args.max_steps, args.num_samples, args.resume))

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "max_steps": args.max_steps,
        "num_samples": args.num_samples,
        "datasets": dataset_results,
    }
    write_comparison(results)
    print("\nDONE. Summary:")
    for r in dataset_results:
        print(f"  {r['dataset']:9s}  base={r['base']['accuracy_pct']:.1f}%  "
              f"finetuned={r['finetuned']['accuracy_pct']:.1f}%  "
              f"({r['improvement_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
