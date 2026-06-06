"""Rebuild combined RESULTS.json + COMPARISON.md from whatever eval CSVs exist.

Scans experiments/<dataset>/{base_eval.csv, finetuned_eval.csv} for every
dataset and writes a single combined summary. Safe to run repeatedly and in any
order — useful after downloading a second dataset's results from Colab so the
PubMedQA and MedMCQA numbers live in one table instead of overwriting each other.

Usage:
  python scripts/aggregate_results.py
  python scripts/plot_comparison.py   # redraw the chart afterwards
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments"
RANDOM_BASELINE = {"medmcqa": 25.0, "pubmedqa": 33.3}
MODEL_ID = "unsloth/Llama-3.2-1B"


def _acc(csv: Path) -> dict:
    df = pd.read_csv(csv)
    total = len(df)
    correct = int(df["is_correct"].sum())
    return {"total": total, "correct": correct,
            "accuracy_pct": round(100 * correct / total, 2) if total else 0.0}


def main() -> None:
    datasets = []
    for name in ("medmcqa", "pubmedqa"):
        base_csv = EXPERIMENTS / name / "base_eval.csv"
        ft_csv = EXPERIMENTS / name / "finetuned_eval.csv"
        if not (base_csv.exists() and ft_csv.exists()):
            print(f"skip {name}: need both base_eval.csv and finetuned_eval.csv")
            continue
        base, ft = _acc(base_csv), _acc(ft_csv)
        datasets.append({
            "dataset": name,
            "random_baseline_pct": RANDOM_BASELINE[name],
            "base": base,
            "finetuned": ft,
            "improvement_pct": round(ft["accuracy_pct"] - base["accuracy_pct"], 2),
            "base_eval_csv": str((EXPERIMENTS / name / "base_eval.csv").relative_to(PROJECT_ROOT)),
            "finetuned_eval_csv": str((EXPERIMENTS / name / "finetuned_eval.csv").relative_to(PROJECT_ROOT)),
        })

    if not datasets:
        print("No complete dataset results found. Nothing written.")
        return

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "datasets": datasets,
    }
    (EXPERIMENTS / "RESULTS.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Base vs Fine-Tuned Comparison",
        "",
        "| Dataset | Random | Base | Fine-tuned | Improvement |",
        "|---------|-------:|-----:|-----------:|------------:|",
    ]
    for r in datasets:
        lines.append(
            f"| {r['dataset']} | {r['random_baseline_pct']}% "
            f"| {r['base']['accuracy_pct']}% ({r['base']['correct']}/{r['base']['total']}) "
            f"| {r['finetuned']['accuracy_pct']}% ({r['finetuned']['correct']}/{r['finetuned']['total']}) "
            f"| {r['improvement_pct']:+.2f}% |"
        )
    (EXPERIMENTS / "COMPARISON.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote combined results for: {', '.join(d['dataset'] for d in datasets)}")


if __name__ == "__main__":
    main()
