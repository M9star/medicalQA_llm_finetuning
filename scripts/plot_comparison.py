"""Plot base vs fine-tuned accuracy from experiments/RESULTS.json.

Saves experiments/comparison.png — a grouped bar chart per dataset with the
random baseline marked, used in docs/RESULTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "experiments" / "RESULTS.json"
OUT = PROJECT_ROOT / "experiments" / "comparison.png"


def main() -> None:
    data = json.loads(RESULTS.read_text())
    datasets = data["datasets"]

    names = [d["dataset"] for d in datasets]
    base = [d["base"]["accuracy_pct"] for d in datasets]
    finetuned = [d["finetuned"]["accuracy_pct"] for d in datasets]
    random_bl = [d["random_baseline_pct"] for d in datasets]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(6, 2.4 * len(names)), 5))
    b1 = ax.bar(x - width / 2, base, width, label="Base model", color="#cbd5e0", edgecolor="white")
    b2 = ax.bar(x + width / 2, finetuned, width, label="Fine-tuned", color="#4299e1", edgecolor="white")

    # random baseline markers
    for i, rb in enumerate(random_bl):
        ax.hlines(rb, x[i] - 0.45, x[i] + 0.45, colors="#e53e3e", linestyles="--", linewidth=1.5,
                  label="Random baseline" if i == 0 else None)

    ax.bar_label(b1, fmt="%.0f%%", padding=3, fontsize=10, color="#4a5568")
    ax.bar_label(b2, fmt="%.0f%%", padding=3, fontsize=10, fontweight="bold", color="#2b6cb0")

    # improvement annotation above each fine-tuned bar
    for i, d in enumerate(datasets):
        imp = d["improvement_pct"]
        ax.annotate(f"+{imp:.0f}%", xy=(x[i] + width / 2, finetuned[i]),
                    xytext=(x[i] + width / 2, finetuned[i] + 8),
                    ha="center", fontsize=11, fontweight="bold", color="#2f855a")

    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names], fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Base vs Fine-Tuned Accuracy (LoRA, 200 steps)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
