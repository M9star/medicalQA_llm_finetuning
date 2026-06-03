"""Dataset visualization — saves plots to an output directory."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from .data import (
    load_prepared_dataset,
    top_values,
    word_length_stats,
)

logger = logging.getLogger(__name__)

_MEDMCQA_RE = re.compile(r"The correct answer is ([A-D])\)")
_PUBMEDQA_RE = re.compile(r"^(Yes|No|Maybe)", re.IGNORECASE)


def _savefig(fig, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.info("Saved %s → %s", title, path)


def _extract_medmcqa_answers(dataset) -> Counter:
    counts: Counter = Counter()
    for ex in dataset:
        m = _MEDMCQA_RE.search(ex.get("output", ""))
        if m:
            counts[m.group(1)] += 1
    return counts


def _extract_pubmedqa_answers(dataset) -> Counter:
    counts: Counter = Counter()
    for ex in dataset:
        m = _PUBMEDQA_RE.match(ex.get("output", ""))
        if m:
            counts[m.group(1).capitalize()] += 1
    return counts


def plot_answer_distribution(counts: Counter, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = sorted(counts)
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Answer")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_path, title)
    plt.close(fig)


def plot_length_histogram(dataset, field: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    lengths = [len(ex[field].split()) for ex in dataset if ex.get(field)]
    stats = word_length_stats(dataset, field)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=40, color="teal", edgecolor="white", alpha=0.85)
    ax.axvline(stats["mean"], color="tomato", linewidth=1.5, linestyle="--", label=f"mean={stats['mean']:.1f}")
    ax.axvline(stats["median"], color="gold", linewidth=1.5, linestyle="--", label=f"median={stats['median']}")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_path, title)
    plt.close(fig)


def plot_subject_distribution(dataset, field: str, limit: int, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    items = top_values(dataset, field, limit=limit)
    subjects = [s for s, _ in reversed(items)]
    counts = [c for _, c in reversed(items)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(subjects, counts, color="mediumseagreen", edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_path, title)
    plt.close(fig)


def plot_text_length_histogram(dataset, title: str, output_path: Path, char_limit: int = 4000) -> None:
    import matplotlib.pyplot as plt

    lengths = [len(ex["text"]) for ex in dataset if ex.get("text")]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=40, color="slateblue", edgecolor="white", alpha=0.85)
    ax.axvline(char_limit, color="tomato", linewidth=1.5, linestyle="--", label=f"max_seq limit ({char_limit} chars)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Text length (chars)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_path, title)
    plt.close(fig)


def visualize_datasets(
    datasets: tuple[str, ...],
    prepared_data_dir: Path,
    output_dir: Path,
) -> None:
    if "medmcqa" in datasets:
        path = prepared_data_dir / "medmcqa_alpaca"
        dd = load_prepared_dataset(path)
        train = dd["train"]

        plot_answer_distribution(
            _extract_medmcqa_answers(train),
            "MedMCQA — Answer Distribution (train)",
            output_dir / "medmcqa_answer_distribution.png",
        )
        plot_length_histogram(
            train, "instruction",
            "MedMCQA — Question Length Distribution (train)",
            output_dir / "medmcqa_question_lengths.png",
        )
        plot_subject_distribution(
            train, "subject",
            limit=15,
            title="MedMCQA — Top 15 Subjects (train)",
            output_path=output_dir / "medmcqa_subject_distribution.png",
        )
        plot_text_length_histogram(
            train,
            "MedMCQA — Full Text Length Distribution (train)",
            output_dir / "medmcqa_text_lengths.png",
        )

    if "pubmedqa" in datasets:
        path = prepared_data_dir / "pubmedqa_alpaca"
        dd = load_prepared_dataset(path)
        train = dd["train"]

        plot_answer_distribution(
            _extract_pubmedqa_answers(train),
            "PubMedQA — Answer Distribution (train)",
            output_dir / "pubmedqa_answer_distribution.png",
        )
        plot_length_histogram(
            train, "instruction",
            "PubMedQA — Question Length Distribution (train)",
            output_dir / "pubmedqa_question_lengths.png",
        )
        plot_text_length_histogram(
            train,
            "PubMedQA — Full Text Length Distribution (train)",
            output_dir / "pubmedqa_text_lengths.png",
        )
