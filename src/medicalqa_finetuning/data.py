"""Dataset loading, formatting, quality checking, and preparation workflows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from .config import DatasetConfig

logger = logging.getLogger(__name__)

ANSWER_LABELS = ["A", "B", "C", "D"]

MEDICAL_SYSTEM_PROMPT = (
    "You are a helpful medical AI assistant with expertise in clinical medicine "
    "and biomedical research."
)


# ==========================================
# 1. Dataset Loading Helpers
# ==========================================

def load_pubmedqa(config: DatasetConfig):
    """Load PubMedQA labeled examples from Hugging Face."""

    logger.info("Loading PubMedQA dataset: %s/%s", config.pubmedqa_name, config.pubmedqa_config)
    return load_dataset(config.pubmedqa_name, config.pubmedqa_config, trust_remote_code=True)


def load_medmcqa(config: DatasetConfig):
    """Load MedMCQA from Hugging Face."""

    logger.info("Loading MedMCQA dataset: %s", config.medmcqa_name)
    return load_dataset(config.medmcqa_name)


def select_shuffled(dataset, size: int, seed: int):
    """Shuffle and select at most ``size`` examples."""

    actual_size = min(size, len(dataset))
    return dataset.shuffle(seed=seed).select(range(actual_size))


def split_pubmedqa_train_validation(raw_pubmedqa, config: DatasetConfig) -> DatasetDict:
    """Create train/validation splits for PubMedQA, which ships as train-only."""

    shuffled = raw_pubmedqa["train"].shuffle(seed=config.seed)
    train_size = min(config.pubmedqa_train_size, len(shuffled))
    validation_end = min(train_size + config.pubmedqa_validation_size, len(shuffled))
    return DatasetDict(
        {
            "train": shuffled.select(range(train_size)),
            "validation": shuffled.select(range(train_size, validation_end)),
        }
    )


def build_medmcqa_subset(raw_medmcqa, config: DatasetConfig) -> DatasetDict:
    """Create a manageable MedMCQA train/validation subset."""

    return DatasetDict(
        {
            "train": select_shuffled(raw_medmcqa["train"], config.medmcqa_train_size, config.seed),
            "validation": select_shuffled(
                raw_medmcqa["validation"], config.medmcqa_validation_size, config.seed
            ),
        }
    )


def map_dataset(dataset: DatasetDict, formatter: Callable, remove_columns: list[str] | None = None):
    """Apply a formatter across all splits."""

    columns = remove_columns or dataset["train"].column_names
    return dataset.map(formatter, remove_columns=columns)


def load_prepared_dataset(path):
    """Load a dataset previously saved by ``prepare_datasets``."""

    logger.info("Loading prepared dataset from %s", path)
    return load_from_disk(str(path))


# ==========================================
# 2. Instruction Formatters
# ==========================================

def format_pubmedqa_alpaca(example):
    """Convert a PubMedQA example to Alpaca instruction format."""

    contexts = example["context"]["contexts"] if example["context"]["contexts"] else []
    context = " ".join(contexts) if contexts else "No context provided."
    instruction = (
        "You are a medical expert. Based on the provided research context, answer "
        "the following yes/no/maybe question. Provide a brief explanation for your answer."
    )
    input_text = f"Context: {context}\n\nQuestion: {example['question']}"
    answer_map = {"yes": "Yes", "no": "No", "maybe": "Maybe"}
    answer = answer_map.get(example["final_decision"], example["final_decision"])
    long_answer = example.get("long_answer", "")
    response = f"{answer}. {long_answer}" if long_answer else answer
    return build_alpaca_record(instruction, input_text, response)


def format_medmcqa_alpaca(example):
    """Convert a MedMCQA example to Alpaca instruction format."""

    instruction = (
        "You are a medical expert. Answer the following multiple-choice medical question. "
        "Choose the correct option and provide a brief explanation."
    )
    options = f"A) {example['opa']}\nB) {example['opb']}\nC) {example['opc']}\nD) {example['opd']}"
    input_text = f"Question: {example['question']}\n\nOptions:\n{options}"
    correct_answer = ANSWER_LABELS[example["cop"]]
    correct_option = [example["opa"], example["opb"], example["opc"], example["opd"]][example["cop"]]
    explanation = example.get("exp", "") or ""
    if explanation:
        response = f"The correct answer is {correct_answer}) {correct_option}.\n\nExplanation: {explanation}"
    else:
        response = f"The correct answer is {correct_answer}) {correct_option}."

    record = build_alpaca_record(instruction, input_text, response)
    record["subject"] = example.get("subject_name")
    record["topic"] = example.get("topic_name")
    return record


def build_alpaca_record(instruction: str, input_text: str, response: str) -> dict:
    """Build a complete Alpaca-style record."""

    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
    return {"instruction": instruction, "input": input_text, "output": response, "text": text}


def format_chatml(instruction: str, input_text: str, response: str) -> dict:
    """Convert instruction/input/response fields to ChatML messages."""

    user_message = f"{instruction}\n\n{input_text}" if input_text else instruction
    return {
        "messages": [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response},
        ]
    }


def format_medmcqa_chatml(example):
    """Convert MedMCQA to ChatML format."""

    alpaca = format_medmcqa_alpaca(example)
    return format_chatml(alpaca["instruction"], alpaca["input"], alpaca["output"])


def format_pubmedqa_eval(example):
    """Build a PubMedQA prompt and answer for evaluation."""

    record = format_pubmedqa_alpaca(example)
    prompt = record["text"].split("### Response:")[0] + "### Response:\n"
    return {"prompt": prompt, "correct_answer": example["final_decision"], "question": example["question"]}


def format_medmcqa_eval(example):
    """Build a MedMCQA prompt and answer for evaluation."""

    record = format_medmcqa_alpaca(example)
    prompt = record["text"].split("### Response:")[0] + "### Response:\n"
    return {
        "prompt": prompt,
        "correct_answer": ANSWER_LABELS[example["cop"]],
        "question": example["question"],
        "subject": example.get("subject_name"),
    }


# ==========================================
# 3. Dataset Analysis & Quality Control
# ==========================================

def answer_distribution(dataset, field: str) -> Counter:
    """Count answer values for a split."""

    return Counter(example[field] for example in dataset)


def word_length_stats(dataset, field: str = "question") -> dict:
    """Return min/max/mean/median word length statistics."""

    lengths = [len(example[field].split()) for example in dataset if example.get(field)]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    ordered = sorted(lengths)
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "median": ordered[len(ordered) // 2],
    }


def top_values(dataset, field: str, limit: int = 10) -> list[tuple[str, int]]:
    """Return the most common values for a field."""

    return Counter(example[field] for example in dataset if example.get(field)).most_common(limit)


def quality_report(dataset_dict, split: str = "train", text_limit: int = 4000) -> dict:
    """Check required fields, text lengths, long examples, and duplicates."""

    dataset = dataset_dict[split]
    text_lengths = [len(example["text"]) for example in dataset]
    unique_texts = {example["text"] for example in dataset}
    total = len(dataset)
    very_long = sum(1 for length in text_lengths if length > text_limit)
    return {
        "split": split,
        "total": total,
        "empty_instruction": sum(1 for ex in dataset if not ex.get("instruction")),
        "empty_input": sum(1 for ex in dataset if not ex.get("input")),
        "empty_output": sum(1 for ex in dataset if not ex.get("output")),
        "min_text_chars": min(text_lengths) if text_lengths else 0,
        "max_text_chars": max(text_lengths) if text_lengths else 0,
        "mean_text_chars": sum(text_lengths) / total if total else 0,
        "over_text_limit": very_long,
        "over_text_limit_pct": (very_long / total * 100) if total else 0,
        "duplicates": total - len(unique_texts),
    }


# ==========================================
# 4. Preparation Pipeline
# ==========================================

def prepare_datasets(config: DatasetConfig, datasets: tuple[str, ...] = ("medmcqa", "pubmedqa")) -> dict[str, Path]:
    """Prepare selected datasets and save them to disk."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if "medmcqa" in datasets:
        raw_medmcqa = load_medmcqa(config)
        subset = build_medmcqa_subset(raw_medmcqa, config)
        formatted = map_dataset(subset, format_medmcqa_alpaca)
        path = config.output_dir / "medmcqa_alpaca"
        formatted.save_to_disk(str(path))
        written["medmcqa"] = path
        logger.info("Saved MedMCQA dataset to %s", path)
        _log_quality_reports(formatted)

    if "pubmedqa" in datasets:
        raw_pubmedqa = load_pubmedqa(config)
        subset = split_pubmedqa_train_validation(raw_pubmedqa, config)
        formatted = map_dataset(subset, format_pubmedqa_alpaca)
        path = config.output_dir / "pubmedqa_alpaca"
        formatted.save_to_disk(str(path))
        written["pubmedqa"] = path
        logger.info("Saved PubMedQA dataset to %s", path)
        _log_quality_reports(formatted)

    return written


def _log_quality_reports(dataset_dict) -> None:
    for split in dataset_dict:
        report = quality_report(dataset_dict, split)
        logger.info(
            "%s quality: total=%s empty_output=%s duplicates=%s mean_chars=%.0f over_4000=%s (%.1f%%)",
            split,
            report["total"],
            report["empty_output"],
            report["duplicates"],
            report["mean_text_chars"],
            report["over_text_limit"],
            report["over_text_limit_pct"],
        )
