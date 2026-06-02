"""Evaluation workflows for MedMCQA and PubMedQA."""

from __future__ import annotations

import logging
import re

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .config import EvaluationConfig
from .data import format_medmcqa_eval, format_pubmedqa_eval
from .modeling import generate_response, load_model, load_tokenizer, print_device_summary, require_training_dependencies

logger = logging.getLogger(__name__)


def extract_medmcqa_answer(response: str) -> str | None:
    """Extract an A/B/C/D answer from a generated response."""

    patterns = [
        r"correct answer is\s*([A-Da-d])",
        r"answer is\s*([A-Da-d])",
        r"^\s*([A-Da-d])\)",
        r"^\s*([A-Da-d])\s",
        r"\b([A-Da-d])\)\s",
        r"Option\s*([A-Da-d])",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    for char in response[:50]:
        if char.upper() in {"A", "B", "C", "D"}:
            return char.upper()
    return None


def extract_pubmedqa_answer(response: str) -> str | None:
    """Extract yes/no/maybe from a generated response."""

    response_lower = response.lower().strip()
    if response_lower.startswith("yes"):
        return "yes"
    if response_lower.startswith("no"):
        return "no"
    if response_lower.startswith("maybe"):
        return "maybe"
    first_part = response_lower[:100]
    if "yes." in first_part or "yes," in first_part or "yes " in first_part:
        return "yes"
    if "no." in first_part or "no," in first_part or "no " in first_part:
        return "no"
    if "maybe" in first_part:
        return "maybe"
    return None


def load_eval_data(config: EvaluationConfig):
    """Load and format evaluation examples."""

    if config.dataset == "medmcqa":
        raw = load_dataset("openlifescienceai/medmcqa")
        split = raw[config.split].shuffle(seed=config.seed).select(range(min(config.num_samples, len(raw[config.split]))))
        return split.map(format_medmcqa_eval), extract_medmcqa_answer
    if config.dataset == "pubmedqa":
        raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
        split = raw["train"].shuffle(seed=config.seed).select(range(min(config.num_samples, len(raw["train"]))))
        return split.map(format_pubmedqa_eval), extract_pubmedqa_answer
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def load_model_for_eval(config: EvaluationConfig):
    """Load base model and optionally attach a LoRA adapter."""

    deps = require_training_dependencies()
    device_info = print_device_summary(config.device, config.quantization)
    tokenizer = load_tokenizer(config.model_id)
    model = load_model(config.model_id, device_info)
    if config.adapter_path:
        from peft import PeftModel

        logger.info("Loading adapter from %s", config.adapter_path)
        model = PeftModel.from_pretrained(model, str(config.adapter_path), is_trainable=False)
    model.eval()
    return model, tokenizer


def evaluate(config: EvaluationConfig) -> tuple[float, pd.DataFrame]:
    """Run accuracy evaluation and save per-example results."""

    test_data, extractor = load_eval_data(config)
    model, tokenizer = load_model_for_eval(config)
    rows = []
    correct = 0
    for example in tqdm(test_data, desc=f"Evaluating {config.dataset}"):
        response = generate_response(model, tokenizer, example["prompt"], config.max_new_tokens)
        predicted = extractor(response)
        is_correct = predicted == example["correct_answer"]
        correct += int(is_correct)
        rows.append(
            {
                "question": example["question"][:200],
                "subject": example.get("subject"),
                "correct_answer": example["correct_answer"],
                "predicted": predicted,
                "is_correct": is_correct,
                "response": response[:500],
            }
        )

    results = pd.DataFrame(rows)
    accuracy = correct / len(results) if len(results) else 0
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_csv, index=False)
    logger.info("Accuracy: %.2f%% (%s/%s)", accuracy * 100, correct, len(results))
    logger.info("Saved evaluation results to %s", config.output_csv)
    return accuracy, results
