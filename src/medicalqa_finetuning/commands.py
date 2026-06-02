"""Command entrypoints for dataset prep, training, and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DatasetConfig, EvaluationConfig, TrainingConfig, resolve_device, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical QA dataset preparation and fine-tuning tools.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare Alpaca-formatted datasets.")
    prepare.add_argument("--dataset", choices=["all", "medmcqa", "pubmedqa"], default="all")
    prepare.add_argument("--output-dir", type=Path, default=DatasetConfig().output_dir)
    prepare.add_argument("--medmcqa-train-size", type=int, default=DatasetConfig().medmcqa_train_size)
    prepare.add_argument("--medmcqa-validation-size", type=int, default=DatasetConfig().medmcqa_validation_size)
    prepare.add_argument("--pubmedqa-train-size", type=int, default=DatasetConfig().pubmedqa_train_size)
    prepare.add_argument("--pubmedqa-validation-size", type=int, default=DatasetConfig().pubmedqa_validation_size)

    analyze = subparsers.add_parser("analyze", help="Print lightweight raw dataset statistics.")
    analyze.add_argument("--dataset", choices=["medmcqa", "pubmedqa"], default="medmcqa")

    device_parser = subparsers.add_parser("check-device", help="Show whether CUDA, MPS, or CPU will be used.")
    device_parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    device_parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default="auto")

    train_parser = subparsers.add_parser("train", help="Fine-tune with LoRA/SFT.")
    train_parser.add_argument("--model-id", default=TrainingConfig().model_id)
    train_parser.add_argument("--dataset-path", type=Path, default=TrainingConfig().dataset_path)
    train_parser.add_argument("--output-dir", type=Path, default=TrainingConfig().output_dir)
    train_parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=TrainingConfig().device)
    train_parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default=TrainingConfig().quantization)
    train_parser.add_argument("--max-steps", type=int, default=TrainingConfig().max_steps)
    train_parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    train_parser.add_argument("--learning-rate", type=float, default=TrainingConfig().learning_rate)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a base model or adapter.")
    eval_parser.add_argument("--dataset", choices=["medmcqa", "pubmedqa"], default=EvaluationConfig().dataset)
    eval_parser.add_argument("--model-id", default=EvaluationConfig().model_id)
    eval_parser.add_argument("--adapter-path", type=Path, default=EvaluationConfig().adapter_path)
    eval_parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=EvaluationConfig().device)
    eval_parser.add_argument("--quantization", choices=["auto", "4bit", "none"], default=EvaluationConfig().quantization)
    eval_parser.add_argument("--num-samples", type=int, default=EvaluationConfig().num_samples)
    eval_parser.add_argument("--output-csv", type=Path, default=EvaluationConfig().output_csv)
    eval_parser.add_argument("--max-new-tokens", type=int, default=EvaluationConfig().max_new_tokens)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.command == "prepare":
        from .data import prepare_datasets

        config = replace(
            DatasetConfig(),
            output_dir=args.output_dir,
            medmcqa_train_size=args.medmcqa_train_size,
            medmcqa_validation_size=args.medmcqa_validation_size,
            pubmedqa_train_size=args.pubmedqa_train_size,
            pubmedqa_validation_size=args.pubmedqa_validation_size,
        )
        selected = ("medmcqa", "pubmedqa") if args.dataset == "all" else (args.dataset,)
        prepare_datasets(config, selected)
        return

    if args.command == "analyze":
        _run_analysis(args.dataset)
        return

    if args.command == "check-device":
        device_info = resolve_device(args.device, args.quantization)
        print(device_info.description)
        return

    if args.command == "train":
        from .train import train

        config = replace(
            TrainingConfig(),
            model_id=args.model_id,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            device=args.device,
            quantization=args.quantization,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        train(config)
        return

    if args.command == "evaluate":
        from .evaluate import evaluate

        config = replace(
            EvaluationConfig(),
            dataset=args.dataset,
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            device=args.device,
            quantization=args.quantization,
            num_samples=args.num_samples,
            output_csv=args.output_csv,
            max_new_tokens=args.max_new_tokens,
        )
        evaluate(config)


def _run_analysis(dataset_name: str) -> None:
    from .data import answer_distribution, load_medmcqa, load_pubmedqa, top_values, word_length_stats

    dataset_config = DatasetConfig()
    if dataset_name == "medmcqa":
        dataset = load_medmcqa(dataset_config)
        train_split = dataset["train"]
        print("MedMCQA answer distribution:", answer_distribution(train_split, "cop"))
        print("MedMCQA question lengths:", word_length_stats(train_split, "question"))
        print("Top MedMCQA subjects:", top_values(train_split, "subject_name", limit=10))
    else:
        dataset = load_pubmedqa(dataset_config)
        train_split = dataset["train"]
        print("PubMedQA answer distribution:", answer_distribution(train_split, "final_decision"))
        print("PubMedQA question lengths:", word_length_stats(train_split, "question"))


if __name__ == "__main__":
    main()
