"""QLoRA fine-tuning workflow."""

from __future__ import annotations

import logging

from .config import TrainingConfig
from .data import load_prepared_dataset
from .modeling import apply_lora, count_trainable_parameters, load_model, load_tokenizer, print_device_summary, require_training_dependencies

logger = logging.getLogger(__name__)


def train(config: TrainingConfig):
    """Fine-tune a base model on a prepared Alpaca dataset."""

    deps = require_training_dependencies()
    device_info = print_device_summary(config.device, config.quantization)
    dataset = load_prepared_dataset(config.dataset_path)
    tokenizer = load_tokenizer(config.model_id)
    model = apply_lora(load_model(config.model_id, device_info, gradient_checkpointing=True), config, device_info)
    parameter_counts = count_trainable_parameters(model)
    logger.info(
        "Trainable parameters: %s/%s (%.2f%%)",
        f"{parameter_counts['trainable']:,}",
        f"{parameter_counts['total']:,}",
        parameter_counts["trainable_pct"],
    )

    training_args = deps["SFTConfig"](
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim=device_info.optim,
        bf16=device_info.bf16,
        fp16=device_info.fp16,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        max_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to=config.report_to,
        seed=42,
    )
    trainer = deps["SFTTrainer"](
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    logger.info(
        "Starting training: model=%s train=%s validation=%s max_steps=%s",
        config.model_id,
        len(dataset["train"]),
        len(dataset["validation"]),
        config.max_steps,
    )
    resume_checkpoint = _latest_checkpoint(config.output_dir) if config.resume else None
    if config.resume:
        if resume_checkpoint:
            logger.info("Resuming training from checkpoint: %s", resume_checkpoint)
        else:
            logger.warning("--resume set but no checkpoint found in %s; training from scratch.", config.output_dir)
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    adapter_path = config.output_dir / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Saved LoRA adapter to %s", adapter_path)
    return result


def _latest_checkpoint(output_dir) -> str | None:
    """Return the path to the highest-numbered checkpoint-* dir, or None."""

    from pathlib import Path

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    checkpoints = [
        p for p in output_dir.glob("checkpoint-*")
        if p.is_dir() and p.name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
    return str(latest)
