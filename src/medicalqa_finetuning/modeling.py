"""Model, tokenizer, LoRA, and generation helpers."""

from __future__ import annotations

import logging

from .config import DeviceInfo, resolve_device

logger = logging.getLogger(__name__)


def require_training_dependencies():
    """Import optional training dependencies only when needed."""

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install them with: "
            "pip install -r requirements.txt"
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def print_device_summary(preferred: str = "auto", quantization: str = "auto") -> DeviceInfo:
    """Resolve and log the selected hardware backend."""

    device_info = resolve_device(preferred, quantization)
    logger.info(device_info.description)
    return device_info


def load_tokenizer(model_id: str):
    """Load tokenizer and ensure a pad token is set."""

    deps = require_training_dependencies()
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def build_bnb_config(device_info: DeviceInfo):
    """Build the default 4-bit quantization config."""

    if not device_info.use_4bit:
        return None
    deps = require_training_dependencies()
    return deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=device_info.torch_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_model(model_id: str, device_info: DeviceInfo, gradient_checkpointing: bool = False):
    """Load a causal LM with the resolved device and quantization strategy."""

    deps = require_training_dependencies()
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": device_info.torch_dtype,
    }
    if device_info.device_map:
        load_kwargs["device_map"] = device_info.device_map
    if device_info.use_4bit:
        load_kwargs["quantization_config"] = build_bnb_config(device_info)

    model = deps["AutoModelForCausalLM"].from_pretrained(model_id, **load_kwargs)
    if not device_info.device_map:
        model = model.to(device_info.device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def apply_lora(model, config, device_info: DeviceInfo):
    """Prepare a model for LoRA fine-tuning."""

    deps = require_training_dependencies()
    lora_config = deps["LoraConfig"](
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=deps["TaskType"].CAUSAL_LM,
        target_modules=list(config.target_modules),
    )
    if device_info.use_4bit:
        model = deps["prepare_model_for_kbit_training"](model, use_gradient_checkpointing=True)
    return deps["get_peft_model"](model, lora_config)


def count_trainable_parameters(model) -> dict:
    """Return total/trainable parameter counts."""

    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return {"total": total, "trainable": trainable, "trainable_pct": 100 * trainable / total if total else 0}


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a deterministic response for evaluation."""

    deps = require_training_dependencies()
    torch = deps["torch"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
