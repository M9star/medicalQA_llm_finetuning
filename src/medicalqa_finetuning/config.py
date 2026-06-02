"""Central project configuration, device resolution, and logging setup."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset names, split sizes, and local artifact paths."""

    seed: int = 42
    output_dir: Path = PROJECT_ROOT / "prepared_data"
    pubmedqa_name: str = "qiaojin/PubMedQA"
    pubmedqa_config: str = "pqa_labeled"
    medmcqa_name: str = "openlifescienceai/medmcqa"
    medmcqa_train_size: int = 10_000
    medmcqa_validation_size: int = 500
    pubmedqa_train_size: int = 800
    pubmedqa_validation_size: int = 100


@dataclass(frozen=True)
class TrainingConfig:
    """Default LoRA/SFT training configuration."""

    model_id: str = "unsloth/Llama-3.2-1B"
    dataset_path: Path = PROJECT_ROOT / "prepared_data" / "medmcqa_alpaca"
    output_dir: Path = PROJECT_ROOT / "medical_llm_finetuned"
    device: str = "auto"
    quantization: str = "auto"
    max_seq_length: int = 1024
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: int = 10
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )
    report_to: str = "none"


@dataclass(frozen=True)
class EvaluationConfig:
    """Default evaluation configuration."""

    model_id: str = "unsloth/Llama-3.2-1B"
    adapter_path: Path | None = None
    device: str = "auto"
    quantization: str = "auto"
    dataset: str = "medmcqa"
    split: str = "validation"
    num_samples: int = 50
    max_new_tokens: int = 150
    output_csv: Path = PROJECT_ROOT / "evaluation_results.csv"
    seed: int = 42


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved hardware strategy for model loading, training, and evaluation."""

    backend: str
    device: str
    torch_dtype: object
    use_4bit: bool
    device_map: str | None
    optim: str
    bf16: bool
    fp16: bool
    description: str


def resolve_device(preferred: str = "auto", quantization: str = "auto") -> DeviceInfo:
    """Resolve the best available device and a compatible precision strategy.

    CUDA uses 4-bit QLoRA by default. Apple MPS does not support bitsandbytes
    4-bit quantization, so it falls back to normal LoRA on MPS. CPU is intended
    for smoke tests and small debugging runs.
    """

    import torch

    preferred = preferred.lower()
    quantization = quantization.lower()
    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, mps, cpu")
    if quantization not in {"auto", "4bit", "none"}:
        raise ValueError("--quantization must be one of: auto, 4bit, none")

    backend = _select_backend(torch, preferred)
    if quantization == "4bit" and backend != "cuda":
        raise ValueError("4-bit bitsandbytes quantization is only supported on CUDA in this project.")

    use_4bit = backend == "cuda" and quantization in {"auto", "4bit"}

    if backend == "cuda":
        bf16 = bool(torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if bf16 else torch.float16
        return DeviceInfo(
            backend="cuda",
            device="cuda",
            torch_dtype=dtype,
            use_4bit=use_4bit,
            device_map="auto",
            optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
            bf16=bf16,
            fp16=not bf16,
            description=_cuda_description(torch, use_4bit),
        )

    if backend == "mps":
        return DeviceInfo(
            backend="mps",
            device="mps",
            torch_dtype=torch.float16,
            use_4bit=False,
            device_map=None,
            optim="adamw_torch",
            bf16=False,
            fp16=False,
            description="Apple Metal MPS detected. Using standard LoRA without bitsandbytes 4-bit.",
        )

    return DeviceInfo(
        backend="cpu",
        device="cpu",
        torch_dtype=torch.float32,
        use_4bit=False,
        device_map=None,
        optim="adamw_torch",
        bf16=False,
        fp16=False,
        description="No GPU detected. Using CPU fallback; training will be very slow.",
    )


def _select_backend(torch, preferred: str) -> str:
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return "cuda"
    if preferred == "mps":
        if not _mps_available(torch):
            raise RuntimeError("MPS was requested, but Apple Metal backend is not available.")
        return "mps"
    if preferred == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def _mps_available(torch) -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _cuda_description(torch, use_4bit: bool) -> str:
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    mode = "4-bit QLoRA" if use_4bit else "standard LoRA"
    return f"CUDA GPU detected: {name} ({total_gb:.1f} GB). Using {mode}."


def setup_logging(level: str = "INFO") -> None:
    """Configure a concise root logger for command-line runs."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
