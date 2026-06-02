# MedicalQA LLM Fine-Tuning

Production-ready utilities for preparing, fine-tuning, and evaluating instruction-tuned medical question-answering models with PubMedQA and MedMCQA.

The original notebook-style `main.py` has been split into a maintainable Python package with explicit command-line workflows, reusable modules, logging, configuration objects, and lightweight tests.

## Features

- Load PubMedQA and MedMCQA from Hugging Face datasets
- Convert examples to Alpaca instruction format
- Run dataset quality checks and basic analysis
- Fine-tune causal language models with CUDA QLoRA or Apple MPS LoRA and TRL `SFTTrainer`
- Auto-detect CUDA, Apple MPS, or CPU before model loading
- Evaluate base or adapter models on MedMCQA and PubMedQA
- Save prepared datasets, LoRA adapters, and evaluation CSVs

## Project Structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── check_device.py
│   ├── evaluate.py
│   ├── prepare_dataset.py
│   └── train.py
├── src/
│   └── medicalqa_finetuning/
│       ├── commands.py    # main command entrypoint
│       ├── config.py      # dataclasses, logging, device resolver
│       ├── data.py        # load/format/analyze/prepare datasets
│       ├── train.py       # LoRA training flow
│       ├── evaluate.py    # evaluation flow and metrics CSV
│       ├── modeling.py    # model/tokenizer/LoRA helpers
│       └── __init__.py
└── tests/
    └── test_formatter.py
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Prepare both datasets:

```bash
python scripts/prepare_dataset.py
```

Prepare only PubMedQA with custom split sizes:

```bash
python scripts/prepare_dataset.py --dataset pubmedqa --pubmedqa-train-size 800 --pubmedqa-validation-size 100
```

Check whether training will use CUDA, MPS, or CPU:

```bash
python scripts/check_device.py
```

Run exploratory analysis without preparing artifacts:

```bash
medicalqa analyze --dataset medmcqa
```

Train a LoRA adapter on the prepared MedMCQA dataset:

```bash
python scripts/train.py \
  --model-id unsloth/Llama-3.2-1B \
  --dataset-path prepared_data/medmcqa_alpaca \
  --output-dir medical_llm_finetuned \
  --max-steps 10
```

Evaluate an adapter:

```bash
python scripts/evaluate.py \
  --dataset medmcqa \
  --model-id unsloth/Llama-3.2-1B \
  --adapter-path medical_llm_finetuned/final_adapter \
  --num-samples 50 \
  --output-csv evaluation_results.csv
```

After `pip install -e .`, use the package command:

```bash
medicalqa prepare --dataset all
medicalqa check-device
medicalqa train --max-steps 10
medicalqa evaluate --dataset pubmedqa --adapter-path pubmedqa_llm_finetuned/final_adapter
```

## Device Support

The project uses `src/medicalqa_finetuning/config.py` to select the safest training mode:

- CUDA: uses 4-bit QLoRA by default with bitsandbytes.
- Apple MPS: uses regular LoRA on MPS because bitsandbytes 4-bit is not supported there.
- CPU: works as a slow fallback for debugging and tiny smoke tests.

You can override the automatic choice:

```bash
python scripts/train.py --device cuda --quantization 4bit
python scripts/train.py --device mps --quantization none
python scripts/evaluate.py --device cpu --quantization none --num-samples 5
```

## Configuration

Defaults live in `src/medicalqa_finetuning/config.py`:

- `DatasetConfig` controls dataset names, split sizes, output paths, and seed.
- `TrainingConfig` controls model ID, LoRA parameters, batch size, learning rate, sequence length, and checkpointing cadence.
- `EvaluationConfig` controls dataset, adapter path, sample count, generation length, and output CSV path.

Command-line flags override the most common defaults.

## Outputs

Generated artifacts are ignored by git:

- `prepared_data/medmcqa_alpaca`
- `prepared_data/pubmedqa_alpaca`
- `medical_llm_finetuned/final_adapter`
- `pubmedqa_llm_finetuned/final_adapter`
- `evaluation_results.csv`

## Development

Run a syntax check:

```bash
python -m compileall src scripts tests
```

Run tests:

```bash
pytest
```

## Notes

Training downloads models and datasets and requires substantial memory. CUDA is the best path for 4-bit QLoRA. Apple Silicon can run smaller LoRA jobs through MPS, but you should reduce model size, batch size, and sequence length if memory is tight. The default `--max-steps 10` is intentionally a smoke-test setting; increase it for real fine-tuning.
