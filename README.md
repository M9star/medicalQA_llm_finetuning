# MedicalQA LLM Fine-Tuning

Prepare, fine-tune, evaluate, and serve instruction-tuned medical
question-answering models on **PubMedQA** and **MedMCQA**, using LoRA / 4-bit
QLoRA on top of `unsloth/Llama-3.2-1B`.

The project covers the full lifecycle: data prep → analysis → visualization →
fine-tuning (on GPU) → base-vs-fine-tuned evaluation → an interactive quiz web
app that serves the fine-tuned models.

## Results

Fine-tuning lifts both tasks above their base model and random baselines, with
<1% of parameters trained (LoRA, 200 steps, 4-bit QLoRA on GPU):

| Dataset | Random | Base | Fine-tuned | Improvement |
|---------|-------:|-----:|-----------:|------------:|
| PubMedQA | 33.3% | 26% | **64%** | **+38%** |
| MedMCQA | 25% | 26% | **32%** | **+6%** |

![Base vs fine-tuned accuracy](experiments/comparison.png)

Details: [`docs/RESULTS.md`](docs/RESULTS.md) · [`docs/FINETUNING.md`](docs/FINETUNING.md)

## Features

- Load PubMedQA & MedMCQA from Hugging Face and convert to Alpaca instruction format
- Dataset quality checks, statistics, and visual plots
- Auto-detect CUDA / Apple MPS / CPU and pick a safe precision strategy
- Fine-tune with LoRA (MPS) or 4-bit QLoRA (CUDA) via TRL `SFTTrainer`
- Resume interrupted training from checkpoints (`--resume`)
- Base-vs-fine-tuned evaluation with per-question CSVs and a comparison plot
- One-line GPU training on Google Colab
- FastAPI **quiz web app**: dataset questions + AI explanations from the fine-tuned model

## Project structure

```text
.
├── README.md
├── pyproject.toml / requirements.txt
├── docs/
│   ├── FINETUNING.md          # full pipeline, GPU/Colab, resume, MPS-failure notes
│   └── RESULTS.md             # results summary + plot
├── scripts/
│   ├── prepare_dataset.py     # download + format datasets
│   ├── check_device.py        # hardware backend
│   ├── visualize_dataset.py   # exploratory plots
│   ├── train.py               # LoRA fine-tuning (+ --resume)
│   ├── evaluate.py            # accuracy eval -> CSV
│   ├── serve.py               # FastAPI quiz server
│   ├── run_experiment.py      # base eval -> train -> fine-tuned eval -> comparison
│   ├── aggregate_results.py   # merge per-dataset results into one table
│   ├── plot_comparison.py     # base-vs-fine-tuned bar chart
│   ├── colab_train.sh         # one-line GPU training on Colab
│   └── colab_resume.sh        # resume training from a checkpoint on Colab
├── src/medicalqa_finetuning/
│   ├── commands.py            # unified CLI entrypoint
│   ├── config.py              # dataclasses, logging, device resolver
│   ├── data.py                # load/format/analyze/prepare datasets
│   ├── modeling.py            # model/tokenizer/LoRA/generation helpers
│   ├── train.py               # training flow
│   ├── evaluate.py            # evaluation + answer extraction
│   ├── visualize.py           # plotting
│   └── api.py                 # FastAPI quiz app
├── experiments/               # results (CSV/JSON/PNG tracked; weights gitignored)
└── tests/
```

## Installation

Uses [`uv`](https://github.com/astral-sh/uv) with Python 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/bin/python
source .venv/bin/activate
```

## Workflow

```bash
# 1. Prepare datasets (Alpaca format)
python scripts/prepare_dataset.py

# 2. Inspect hardware + data
python scripts/check_device.py
python scripts/visualize_dataset.py          # plots/ -> answer dist, lengths, subjects

# 3. Fine-tune + compare (base vs fine-tuned). Use a GPU — see note below.
python scripts/run_experiment.py --datasets medmcqa pubmedqa --max-steps 200

# 4. Regenerate the comparison plot from results
python scripts/plot_comparison.py

# 5. Serve the quiz with a fine-tuned adapter
python scripts/serve.py --adapter-path experiments/pubmedqa/final_adapter
```

Then open **http://127.0.0.1:8000** — pick a task, answer a question, and click
**Get AI Explanation** to see the fine-tuned model explain the answer.

## ⚠️ Train on GPU, not Apple MPS

Fine-tuning on Apple MPS **diverges to NaN weights** (fp16 instability — the
model ends up emitting `!!!!!!!` and scores 0%). MPS is fine for the quiz and
base-model evaluation, **but not for training.** Train on a CUDA GPU instead,
where the project auto-uses stable 4-bit QLoRA.

The easy path is **Google Colab's free GPU** — run in one cell
(Runtime → Change runtime type → GPU):

```bash
# both datasets
!curl -sSL https://raw.githubusercontent.com/M9star/medicalQA_llm_finetuning/main/scripts/colab_train.sh | bash
```

See [`docs/FINETUNING.md`](docs/FINETUNING.md) for resuming interrupted runs and
moving adapters back to your machine.

## CLI (after `uv pip install -e .`)

```bash
medicalqa prepare --dataset all
medicalqa analyze --dataset medmcqa
medicalqa visualize
medicalqa check-device
medicalqa train --max-steps 200 --resume
medicalqa evaluate --dataset pubmedqa --adapter-path experiments/pubmedqa/final_adapter
medicalqa serve --adapter-path experiments/medmcqa/final_adapter
```

## Configuration

Defaults live in `src/medicalqa_finetuning/config.py`:
`DatasetConfig` (names, split sizes, seed), `TrainingConfig` (model, LoRA params,
batch/lr/steps, resume), `EvaluationConfig` (dataset, adapter, samples, output CSV).
CLI flags override the common ones.

## What is and isn't in git

Tracked: all source, scripts, docs, and **lightweight results** (eval CSVs,
`RESULTS.json`, `COMPARISON.md`, `comparison.png`).

Gitignored (large / reproducible): model adapter weights
(`experiments/**/final_adapter/`), checkpoints, prepared datasets, `.venv/`,
and downloaded base models. Adapters are reproducible via Colab; back them up
to Google Drive if you want to keep them without retraining (see below).

## Tests

```bash
pytest
python -m compileall src scripts
```
