# Fine-Tuning & Evaluation Pipeline

This document explains how the project fine-tunes medical QA adapters, how it
compares the base model against the fine-tuned model, and how those adapters
plug into the quiz web app as a use case.

---

## 1. Overview

The goal is to take a small general-purpose language model
(`unsloth/Llama-3.2-1B`) and specialize it for two medical QA tasks using
**LoRA** (Low-Rank Adaptation) — training <1% of the parameters while the base
weights stay frozen.

| Task | Dataset | Answer space | Random baseline |
|------|---------|--------------|-----------------|
| Multiple choice | MedMCQA | A / B / C / D | 25% |
| Research yes/no | PubMedQA | yes / no / maybe | 33.3% |

Each task gets its **own adapter**, so the quiz can load the right specialist
for each mode.

---

## 2. Pipeline stages

```
prepare_dataset.py  ->  prepared_data/{medmcqa,pubmedqa}_alpaca/   (Alpaca-formatted)
        |
        v
run_experiment.py   ->  for each dataset:
        |                  1. evaluate BASE model       (baseline accuracy)
        |                  2. train LoRA adapter         (200 steps)
        |                  3. evaluate FINE-TUNED model  (final accuracy)
        v
experiments/        ->  adapters + eval CSVs + RESULTS.json + COMPARISON.md
        |
        v
serve.py --adapter-path experiments/<task>/final_adapter
                    ->  quiz UI generates explanations from the fine-tuned model
```

---

## 3. The data format (Alpaca)

Every training example is a single `text` string in Alpaca instruction format:

```
### Instruction:
You are a medical expert. Answer the following multiple-choice medical question...

### Input:
Question: <question>

Options:
A) ...
B) ...
C) ...
D) ...

### Response:
The correct answer is A) ...

Explanation: ...
```

Training teaches the model to continue from `### Response:` in this exact shape.
That is why, after fine-tuning, the model reliably starts its answer with
`The correct answer is X)` (MedMCQA) or `Yes/No/Maybe` (PubMedQA) — which is what
the answer extractor in `evaluate.py` looks for.

---

## 4. LoRA configuration

Defined in `src/medicalqa_finetuning/config.py` (`TrainingConfig`):

| Setting | Value | Notes |
|---------|-------|-------|
| `lora_r` | 8 | rank of the low-rank update |
| `lora_alpha` | 16 | scaling |
| `lora_dropout` | 0.05 | |
| `target_modules` | q,k,v,o,gate,up,down proj | all attention + MLP projections |
| `max_steps` | 200 (this run) | optimizer steps |
| `batch_size` | 4 | per device |
| `gradient_accumulation_steps` | 4 | effective batch = 16 |
| `learning_rate` | 2e-4 | with cosine schedule, 3% warmup |
| `max_seq_length` | 1024 | |

Result: **~5.6M trainable parameters out of ~1.24B total (0.45%)**.

### Hardware note (Apple MPS)
On Apple Silicon the project uses **standard LoRA in fp16** — `bitsandbytes`
4-bit QLoRA is CUDA-only. On an NVIDIA GPU the same code path automatically
switches to 4-bit QLoRA (see `resolve_device` in `config.py`).

---

## 5. Running the experiment

```bash
# Full run used for the committed results: both datasets, 200 steps, 50 eval samples
python scripts/run_experiment.py --datasets medmcqa pubmedqa --max-steps 200 --num-samples 50

# Quick smoke test
python scripts/run_experiment.py --datasets medmcqa --max-steps 50 --num-samples 30
```

Each phase (base eval / train / fine-tuned eval) runs as an **isolated
subprocess** so GPU/MPS memory is fully released between steps.

### Output layout

```
experiments/
  medmcqa/
    final_adapter/        # LoRA adapter weights (gitignored - regenerate with the script)
    checkpoint-*/         # intermediate training checkpoints (gitignored)
    base_eval.csv         # per-question results, base model
    finetuned_eval.csv    # per-question results, fine-tuned model
  pubmedqa/
    final_adapter/
    base_eval.csv
    finetuned_eval.csv
  RESULTS.json            # machine-readable summary (tracked)
  COMPARISON.md           # human-readable comparison table (tracked)
```

> The adapter weights and checkpoints are **gitignored** because they are large
> and reproducible. The lightweight CSVs, `RESULTS.json`, and `COMPARISON.md`
> are committed so the results are visible without re-running training.

---

## 6. How evaluation works

`evaluate.py` loads a fixed validation sample (seeded for reproducibility),
generates a deterministic (greedy) response for each question, extracts the
predicted answer with regex, and compares to the gold label.

- **MedMCQA** extractor looks for `correct answer is X`, `answer is X`, a leading
  `X)`, etc.
- **PubMedQA** extractor looks for a leading `yes/no/maybe` or the first such
  keyword in the response.

Accuracy = correct / total. Per-question rows (question, gold, predicted,
is_correct, raw response) are saved to the CSV for inspection.

---

## 7. Results

See **[`experiments/COMPARISON.md`](../experiments/COMPARISON.md)** for the
auto-generated table, and `experiments/RESULTS.json` for the raw numbers.

Expected pattern: the base model sits near the random baseline; the fine-tuned
model improves both **accuracy** and, just as importantly, **answer formatting**
— it stops rambling and starts answering in the trained shape, which is what
makes the extractor (and the quiz) reliable.

---

## 8. Use case: the quiz web app

The quiz (`src/medicalqa_finetuning/api.py`, served by `scripts/serve.py`) is the
end-user surface for these adapters:

1. **Questions** are pulled directly from the prepared datasets — no model
   needed, always correct.
2. The user picks an answer and gets instant correct/wrong feedback.
3. **"Get AI Explanation"** calls the model to explain *why* the correct answer
   is right and why the others are wrong.

To serve explanations from a **fine-tuned** adapter instead of the base model:

```bash
# MedMCQA specialist
python scripts/serve.py --adapter-path experiments/medmcqa/final_adapter

# PubMedQA specialist
python scripts/serve.py --adapter-path experiments/pubmedqa/final_adapter
```

The same UI, same endpoints — only the quality of the generated explanation
changes. This lets you directly compare base-model vs fine-tuned explanations in
the exact interface end users see.

> Note: the current quiz serves a single adapter at a time. A future improvement
> is to load both adapters and switch per task (`medmcqa` vs `pubmedqa`)
> automatically.

---

## 8b. Training on a GPU (Google Colab) — required

> **Important: do not fine-tune on Apple MPS.** Local MPS training diverges to
> **NaN weights** (the model is loaded in fp16 but MPS has no bf16 and HF
> mixed-precision loss scaling isn't applied). Symptom: the fine-tuned model
> emits `!!!!!!!` and accuracy collapses to 0%. Confirmed on MedMCQA:
> base = 28%, local MPS fine-tune = 0% (broken). MPS is also severely
> thermal-throttled (~45s/step → ~360s/step). **Train on a CUDA GPU instead**,
> where the project auto-uses stable **4-bit QLoRA**.

Run this in a **single Colab cell** (Runtime → Change runtime type → GPU first):

```bash
# Both datasets
!curl -sSL https://raw.githubusercontent.com/M9star/medicalQA_llm_finetuning/main/scripts/colab_train.sh | bash

# One dataset only
!curl -sSL .../scripts/colab_train.sh | DATASETS=medmcqa bash
```

`scripts/colab_train.sh` clones the repo, installs deps, prepares the data, runs
base eval → fine-tune → fine-tuned eval per dataset, and zips the adapters +
comparison for download. Nothing runs on your local machine. After downloading,
unzip `experiments/` into the project and serve as in section 8.

> MPS is still perfectly fine for **inference / the quiz** and for **base-model
> evaluation** — just not for training.

### Resuming an interrupted run

Training saves a `checkpoint-<N>/` every 50 steps (with optimizer, scheduler,
RNG, and step state). If a run is interrupted, download/keep that checkpoint
folder and continue later with `--resume`, which picks up from the latest
checkpoint in the output dir instead of restarting from step 0.

On Colab, after copying the checkpoint zip into the session (from Drive or
`files.upload()`):

```bash
!curl -sSL https://raw.githubusercontent.com/M9star/medicalQA_llm_finetuning/main/scripts/colab_resume.sh | bash
```

`scripts/colab_resume.sh` restores the `checkpoint-<N>/` into
`experiments/<dataset>/`, then runs `run_experiment.py --resume` to finish
training and evaluate. Override the source zip / dataset via env vars, e.g.
`CKPT_ZIP=/content/my.zip DATASET=medmcqa`.

> Resume requires the **full** `checkpoint-<N>/` folder (optimizer.pt,
> scheduler.pt, rng_state.pth, trainer_state.json + adapter files) — the adapter
> weights alone cannot resume. Data prep is deterministic (seed 42), so the
> training set reproduces exactly across sessions.

---

## 9. Reproducing from scratch

```bash
# 1. Environment
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/bin/python

# 2. Data
python scripts/prepare_dataset.py

# 3. Fine-tune + compare
python scripts/run_experiment.py

# 4. Serve the quiz with a fine-tuned adapter
python scripts/serve.py --adapter-path experiments/medmcqa/final_adapter
```
