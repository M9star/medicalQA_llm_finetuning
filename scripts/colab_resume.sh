#!/usr/bin/env bash
#
# Resume an interrupted fine-tune on Colab from a downloaded checkpoint zip.
#
# The checkpoint zip is the one saved when training was interrupted (it contains
# a checkpoint-<N>/ folder with optimizer.pt, scheduler.pt, rng_state.pth,
# trainer_state.json, and the adapter weights).
#
# 1. Put the checkpoint zip somewhere Colab can read it, e.g. copy from Drive:
#      !cp /content/drive/MyDrive/medicalqa/medmcqa_adapters_checkpoint-100.zip /content/
#    or upload it with: from google.colab import files; files.upload()
#
# 2. Run (Runtime -> GPU first):
#      !curl -sSL https://raw.githubusercontent.com/M9star/medicalQA_llm_finetuning/main/scripts/colab_resume.sh | bash
#
# Override defaults via env vars, e.g.:
#      !... | CKPT_ZIP=/content/my.zip DATASET=medmcqa bash

set -euo pipefail

REPO_URL="https://github.com/M9star/medicalQA_llm_finetuning.git"
REPO_DIR="medicalQA_llm_finetuning"
DATASET="${DATASET:-medmcqa}"
MAX_STEPS="${MAX_STEPS:-200}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
CKPT_ZIP="${CKPT_ZIP:-/content/medmcqa_adapters_checkpoint-100.zip}"

echo "========================================================"
echo " Resume $DATASET -> max_steps=$MAX_STEPS  from: $CKPT_ZIP"
echo "========================================================"

nvidia-smi || echo "WARNING: no GPU detected."

if [ ! -f "$CKPT_ZIP" ]; then
  echo "ERROR: checkpoint zip not found at $CKPT_ZIP"
  echo "Copy it from Drive or upload it first, or set CKPT_ZIP=/path/to/zip."
  exit 1
fi

# Clone + install
[ -d "$REPO_DIR" ] || git clone "$REPO_URL"
cd "$REPO_DIR"
pip install -q -U transformers peft trl accelerate bitsandbytes datasets

# Prepare data (deterministic — same seed reproduces the exact training set)
python scripts/prepare_dataset.py --dataset "$DATASET"

# Restore the checkpoint into experiments/<dataset>/checkpoint-<N>/
mkdir -p "experiments/$DATASET"
TMP="$(mktemp -d)"
unzip -o -q "$CKPT_ZIP" -d "$TMP"
CKPT_DIR="$(find "$TMP" -type d -name 'checkpoint-*' | sort | tail -1)"
if [ -z "$CKPT_DIR" ]; then
  echo "ERROR: no checkpoint-* folder inside $CKPT_ZIP"
  exit 1
fi
cp -r "$CKPT_DIR" "experiments/$DATASET/"
echo "Restored $(basename "$CKPT_DIR") -> experiments/$DATASET/"

# Resume: base eval -> train (continues from checkpoint) -> fine-tuned eval
python scripts/run_experiment.py --datasets "$DATASET" \
  --max-steps "$MAX_STEPS" --num-samples "$NUM_SAMPLES" --resume

# Bundle final adapter + results
ZIP="/content/medicalqa_adapters.zip"
[ -d /content ] || ZIP="$(pwd)/medicalqa_adapters.zip"
zip -r -q "$ZIP" experiments -x 'experiments/**/checkpoint-*/*' || true

echo ""
echo "========================================================"
echo " DONE — resumed to step $MAX_STEPS."
cat experiments/COMPARISON.md || true
echo ""
echo " Adapter + results zipped at: $ZIP"
echo "========================================================"
