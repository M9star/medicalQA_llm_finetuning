#!/usr/bin/env bash
#
# Train the medical QA LoRA adapters on a CUDA GPU (e.g. Google Colab free T4).
#
# WHY: fine-tuning on Apple MPS diverges to NaN weights (fp16 instability) and is
# thermal-throttled. CUDA training is stable (auto 4-bit QLoRA) and fast.
#
# Run it in ONE Colab cell (Runtime -> Change runtime type -> GPU first):
#   !curl -sSL https://raw.githubusercontent.com/M9star/medicalQA_llm_finetuning/main/scripts/colab_train.sh | bash
#
# Trains BOTH datasets by default. To train one only:
#   !curl -sSL .../scripts/colab_train.sh | DATASETS=medmcqa bash
#
# Leaves adapters + comparison in experiments/ and zips them for download.
# Does NOT touch your local machine — everything happens in the Colab session.

set -euo pipefail

REPO_URL="https://github.com/M9star/medicalQA_llm_finetuning.git"
REPO_DIR="medicalQA_llm_finetuning"
DATASETS="${DATASETS:-medmcqa pubmedqa}"
MAX_STEPS="${MAX_STEPS:-200}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

echo "========================================================"
echo " Fine-tuning: [$DATASETS]  max_steps=$MAX_STEPS  samples=$NUM_SAMPLES"
echo "========================================================"

# 1. GPU check (non-fatal if nvidia-smi missing)
nvidia-smi || echo "WARNING: no GPU detected — training will be slow/unstable."

# 2. Clone (skip if already present)
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL"
fi
cd "$REPO_DIR"

# 3. Install only what's needed; keep Colab's CUDA torch intact
pip install -q -U transformers peft trl accelerate bitsandbytes datasets

# 4. Prepare datasets in Alpaca format
# shellcheck disable=SC2086
python scripts/prepare_dataset.py --dataset all

# 5. Confirm backend (should report 4-bit QLoRA on CUDA)
python scripts/check_device.py

# 6. Base eval -> fine-tune -> fine-tuned eval -> comparison, per dataset
# shellcheck disable=SC2086
python scripts/run_experiment.py --datasets $DATASETS \
  --max-steps "$MAX_STEPS" --num-samples "$NUM_SAMPLES"

# 7. Bundle adapters + results for download
ZIP="/content/medicalqa_adapters.zip"
[ -d /content ] || ZIP="$(pwd)/medicalqa_adapters.zip"
zip -r -q "$ZIP" experiments -x 'experiments/**/checkpoint-*/*' || true

echo ""
echo "========================================================"
echo " DONE."
cat experiments/COMPARISON.md || true
echo ""
echo " Adapters + results zipped at: $ZIP"
echo " Download it, unzip 'experiments/' into the project locally, then serve:"
echo "   python scripts/serve.py --adapter-path experiments/medmcqa/final_adapter"
echo "========================================================"
