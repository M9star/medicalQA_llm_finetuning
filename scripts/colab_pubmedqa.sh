#!/usr/bin/env bash
#
# Train the PubMedQA LoRA adapter on a CUDA GPU (e.g. Google Colab free T4).
#
# Run it in ONE Colab cell with:
#   !curl -sSL https://raw.githubusercontent.com/M9star/medicalQA_llm_finetuning/main/scripts/colab_pubmedqa.sh | bash
#
# It clones the repo, installs deps, prepares PubMedQA, runs base eval ->
# fine-tune -> fine-tuned eval, and leaves the adapter + comparison in
# experiments/pubmedqa/. On CUDA the project auto-uses 4-bit QLoRA.
#
# Does NOT touch your local machine — everything happens in the Colab session.

set -euo pipefail

REPO_URL="https://github.com/M9star/medicalQA_llm_finetuning.git"
REPO_DIR="medicalQA_llm_finetuning"
MAX_STEPS="${MAX_STEPS:-200}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

echo "========================================================"
echo " PubMedQA fine-tuning  |  max_steps=$MAX_STEPS  samples=$NUM_SAMPLES"
echo "========================================================"

# 1. GPU check (non-fatal if nvidia-smi missing)
nvidia-smi || echo "WARNING: no GPU detected — training will be very slow."

# 2. Clone (skip if already present)
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL"
fi
cd "$REPO_DIR"

# 3. Install only what's needed; keep Colab's CUDA torch intact
pip install -q -U transformers peft trl accelerate bitsandbytes datasets

# 4. Prepare PubMedQA in Alpaca format
python scripts/prepare_dataset.py --dataset pubmedqa

# 5. Confirm backend (should report 4-bit QLoRA on CUDA)
python scripts/check_device.py

# 6. Base eval -> fine-tune -> fine-tuned eval -> comparison
python scripts/run_experiment.py --datasets pubmedqa \
  --max-steps "$MAX_STEPS" --num-samples "$NUM_SAMPLES"

# 7. Bundle the adapter + results for download
ZIP="/content/pubmedqa_adapter.zip"
[ -d /content ] || ZIP="$(pwd)/pubmedqa_adapter.zip"
zip -r -q "$ZIP" experiments/pubmedqa/final_adapter \
  experiments/pubmedqa/base_eval.csv \
  experiments/pubmedqa/finetuned_eval.csv \
  experiments/RESULTS.json experiments/COMPARISON.md || true

echo ""
echo "========================================================"
echo " DONE."
echo " Comparison:"
cat experiments/COMPARISON.md || true
echo ""
echo " Adapter + results zipped at: $ZIP"
echo " Download it, unzip into experiments/pubmedqa/final_adapter locally, then:"
echo "   python scripts/serve.py --adapter-path experiments/pubmedqa/final_adapter"
echo "========================================================"
