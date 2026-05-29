#!/bin/bash
set -euo pipefail

# --- SETUP ENVIRONMENT ---
echo "--- Starting job on node $(hostname) ---"

setup_conda_env() {
  if [[ -z "${CONDA_ENV_NAME:-}" && -z "${CONDA_ENV:-}" ]]; then
    echo "Using current Python environment: ${CONDA_DEFAULT_ENV:-system}"
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "Warning: conda command not found; using current environment."
    return
  fi

  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    source "${conda_base}/etc/profile.d/conda.sh"
  fi

  target_env="${CONDA_ENV_NAME:-${CONDA_ENV}}"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${target_env}" ]]; then
    conda activate "${target_env}" || echo "Warning: could not activate conda env '${target_env}'; using current environment."
  fi
}

setup_conda_env
echo "Python: $(command -v python)"
python - <<'PY'
import sys
print(f"Python version: {sys.version.split()[0]}")
PY

# Check GPU status (Optional, but good for logs)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "Warning: nvidia-smi not found."
fi

cd "$(dirname "$0")"
mkdir -p slurm_logs weights

export FAIRVISION_DATA_ROOT="${FAIRVISION_DATA_ROOT:-../Datasets/FairVision}"
export RETFOUND_OCT_BACKBONE_WEIGHTS="${RETFOUND_OCT_BACKBONE_WEIGHTS:-weights/RETFound_mae_natureOCT.pth}"
export RETFOUND_OCT_MODEL_WEIGHTS="${RETFOUND_OCT_MODEL_WEIGHTS:-weights/oct_model_best.pth}"
export RETFOUND_EPOCHS="${RETFOUND_EPOCHS:-60}"
export RETFOUND_BATCH_SIZE="${RETFOUND_BATCH_SIZE:-64}"
export RETFOUND_NUM_WORKERS="${RETFOUND_NUM_WORKERS:-16}"
export RETFOUND_OCT_REPRESENTATION="${RETFOUND_OCT_REPRESENTATION:-center}"
export RETFOUND_FREEZE_BACKBONE="${RETFOUND_FREEZE_BACKBONE:-1}"
export FAIRVISION_STRICT_ALIGNMENT="${FAIRVISION_STRICT_ALIGNMENT:-1}"

# --- EXECUTE PYTHON SCRIPT ---
echo "Running main training script: linear_probing_oct3.py"
echo "FAIRVISION_DATA_ROOT=${FAIRVISION_DATA_ROOT}"
echo "RETFOUND_OCT_BACKBONE_WEIGHTS=${RETFOUND_OCT_BACKBONE_WEIGHTS}"
echo "RETFOUND_OCT_MODEL_WEIGHTS=${RETFOUND_OCT_MODEL_WEIGHTS}"
echo "RETFOUND_RESUME=${RETFOUND_RESUME:-1}"
echo "RETFOUND_OCT_REPRESENTATION=${RETFOUND_OCT_REPRESENTATION}"
echo "RETFOUND_FREEZE_BACKBONE=${RETFOUND_FREEZE_BACKBONE}"
echo "FAIRVISION_STRICT_ALIGNMENT=${FAIRVISION_STRICT_ALIGNMENT}"

python -u -m VisionAgent.linear_probing_oct3

echo "--- Job finished successfully ---"
