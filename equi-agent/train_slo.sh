#!/bin/bash
set -euo pipefail

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

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "Warning: nvidia-smi not found."
fi

cd "$(dirname "$0")"
mkdir -p slurm_logs weights

FAIRVISION_DATA_ROOT="${FAIRVISION_DATA_ROOT:-../Datasets/FairVision}"
MIRAGE_DIR="${MIRAGE_DIR:-VisionAgent/MIRAGE}"
MIRAGE_SLO_MODEL_WEIGHTS="${MIRAGE_SLO_MODEL_WEIGHTS:-weights/slo_model_best.pth}"
MIRAGE_EPOCHS="${MIRAGE_EPOCHS:-60}"
MIRAGE_BATCH_SIZE="${MIRAGE_BATCH_SIZE:-64}"
MIRAGE_NUM_WORKERS="${MIRAGE_NUM_WORKERS:-16}"
MIRAGE_RESUME="${MIRAGE_RESUME:-1}"
export FAIRVISION_STRICT_ALIGNMENT="${FAIRVISION_STRICT_ALIGNMENT:-1}"

resume_arg="--resume"
case "${MIRAGE_RESUME,,}" in
  0|false|no|n|off)
    resume_arg="--no-resume"
    ;;
esac

echo "Running MIRAGE SLO FairVision training"
echo "FAIRVISION_DATA_ROOT=${FAIRVISION_DATA_ROOT}"
echo "MIRAGE_DIR=${MIRAGE_DIR}"
echo "MIRAGE_SLO_MODEL_WEIGHTS=${MIRAGE_SLO_MODEL_WEIGHTS}"
echo "MIRAGE_RESUME=${MIRAGE_RESUME}"
echo "FAIRVISION_STRICT_ALIGNMENT=${FAIRVISION_STRICT_ALIGNMENT}"

python -u scripts/train_fairvision_mirage_slo.py \
  --data-root "${FAIRVISION_DATA_ROOT}" \
  --mirage-dir "${MIRAGE_DIR}" \
  --output-weights "${MIRAGE_SLO_MODEL_WEIGHTS}" \
  --epochs "${MIRAGE_EPOCHS}" \
  --batch-size "${MIRAGE_BATCH_SIZE}" \
  --num-workers "${MIRAGE_NUM_WORKERS}" \
  "${resume_arg}"

echo "--- Job finished successfully ---"
