#!/bin/bash
#
#SBATCH --job-name=SLO_MIRAGE_Probing
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

set -euo pipefail

echo "--- Starting job on node $(hostname) ---"

source $(conda info --base)/etc/profile.d/conda.sh
module load anaconda
conda activate "${CONDA_ENV:-retfound}"

nvidia-smi

cd "$(dirname "$0")"
mkdir -p slurm_logs weights

FAIRVISION_DATA_ROOT="${FAIRVISION_DATA_ROOT:-../Datasets/FairVision}"
MIRAGE_DIR="${MIRAGE_DIR:-VisionAgent/MIRAGE}"
MIRAGE_SLO_MODEL_WEIGHTS="${MIRAGE_SLO_MODEL_WEIGHTS:-weights/slo_model_best.pth}"
MIRAGE_EPOCHS="${MIRAGE_EPOCHS:-60}"
MIRAGE_BATCH_SIZE="${MIRAGE_BATCH_SIZE:-64}"
MIRAGE_NUM_WORKERS="${MIRAGE_NUM_WORKERS:-16}"

echo "Running MIRAGE SLO FairVision training"
echo "FAIRVISION_DATA_ROOT=${FAIRVISION_DATA_ROOT}"
echo "MIRAGE_DIR=${MIRAGE_DIR}"
echo "MIRAGE_SLO_MODEL_WEIGHTS=${MIRAGE_SLO_MODEL_WEIGHTS}"

python -u scripts/train_fairvision_mirage_slo.py \
  --data-root "${FAIRVISION_DATA_ROOT}" \
  --mirage-dir "${MIRAGE_DIR}" \
  --output-weights "${MIRAGE_SLO_MODEL_WEIGHTS}" \
  --epochs "${MIRAGE_EPOCHS}" \
  --batch-size "${MIRAGE_BATCH_SIZE}" \
  --num-workers "${MIRAGE_NUM_WORKERS}"

echo "--- Job finished successfully ---"
