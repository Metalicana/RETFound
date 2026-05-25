#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-Datasets}"
MANIFEST_ROOT="${MANIFEST_ROOT:-equi-agent/outputs/manifests}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-equi-agent/outputs/checkpoints}"
TABLES_ROOT="${TABLES_ROOT:-equi-agent/outputs/tables}"
TASKS="${TASKS:-glaucoma_detection progression_forecasting}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1}"
VISIONFM_ROOT="${VISIONFM_ROOT:-Foundation_Models/VisionFM-main}"
VISIONFM_WEIGHTS="${VISIONFM_WEIGHTS:-}"
CHECKPOINT_KEY="${CHECKPOINT_KEY:-teacher}"
ARCH="${ARCH:-vit_base}"
PATCH_SIZE="${PATCH_SIZE:-16}"
FEATURE_BLOCKS="${FEATURE_BLOCKS:-4}"
MAX_ITER="${MAX_ITER:-5000}"
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"

if [[ -z "${VISIONFM_WEIGHTS}" ]]; then
  echo "VISIONFM_WEIGHTS must point to VFM_OCT_weights.pth." >&2
  exit 1
fi

python equi-agent/scripts/build_manifests.py \
  --datasets-root "${DATASETS_ROOT}" \
  --out-dir "${MANIFEST_ROOT}"

PATH_ARGS=()
if [[ -n "${PATH_PREFIX_FROM}" ]]; then
  PATH_ARGS+=(--path-prefix-from "${PATH_PREFIX_FROM}" --path-prefix-to "${PATH_PREFIX_TO}")
fi

COMMON_ARGS=(
  --visionfm-root "${VISIONFM_ROOT}"
  --pretrained-weights "${VISIONFM_WEIGHTS}"
  --checkpoint-key "${CHECKPOINT_KEY}"
  --arch "${ARCH}"
  --patch-size "${PATCH_SIZE}"
  --feature-blocks "${FEATURE_BLOCKS}"
  --max-iter "${MAX_ITER}"
  --threshold-metric "${THRESHOLD_METRIC}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
)

for task in ${TASKS}; do
  prediction_file="${PREDICTIONS_ROOT}/gdp_${task}_visionfm_oct.csv"
  checkpoint_file="${CHECKPOINT_ROOT}/gdp_${task}_visionfm_oct_linear_probe.pkl"
  metrics_dir="${METRICS_ROOT}/exp8_gdp_${task}_visionfm_oct"

  python equi-agent/scripts/predict_gdp_visionfm_oct.py \
    --task "${task}" \
    --checkpoint "${checkpoint_file}" \
    --out "${prediction_file}" \
    "${COMMON_ARGS[@]}" \
    "${PATH_ARGS[@]}"

  python equi-agent/scripts/evaluate_predictions.py \
    --predictions "${prediction_file}" \
    --out-dir "${metrics_dir}"
done

python equi-agent/scripts/build_manuscript_tables.py \
  --metrics-root "${METRICS_ROOT}" \
  --out-dir "${TABLES_ROOT}"

echo "GDP VisionFM OCT complete."
echo "Predictions: ${PREDICTIONS_ROOT}/gdp_*_visionfm_oct.csv"
echo "Metrics: ${METRICS_ROOT}/exp8_gdp_*_visionfm_oct"
