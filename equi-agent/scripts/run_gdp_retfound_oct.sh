#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-Datasets}"
MANIFEST_ROOT="${MANIFEST_ROOT:-equi-agent/outputs/manifests}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-equi-agent/outputs/checkpoints}"
TABLES_ROOT="${TABLES_ROOT:-equi-agent/outputs/tables}"
TASKS="${TASKS:-glaucoma_detection progression_forecasting}"
GDP_PROGRESSION_TARGET="${GDP_PROGRESSION_TARGET:-md_fast_no_p_cut}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1}"
RETFOUND_BACKBONE_WEIGHTS="${RETFOUND_BACKBONE_WEIGHTS:-equi-agent/weights/RETFound_mae_natureOCT.pth}"
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"

python equi-agent/scripts/build_manifests.py \
  --datasets-root "${DATASETS_ROOT}" \
  --out-dir "${MANIFEST_ROOT}"

PATH_ARGS=()
if [[ -n "${PATH_PREFIX_FROM}" ]]; then
  PATH_ARGS+=(--path-prefix-from "${PATH_PREFIX_FROM}" --path-prefix-to "${PATH_PREFIX_TO}")
fi

for task in ${TASKS}; do
  task_suffix="${task}"
  MANIFEST_ARGS=()
  if [[ "${task}" == "progression_forecasting" ]]; then
    task_suffix="${task}_${GDP_PROGRESSION_TARGET}"
    MANIFEST_ARGS+=(--manifest-file "${MANIFEST_ROOT}/gdp_progression_forecasting_${GDP_PROGRESSION_TARGET}.csv")
  fi

  prediction_file="${PREDICTIONS_ROOT}/gdp_${task_suffix}_retfound_oct.csv"
  checkpoint_file="${CHECKPOINT_ROOT}/gdp_${task_suffix}_retfound_oct_linear_probe.pkl"
  metrics_dir="${METRICS_ROOT}/exp8_gdp_${task_suffix}_retfound_oct"

  python equi-agent/scripts/predict_gdp_retfound_oct.py \
    --task "${task}" \
    "${MANIFEST_ARGS[@]}" \
    --mode linear-probe \
    --backbone-weights "${RETFOUND_BACKBONE_WEIGHTS}" \
    --threshold-metric "${THRESHOLD_METRIC}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --checkpoint "${checkpoint_file}" \
    --out "${prediction_file}" \
    "${PATH_ARGS[@]}"

  python equi-agent/scripts/evaluate_predictions.py \
    --predictions "${prediction_file}" \
    --out-dir "${metrics_dir}"
done

python equi-agent/scripts/build_manuscript_tables.py \
  --metrics-root "${METRICS_ROOT}" \
  --out-dir "${TABLES_ROOT}"

echo "GDP RETFound OCT complete."
echo "Predictions: ${PREDICTIONS_ROOT}/gdp_*_retfound_oct.csv"
echo "Metrics: ${METRICS_ROOT}/exp8_gdp_*_retfound_oct"
