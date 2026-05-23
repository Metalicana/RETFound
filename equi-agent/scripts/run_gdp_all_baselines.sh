#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-Datasets}"
MANIFEST_ROOT="${MANIFEST_ROOT:-equi-agent/outputs/manifests}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
TABLES_ROOT="${TABLES_ROOT:-equi-agent/outputs/tables}"
TASKS="${TASKS:-glaucoma_detection progression_forecasting}"
FEATURE_SETS="${FEATURE_SETS:-rnflt clinical rnflt_clinical bscan bscan_clinical all}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1}"

python equi-agent/scripts/build_manifests.py \
  --datasets-root "${DATASETS_ROOT}" \
  --out-dir "${MANIFEST_ROOT}"

for task in ${TASKS}; do
  for feature_set in ${FEATURE_SETS}; do
    prediction_file="${PREDICTIONS_ROOT}/gdp_${task}_${feature_set}_logreg.csv"
    metrics_dir="${METRICS_ROOT}/exp8_gdp_${task}_${feature_set}"

    python equi-agent/scripts/predict_gdp_classical_baselines.py \
      --manifest-dir "${MANIFEST_ROOT}" \
      --task "${task}" \
      --feature-set "${feature_set}" \
      --threshold-metric "${THRESHOLD_METRIC}" \
      --out "${prediction_file}"

    python equi-agent/scripts/evaluate_predictions.py \
      --predictions "${prediction_file}" \
      --out-dir "${metrics_dir}"
  done
done

python equi-agent/scripts/build_manuscript_tables.py \
  --metrics-root "${METRICS_ROOT}" \
  --out-dir "${TABLES_ROOT}"

echo "GDP baselines complete."
echo "Predictions: ${PREDICTIONS_ROOT}/gdp_*_logreg.csv"
echo "Metrics: ${METRICS_ROOT}/exp8_gdp_*"
echo "Tables: ${TABLES_ROOT}"
