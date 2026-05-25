#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-Datasets}"
MANIFEST_ROOT="${MANIFEST_ROOT:-equi-agent/outputs/manifests}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
FEATURE_SETS="${FEATURE_SETS:-rnflt clinical rnflt_clinical bscan bscan_clinical all}"
PROGRESSION_TARGETS="${PROGRESSION_TARGETS:-td_pointwise md_fast md td_pointwise_no_p_cut vfi md_fast_no_p_cut}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1}"

python equi-agent/scripts/build_manifests.py \
  --datasets-root "${DATASETS_ROOT}" \
  --out-dir "${MANIFEST_ROOT}" \
  --datasets gdp

for target in ${PROGRESSION_TARGETS}; do
  manifest_file="${MANIFEST_ROOT}/gdp_progression_forecasting_${target}.csv"
  for feature_set in ${FEATURE_SETS}; do
    prediction_file="${PREDICTIONS_ROOT}/gdp_progression_forecasting_${target}_${feature_set}_logreg.csv"
    metrics_dir="${METRICS_ROOT}/exp8_gdp_progression_forecasting_${target}_${feature_set}"

    python equi-agent/scripts/predict_gdp_classical_baselines.py \
      --task progression_forecasting \
      --manifest-file "${manifest_file}" \
      --model-prefix "gdp_progression_forecasting_${target}" \
      --feature-set "${feature_set}" \
      --threshold-metric "${THRESHOLD_METRIC}" \
      --out "${prediction_file}"

    python equi-agent/scripts/evaluate_predictions.py \
      --predictions "${prediction_file}" \
      --out-dir "${metrics_dir}"
  done
done

echo "GDP target-specific progression baselines complete."
echo "Targets: ${PROGRESSION_TARGETS}"
echo "Predictions: ${PREDICTIONS_ROOT}/gdp_progression_forecasting_*_logreg.csv"
echo "Metrics: ${METRICS_ROOT}/exp8_gdp_progression_forecasting_*_logreg"
