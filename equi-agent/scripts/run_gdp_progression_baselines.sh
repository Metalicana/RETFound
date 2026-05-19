#!/usr/bin/env bash
set -euo pipefail

METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
TABLES_ROOT="${TABLES_ROOT:-equi-agent/outputs/tables}"
FEATURE_SETS="${FEATURE_SETS:-rnflt clinical rnflt_clinical bscan bscan_clinical all}"

for feature_set in ${FEATURE_SETS}; do
  prediction_file="${PREDICTIONS_ROOT}/gdp_progression_${feature_set}_logreg.csv"
  metrics_dir="${METRICS_ROOT}/exp8_gdp_progression_${feature_set}"

  python equi-agent/scripts/predict_gdp_classical_baselines.py \
    --task progression_forecasting \
    --feature-set "${feature_set}" \
    --out "${prediction_file}"

  python equi-agent/scripts/evaluate_predictions.py \
    --predictions "${prediction_file}" \
    --out-dir "${metrics_dir}"
done

python equi-agent/scripts/build_manuscript_tables.py \
  --metrics-root "${METRICS_ROOT}" \
  --out-dir "${TABLES_ROOT}"

echo "GDP progression baselines complete."
echo "Longitudinal table: ${TABLES_ROOT}/exp8_table11_longitudinal_glaucoma.md"
