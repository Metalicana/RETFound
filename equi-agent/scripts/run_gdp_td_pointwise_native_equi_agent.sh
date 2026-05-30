#!/usr/bin/env bash
set -euo pipefail

# End-to-end GDP progression arbitration for the less-sparse td_pointwise_no_p_cut
# target. This workflow assumes the native Harvard-GDP RNFLT+TDS EfficientNet
# checkpoint has already been exported to the prediction CSV below.

TARGET="${GDP_PROGRESSION_TARGET:-td_pointwise_no_p_cut}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
NATIVE_MODEL="${NATIVE_GDP_MODEL:-gdp_native_rnflt_tds_efficientnet}"
PREDICTION_PREFIX="${PREDICTION_PREFIX:-gdp_progression_forecasting_${TARGET}}"
METRICS_PREFIX="${METRICS_PREFIX:-exp8_gdp_progression_forecasting_${TARGET}}"
NATIVE_PREDICTIONS="${PREDICTIONS_ROOT}/${PREDICTION_PREFIX}_${NATIVE_MODEL}.csv"
NATIVE_METRICS_DIR="${METRICS_ROOT}/${METRICS_PREFIX}_${NATIVE_MODEL}"
NATIVE_AGGREGATE="${NATIVE_METRICS_DIR}/${PREDICTION_PREFIX}_${NATIVE_MODEL}_aggregate.csv"

if [[ ! -f "$NATIVE_PREDICTIONS" ]]; then
  echo "Missing native GDP prediction CSV: $NATIVE_PREDICTIONS" >&2
  echo "Export the trained Harvard-GDP RNFLT+TDS EfficientNet checkpoint to this CSV before running live Equi-Agent." >&2
  exit 2
fi

if [[ ! -f "$NATIVE_AGGREGATE" ]]; then
  python equi-agent/scripts/evaluate_predictions.py \
    --predictions "$NATIVE_PREDICTIONS" \
    --out-dir "$NATIVE_METRICS_DIR" \
    --min-positive 5 \
    --min-negative 5
fi

MODELS="${MODELS:-$NATIVE_MODEL rnflt_logreg clinical_logreg bscan_logreg retfound_oct}"
REFERENCE_STRATEGY="${REFERENCE_STRATEGY:-best_auroc}"
LOCK_REFERENCE_PREDICTION="${LOCK_REFERENCE_PREDICTION:-1}"
REQUIRE_NATIVE_GDP_MODEL="${REQUIRE_NATIVE_GDP_MODEL:-1}"
MAX_CASES="${MAX_CASES:-0}"
PREDICTION_PREFIX="$PREDICTION_PREFIX"
METRICS_PREFIX="$METRICS_PREFIX"

export MODELS
export REFERENCE_STRATEGY
export LOCK_REFERENCE_PREDICTION
export REQUIRE_NATIVE_GDP_MODEL
export MAX_CASES
export PREDICTION_PREFIX
export METRICS_PREFIX

bash equi-agent/scripts/run_gdp_progression_equi_agent_live.sh
