#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
OUT_DIR="${OUT_DIR:-equi-agent/outputs/equi_agent_gdp_progression_live}"
EVAL_DIR="${EVAL_DIR:-equi-agent/outputs/metrics/exp8_gdp_progression_forecasting_equi_agent_longitudinal}"
MODELS="${MODELS:-retfound_oct visionfm_oct urfound_oct}"
MAX_CASES="${MAX_CASES:-0}"
SEED_OFFSET="${SEED_OFFSET:-0}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-${OPENAI_MODEL:-gpt-5.1}}"
API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_CLASSICAL_BASELINES="${INCLUDE_CLASSICAL_BASELINES:-0}"
SAMPLE_RANDOM="${SAMPLE_RANDOM:-0}"
REQUEST_SLEEP_SEC="${REQUEST_SLEEP_SEC:-0}"
REFERENCE_STRATEGY="${REFERENCE_STRATEGY:-${GDP_AGENT_REFERENCE_STRATEGY:-weighted}}"
LOCK_REFERENCE_PREDICTION="${LOCK_REFERENCE_PREDICTION:-${GDP_AGENT_LOCK_REFERENCE_PREDICTION:-0}}"

args=(
  --predictions-root "$PREDICTIONS_ROOT"
  --metrics-root "$METRICS_ROOT"
  --out-dir "$OUT_DIR"
  --models $MODELS
  --max-cases "$MAX_CASES"
  --seed-offset "$SEED_OFFSET"
  --random-seed "$RANDOM_SEED"
  --deployment "$DEPLOYMENT"
  --api-version "$API_VERSION"
  --request-sleep-sec "$REQUEST_SLEEP_SEC"
  --reference-strategy "$REFERENCE_STRATEGY"
)

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

if [[ "$INCLUDE_CLASSICAL_BASELINES" == "1" ]]; then
  args+=(--include-classical-baselines)
fi

if [[ "$SAMPLE_RANDOM" == "1" ]]; then
  args+=(--sample-random)
fi

if [[ "$LOCK_REFERENCE_PREDICTION" == "1" ]]; then
  args+=(--lock-reference-prediction)
fi

python equi-agent/scripts/run_equi_agent_gdp_progression_live.py "${args[@]}"

if python -c "import pandas" >/dev/null 2>&1; then
  python equi-agent/scripts/evaluate_predictions.py \
    --predictions "$OUT_DIR/equi_agent_gdp_progression_predictions.csv" \
    --out-dir "$EVAL_DIR" \
    --min-positive 5 \
    --min-negative 5
else
  echo "Skipping evaluate_predictions.py because pandas is not installed in this environment."
  echo "Predictions written to: $OUT_DIR/equi_agent_gdp_progression_predictions.csv"
fi
