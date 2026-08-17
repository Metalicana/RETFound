#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-equi-agent/outputs/baselines/gdp_progression_llm_v1}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
TARGETS="md vfi td_pointwise md_fast md_fast_no_p_cut td_pointwise_no_p_cut"
MODELS="${MODELS:-gpt-5.1 gpt-5.6-luna claude-haiku-4.5}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$RUN_ROOT"
COLLECTOR_MODEL_SLUGS=()

slug_for_model() {
  case "$1" in
    gpt-5.1) echo "gpt51" ;;
    gpt-5.6-luna) echo "gpt56_luna" ;;
    claude-haiku-4.5) echo "claude_haiku45" ;;
    *) echo "Unsupported model: $1" >&2; return 1 ;;
  esac
}

deployment_for_model() {
  case "$1" in
    gpt-5.1) echo "${GPT51_DEPLOYMENT:-gpt-5.1}" ;;
    gpt-5.6-luna) echo "${GPT56_DEPLOYMENT:-gpt-5.6-luna}" ;;
    claude-haiku-4.5) echo "${CLAUDE_HAIKU45_DEPLOYMENT:-claude-haiku-4-5}" ;;
    *) echo "Unsupported model: $1" >&2; return 1 ;;
  esac
}

for model in $MODELS; do
  COLLECTOR_MODEL_SLUGS+=("$(slug_for_model "$model")")
done

for target in $TARGETS; do
  manifest="equi-agent/outputs/manifests/gdp_progression_forecasting_${target}.csv"
  [[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 1; }
done

for model in $MODELS; do
  slug="$(slug_for_model "$model")"
  deployment="$(deployment_for_model "$model")"
  out_dir="$RUN_ROOT/$slug"

  echo
  echo "=== all six targets; model=$model deployment=$deployment ==="
  "$PYTHON_BIN" equi-agent/scripts/run_gdp_progression_llm_multitarget_baseline.py \
    --manifests-root equi-agent/outputs/manifests \
    --out-dir "$out_dir" \
    --model "$model" \
    --deployment "$deployment"

  "$PYTHON_BIN" - "$out_dir/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if not summary.get("complete_locked_cohort"):
    raise SystemExit(
        f"Incomplete locked cohort: completed_calls={summary.get('completed_calls')} "
        f"missing_calls={summary.get('missing_calls')}"
    )
PY

  for target in $TARGETS; do
    "$PYTHON_BIN" equi-agent/scripts/evaluate_predictions.py \
      --predictions "$out_dir/predictions_${target}.csv" \
      --out-dir "$METRICS_ROOT/exp8_gdp_progression_forecasting_${target}_llm_${slug}"
  done
done

"$PYTHON_BIN" equi-agent/scripts/collect_gdp_progression_llm_results.py \
  --run-root "$RUN_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --out-dir "$RUN_ROOT/collected_results" \
  --targets $TARGETS \
  --model-slugs "${COLLECTOR_MODEL_SLUGS[@]}"

echo
echo "GDP progression LLM baseline suite complete."
