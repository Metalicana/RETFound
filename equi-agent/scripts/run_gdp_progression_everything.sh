#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-equi-agent/outputs/gdp_progression_everything_v1}"
LOG_FILE="$RUN_ROOT/master.log"
PID_FILE="$RUN_ROOT/master.pid"
STATE_FILE="$RUN_ROOT/current_stage.txt"
RESULTS_DIR="$RUN_ROOT/complete_results"

launch() {
  mkdir -p "$RUN_ROOT"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running: PID=$(cat "$PID_FILE")"
    echo "Log: $LOG_FILE"
    return 0
  fi
  nohup bash "$0" --execute >> "$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Started complete GDP progression suite: PID=$pid"
  echo "Monitor: bash $0 --status"
  echo "Log: $LOG_FILE"
}

status() {
  echo "run_root=$RUN_ROOT"
  if [[ -f "$STATE_FILE" ]]; then
    echo "stage=$(cat "$STATE_FILE")"
  else
    echo "stage=not_started"
  fi
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      ps -p "$pid" -o pid,etime,%cpu,%mem,cmd
    else
      echo "process_not_running pid=$pid"
    fi
  fi
  if [[ -f "$RESULTS_DIR/completion_status.json" ]]; then
    cat "$RESULTS_DIR/completion_status.json"
  fi
  if [[ -f "$LOG_FILE" ]]; then
    echo "--- last 40 log lines ---"
    tail -n 40 "$LOG_FILE"
  fi
}

audit() {
  mkdir -p "$RESULTS_DIR"
  python equi-agent/scripts/collect_gdp_progression_complete_results.py \
    --out-dir "$RESULTS_DIR" \
    --allow-incomplete
}

case "${1:---execute}" in
  --launch)
    launch
    exit 0
    ;;
  --status)
    status
    exit 0
    ;;
  --audit)
    audit
    exit 0
    ;;
  --execute)
    ;;
  *)
    echo "Usage: $0 [--launch|--status|--audit|--execute]" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT" "$RESULTS_DIR"
echo $$ > "$PID_FILE"

CURRENT_STAGE="preflight"
on_error() {
  local exit_code=$?
  printf 'FAILED stage=%s line=%s exit=%s utc=%s\n' \
    "$CURRENT_STAGE" "${BASH_LINENO[0]:-unknown}" "$exit_code" "$(date -u +%FT%TZ)" \
    > "$STATE_FILE"
  echo "FAILED: stage=$CURRENT_STAGE exit=$exit_code" >&2
  echo "Resume with: bash $0 --launch" >&2
  exit "$exit_code"
}
trap on_error ERR

mark_stage() {
  CURRENT_STAGE="$1"
  printf 'RUNNING stage=%s utc=%s\n' "$CURRENT_STAGE" "$(date -u +%FT%TZ)" > "$STATE_FILE"
  echo
  echo "=== $CURRENT_STAGE ==="
}

TARGETS=(md vfi td_pointwise md_fast md_fast_no_p_cut td_pointwise_no_p_cut)
FEATURE_SETS=(rnflt clinical bscan rnflt_clinical bscan_clinical all)
EXPECTED_POSITIVES=(18 19 18 4 6 60)
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-equi-agent/outputs/manifests}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-equi-agent/outputs/checkpoints}"
LLM_ROOT="${LLM_ROOT:-equi-agent/outputs/baselines/gdp_progression_llm_v1}"
AGENT_ROOT="${AGENT_ROOT:-equi-agent/outputs/equi_agent_gdp_progression_multitarget_live}"
DATASETS_ROOT="${DATASETS_ROOT:-Datasets}"
GPU="${GPU:-0}"
PYTHON_ENV="${PYTHON_ENV:-retfound}"
RETFOUND_WEIGHTS="${RETFOUND_WEIGHTS:-equi-agent/weights/RETFound_mae_natureOCT.pth}"
LLM_MODELS="${LLM_MODELS:-gpt-5.1 gpt-5.6-luna claude-haiku-4.5}"
AGENT_DEPLOYMENT="${AGENT_DEPLOYMENT:-gpt-5.1}"
REFERENCE_STRATEGY="${REFERENCE_STRATEGY:-weighted}"

run_python() {
  conda run --no-capture-output -n "$PYTHON_ENV" python "$@"
}

csv_complete() {
  local path=$1
  local positives=$2
  [[ -f "$path" ]] || return 1
  python -c 'import csv,sys
rows=list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
positives=sum(str(row.get("y_true", "")).strip() in {"1", "1.0"} for row in rows)
raise SystemExit(0 if len(rows)==200 and positives==int(sys.argv[2]) else 1)' \
    "$path" "$positives"
}

metric_complete() {
  local path=$1
  [[ -s "$path" ]] || return 1
  python -c 'import csv,sys
rows=list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
raise SystemExit(0 if len(rows)==1 and int(float(rows[0].get("n", 0)))==200 else 1)' \
    "$path"
}

evaluate_if_missing() {
  local predictions=$1
  local metrics_dir=$2
  local aggregate_name=$3
  if ! metric_complete "$metrics_dir/$aggregate_name"; then
    run_python equi-agent/scripts/evaluate_predictions.py \
      --predictions "$predictions" \
      --out-dir "$metrics_dir" \
      --min-positive 2 \
      --min-negative 5
  fi
}

mark_stage "preflight"
command -v conda >/dev/null
conda env list | grep -qE "^${PYTHON_ENV}[[:space:]]"
[[ -f "$RETFOUND_WEIGHTS" ]]
[[ -d "$DATASETS_ROOT/GDP" ]]
run_python -c 'import numpy,pandas,sklearn,torch,torchvision; print("runtime imports ok")'
run_python -c 'import os,sys
from dotenv import load_dotenv
load_dotenv()
models=sys.argv[1].split()
if any(model.startswith("gpt-") for model in models):
    import openai
    assert os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE"), "missing Azure OpenAI endpoint"
    assert os.getenv("AZURE_OPENAI_API_KEY"), "missing AZURE_OPENAI_API_KEY"
if "claude-haiku-4.5" in models:
    from anthropic import AnthropicFoundry
    assert os.getenv("ANTHROPIC_FOUNDRY_BASE_URL") or os.getenv("AZURE_AI_ANTHROPIC_ENDPOINT"), "missing Claude Foundry endpoint"
    assert os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"), "missing Claude API key"
print("API clients and credentials ok")' "$LLM_MODELS"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT" "$MANIFESTS_ROOT" "$CHECKPOINTS_ROOT" "$LLM_ROOT" "$AGENT_ROOT"

mark_stage "manifests"
run_python equi-agent/scripts/build_manifests.py \
  --datasets-root "$DATASETS_ROOT" \
  --datasets gdp \
  --out-dir "$MANIFESTS_ROOT"
for index in "${!TARGETS[@]}"; do
  target="${TARGETS[$index]}"
  manifest="$MANIFESTS_ROOT/gdp_progression_forecasting_${target}.csv"
  [[ -f "$manifest" ]]
  rows=$(( $(wc -l < "$manifest") - 1 ))
  [[ "$rows" -eq 500 ]]
done

mark_stage "classical_baselines"
for index in "${!TARGETS[@]}"; do
  target="${TARGETS[$index]}"
  positives="${EXPECTED_POSITIVES[$index]}"
  manifest="$MANIFESTS_ROOT/gdp_progression_forecasting_${target}.csv"
  prefix="gdp_progression_forecasting_${target}"
  for feature_set in "${FEATURE_SETS[@]}"; do
    model="${feature_set}_logreg"
    predictions="$PREDICTIONS_ROOT/${prefix}_${model}.csv"
    metrics_dir="$METRICS_ROOT/exp8_${prefix}_${feature_set}"
    aggregate="${prefix}_${model}_aggregate.csv"
    if ! csv_complete "$predictions" "$positives"; then
      run_python equi-agent/scripts/predict_gdp_classical_baselines.py \
        --task progression_forecasting \
        --manifest-file "$manifest" \
        --model-prefix "$prefix" \
        --feature-set "$feature_set" \
        --threshold-metric f1 \
        --out "$predictions"
    else
      echo "skip complete classical target=$target model=$model"
    fi
    evaluate_if_missing "$predictions" "$metrics_dir" "$aggregate"
  done
done

mark_stage "retfound"
for index in "${!TARGETS[@]}"; do
  target="${TARGETS[$index]}"
  positives="${EXPECTED_POSITIVES[$index]}"
  prefix="gdp_progression_forecasting_${target}"
  predictions="$PREDICTIONS_ROOT/${prefix}_retfound_oct.csv"
  metrics_dir="$METRICS_ROOT/exp8_${prefix}_retfound_oct"
  aggregate="${prefix}_retfound_oct_aggregate.csv"
  if ! csv_complete "$predictions" "$positives"; then
    CUDA_VISIBLE_DEVICES="$GPU" run_python equi-agent/scripts/predict_gdp_retfound_oct.py \
      --task progression_forecasting \
      --manifest-file "$MANIFESTS_ROOT/${prefix}.csv" \
      --mode linear-probe \
      --backbone-weights "$RETFOUND_WEIGHTS" \
      --threshold-metric f1 \
      --batch-size "${RETFOUND_BATCH_SIZE:-32}" \
      --num-workers "${NUM_WORKERS:-4}" \
      --device cuda:0 \
      --checkpoint "$CHECKPOINTS_ROOT/${prefix}_retfound_oct_linear_probe.pkl" \
      --path-prefix-from "${GDP_PATH_PREFIX_FROM:-/Users/metalicana/projects_spring_2026/RETFound}" \
      --path-prefix-to "$REPO_ROOT" \
      --out "$predictions"
  else
    echo "skip complete RETFound target=$target"
  fi
  evaluate_if_missing "$predictions" "$metrics_dir" "$aggregate"
done

mark_stage "native_multitarget"
native_model="gdp_native_rnflt_tds_multitask_efficientnet"
native_missing=0
for index in "${!TARGETS[@]}"; do
  target="${TARGETS[$index]}"
  positives="${EXPECTED_POSITIVES[$index]}"
  predictions="$PREDICTIONS_ROOT/gdp_progression_forecasting_${target}_${native_model}.csv"
  csv_complete "$predictions" "$positives" || native_missing=1
done
if [[ "$native_missing" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" run_python equi-agent/scripts/predict_gdp_native_multitarget.py \
    --manifests-root "$MANIFESTS_ROOT" \
    --predictions-root "$PREDICTIONS_ROOT" \
    --checkpoint "$CHECKPOINTS_ROOT/${native_model}.pt" \
    --summary "$CHECKPOINTS_ROOT/${native_model}_summary.json" \
    --device cuda:0 \
    --batch-size "${NATIVE_BATCH_SIZE:-16}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --epochs "${NATIVE_EPOCHS:-10}" \
    --folds "${NATIVE_FOLDS:-3}" \
    --path-prefix-from "${GDP_PATH_PREFIX_FROM:-/Users/metalicana/projects_spring_2026/RETFound}" \
    --path-prefix-to "$REPO_ROOT"
else
  echo "skip complete six-output native helper"
fi
for index in "${!TARGETS[@]}"; do
  target="${TARGETS[$index]}"
  prefix="gdp_progression_forecasting_${target}"
  predictions="$PREDICTIONS_ROOT/${prefix}_${native_model}.csv"
  metrics_dir="$METRICS_ROOT/exp8_${prefix}_${native_model}"
  evaluate_if_missing "$predictions" "$metrics_dir" "${prefix}_${native_model}_aggregate.csv"
done

mark_stage "standalone_llm_baselines"
PYTHON_BIN="$(conda run -n "$PYTHON_ENV" python -c 'import sys; print(sys.executable)')" \
RUN_ROOT="$LLM_ROOT" \
METRICS_ROOT="$METRICS_ROOT" \
MODELS="$LLM_MODELS" \
GDP_PATH_PREFIX_TO="$REPO_ROOT" \
bash equi-agent/scripts/run_gdp_progression_llm_baselines.sh

mark_stage "equi_agent_multitarget"
CUDA_VISIBLE_DEVICES="$GPU" run_python \
  equi-agent/scripts/run_equi_agent_gdp_progression_multitarget_live.py \
  --predictions-root "$PREDICTIONS_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --out-dir "$AGENT_ROOT" \
  --deployment "$AGENT_DEPLOYMENT" \
  --reference-strategy "$REFERENCE_STRATEGY" \
  --max-retries "${AGENT_MAX_RETRIES:-2}" \
  --retry-sleep-sec "${AGENT_RETRY_SLEEP_SEC:-5}" \
  --request-sleep-sec "${AGENT_REQUEST_SLEEP_SEC:-0}"

for target in "${TARGETS[@]}"; do
  predictions="$AGENT_ROOT/predictions_${target}.csv"
  metrics_dir="$METRICS_ROOT/exp8_gdp_progression_forecasting_${target}_equi_agent_multitarget"
  evaluate_if_missing "$predictions" "$metrics_dir" "predictions_${target}_aggregate.csv"
done

mark_stage "collect_and_verify"
run_python equi-agent/scripts/collect_gdp_progression_complete_results.py \
  --predictions-root "$PREDICTIONS_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --llm-root "$LLM_ROOT" \
  --agent-root "$AGENT_ROOT" \
  --out-dir "$RESULTS_DIR"

printf 'COMPLETE utc=%s results=%s\n' "$(date -u +%FT%TZ)" "$RESULTS_DIR" > "$STATE_FILE"
echo
echo "GDP progression suite COMPLETE."
echo "Markdown: $RESULTS_DIR/gdp_progression_complete_results.md"
echo "LaTeX: $RESULTS_DIR/gdp_progression_complete_tables.tex"
echo "Status: $RESULTS_DIR/completion_status.json"
