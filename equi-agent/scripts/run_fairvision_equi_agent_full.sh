#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
OUT_DIR="${OUT_DIR:-equi-agent/outputs/equi_agent_live_full_tuned_v3}"
TABLES_OUT_DIR="${TABLES_OUT_DIR:-equi-agent/outputs/tables/equi_agent_full_compare}"

TASKS="${TASKS:-amd dr glaucoma}"
MODELS="${MODELS:-retfound_oct mirage_slo flair_slo ret_clip_slo visionfm_slo visionfm_oct retizero_slo urfound_slo urfound_oct}"
DEPLOYMENT="${DEPLOYMENT:-gpt-5.1}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-700}"
MAX_PROBABILITY_ADJUSTMENT="${MAX_PROBABILITY_ADJUSTMENT:-0.10}"
REQUEST_SLEEP_SEC="${REQUEST_SLEEP_SEC:-0}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-5}"
MAX_CASES_PER_TASK="${MAX_CASES_PER_TASK:-0}"
SEED_OFFSET="${SEED_OFFSET:-0}"
SAMPLE_RANDOM="${SAMPLE_RANDOM:-0}"
SAMPLE_STRATIFIED="${SAMPLE_STRATIFIED:-0}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
TARGET_POSITIVE_FRAC="${TARGET_POSITIVE_FRAC:-0.50}"
MIN_POSITIVE_FRAC="${MIN_POSITIVE_FRAC:-0.15}"
MAX_POSITIVE_FRAC="${MAX_POSITIVE_FRAC:-0.85}"
PROMPT_VARIANT="${PROMPT_VARIANT:-${EQUI_AGENT_PROMPT_VARIANT:-current}}"
INCLUDE_IMAGE_TOKENS="${INCLUDE_IMAGE_TOKENS:-0}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-equi-agent/outputs/manifests}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-768}"
SENSITIVITY_THRESHOLD="${SENSITIVITY_THRESHOLD:-0.35}"
NEUTRAL_THRESHOLD="${NEUTRAL_THRESHOLD:-0.50}"
PRECISION_THRESHOLD="${PRECISION_THRESHOLD:-0.65}"
DRY_RUN="${DRY_RUN:-0}"
RUN_COMPARE="${RUN_COMPARE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

COMMON_ARGS=(
  --predictions-root "${PREDICTIONS_ROOT}"
  --metrics-root "${METRICS_ROOT}"
  --manifests-root "${MANIFESTS_ROOT}"
  --out-dir "${OUT_DIR}"
  --tasks ${TASKS}
  --models ${MODELS}
  --max-cases-per-task "${MAX_CASES_PER_TASK}"
  --seed-offset "${SEED_OFFSET}"
  --random-seed "${RANDOM_SEED}"
  --target-positive-frac "${TARGET_POSITIVE_FRAC}"
  --min-positive-frac "${MIN_POSITIVE_FRAC}"
  --max-positive-frac "${MAX_POSITIVE_FRAC}"
  --temperature "${TEMPERATURE}"
  --max-output-tokens "${MAX_OUTPUT_TOKENS}"
  --deployment "${DEPLOYMENT}"
  --max-probability-adjustment "${MAX_PROBABILITY_ADJUSTMENT}"
  --request-sleep-sec "${REQUEST_SLEEP_SEC}"
  --max-retries "${MAX_RETRIES}"
  --retry-sleep-sec "${RETRY_SLEEP_SEC}"
  --prompt-variant "${PROMPT_VARIANT}"
  --max-image-side "${MAX_IMAGE_SIDE}"
  --sensitivity-threshold "${SENSITIVITY_THRESHOLD}"
  --neutral-threshold "${NEUTRAL_THRESHOLD}"
  --precision-threshold "${PRECISION_THRESHOLD}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

if [[ "${SAMPLE_RANDOM}" == "1" ]]; then
  COMMON_ARGS+=(--sample-random)
fi

if [[ "${SAMPLE_STRATIFIED}" == "1" ]]; then
  COMMON_ARGS+=(--sample-stratified)
fi

if [[ "${INCLUDE_IMAGE_TOKENS}" == "1" ]]; then
  COMMON_ARGS+=(--include-image-tokens)
fi

python equi-agent/scripts/run_equi_agent_fairvision_live.py "${COMMON_ARGS[@]}"

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  python equi-agent/scripts/summarize_equi_agent_live.py \
    --predictions "${OUT_DIR}/equi_agent_live_predictions.csv" \
    --out-json "${OUT_DIR}/equi_agent_live_metrics_summary.json" \
    --out-csv "${OUT_DIR}/equi_agent_live_metrics_summary.csv"
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
  python equi-agent/scripts/compare_equi_agent_subset.py \
    --equi-predictions "${OUT_DIR}/equi_agent_live_predictions.csv" \
    --predictions-root "${PREDICTIONS_ROOT}" \
    --global-priors "${METRICS_ROOT}/validation_subgroup_priors_global.csv" \
    --out-dir "${TABLES_OUT_DIR}"
fi

echo "Full FairVision Equi-Agent run complete."
echo "Predictions: ${OUT_DIR}/equi_agent_live_predictions.csv"
echo "Summary: ${OUT_DIR}/equi_agent_live_summary.json"
echo "Metrics summary: ${OUT_DIR}/equi_agent_live_metrics_summary.csv"
echo "Exact-case comparison: ${TABLES_OUT_DIR}/subset_comparison_metrics.csv"
