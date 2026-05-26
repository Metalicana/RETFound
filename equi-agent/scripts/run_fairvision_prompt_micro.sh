#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-equi-agent/outputs/manifests}"
TABLES_ROOT="${TABLES_ROOT:-equi-agent/outputs/tables}"
OUT_ROOT="${OUT_ROOT:-equi-agent/outputs/prompt_micro}"

TASKS="${TASKS:-amd dr glaucoma}"
MODELS="${MODELS:-retfound_oct mirage_slo flair_slo ret_clip_slo visionfm_slo visionfm_oct retizero_slo urfound_slo urfound_oct}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-current visual_first f1_rescue}"
CASES_PER_TASK="${CASES_PER_TASK:-100}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
SEED_OFFSET="${SEED_OFFSET:-0}"
DEPLOYMENT="${DEPLOYMENT:-gpt-5.1}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-700}"
MAX_PROBABILITY_ADJUSTMENT="${MAX_PROBABILITY_ADJUSTMENT:-0.10}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-768}"
INCLUDE_IMAGE_TOKENS="${INCLUDE_IMAGE_TOKENS:-1}"
DRY_RUN="${DRY_RUN:-0}"
REQUEST_SLEEP_SEC="${REQUEST_SLEEP_SEC:-0}"
TARGET_POSITIVE_FRAC="${TARGET_POSITIVE_FRAC:-0.50}"
SENSITIVITY_THRESHOLD="${SENSITIVITY_THRESHOLD:-0.35}"
NEUTRAL_THRESHOLD="${NEUTRAL_THRESHOLD:-0.50}"
PRECISION_THRESHOLD="${PRECISION_THRESHOLD:-0.65}"

for variant in ${PROMPT_VARIANTS}; do
  out_dir="${OUT_ROOT}/${variant}_n${CASES_PER_TASK}_seed${RANDOM_SEED}"
  table_dir="${TABLES_ROOT}/prompt_micro_${variant}_n${CASES_PER_TASK}_seed${RANDOM_SEED}"

  args=(
    --predictions-root "${PREDICTIONS_ROOT}"
    --metrics-root "${METRICS_ROOT}"
    --manifests-root "${MANIFESTS_ROOT}"
    --out-dir "${out_dir}"
    --tasks ${TASKS}
    --models ${MODELS}
    --max-cases-per-task "${CASES_PER_TASK}"
    --sample-stratified
    --target-positive-frac "${TARGET_POSITIVE_FRAC}"
    --seed-offset "${SEED_OFFSET}"
    --random-seed "${RANDOM_SEED}"
    --temperature "${TEMPERATURE}"
    --max-output-tokens "${MAX_OUTPUT_TOKENS}"
    --deployment "${DEPLOYMENT}"
    --max-probability-adjustment "${MAX_PROBABILITY_ADJUSTMENT}"
    --request-sleep-sec "${REQUEST_SLEEP_SEC}"
    --prompt-variant "${variant}"
    --max-image-side "${MAX_IMAGE_SIDE}"
    --sensitivity-threshold "${SENSITIVITY_THRESHOLD}"
    --neutral-threshold "${NEUTRAL_THRESHOLD}"
    --precision-threshold "${PRECISION_THRESHOLD}"
  )

  if [[ "${INCLUDE_IMAGE_TOKENS}" == "1" ]]; then
    args+=(--include-image-tokens)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
  fi

  python equi-agent/scripts/run_equi_agent_fairvision_live.py "${args[@]}"

  python equi-agent/scripts/summarize_equi_agent_live.py \
    --predictions "${out_dir}/equi_agent_live_predictions.csv" \
    --out-json "${out_dir}/equi_agent_live_metrics_summary.json" \
    --out-csv "${out_dir}/equi_agent_live_metrics_summary.csv"

  python equi-agent/scripts/compare_equi_agent_subset.py \
    --equi-predictions "${out_dir}/equi_agent_live_predictions.csv" \
    --predictions-root "${PREDICTIONS_ROOT}" \
    --global-priors "${METRICS_ROOT}/validation_subgroup_priors_global.csv" \
    --out-dir "${table_dir}"

  echo "Prompt micro complete: variant=${variant}"
  echo "Metrics: ${out_dir}/equi_agent_live_metrics_summary.csv"
  echo "Comparison: ${table_dir}/subset_comparison_metrics.csv"
done
