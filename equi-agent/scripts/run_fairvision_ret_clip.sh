#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-amd dr glaucoma}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-equi-agent/outputs/checkpoints}"
RET_CLIP_ROOT="${RET_CLIP_ROOT:-Foundation_Models/RET-CLIP-main}"
RET_CLIP_WEIGHTS="${RET_CLIP_WEIGHTS:-}"
VISION_MODEL="${VISION_MODEL:-ViT-B-16}"
TEXT_MODEL="${TEXT_MODEL:-RoBERTa-wwm-ext-base-chinese}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-balanced_accuracy}"
FAIRVISION_AMD_STAGES="${FAIRVISION_AMD_STAGES:-1}"
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"

if [[ -z "${RET_CLIP_WEIGHTS}" ]]; then
  echo "RET_CLIP_WEIGHTS must point to the RET-CLIP pretrained checkpoint." >&2
  exit 1
fi

COMMON_ARGS=(
  --ret-clip-root "${RET_CLIP_ROOT}"
  --ret-clip-weights "${RET_CLIP_WEIGHTS}"
  --vision-model "${VISION_MODEL}"
  --text-model "${TEXT_MODEL}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${PATH_PREFIX_FROM}" ]]; then
  COMMON_ARGS+=(--path-prefix-from "${PATH_PREFIX_FROM}" --path-prefix-to "${PATH_PREFIX_TO}")
fi

for task in ${TASKS}; do
  task_args=()
  if [[ "${task}" == "amd" && ("${FAIRVISION_AMD_STAGES}" == "1" || "${FAIRVISION_AMD_STAGES}" == "true") ]]; then
    task_args+=(--fairvision-amd-stages)
  fi
  stem="fairvision_${task}_ret_clip_slo"
  val_file="${PREDICTIONS_ROOT}/${stem}_val.csv"
  test_file="${PREDICTIONS_ROOT}/${stem}_test.csv"
  thresholded_file="${PREDICTIONS_ROOT}/${stem}_test_thresholded.csv"
  thresholds_file="${METRICS_ROOT}/thresholds_${stem}.csv"
  metrics_dir="${METRICS_ROOT}/exp2_ret_clip_slo_${task}"
  checkpoint_file="${CHECKPOINT_ROOT}/${stem}_linear_probe.pkl"

  python equi-agent/scripts/train_fairvision_ret_clip.py \
    --task "${task}" \
    --checkpoint "${checkpoint_file}" \
    --out-val "${val_file}" \
    --out-test "${test_file}" \
    "${task_args[@]}" \
    "${COMMON_ARGS[@]}"

  python equi-agent/scripts/tune_thresholds.py \
    --validation "${val_file}" \
    --test "${test_file}" \
    --metric "${THRESHOLD_METRIC}" \
    --thresholds-out "${thresholds_file}" \
    --test-out "${thresholded_file}"

  python equi-agent/scripts/evaluate_predictions.py \
    --predictions "${thresholded_file}" \
    --out-dir "${metrics_dir}"
done

combined_thresholded_file="${PREDICTIONS_ROOT}/fairvision_ret_clip_slo_test_thresholded.csv"
first_task=true
for task in ${TASKS}; do
  task_file="${PREDICTIONS_ROOT}/fairvision_${task}_ret_clip_slo_test_thresholded.csv"
  if [[ "${first_task}" == "true" ]]; then
    head -n 1 "${task_file}" > "${combined_thresholded_file}"
    first_task=false
  fi
  tail -n +2 "${task_file}" >> "${combined_thresholded_file}"
done

python equi-agent/scripts/evaluate_predictions.py \
  --predictions "${combined_thresholded_file}" \
  --out-dir "${METRICS_ROOT}/exp2_ret_clip_slo"

python equi-agent/scripts/build_manuscript_tables.py \
  --metrics-root "${METRICS_ROOT}" \
  --out-dir equi-agent/outputs/tables

echo "RET-CLIP FairVision experiment complete."
