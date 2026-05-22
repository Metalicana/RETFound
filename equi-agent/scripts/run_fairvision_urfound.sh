#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-amd dr glaucoma}"
MODALITY="${MODALITY:-slo}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-equi-agent/outputs/checkpoints}"
URFOUND_ROOT="${URFOUND_ROOT:-Foundation_Models/UrFound-main}"
URFOUND_WEIGHTS="${URFOUND_WEIGHTS:-}"
URFOUND_MODEL="${URFOUND_MODEL:-vit_base_patch16}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-balanced_accuracy}"
MAX_ITER="${MAX_ITER:-5000}"
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"

if [[ "${MODALITY}" != "slo" && "${MODALITY}" != "oct" ]]; then
  echo "MODALITY must be 'slo' or 'oct'." >&2
  exit 1
fi

if [[ -z "${URFOUND_WEIGHTS}" ]]; then
  echo "URFOUND_WEIGHTS must point to a downloaded UrFound checkpoint." >&2
  exit 1
fi

COMMON_ARGS=(
  --urfound-root "${URFOUND_ROOT}"
  --pretrained-weights "${URFOUND_WEIGHTS}"
  --model "${URFOUND_MODEL}"
  --modality "${MODALITY}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --max-iter "${MAX_ITER}"
)

if [[ -n "${PATH_PREFIX_FROM}" ]]; then
  COMMON_ARGS+=(--path-prefix-from "${PATH_PREFIX_FROM}" --path-prefix-to "${PATH_PREFIX_TO}")
fi

for task in ${TASKS}; do
  stem="fairvision_${task}_urfound_${MODALITY}"
  val_file="${PREDICTIONS_ROOT}/${stem}_val.csv"
  test_file="${PREDICTIONS_ROOT}/${stem}_test.csv"
  thresholded_file="${PREDICTIONS_ROOT}/${stem}_test_thresholded.csv"
  thresholds_file="${METRICS_ROOT}/thresholds_${stem}.csv"
  metrics_dir="${METRICS_ROOT}/exp2_urfound_${MODALITY}_${task}"
  checkpoint_file="${CHECKPOINT_ROOT}/${stem}_linear_probe.pkl"

  python equi-agent/scripts/train_fairvision_urfound.py \
    --task "${task}" \
    --checkpoint "${checkpoint_file}" \
    --out-val "${val_file}" \
    --out-test "${test_file}" \
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

combined_thresholded_file="${PREDICTIONS_ROOT}/fairvision_urfound_${MODALITY}_test_thresholded.csv"
first_task=true
for task in ${TASKS}; do
  task_file="${PREDICTIONS_ROOT}/fairvision_${task}_urfound_${MODALITY}_test_thresholded.csv"
  if [[ "${first_task}" == "true" ]]; then
    head -n 1 "${task_file}" > "${combined_thresholded_file}"
    first_task=false
  fi
  tail -n +2 "${task_file}" >> "${combined_thresholded_file}"
done

python equi-agent/scripts/evaluate_predictions.py \
  --predictions "${combined_thresholded_file}" \
  --out-dir "${METRICS_ROOT}/exp2_urfound_${MODALITY}"

python equi-agent/scripts/build_manuscript_tables.py \
  --metrics-root "${METRICS_ROOT}" \
  --out-dir equi-agent/outputs/tables

echo "UrFound ${MODALITY} FairVision experiment complete."
