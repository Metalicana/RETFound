#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-amd dr glaucoma}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-equi-agent/outputs/checkpoints}"
RETIZERO_ROOT="${RETIZERO_ROOT:-Foundation_Models/RetiZero-main}"
RETIZERO_WEIGHTS="${RETIZERO_WEIGHTS:-${RETIZERO_ROOT}/pretrained/RetiZero.pth}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-balanced_accuracy}"
MAX_ITER="${MAX_ITER:-5000}"
PROBE_KIND="${PROBE_KIND:-torch_mlp}"
PROBE_EPOCHS="${PROBE_EPOCHS:-60}"
PROBE_LR="${PROBE_LR:-0.001}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.0001}"
PROBE_HIDDEN_DIM="${PROBE_HIDDEN_DIM:-256}"
PROBE_DROPOUT="${PROBE_DROPOUT:-0.2}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
FAIRVISION_AMD_STAGES="${FAIRVISION_AMD_STAGES:-1}"
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"

COMMON_ARGS=(
  --retizero-root "${RETIZERO_ROOT}"
  --pretrained-weights "${RETIZERO_WEIGHTS}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --max-iter "${MAX_ITER}"
  --probe-kind "${PROBE_KIND}"
  --probe-epochs "${PROBE_EPOCHS}"
  --probe-lr "${PROBE_LR}"
  --probe-weight-decay "${PROBE_WEIGHT_DECAY}"
  --probe-hidden-dim "${PROBE_HIDDEN_DIM}"
  --probe-dropout "${PROBE_DROPOUT}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
)

if [[ -n "${PATH_PREFIX_FROM}" ]]; then
  COMMON_ARGS+=(--path-prefix-from "${PATH_PREFIX_FROM}" --path-prefix-to "${PATH_PREFIX_TO}")
fi

for task in ${TASKS}; do
  task_args=()
  if [[ "${task}" == "amd" && ("${FAIRVISION_AMD_STAGES}" == "1" || "${FAIRVISION_AMD_STAGES}" == "true") ]]; then
    task_args+=(--fairvision-amd-stages)
  fi
  stem="fairvision_${task}_retizero_slo"
  val_file="${PREDICTIONS_ROOT}/${stem}_val.csv"
  test_file="${PREDICTIONS_ROOT}/${stem}_test.csv"
  thresholded_file="${PREDICTIONS_ROOT}/${stem}_test_thresholded.csv"
  thresholds_file="${METRICS_ROOT}/thresholds_${stem}.csv"
  metrics_dir="${METRICS_ROOT}/exp2_retizero_slo_${task}"
  if [[ "${PROBE_KIND}" == "torch_mlp" ]]; then
    checkpoint_file="${CHECKPOINT_ROOT}/${stem}_torch_probe.pt"
  else
    checkpoint_file="${CHECKPOINT_ROOT}/${stem}_linear_probe.pkl"
  fi

  python equi-agent/scripts/train_fairvision_retizero.py \
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

combined_thresholded_file="${PREDICTIONS_ROOT}/fairvision_retizero_slo_test_thresholded.csv"
first_task=true
for task in ${TASKS}; do
  task_file="${PREDICTIONS_ROOT}/fairvision_${task}_retizero_slo_test_thresholded.csv"
  if [[ "${first_task}" == "true" ]]; then
    head -n 1 "${task_file}" > "${combined_thresholded_file}"
    first_task=false
  fi
  tail -n +2 "${task_file}" >> "${combined_thresholded_file}"
done

python equi-agent/scripts/evaluate_predictions.py \
  --predictions "${combined_thresholded_file}" \
  --out-dir "${METRICS_ROOT}/exp2_retizero_slo"

python equi-agent/scripts/build_manuscript_tables.py \
  --metrics-root "${METRICS_ROOT}" \
  --out-dir equi-agent/outputs/tables

echo "RetiZero SLO FairVision experiment complete."
