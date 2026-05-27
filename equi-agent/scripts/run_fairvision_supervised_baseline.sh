#!/usr/bin/env bash
set -euo pipefail

ARCH="${ARCH:-resnet50}"
MODALITY="${MODALITY:-oct}"
TASKS="${TASKS:-amd dr glaucoma}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
IMAGENET_WEIGHTS="${IMAGENET_WEIGHTS:-1}"
OCT_REPRESENTATION="${OCT_REPRESENTATION:-center}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"
BALANCED_SAMPLER="${BALANCED_SAMPLER:-0}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-equi-agent/outputs/checkpoints}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-balanced_accuracy}"
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"

PATH_ARGS=()
if [[ -n "${PATH_PREFIX_FROM}" ]]; then
  PATH_ARGS+=(--path-prefix-from "${PATH_PREFIX_FROM}" --path-prefix-to "${PATH_PREFIX_TO}")
fi

WEIGHT_ARGS=()
if [[ "${IMAGENET_WEIGHTS}" == "1" || "${IMAGENET_WEIGHTS}" == "true" ]]; then
  WEIGHT_ARGS+=(--imagenet-weights)
fi

EXTRA_ARGS=()
if [[ "${FREEZE_BACKBONE}" == "1" || "${FREEZE_BACKBONE}" == "true" ]]; then
  EXTRA_ARGS+=(--freeze-backbone)
fi
if [[ "${BALANCED_SAMPLER}" == "1" || "${BALANCED_SAMPLER}" == "true" ]]; then
  EXTRA_ARGS+=(--balanced-sampler)
fi

for task in ${TASKS}; do
  stem="fairvision_${task}_${ARCH}_${MODALITY}_${OCT_REPRESENTATION}_supervised"
  val_file="${PREDICTIONS_ROOT}/${stem}_val.csv"
  test_file="${PREDICTIONS_ROOT}/${stem}_test.csv"
  thresholded_file="${PREDICTIONS_ROOT}/${stem}_test_thresholded.csv"
  thresholds_file="${METRICS_ROOT}/thresholds_${stem}.csv"
  metrics_dir="${METRICS_ROOT}/exp2_supervised_${ARCH}_${MODALITY}_${task}"
  checkpoint_file="${CHECKPOINT_ROOT}/${stem}.pth"

  python equi-agent/scripts/train_fairvision_supervised_baseline.py \
    --task "${task}" \
    --modality "${MODALITY}" \
    --arch "${ARCH}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --image-size "${IMAGE_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --oct-representation "${OCT_REPRESENTATION}" \
    --device "${DEVICE}" \
    --checkpoint "${checkpoint_file}" \
    --out-val "${val_file}" \
    --out-test "${test_file}" \
    "${EXTRA_ARGS[@]}" \
    "${WEIGHT_ARGS[@]}" \
    "${PATH_ARGS[@]}"

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

echo "Supervised FairVision baseline complete: ARCH=${ARCH} MODALITY=${MODALITY} OCT_REPRESENTATION=${OCT_REPRESENTATION} IMAGE_SIZE=${IMAGE_SIZE}"
