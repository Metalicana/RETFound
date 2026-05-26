#!/usr/bin/env bash
set -euo pipefail

FAIRVISION_REPO="${FAIRVISION_REPO:-../FairVision-main}"
DATASET_DIR="${DATASET_DIR:-Datasets/FairVision}"
OUT_ROOT="${OUT_ROOT:-equi-agent/outputs/fairvision_official_3d}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
STAGED_DATASET_DIR="${STAGED_DATASET_DIR:-${OUT_ROOT}/staged_fairvision}"
STAGE_COPY="${STAGE_COPY:-false}"

TASKS="${TASKS:-amd dr}"
MODEL_TYPE="${MODEL_TYPE:-resnet18}"
MODALITY_TYPE="${MODALITY_TYPE:-oct_bscans_3d}"
CONV_TYPE="${CONV_TYPE:-Conv3d}"
ATTRIBUTE_TYPE="${ATTRIBUTE_TYPE:-race}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MOMENTUM="${MOMENTUM:-0.1}"
IMAGE_SIZE="${IMAGE_SIZE:-200}"
WORKERS="${WORKERS:-8}"
BOOTSTRAP_REPEAT_TIMES="${BOOTSTRAP_REPEAT_TIMES:-100}"
THRESHOLD="${THRESHOLD:-0.5}"
SORT_FILES="${SORT_FILES:-false}"

mkdir -p "${OUT_ROOT}" "${PREDICTIONS_ROOT}" "${METRICS_ROOT}"
RETFOUND_ROOT="$(pwd)"

stage_args=()
if [[ "${STAGE_COPY}" == "true" ]]; then
  stage_args+=(--copy)
fi
python equi-agent/scripts/stage_fairvision_official_layout.py \
  --source-root "${DATASET_DIR}" \
  --out-root "${STAGED_DATASET_DIR}" \
  --tasks ${TASKS} \
  "${stage_args[@]}"

run_one_task() {
  local task="$1"
  local disease_dir script_name model_name result_dir result_dir_final perf_file pred_raw metrics_dir sort_arg

  case "${task}" in
    amd)
      disease_dir="${STAGED_DATASET_DIR}/AMD"
      script_name="scripts/train_amd_fair_3d.py"
      model_name="fairvision_official_3d_resnet18_amd"
      ;;
    dr)
      disease_dir="${STAGED_DATASET_DIR}/DR"
      script_name="scripts/train_dr_fair_3d.py"
      model_name="fairvision_official_3d_resnet18_dr"
      ;;
    *)
      echo "Unsupported task: ${task}. Supported: amd dr" >&2
      exit 1
      ;;
  esac

  result_dir="${OUT_ROOT}/${task}_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${CONV_TYPE}"
  perf_file="${task}_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${CONV_TYPE}.csv"
  pred_raw="${PREDICTIONS_ROOT}/fairvision_official_${task}_3d_test.csv"
  metrics_dir="${METRICS_ROOT}/fairvision_official_${task}_3d"

  mkdir -p "${result_dir}"

  (
    cd "${FAIRVISION_REPO}"
    python "${script_name}" \
      --data_dir "${RETFOUND_ROOT}/${disease_dir}" \
      --result_dir "${RETFOUND_ROOT}/${result_dir}" \
      --model_type "${MODEL_TYPE}" \
      --image_size "${IMAGE_SIZE}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --momentum "${MOMENTUM}" \
      --batch_size "${BATCH_SIZE}" \
      --task cls \
      --epochs "${EPOCHS}" \
      --workers "${WORKERS}" \
      --modality_types "${MODALITY_TYPE}" \
      --perf_file "${perf_file}" \
      --attribute_type "${ATTRIBUTE_TYPE}" \
      --conv_type "${CONV_TYPE}" \
      --bootstrap_repeat_times "${BOOTSTRAP_REPEAT_TIMES}"
  )

  result_dir_final="${result_dir}"
  if [[ ! -f "${result_dir_final}/pred_gt_best_epoch.npz" ]]; then
    result_dir_final="$(find "${OUT_ROOT}" -maxdepth 1 -type d -name "$(basename "${result_dir}")_seed*_auc*" | sort | tail -n 1)"
  fi
  if [[ -z "${result_dir_final}" || ! -f "${result_dir_final}/pred_gt_best_epoch.npz" ]]; then
    echo "Could not find pred_gt_best_epoch.npz for ${task} under ${result_dir} or renamed result directory." >&2
    exit 1
  fi

  sort_arg=()
  if [[ "${SORT_FILES}" == "true" ]]; then
    sort_arg+=(--sort-files)
  fi

  python equi-agent/scripts/convert_fairvision_official_predictions.py \
    --task "${task}" \
    --data-dir "${disease_dir}" \
    --pred-npz "${result_dir_final}/pred_gt_best_epoch.npz" \
    --out "${pred_raw}" \
    --model-name "${model_name}" \
    --threshold "${THRESHOLD}" \
    "${sort_arg[@]}"

  python equi-agent/scripts/evaluate_predictions.py \
    --predictions "${pred_raw}" \
    --out-dir "${metrics_dir}"
}

for task in ${TASKS}; do
  run_one_task "${task}"
done

combined="${PREDICTIONS_ROOT}/fairvision_official_3d_test.csv"
first_task=true
for task in ${TASKS}; do
  task_file="${PREDICTIONS_ROOT}/fairvision_official_${task}_3d_test.csv"
  if [[ "${first_task}" == "true" ]]; then
    head -n 1 "${task_file}" > "${combined}"
    first_task=false
  fi
  tail -n +2 "${task_file}" >> "${combined}"
done

python equi-agent/scripts/evaluate_predictions.py \
  --predictions "${combined}" \
  --out-dir "${METRICS_ROOT}/fairvision_official_3d"

echo "FairVision official 3D baselines complete: ${combined}"
