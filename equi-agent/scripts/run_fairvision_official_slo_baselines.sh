#!/usr/bin/env bash
set -euo pipefail

FAIRVISION_REPO="${FAIRVISION_REPO:-../FairVision-main}"
DATASET_DIR="${DATASET_DIR:-Datasets/FairVision}"
OUT_ROOT="${OUT_ROOT:-equi-agent/outputs/fairvision_official}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
STAGED_DATASET_DIR="${STAGED_DATASET_DIR:-${OUT_ROOT}/staged_fairvision}"
STAGE_COPY="${STAGE_COPY:-false}"

TASKS="${TASKS:-amd dr}"
MODEL_TYPE="${MODEL_TYPE:-ViT-B}"
MODALITY_TYPE="${MODALITY_TYPE:-slo_fundus}"
ATTRIBUTE_TYPE="${ATTRIBUTE_TYPE:-race}"
VIT_WEIGHTS="${VIT_WEIGHTS:-imagenet}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-50}"
BLR="${BLR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LAYER_DECAY="${LAYER_DECAY:-0.55}"
DROP_PATH="${DROP_PATH:-0.1}"
WORKERS="${WORKERS:-8}"
BOOTSTRAP_REPEAT_TIMES="${BOOTSTRAP_REPEAT_TIMES:-100}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-balanced_accuracy}"
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
      script_name="scripts/train_amd_fair.py"
      model_name="fairvision_official_vit_slo_amd"
      ;;
    dr)
      disease_dir="${STAGED_DATASET_DIR}/DR"
      script_name="scripts/train_dr_fair.py"
      model_name="fairvision_official_vit_slo_dr"
      ;;
    *)
      echo "Unsupported task: ${task}. Supported: amd dr" >&2
      exit 1
      ;;
  esac

  result_dir="${OUT_ROOT}/${task}_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${VIT_WEIGHTS}"
  perf_file="${task}_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv"
  pred_raw="${PREDICTIONS_ROOT}/fairvision_official_${task}_vit_slo_test.csv"
  metrics_dir="${METRICS_ROOT}/fairvision_official_${task}_vit_slo"

  mkdir -p "${result_dir}"

  (
    cd "${FAIRVISION_REPO}"
    python "${script_name}" \
      --epochs "${EPOCHS}" \
      --workers "${WORKERS}" \
      --batch_size "${BATCH_SIZE}" \
      --blr "${BLR}" \
      --min_lr "${MIN_LR}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --layer_decay "${LAYER_DECAY}" \
      --drop_path "${DROP_PATH}" \
      --data_dir "${RETFOUND_ROOT}/${disease_dir}" \
      --result_dir "${RETFOUND_ROOT}/${result_dir}" \
      --model_type "${MODEL_TYPE}" \
      --modality_types "${MODALITY_TYPE}" \
      --perf_file "${perf_file}" \
      --vit_weights "${VIT_WEIGHTS}" \
      --attribute_type "${ATTRIBUTE_TYPE}" \
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

combined="${PREDICTIONS_ROOT}/fairvision_official_vit_slo_test.csv"
first_task=true
for task in ${TASKS}; do
  task_file="${PREDICTIONS_ROOT}/fairvision_official_${task}_vit_slo_test.csv"
  if [[ "${first_task}" == "true" ]]; then
    head -n 1 "${task_file}" > "${combined}"
    first_task=false
  fi
  tail -n +2 "${task_file}" >> "${combined}"
done

python equi-agent/scripts/evaluate_predictions.py \
  --predictions "${combined}" \
  --out-dir "${METRICS_ROOT}/fairvision_official_vit_slo"

echo "FairVision official SLO baselines complete: ${combined}"
