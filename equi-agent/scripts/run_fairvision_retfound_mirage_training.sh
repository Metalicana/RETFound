#!/usr/bin/env bash
set -euo pipefail

RETFOUND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EQUI_AGENT_DIR="${RETFOUND_ROOT}/equi-agent"

FAIRVISION_DATA_ROOT="${FAIRVISION_DATA_ROOT:-${RETFOUND_ROOT}/Datasets/FairVision}"
RUN_RETFOUND="${RUN_RETFOUND:-1}"
RUN_MIRAGE="${RUN_MIRAGE:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
EXPORT_RETFOUND="${EXPORT_RETFOUND:-${RUN_RETFOUND}}"
EXPORT_MIRAGE="${EXPORT_MIRAGE:-${RUN_MIRAGE}}"
DEVICE="${DEVICE:-cuda}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${EQUI_AGENT_DIR}/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-${EQUI_AGENT_DIR}/outputs/metrics}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-balanced_accuracy}"

RETFOUND_OCT_BACKBONE_WEIGHTS="${RETFOUND_OCT_BACKBONE_WEIGHTS:-${EQUI_AGENT_DIR}/weights/RETFound_mae_natureOCT.pth}"
RETFOUND_OCT_MODEL_WEIGHTS="${RETFOUND_OCT_MODEL_WEIGHTS:-${EQUI_AGENT_DIR}/weights/oct_model_best.pth}"
RETFOUND_EPOCHS="${RETFOUND_EPOCHS:-60}"
RETFOUND_BATCH_SIZE="${RETFOUND_BATCH_SIZE:-64}"
RETFOUND_NUM_WORKERS="${RETFOUND_NUM_WORKERS:-16}"

MIRAGE_DIR="${MIRAGE_DIR:-${EQUI_AGENT_DIR}/VisionAgent/MIRAGE}"
MIRAGE_SLO_MODEL_WEIGHTS="${MIRAGE_SLO_MODEL_WEIGHTS:-${EQUI_AGENT_DIR}/weights/slo_model_best.pth}"
MIRAGE_EPOCHS="${MIRAGE_EPOCHS:-60}"
MIRAGE_BATCH_SIZE="${MIRAGE_BATCH_SIZE:-64}"
MIRAGE_NUM_WORKERS="${MIRAGE_NUM_WORKERS:-16}"

mkdir -p "${EQUI_AGENT_DIR}/weights"
mkdir -p "${PREDICTIONS_ROOT}" "${METRICS_ROOT}"

if [[ "${RUN_RETFOUND}" == "1" ]]; then
  (
    cd "${EQUI_AGENT_DIR}"
    FAIRVISION_DATA_ROOT="${FAIRVISION_DATA_ROOT}" \
    RETFOUND_OCT_BACKBONE_WEIGHTS="${RETFOUND_OCT_BACKBONE_WEIGHTS}" \
    RETFOUND_OCT_MODEL_WEIGHTS="${RETFOUND_OCT_MODEL_WEIGHTS}" \
    RETFOUND_EPOCHS="${RETFOUND_EPOCHS}" \
    RETFOUND_BATCH_SIZE="${RETFOUND_BATCH_SIZE}" \
    RETFOUND_NUM_WORKERS="${RETFOUND_NUM_WORKERS}" \
      python -u -m VisionAgent.linear_probing_oct3
  )
fi

if [[ "${RUN_MIRAGE}" == "1" ]]; then
  python -u "${EQUI_AGENT_DIR}/scripts/train_fairvision_mirage_slo.py" \
    --data-root "${FAIRVISION_DATA_ROOT}" \
    --mirage-dir "${MIRAGE_DIR}" \
    --output-weights "${MIRAGE_SLO_MODEL_WEIGHTS}" \
    --epochs "${MIRAGE_EPOCHS}" \
    --batch-size "${MIRAGE_BATCH_SIZE}" \
    --num-workers "${MIRAGE_NUM_WORKERS}"
fi

if [[ "${RUN_EXPORT}" == "1" && "${EXPORT_RETFOUND}" == "1" ]]; then
  retfound_val="${PREDICTIONS_ROOT}/fairvision_oct_retfound_val.csv"
  retfound_test="${PREDICTIONS_ROOT}/fairvision_oct_retfound_test.csv"
  retfound_thresholds="${METRICS_ROOT}/thresholds_fairvision_oct_retfound.csv"
  retfound_thresholded="${PREDICTIONS_ROOT}/fairvision_oct_retfound_test_thresholded.csv"

  python -u "${EQUI_AGENT_DIR}/scripts/predict_fairvision_oct.py" \
    --manifest-dir "${EQUI_AGENT_DIR}/outputs/manifests" \
    --weights "${RETFOUND_OCT_MODEL_WEIGHTS}" \
    --backbone-weights "${RETFOUND_OCT_BACKBONE_WEIGHTS}" \
    --split val \
    --device "${DEVICE}" \
    --out "${retfound_val}"

  python -u "${EQUI_AGENT_DIR}/scripts/predict_fairvision_oct.py" \
    --manifest-dir "${EQUI_AGENT_DIR}/outputs/manifests" \
    --weights "${RETFOUND_OCT_MODEL_WEIGHTS}" \
    --backbone-weights "${RETFOUND_OCT_BACKBONE_WEIGHTS}" \
    --split test \
    --device "${DEVICE}" \
    --out "${retfound_test}"

  python -u "${EQUI_AGENT_DIR}/scripts/tune_thresholds.py" \
    --validation "${retfound_val}" \
    --test "${retfound_test}" \
    --metric "${THRESHOLD_METRIC}" \
    --thresholds-out "${retfound_thresholds}" \
    --test-out "${retfound_thresholded}"

  python -u "${EQUI_AGENT_DIR}/scripts/evaluate_predictions.py" \
    --predictions "${retfound_thresholded}" \
    --out-dir "${METRICS_ROOT}/exp2_retfound_oct"
fi

if [[ "${RUN_EXPORT}" == "1" && "${EXPORT_MIRAGE}" == "1" ]]; then
  mirage_val="${PREDICTIONS_ROOT}/fairvision_slo_mirage_val.csv"
  mirage_test="${PREDICTIONS_ROOT}/fairvision_slo_mirage_test.csv"
  mirage_thresholds="${METRICS_ROOT}/thresholds_fairvision_slo_mirage.csv"
  mirage_thresholded="${PREDICTIONS_ROOT}/fairvision_slo_mirage_test_thresholded.csv"

  python -u "${EQUI_AGENT_DIR}/scripts/predict_fairvision_slo.py" \
    --manifest-dir "${EQUI_AGENT_DIR}/outputs/manifests" \
    --weights "${MIRAGE_SLO_MODEL_WEIGHTS}" \
    --mirage-dir "${MIRAGE_DIR}" \
    --split val \
    --device "${DEVICE}" \
    --out "${mirage_val}"

  python -u "${EQUI_AGENT_DIR}/scripts/predict_fairvision_slo.py" \
    --manifest-dir "${EQUI_AGENT_DIR}/outputs/manifests" \
    --weights "${MIRAGE_SLO_MODEL_WEIGHTS}" \
    --mirage-dir "${MIRAGE_DIR}" \
    --split test \
    --device "${DEVICE}" \
    --out "${mirage_test}"

  python -u "${EQUI_AGENT_DIR}/scripts/tune_thresholds.py" \
    --validation "${mirage_val}" \
    --test "${mirage_test}" \
    --metric "${THRESHOLD_METRIC}" \
    --thresholds-out "${mirage_thresholds}" \
    --test-out "${mirage_thresholded}"

  python -u "${EQUI_AGENT_DIR}/scripts/evaluate_predictions.py" \
    --predictions "${mirage_thresholded}" \
    --out-dir "${METRICS_ROOT}/exp2_mirage_slo"
fi

if [[ "${RUN_EXPORT}" == "1" ]]; then
  python -u "${EQUI_AGENT_DIR}/scripts/build_manuscript_tables.py" \
    --metrics-root "${METRICS_ROOT}" \
    --out-dir "${EQUI_AGENT_DIR}/outputs/tables"
fi

echo "RETFound/MIRAGE FairVision training complete."
