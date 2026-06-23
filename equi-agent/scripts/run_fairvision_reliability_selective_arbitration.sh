#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-equi-agent/outputs/predictions}"
METRICS_ROOT="${METRICS_ROOT:-equi-agent/outputs/metrics}"
OUT_DIR="${OUT_DIR:-equi-agent/outputs/fairvision_reliability_selective_arbitration}"
TASKS="${TASKS:-amd dr glaucoma}"
MODELS="${MODELS:-retfound_oct mirage_slo flair_slo ret_clip_slo visionfm_slo visionfm_oct retizero_slo urfound_slo urfound_oct}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUBGROUP_PRIORS="${SUBGROUP_PRIORS:-${METRICS_ROOT}/validation_subgroup_priors.csv}"
GLOBAL_PRIORS="${GLOBAL_PRIORS:-${METRICS_ROOT}/validation_subgroup_priors_global.csv}"
SUBGROUP_SHRINKAGE_K="${SUBGROUP_SHRINKAGE_K:-50}"
CONFORMAL_SHRINKAGE_K="${CONFORMAL_SHRINKAGE_K:-50}"
CONFORMAL_ALPHA="${CONFORMAL_ALPHA:-0.10}"
BETA="${BETA:-4.0}"
GAMMA="${GAMMA:-0.75}"
DELTA="${DELTA:-0.50}"
WEIGHT_AUROC="${WEIGHT_AUROC:-0.45}"
WEIGHT_F1="${WEIGHT_F1:-0.35}"
WEIGHT_ECE="${WEIGHT_ECE:-0.10}"
WEIGHT_FNR="${WEIGHT_FNR:-0.07}"
WEIGHT_FPR="${WEIGHT_FPR:-0.03}"
CLOSE_CALL_MARGIN="${CLOSE_CALL_MARGIN:-0.08}"
DISAGREEMENT_RATE_THRESHOLD="${DISAGREEMENT_RATE_THRESHOLD:-0.25}"
LOW_RELIABILITY_THRESHOLD="${LOW_RELIABILITY_THRESHOLD:-0.35}"
RUN_COUNTERFACTUALS="${RUN_COUNTERFACTUALS:-0}"
RUN_REPORT="${RUN_REPORT:-1}"

ARGS=(
  --predictions-root "${PREDICTIONS_ROOT}"
  --metrics-root "${METRICS_ROOT}"
  --out-dir "${OUT_DIR}"
  --tasks ${TASKS}
  --models ${MODELS}
  --subgroup-priors "${SUBGROUP_PRIORS}"
  --global-priors "${GLOBAL_PRIORS}"
  --subgroup-shrinkage-k "${SUBGROUP_SHRINKAGE_K}"
  --conformal-shrinkage-k "${CONFORMAL_SHRINKAGE_K}"
  --conformal-alpha "${CONFORMAL_ALPHA}"
  --beta "${BETA}"
  --gamma "${GAMMA}"
  --delta "${DELTA}"
  --weight-auroc "${WEIGHT_AUROC}"
  --weight-f1 "${WEIGHT_F1}"
  --weight-ece "${WEIGHT_ECE}"
  --weight-fnr "${WEIGHT_FNR}"
  --weight-fpr "${WEIGHT_FPR}"
  --close-call-margin "${CLOSE_CALL_MARGIN}"
  --disagreement-rate-threshold "${DISAGREEMENT_RATE_THRESHOLD}"
  --low-reliability-threshold "${LOW_RELIABILITY_THRESHOLD}"
)

if [[ "${RUN_COUNTERFACTUALS}" == "1" ]]; then
  ARGS+=(--run-counterfactuals)
fi

"${PYTHON_BIN}" equi-agent/scripts/run_fairvision_reliability_selective_arbitration.py "${ARGS[@]}"

if [[ "${RUN_REPORT}" == "1" ]]; then
  "${PYTHON_BIN}" equi-agent/scripts/summarize_fairvision_reliability_selective_arbitration.py \
    --out-dir "${OUT_DIR}"
fi

echo "Reliability-conditioned selective arbitration complete."
echo "Predictions: ${OUT_DIR}/selective_arbitration_predictions.csv"
echo "Ablations: ${OUT_DIR}/selective_arbitration_ablation_metrics.csv"
echo "Risk-coverage: ${OUT_DIR}/risk_coverage_curve.csv"
