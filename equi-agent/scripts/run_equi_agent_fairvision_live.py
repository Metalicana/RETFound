from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from smoke_equi_agent_arbitration import (
    DEFAULT_MODELS,
    TASKS,
    case_metadata,
    estimate_tokens,
    fnum,
    load_predictions,
    load_priors,
    mock_arbitrate,
    read_csv,
    select_cases,
    write_csv,
    write_jsonl,
)


OUTPUT_FIELDS = [
    "patient_id",
    "eye_id",
    "visit_id",
    "image_id",
    "dataset",
    "task",
    "model_name",
    "y_true",
    "y_prob",
    "y_pred",
    "applied_threshold",
    "split",
    "race",
    "ethnicity",
    "sex_gender",
    "age",
    "age_group",
    "positive_votes",
    "num_models",
    "disagreement",
    "close_call",
    "safety_decision",
    "primary_model",
    "confidence",
    "calibration_action",
    "escalate_to_human",
    "vision_evidence_summary",
    "equity_reliability_concern",
    "equity_threshold_policy",
    "equity_model_reliability_table",
    "orchestrator_rationale",
    "safety_reasons",
    "agent_trace_json",
    "llm_provider",
    "llm_deployment",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "retry_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live Equi-Agent arbitration over FairVision prediction CSVs using "
            "validation-derived subgroup priors. Use --dry-run to validate schemas "
            "without calling Azure/OpenAI."
        )
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument("--metrics-root", type=Path, default=Path("equi-agent/outputs/metrics"))
    parser.add_argument("--manifests-root", type=Path, default=Path("equi-agent/outputs/manifests"))
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/equi_agent_live"))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-cases-per-task", type=int, default=3, help="Use <=0 to process all common cases.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset into each task's shared case list.")
    parser.add_argument("--sample-random", action="store_true", help="Randomly sample common cases per task instead of taking sorted cases.")
    parser.add_argument(
        "--case-order",
        choices=["sorted", "source"],
        default="sorted",
        help=(
            "Order for non-random, non-stratified case selection. 'sorted' preserves the existing stable case-id order; "
            "'source' follows the first requested model's CSV row order after intersecting requested models, which is "
            "closer to the legacy OphthalmicAgent first-N test-row protocol."
        ),
    )
    parser.add_argument(
        "--sample-stratified",
        action="store_true",
        help="Randomly sample common cases per task with a controlled y_true class mix.",
    )
    parser.add_argument("--random-seed", type=int, default=2026, help="Seed used with --sample-random.")
    parser.add_argument(
        "--target-positive-frac",
        type=float,
        default=0.50,
        help="Target positive fraction for --sample-stratified.",
    )
    parser.add_argument(
        "--min-positive-frac",
        type=float,
        default=0.15,
        help="Minimum allowed positive fraction for --sample-stratified.",
    )
    parser.add_argument(
        "--max-positive-frac",
        type=float,
        default=0.85,
        help="Maximum allowed positive fraction for --sample-stratified.",
    )
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument(
        "--prompt-variant",
        choices=[
            "current",
            "visual_first",
            "f1_rescue",
            "diagnosis_tuned",
            "diagnosis_tuned_v2",
            "ophthalmic_agent_style",
        ],
        default=os.getenv("EQUI_AGENT_PROMPT_VARIANT", "current"),
        help="Prompt policy used by the arbitration LLM. Use micro experiments to compare variants.",
    )
    parser.add_argument("--sensitivity-threshold", type=float, default=0.35)
    parser.add_argument("--neutral-threshold", type=float, default=0.50)
    parser.add_argument("--precision-threshold", type=float, default=0.65)
    parser.add_argument(
        "--include-image-tokens",
        action="store_true",
        help="Attach OCT and SLO/Fundus images from FairVision NPZ files to the live vision-specialist prompt.",
    )
    parser.add_argument(
        "--path-prefix-from",
        default=os.getenv("FAIRVISION_PATH_PREFIX_FROM", "/home/ab575577/RETFound"),
        help="Optional stale prefix in manifest image_path values to replace before loading NPZ files.",
    )
    parser.add_argument(
        "--path-prefix-to",
        default=os.getenv("FAIRVISION_PATH_PREFIX_TO", str(Path.cwd())),
        help="Replacement prefix used with --path-prefix-from.",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=768,
        help="Maximum side length for JPEG images attached to the LLM prompt.",
    )
    parser.add_argument(
        "--max-probability-adjustment",
        type=float,
        default=0.10,
        help="Clamp LLM final_probability to deterministic weighted_probability +/- this margin. Use <0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build evidence packets and mock outputs without API calls.")
    parser.add_argument("--provider", choices=["auto", "azure", "openai"], default="auto")
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    parser.add_argument("--request-sleep-sec", type=float, default=0.0, help="Optional delay between live API calls.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per case for transient API or JSON parse failures.")
    parser.add_argument("--retry-sleep-sec", type=float, default=5.0, help="Base delay before retrying failed live calls.")
    parser.add_argument("--chars-per-token", type=float, default=4.0, help="Dry-run usage estimate only.")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def case_id(meta: dict[str, Any]) -> str:
    return "|".join(str(meta.get(col, "")) for col in ["patient_id", "eye_id", "visit_id", "image_id", "task"])


def build_evidence_packet(meta: dict[str, Any], arbitration: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": {
            "case_id": case_id(meta),
            "dataset": meta.get("dataset", ""),
            "task": meta.get("task", ""),
        },
        "patient_metadata_for_reliability_only": {
            "race": meta.get("race", ""),
            "ethnicity": meta.get("ethnicity", ""),
            "sex_gender": meta.get("sex_gender", ""),
            "age": meta.get("age", ""),
            "age_group": meta.get("age_group", ""),
        },
        "foundation_model_outputs": [
            {
                "model": row["model"],
                "probability": round(row["prob"], 6),
                "binary_prediction": row["pred"],
                "validation_trust_weight": round(row["weight"], 6),
                "prior_attribute": row["prior_attribute"],
                "prior_subgroup": row["prior_subgroup"],
                "prior_balanced_accuracy": round(row["prior_balanced_accuracy"], 6),
                "prior_false_positive_rate": round(row["prior_fpr"], 6),
                "prior_false_negative_rate": round(row["prior_fnr"], 6),
                "prior_unstable": row["prior_unstable"],
            }
            for row in arbitration.get("model_rows", [])
        ],
        "deterministic_reference": {
            "weighted_probability": round(arbitration["final_prob"], 6),
            "weighted_prediction": arbitration["final_pred"],
            "positive_votes": arbitration["positive_votes"],
            "num_models": arbitration["num_models"],
            "disagreement": arbitration["disagreement"],
            "close_call": arbitration["close_call"],
            "safety_decision": arbitration["safety_decision"],
        },
    }


def manifest_file(manifests_root: Path, task: str) -> Path:
    return manifests_root / f"fairvision_{task}.csv"


def load_manifests(manifests_root: Path, tasks: list[str]) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, Any]]]:
    by_image_task: dict[tuple[str, str], dict[str, str]] = {}
    loaded = []
    for task in tasks:
        path = manifest_file(manifests_root, task)
        if not path.exists():
            loaded.append({"task": task, "path": str(path), "rows": 0, "missing": True})
            continue
        rows = read_csv(path)
        for row in rows:
            by_image_task[(row.get("image_id", ""), task)] = row
        loaded.append({"task": task, "path": str(path), "rows": len(rows), "missing": False})
    return by_image_task, loaded


def resolve_manifest_path(path_value: str, prefix_from: str | None, prefix_to: str | None) -> Path:
    path_text = str(path_value or "")
    path = Path(path_text)
    if path.exists():
        return path
    if prefix_from and prefix_to and path_text.startswith(prefix_from):
        rewritten = Path(prefix_to + path_text[len(prefix_from) :])
        if rewritten.exists():
            return rewritten
        return rewritten
    return path


def normalize_to_uint8(array: Any) -> Any:
    import numpy as np

    image = np.asarray(array)
    if image.ndim == 3:
        image = image[image.shape[0] // 2] if image.shape[0] < image.shape[-1] else image[:, :, image.shape[-1] // 2]
    image = image.astype("float32")
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    min_value = float(image[finite].min())
    max_value = float(image[finite].max())
    if max_value <= 1.0 and min_value >= 0.0:
        image = image * 255.0
    elif max_value > min_value:
        image = 255.0 * (image - min_value) / (max_value - min_value)
    return np.clip(image, 0, 255).astype(np.uint8)


def jpeg_data_url(image_array: Any, max_side: int) -> str:
    from PIL import Image

    image = Image.fromarray(normalize_to_uint8(image_array))
    if image.mode not in {"L", "RGB"}:
        image = image.convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def image_payload_for_case(
    meta: dict[str, Any],
    manifest_lookup: dict[tuple[str, str], dict[str, str]],
    prefix_from: str | None,
    prefix_to: str | None,
    max_side: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    task = str(meta.get("task", ""))
    image_id = str(meta.get("image_id", ""))
    row = manifest_lookup.get((image_id, task))
    if row is None:
        return None, {"image_id": image_id, "task": task, "reason": "manifest row not found"}

    npz_path = resolve_manifest_path(row.get("image_path", ""), prefix_from, prefix_to)
    if not npz_path.exists():
        return None, {"image_id": image_id, "task": task, "path": str(npz_path), "reason": "NPZ not found"}

    import numpy as np

    with np.load(npz_path) as data:
        oct_key = row.get("oct_key") or "oct_bscans"
        fundus_key = row.get("fundus_key") or "slo_fundus"
        missing_keys = [key for key in [oct_key, fundus_key] if key not in data]
        if missing_keys:
            return None, {
                "image_id": image_id,
                "task": task,
                "path": str(npz_path),
                "reason": f"missing NPZ keys: {', '.join(missing_keys)}",
            }
        payload = {
            "oct_image_url": jpeg_data_url(data[oct_key], max_side),
            "slo_image_url": jpeg_data_url(data[fundus_key], max_side),
            "source_path": str(npz_path),
            "oct_key": oct_key,
            "fundus_key": fundus_key,
        }
    return payload, None


def prompt_variant_guidance(variant: str) -> str:
    if variant == "visual_first":
        return (
            "PROMPT_VARIANT=visual_first. When images are attached, perform morphology review before reading model votes. "
            "For AMD, require visible drusen/RPE disruption/subretinal fluid/atrophy or a reliable positive source before "
            "raising probability. For DR, require visible hemorrhage/exudate/microaneurysm-like vascular lesions or a reliable "
            "positive source. For glaucoma, use OCT/SLO morphology to adjudicate cup/RNFL-compatible evidence. If image "
            "morphology clearly conflicts with weak near-all-positive model behavior, down-weight the votes."
        )
    if variant == "f1_rescue":
        return (
            "PROMPT_VARIANT=f1_rescue. Optimize forced benchmark F1 while preserving threshold consistency. Do not chase "
            "specificity at the cost of missing most positives. For rare-positive DR, prefer sensitivity_shift when any "
            "validated source with acceptable balanced accuracy is positive or morphology is suspicious. For AMD, avoid "
            "near-all-positive voting as sole evidence, but do not suppress positives when multiple modalities and morphology "
            "support disease. Use precision_shift only when false-positive priors are strong and positive evidence is weak."
        )
    if variant == "diagnosis_tuned":
        return (
            "PROMPT_VARIANT=diagnosis_tuned. Tune the diagnosis arbitration for balanced thresholded performance, not "
            "for maximum positive calls. Preserve threshold consistency. For glaucoma, do not let weak multi-model "
            "positive voting override poor specificity: require either a strong deterministic probability, a reliable "
            "primary source with low false-positive risk, or image/functional support before calling positive. If glaucoma "
            "evidence is mostly positive but comes from sources with high false-positive priors or poor balanced accuracy, "
            "use precision_shift. Do not use sensitivity_shift for glaucoma when the deterministic probability is below "
            "0.50 and only one or two models are positive; in that setting use neutral or precision_shift and escalate. "
            "For DR, avoid dismissing positives just because DR is rare: if at least one lower-FPR OCT-compatible source "
            "or the best-balanced SLO source votes positive, keep neutral or sensitivity_shift rather than precision_shift. "
            "Only use precision_shift for DR when positive evidence comes mainly from high-FPR SLO sources and lower-FPR "
            "OCT-compatible sources are negative. For AMD, keep precision_shift for weak positive calls driven by high-FP "
            "sources, but do not force precision_shift when the deterministic probability is near 0.50-0.65 and the case "
            "has multiple positives across OCT-compatible sources or a strong single OCT-compatible source; use neutral "
            "and escalate instead."
        )
    if variant == "diagnosis_tuned_v2":
        return (
            "PROMPT_VARIANT=diagnosis_tuned_v2. Use diagnosis_tuned, with stricter glaucoma close-call handling. "
            "Avoid both failure modes seen in prompt tuning: do not call broad weak evidence positive, but also do not "
            "down-adjust borderline glaucoma cases into false negatives solely because SLO positives have high FPR. "
            "For glaucoma, if deterministic_reference.weighted_probability is between 0.45 and 0.50, positive_votes is "
            "at least 5, and OCT-compatible evidence is not unanimously strongly negative, keep final_probability at "
            "least 0.50, use neutral threshold, and set escalate_to_human=true. If weighted_probability is below 0.45 "
            "or positive_votes is below 5, do not rescue unless attached morphology supports glaucoma. If weighted_probability "
            "is already above 0.50 but below 0.60, do not up-adjust above 0.52 unless at least two OCT-compatible sources "
            "are strongly positive; keep escalation high. For DR, preserve the diagnosis_tuned improvement: rescue "
            "lower-FPR/OCT-compatible positives and avoid precision_shift when DR evidence is mixed but credible. For AMD, "
            "preserve precision control for high-FPR weak positives, but use neutral plus escalation for moderate "
            "OCT-supported probabilities."
        )
    if variant == "ophthalmic_agent_style":
        return (
            "PROMPT_VARIANT=ophthalmic_agent_style. Mirror the legacy OphthalmicAgent hierarchy while preserving the "
            "current structured JSON output. First, separate disease evidence from model reliability. Second, require "
            "the Equity Agent to build a model-by-model historical FP/FN audit before recommending a threshold. Use this "
            "semantics exactly: low FPR makes a positive vote more credible; high FPR makes a positive vote less credible; "
            "low FNR makes a negative vote more credible; high FNR makes a negative vote less credible. Do not use low FPR "
            "to justify a negative call, and do not use low FNR to justify a positive call. For AMD, use a pathology-signal "
            "mindset: if multiple credible sources or OCT-compatible sources support disease, do not suppress to negative "
            "solely because some high-FPR SLO sources are also positive; use neutral threshold plus escalation for moderate "
            "signals. For DR, preserve sensitivity to credible positives from lower-FPR/OCT-compatible or best-balanced "
            "sources; use precision_shift only when positives are dominated by high-FPR sources and credible sources are "
            "negative. For glaucoma, use the old precision discipline: prefer OCT-compatible structural sources and any "
            "available functional evidence; if visual field/function is unavailable, say unavailable and do not invent it. "
            "Weak SLO-heavy positives should not override more reliable OCT-compatible negatives, but borderline cases with "
            "mixed credible evidence should be forced-label decisions with human escalation rather than abstention. Keep "
            "the JSON compact: do not duplicate the model-by-model audit outside equity_agent.model_reliability_table, "
            "and keep all rationales to one short sentence."
        )
    return "PROMPT_VARIANT=current. Use the default trust-calibration policy."


def task_specific_guidance(task: str) -> str:
    task = task.lower()
    if task == "amd":
        return (
            "CURRENT TASK: AMD diagnosis. Treat the target as binary any-AMD versus no AMD. "
            "Disease evidence should come from model outputs and, when attached, macular morphology such as drusen, "
            "RPE disruption, geographic atrophy, pigmentary change, or fluid. Do not use DR or glaucoma votes as AMD "
            "evidence. For AMD, near-threshold negative calls from sources with high false-negative priors should trigger "
            "sensitivity_shift; weak positives from sources with high false-positive priors should trigger precision_shift. "
            "However, do not convert every high-FPR context into precision_shift: if OCT-compatible sources agree or the "
            "deterministic probability is moderately positive, use neutral plus escalation rather than a 0.65 threshold."
        )
    if task == "dr":
        return (
            "CURRENT TASK: diabetic retinopathy diagnosis. Treat the target as binary DR versus no DR. "
            "Disease evidence should come from model outputs and, when attached, vascular morphology such as microaneurysms, "
            "hemorrhages, exudates, venous beading, or neovascular features. DR is rare in this split, so avoid calling "
            "positive from vague concern alone; however, do not suppress a validated positive model vote when the source has "
            "acceptable false-positive behavior or when multiple sources agree. Prefer sensitivity_shift for near-threshold "
            "negative DR cases when the available sources have high false-negative priors. If lower-FPR OCT-compatible "
            "models vote positive, do not let high-FPR SLO models alone force precision_shift. Do not use AMD or glaucoma "
            "votes as DR evidence."
        )
    if task == "glaucoma":
        return (
            "CURRENT TASK: glaucoma diagnosis. Treat the target as binary glaucoma versus no glaucoma. "
            "Disease evidence should come from model outputs and, when attached, optic nerve/RNFL-compatible morphology such "
            "as increased cupping, rim thinning, RNFL loss, or compatible OCT evidence. If functional visual-field evidence "
            "is unavailable, say unavailable rather than inventing it. Do not use AMD or DR votes as glaucoma evidence. "
            "Do not call glaucoma positive from vote count alone when sources have high false-positive priors or weak balanced "
            "accuracy. Use precision_shift for weak or discordant positive glaucoma evidence, and reserve sensitivity_shift "
            "for borderline negative cases with at least two credible low-FP sources or attached morphology support. If only "
            "one low-FP source is positive and the deterministic probability is below 0.50, do not lower the threshold to "
            "0.35; keep neutral or precision_shift and escalate. For close-call cases with deterministic probability between "
            "0.45 and 0.50 and at least five positive votes, avoid down-adjusting below the deterministic reference solely "
            "because some SLO positives have high FPR; use neutral threshold, high uncertainty, and escalation. Escalate "
            "close calls because structural signs and functional impairment may be discordant."
        )
    return "CURRENT TASK: binary retinal disease diagnosis for the supplied task. Use only current-task evidence."


def build_live_messages(
    evidence_packet: dict[str, Any],
    image_payload: dict[str, Any] | None = None,
    prompt_variant: str = "current",
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    thresholds = thresholds or {"sensitivity_shift": 0.35, "neutral": 0.5, "escalate": 0.5, "precision_shift": 0.65}
    system = (
        "You are Equi-Agent, a professional ophthalmologist-led clinical decision-support system for retinal disease "
        "assessment. You assist real ophthalmologists by synthesizing multimodal retinal evidence, foundation-model "
        "outputs, patient context, validation-derived reliability priors, and safety concerns into a clear diagnostic "
        "recommendation. You are not a generic chatbot and you are not merely a scoreboard; reason like a careful "
        "ophthalmic consultant.\n\n"
        "Use retinal images and current-task model outputs as disease evidence. Use demographic metadata only to select "
        "and interpret validation-derived model reliability priors; never use race, ethnicity, sex/gender, or age as "
        "direct disease evidence. The field binary_prediction was computed using that model's validation-selected "
        "threshold, so it may differ from probability >= 0.5.\n\n"
        "The Equity Agent's role is central and concrete: for each foundation model, compare its raw probability and "
        "validation-threshold binary_prediction against that exact model's false-positive and false-negative track "
        "record for this disease and patient context. If a model says YES but has a high false-positive rate for the "
        "relevant subgroup/context, doubt or down-weight that positive call. If a model says YES and has a low "
        "false-positive rate with stable reliability, trust that positive call more. If a model says NO but has a high "
        "false-negative rate, doubt that negative call and consider a sensitivity shift. If a model says NO and has a "
        "low false-negative rate with stable reliability, trust that negative call more. Translate this model-by-model "
        "track record into a recommended threshold policy and a preferred model/source for the Orchestrator. Treat "
        "prior_unstable=true as statistical instability; treat low balanced accuracy as weak model reliability, not as "
        "subgroup evidence. You must expose this reasoning in agent_trace.equity_agent.model_reliability_table for every "
        "foundation model. Historical FP/FN rates are reliability evidence, not disease evidence: low FPR supports trusting "
        "a positive vote; low FNR supports trusting a negative vote.\n\n"
        "Always return a diagnostic probability and binary label for the retrospective evaluation file. Escalation is a "
        "separate safety/referral flag for human review; it must never erase or replace the diagnostic recommendation. "
        "Return only valid JSON. "
        + task_specific_guidance(str(evidence_packet.get("case", {}).get("task", "")))
        + "\n\n"
        + prompt_variant_guidance(prompt_variant)
    )
    user = {
        "instructions": {
            "internal_agent_protocol": {
                "bio_profiler_agent": (
                    "Summarize metadata only as reliability context for priors. Do not infer disease risk from race, "
                    "ethnicity, sex/gender, or age."
                ),
                "vision_agent": (
                    "If OCT/SLO images are attached, independently inspect the retinal morphology first, then compare "
                    "against raw probabilities, validation-threshold binary predictions, modality patterns, and vote "
                    "disagreement for the current task only. If no images are attached, state that the visual review is "
                    "probability-only. Do not write a per-model table here; the per-model FP/FN audit belongs only in "
                    "equity_agent.model_reliability_table."
                ),
                "equity_agent": (
                    "This is the explicit Equity Agent, matching the OphthalmicAgent design. Read the validation-derived "
                    "FN and FP rates for each available model/source alongside that model's raw probability and "
                    "binary_prediction. Apply simple reliability logic model by model: high-FP positive predictions are "
                    "less trustworthy; low-FP positive predictions are more trustworthy; high-FN negative predictions are "
                    "less trustworthy; low-FN negative predictions are more trustworthy. Identify whether the current "
                    "context has FN risk, FP risk, minimal risk, or unstable evidence. Recommend one threshold policy: "
                    "sensitivity_shift, precision_shift, neutral, or escalate. Recommend the primary_model/source that "
                    "has the best stable track record for its current YES/NO judgment. If subgroup evidence is unstable, "
                    "fall back to global model reliability and say so in the reasoning. Always produce a per-model "
                    "model_reliability_table that cites probability, vote, FPR, FNR, stability, and trust_action. Keep "
                    "each row compact; reason should be at most 8 words."
                ),
                "orchestrator": (
                    "Act as the Lead Ophthalmic Orchestrator. Combine the Vision Agent's morphology review, model outputs, "
                    "and the Equity Agent's FN/FP threshold recommendation. Anchor final_probability on "
                    "deterministic_reference.weighted_probability, but adjust by at most 0.10 when visual morphology, "
                    "the Equity Agent's primary_model recommendation, or validated source disagreement strongly justify it. "
                    f"Choose calibration_action first, then apply its threshold: sensitivity_shift uses "
                    f"{thresholds['sensitivity_shift']:.2f}, neutral/escalate uses {thresholds['neutral']:.2f}, "
                    f"and precision_shift uses {thresholds['precision_shift']:.2f}. final_prediction must be based on "
                    "final_probability >= the applied threshold."
                ),
                "safety_agent": (
                    "Set escalate_to_human=true for close calls around the applied threshold, severe split votes, statistically "
                    "unstable priors, or severe weak reliability. Do not escalate merely because one or two of nine models disagree. "
                    "If the final probability is far from the applied threshold and votes are not severely split, set escalate_to_human=false. "
                    "Always preserve the diagnostic label while separately flagging human-review need."
                ),
            },
            "task": (
                "Return one binary disease recommendation for the requested task. "
                "Even when escalation is warranted, final_prediction must be 0 or 1 for the retrospective results table."
            ),
            "threshold": 0.5,
            "dynamic_thresholds": {
                "sensitivity_shift": thresholds["sensitivity_shift"],
                "neutral": thresholds["neutral"],
                "escalate": thresholds["escalate"],
                "precision_shift": thresholds["precision_shift"],
            },
            "benchmark_rule": (
                "Do not abstain in this retrospective evaluation output. Do not output -1. Do not use escalate_to_human as the diagnosis. "
                "Use it only as a separate safety flag. final_prediction must be threshold-consistent with final_probability "
                "at the applied dynamic threshold."
            ),
            "required_json_schema": {
                "agent_trace": {
                    "bio_profiler": {
                        "direct_risk_from_demographics": "must be false",
                        "reliability_context": "short metadata summary used only for reliability-prior lookup",
                    },
                    "vision_agent": {
                        "visual_review_mode": "image_attached or probability_only",
                        "evidence_summary": "one short current-task image/model evidence summary",
                    },
                    "equity_agent": {
                        "prior_level_used": "intersectional, subgroup, global, or unavailable",
                        "reliability_concern": "false_negative_risk, false_positive_risk, calibration_risk, minimal_risk, or unstable_priors",
                        "threshold_policy": "sensitivity_shift, precision_shift, neutral, or escalate",
                        "recommended_threshold": "numeric threshold",
                        "model_reliability_table": [
                            {
                                "model": "source model name",
                                "probability": "numeric probability supplied",
                                "binary_vote": "0 or 1 supplied",
                                "fpr": "prior false-positive rate",
                                "fnr": "prior false-negative rate",
                                "balanced_accuracy": "prior balanced accuracy",
                                "prior_unstable": "boolean",
                                "trust_action": "up_weight, down_weight, keep, or escalate",
                                "reason": "8 words or fewer linking vote to FP/FN rates",
                            }
                        ],
                        "rationale": "one short sentence citing source FP/FN/prior stability evidence",
                    },
                    "orchestrator": {
                        "reference_probability": "deterministic weighted probability",
                        "probability_adjustment": "numeric adjustment from deterministic reference to final_probability",
                        "applied_threshold": "numeric threshold",
                        "final_prediction_check": "briefly state final_probability >= threshold or < threshold",
                        "rationale": "one short sentence explaining final arbitration",
                    },
                    "safety_agent": {
                        "decision": "ACCEPT or ESCALATE_TO_HUMAN",
                        "uncertainty_level": "low, medium, or high",
                        "escalation_reasons": "list of short reasons",
                    },
                },
                "final_probability": "float from 0 to 1",
                "final_prediction": "0 or 1",
                "confidence": "low, medium, or high",
                "primary_model": "model name or ensemble",
                "calibration_action": "sensitivity_shift, precision_shift, neutral, or escalate",
                "escalate_to_human": "boolean",
                "reasoning": (
                    "40 words or fewer naming which model judgments were trusted or doubted because of subgroup FP/FN"
                ),
            },
        },
        "evidence_packet": evidence_packet,
    }
    user_text = json.dumps(user, sort_keys=True)
    if not image_payload:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    image_context = {
        "attached_images": {
            "oct": "OCT B-scan center slice from the same FairVision NPZ case",
            "slo": "SLO/Fundus image from the same FairVision NPZ case",
            "source_path": image_payload.get("source_path", ""),
        },
        "visual_audit_required": (
            "Perform the legacy OphthalmicAgent-style grounded visual audit before arbitration. For AMD, inspect "
            "RPE/drusen/fluid/atrophy; for DR, inspect hemorrhage/exudate/microaneurysm evidence; for glaucoma, inspect "
            "optic disc cupping/RNFL-compatible signs when visible. Return the same JSON schema, with reasoning briefly "
            "stating whether image morphology supports or conflicts with model evidence."
        ),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(image_context, sort_keys=True) + "\n\n" + user_text},
                {"type": "image_url", "image_url": {"url": image_payload["oct_image_url"]}},
                {"type": "image_url", "image_url": {"url": image_payload["slo_image_url"]}},
            ],
        },
    ]


def message_text_for_estimate(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                parts.append("[attached image]")
        return "\n".join(parts)
    return str(content)


def select_case_keys(
    task_predictions: dict[str, dict[tuple, dict]],
    limit: int,
    offset: int,
    sample_random: bool,
    sample_stratified: bool,
    case_order: str,
    seed: int,
    task: str,
    target_positive_frac: float,
    min_positive_frac: float,
    max_positive_frac: float,
) -> list[tuple]:
    if sample_random and sample_stratified:
        raise ValueError("Use either --sample-random or --sample-stratified, not both.")
    if not sample_random:
        if sample_stratified:
            return select_stratified_cases(
                task_predictions,
                limit,
                offset,
                seed,
                task,
                target_positive_frac,
                min_positive_frac,
                max_positive_frac,
            )
        return select_cases_in_order(task_predictions, limit, offset, case_order)
    all_keys = select_cases_in_order(task_predictions, 10**12, 0, case_order)
    rng = random.Random(f"{seed}:{task}")
    rng.shuffle(all_keys)
    return all_keys[offset : offset + limit]


def select_cases_in_order(
    task_predictions: dict[str, dict[tuple, dict]],
    limit: int,
    offset: int,
    case_order: str,
) -> list[tuple]:
    if case_order == "source":
        if not task_predictions:
            return []
        common = None
        for rows_by_key in task_predictions.values():
            keys = set(rows_by_key)
            common = keys if common is None else common & keys
        if not common:
            return []
        reference_rows = next(iter(task_predictions.values()))
        ordered = [key for key in reference_rows if key in common]
        return ordered[offset : offset + limit]
    return select_cases(task_predictions, limit, offset)


def select_stratified_cases(
    task_predictions: dict[str, dict[tuple, dict]],
    limit: int,
    offset: int,
    seed: int,
    task: str,
    target_positive_frac: float,
    min_positive_frac: float,
    max_positive_frac: float,
) -> list[tuple]:
    if not 0.0 <= min_positive_frac <= target_positive_frac <= max_positive_frac <= 1.0:
        raise ValueError("Require 0 <= min_positive_frac <= target_positive_frac <= max_positive_frac <= 1.")
    all_keys = select_cases(task_predictions, 10**12, 0)
    if limit >= len(all_keys):
        return all_keys[offset : offset + limit]

    reference_rows = next(iter(task_predictions.values()))
    positives = [key for key in all_keys if int(fnum(reference_rows[key].get("y_true"), 0.0)) == 1]
    negatives = [key for key in all_keys if int(fnum(reference_rows[key].get("y_true"), 0.0)) == 0]

    desired_pos = round(limit * target_positive_frac)
    desired_pos = min(desired_pos, len(positives), limit)
    desired_pos = max(desired_pos, limit - len(negatives), 0)
    desired_neg = limit - desired_pos

    positive_frac = desired_pos / limit if limit else 0.0
    if positive_frac < min_positive_frac or positive_frac > max_positive_frac:
        raise ValueError(
            f"Cannot satisfy positive fraction bounds for {task}: requested {limit} cases, "
            f"available positives={len(positives)}, negatives={len(negatives)}, "
            f"resulting positive fraction={positive_frac:.3f}."
        )

    rng = random.Random(f"{seed}:{task}:stratified")
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected = positives[offset : offset + desired_pos] + negatives[offset : offset + desired_neg]
    rng.shuffle(selected)
    return selected


def json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_probability(value: Any, fallback: float, max_adjustment: float | None = None) -> float:
    prob = fnum(value, fallback)
    if max_adjustment is not None and max_adjustment >= 0:
        lower = fallback - max_adjustment
        upper = fallback + max_adjustment
        prob = min(upper, max(lower, prob))
    return min(1.0, max(0.0, prob))


def normalize_action(value: Any) -> str:
    action = str(value or "").strip().lower()
    if "sensitivity" in action:
        return "sensitivity_shift"
    if "precision" in action:
        return "precision_shift"
    if "escalate" in action:
        return "escalate"
    return "neutral"


def threshold_for_action(action: str, thresholds: dict[str, float] | None = None) -> float:
    thresholds = thresholds or {"sensitivity_shift": 0.35, "neutral": 0.5, "escalate": 0.5, "precision_shift": 0.65}
    if action == "sensitivity_shift":
        return thresholds["sensitivity_shift"]
    if action == "precision_shift":
        return thresholds["precision_shift"]
    if action == "escalate":
        return thresholds["escalate"]
    return thresholds["neutral"]


def normalize_prediction(probability: float, applied_threshold: float) -> int:
    return int(probability >= applied_threshold)


def safe_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def list_as_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def parsed_agent_trace(parsed: dict[str, Any]) -> dict[str, Any]:
    trace = parsed.get("agent_trace")
    return trace if isinstance(trace, dict) else {}


def trace_summary_fields(trace: dict[str, Any]) -> dict[str, str]:
    vision = trace.get("vision_agent", {}) if isinstance(trace.get("vision_agent"), dict) else {}
    equity = trace.get("equity_agent", {}) if isinstance(trace.get("equity_agent"), dict) else {}
    orchestrator = trace.get("orchestrator", {}) if isinstance(trace.get("orchestrator"), dict) else {}
    safety = trace.get("safety_agent", {}) if isinstance(trace.get("safety_agent"), dict) else {}
    return {
        "vision_evidence_summary": str(vision.get("evidence_summary", "")),
        "equity_reliability_concern": str(equity.get("reliability_concern", "")),
        "equity_threshold_policy": str(equity.get("threshold_policy", "")),
        "equity_model_reliability_table": safe_json(equity.get("model_reliability_table", [])),
        "orchestrator_rationale": str(orchestrator.get("rationale", "")),
        "safety_reasons": list_as_text(safety.get("escalation_reasons", "")),
        "agent_trace_json": safe_json(trace),
    }


def safety_flag_for_case(arbitration: dict[str, Any], probability: float, applied_threshold: float) -> tuple[bool, str]:
    model_rows = arbitration.get("model_rows", [])
    positive_votes = int(arbitration.get("positive_votes", 0))
    num_models = int(arbitration.get("num_models", len(model_rows)))
    negative_votes = max(0, num_models - positive_votes)
    close_to_threshold = abs(probability - applied_threshold) < 0.075
    severe_split_vote = min(positive_votes, negative_votes) >= 3
    unstable_prior = any(str(row.get("prior_unstable", "")).strip().lower() in {"true", "1"} for row in model_rows)
    weak_reliability = False
    if model_rows:
        mean_balanced_accuracy = mean(float(row.get("prior_balanced_accuracy", 0.5)) for row in model_rows)
        weak_reliability = mean_balanced_accuracy < 0.53 and (close_to_threshold or severe_split_vote)
    escalate = close_to_threshold or severe_split_vote or unstable_prior or weak_reliability
    return escalate, "ESCALATE_TO_HUMAN" if escalate else "ACCEPT"


def dry_run_response(
    arbitration: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, int], str]:
    model_rows = arbitration.get("model_rows", [])
    primary = "ensemble"
    if model_rows:
        primary = max(model_rows, key=lambda row: row.get("weight", 0.0)).get("model", "ensemble")
    high_fnr = any(row.get("prior_fnr", 0.0) >= 0.2 for row in model_rows)
    high_fpr = any(row.get("prior_fpr", 0.0) >= 0.2 and row.get("prior_fnr", 0.0) < 0.2 for row in model_rows)
    action = "escalate" if arbitration["safety_decision"] != "ACCEPT" else "neutral"
    if action == "neutral" and high_fnr:
        action = "sensitivity_shift"
    if action == "neutral" and high_fpr:
        action = "precision_shift"
    parsed = {
        "agent_trace": {
            "bio_profiler": {
                "direct_risk_from_demographics": False,
                "reliability_context": "Dry-run metadata used only for reliability-prior lookup.",
            },
            "vision_agent": {
                "visual_review_mode": "probability_only",
                "evidence_summary": "Dry-run deterministic weighted arbitration; no LLM visual review.",
                "source_assessments": [
                    {
                        "model": row.get("model", ""),
                        "probability": row.get("prob", 0.0),
                        "binary_vote": row.get("pred", 0),
                        "balanced_accuracy": row.get("prior_balanced_accuracy", ""),
                        "fpr": row.get("prior_fpr", ""),
                        "fnr": row.get("prior_fnr", ""),
                        "prior_unstable": row.get("prior_unstable", ""),
                        "trust_action": "keep",
                        "rationale": "Dry-run source assessment from validation priors.",
                    }
                    for row in model_rows
                ],
            },
            "equity_agent": {
                "prior_level_used": "subgroup_or_global",
                "reliability_concern": "false_negative_risk" if high_fnr else ("false_positive_risk" if high_fpr else "minimal_risk"),
                "threshold_policy": action,
                "recommended_threshold": threshold_for_action(action, thresholds),
                "rationale": "Dry-run threshold policy from validation FP/FN priors.",
            },
            "orchestrator": {
                "reference_probability": arbitration["final_prob"],
                "probability_adjustment": 0.0,
                "applied_threshold": threshold_for_action(action, thresholds),
                "final_prediction_check": "Dry-run prediction follows deterministic weighted output.",
                "rationale": "Dry-run deterministic weighted arbitration.",
            },
            "safety_agent": {
                "decision": arbitration["safety_decision"],
                "uncertainty_level": "low" if arbitration["safety_decision"] == "ACCEPT" else "high",
                "escalation_reasons": [] if arbitration["safety_decision"] == "ACCEPT" else ["deterministic safety flag"],
            },
        },
        "final_probability": arbitration["final_prob"],
        "final_prediction": arbitration["final_pred"],
        "confidence": "low" if arbitration["safety_decision"] != "ACCEPT" else "medium",
        "primary_model": primary,
        "calibration_action": action,
        "applied_threshold": threshold_for_action(action, thresholds),
        "escalate_to_human": arbitration["safety_decision"] != "ACCEPT",
        "reasoning": "Dry-run deterministic weighted arbitration using validation priors.",
    }
    return parsed, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, json.dumps(parsed)


def make_client(provider: str, api_version: str):
    load_dotenv()
    if provider == "auto":
        provider = "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE") else "openai"
    if provider == "azure":
        from openai import AzureOpenAI

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for Azure live inference.")
        return (
            provider,
            AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version),
        )

    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY for OpenAI live inference.")
    return provider, OpenAI()


def call_llm(client: Any, deployment: str, messages: list[dict[str, str]], temperature: float, max_output_tokens: int):
    try:
        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
    except TypeError:
        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def response_text(response: Any) -> str:
    return response.choices[0].message.content or ""


def live_response_with_retries(
    client: Any,
    deployment: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep_sec: float,
) -> tuple[dict[str, Any], dict[str, int], str, int]:
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        try:
            response = call_llm(client, deployment, messages, temperature, max_output_tokens)
            raw_text = response_text(response)
            usage = usage_dict(response)
            parsed = json_from_text(raw_text)
            return parsed, usage, raw_text, attempt
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            if retry_sleep_sec > 0:
                time.sleep(retry_sleep_sec * (attempt + 1))
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    thresholds = {
        "sensitivity_shift": args.sensitivity_threshold,
        "neutral": args.neutral_threshold,
        "escalate": args.neutral_threshold,
        "precision_shift": args.precision_threshold,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_task, loaded_files, missing_files = load_predictions(args.predictions_root, args.tasks, args.models)
    subgroup_priors, global_priors = load_priors(args.metrics_root)
    manifest_lookup: dict[tuple[str, str], dict[str, str]] = {}
    loaded_manifests: list[dict[str, Any]] = []
    if args.include_image_tokens:
        manifest_lookup, loaded_manifests = load_manifests(args.manifests_root, args.tasks)

    provider = "dry_run"
    client = None
    if not args.dry_run:
        provider, client = make_client(args.provider, args.api_version)

    prediction_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for task in args.tasks:
        limit = args.max_cases_per_task
        if limit <= 0:
            limit = 10**12
        keys = select_case_keys(
            by_task[task],
            limit,
            args.seed_offset,
            args.sample_random,
            args.sample_stratified,
            args.case_order,
            args.random_seed,
            task,
            args.target_positive_frac,
            args.min_positive_frac,
            args.max_positive_frac,
        )
        for key in keys:
            rows_by_model = {
                model: rows_by_key[key]
                for model, rows_by_key in by_task[task].items()
                if key in rows_by_key
            }
            meta = case_metadata(rows_by_model)
            arbitration = mock_arbitrate(task, rows_by_model, meta, subgroup_priors, global_priors)
            evidence_packet = build_evidence_packet(meta, arbitration)
            image_payload = None
            image_error = None
            if args.include_image_tokens:
                image_payload, image_error = image_payload_for_case(
                    meta,
                    manifest_lookup,
                    args.path_prefix_from,
                    args.path_prefix_to,
                    args.max_image_side,
                )
                evidence_packet["retinal_image_context"] = (
                    {
                        "images_attached": True,
                        "source_path": image_payload.get("source_path", "") if image_payload else "",
                        "oct_key": image_payload.get("oct_key", "") if image_payload else "",
                        "fundus_key": image_payload.get("fundus_key", "") if image_payload else "",
                    }
                    if image_payload
                    else {
                        "images_attached": False,
                        "error": image_error,
                    }
                )
            messages = build_live_messages(
                evidence_packet,
                image_payload,
                prompt_variant=args.prompt_variant,
                thresholds=thresholds,
            )

            try:
                if args.dry_run:
                    parsed, usage, raw_text = dry_run_response(arbitration, thresholds)
                    usage["prompt_tokens"] = sum(
                        estimate_tokens(message_text_for_estimate(m), args.chars_per_token)
                        for m in messages
                    )
                    usage["completion_tokens"] = estimate_tokens(raw_text, args.chars_per_token)
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                else:
                    parsed, usage, raw_text, retry_count = live_response_with_retries(
                        client,
                        args.deployment,
                        messages,
                        args.temperature,
                        args.max_output_tokens,
                        args.max_retries,
                        args.retry_sleep_sec,
                    )
                    usage["retry_count"] = retry_count
                    if args.request_sleep_sec > 0:
                        time.sleep(args.request_sleep_sec)

                final_prob = clamp_probability(
                    parsed.get("final_probability"),
                    arbitration["final_prob"],
                    args.max_probability_adjustment,
                )
                calibration_action = normalize_action(parsed.get("calibration_action"))
                applied_threshold = threshold_for_action(calibration_action, thresholds)
                final_pred = normalize_prediction(final_prob, applied_threshold)
                escalate_to_human, safety_decision = safety_flag_for_case(arbitration, final_prob, applied_threshold)
                agent_trace = parsed_agent_trace(parsed)
                trace_fields = trace_summary_fields(agent_trace)
                prediction_rows.append(
                    {
                        **meta,
                        "model_name": "equi_agent_live" if not args.dry_run else "equi_agent_live_dry_run",
                        "y_prob": f"{final_prob:.6f}",
                        "y_pred": final_pred,
                        "applied_threshold": f"{applied_threshold:.2f}",
                        "split": "test",
                        "positive_votes": arbitration["positive_votes"],
                        "num_models": arbitration["num_models"],
                        "disagreement": arbitration["disagreement"],
                        "close_call": arbitration["close_call"],
                        "safety_decision": safety_decision,
                        "primary_model": parsed.get("primary_model", ""),
                        "confidence": parsed.get("confidence", ""),
                        "calibration_action": calibration_action,
                        "escalate_to_human": escalate_to_human,
                        **trace_fields,
                        "llm_provider": provider,
                        "llm_deployment": args.deployment,
                        "retry_count": usage.get("retry_count", 0),
                        **usage,
                    }
                )
                usage_rows.append(
                    {
                        **meta,
                        "llm_provider": provider,
                        "llm_deployment": args.deployment,
                        "retry_count": usage.get("retry_count", 0),
                        **usage,
                    }
                )
                raw_rows.append(
                    {
                        **meta,
                        "llm_provider": provider,
                        "llm_deployment": args.deployment,
                        "messages": messages,
                        "evidence_packet": evidence_packet,
                        "parsed_response": parsed,
                        "agent_trace": agent_trace,
                        "normalized_response": {
                            "final_probability": final_prob,
                            "final_prediction": final_pred,
                            "applied_threshold": applied_threshold,
                            "calibration_action": calibration_action,
                            "escalate_to_human": escalate_to_human,
                            "safety_decision": safety_decision,
                        },
                        "raw_response": raw_text,
                        "retry_count": usage.get("retry_count", 0),
                    }
                )
                case_rows.append({**meta, **arbitration, "evidence_packet": evidence_packet})
            except Exception as exc:
                error_rows.append(
                    {
                        **meta,
                        "llm_provider": provider,
                        "llm_deployment": args.deployment,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "messages": messages,
                        "evidence_packet": evidence_packet,
                    }
                )

    write_csv(args.out_dir / "equi_agent_live_predictions.csv", prediction_rows, OUTPUT_FIELDS)
    write_csv(
        args.out_dir / "equi_agent_live_usage.csv",
        usage_rows,
        [
            "patient_id",
            "eye_id",
            "visit_id",
            "image_id",
            "dataset",
            "task",
            "y_true",
            "race",
            "ethnicity",
            "sex_gender",
            "age",
            "age_group",
            "llm_provider",
            "llm_deployment",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "retry_count",
        ],
    )
    write_jsonl(args.out_dir / "equi_agent_live_raw_responses.jsonl", raw_rows)
    write_jsonl(
        args.out_dir / "equi_agent_live_agent_trace.jsonl",
        [
            {
                **{key: row.get(key, "") for key in [
                    "patient_id",
                    "eye_id",
                    "visit_id",
                    "image_id",
                    "dataset",
                    "task",
                    "y_true",
                    "race",
                    "ethnicity",
                    "sex_gender",
                    "age",
                    "age_group",
                ]},
                "agent_trace": row.get("agent_trace", {}),
                "normalized_response": row.get("normalized_response", {}),
                "parsed_response": row.get("parsed_response", {}),
            }
            for row in raw_rows
        ],
    )
    write_jsonl(args.out_dir / "equi_agent_live_errors.jsonl", error_rows)
    write_jsonl(args.out_dir / "equi_agent_live_cases.jsonl", case_rows)
    write_jsonl(args.out_dir / "equi_agent_live_loaded_files.jsonl", loaded_files)
    if loaded_manifests:
        write_jsonl(args.out_dir / "equi_agent_live_loaded_manifests.jsonl", loaded_manifests)
    write_jsonl(args.out_dir / "equi_agent_live_missing_files.jsonl", missing_files)

    model_prior_rows = [
        model_row
        for case_row in case_rows
        for model_row in case_row.get("model_rows", [])
    ]
    model_prior_hits = sum(1 for row in model_prior_rows if row.get("prior_attribute") != "NONE")
    total_prompt = sum(int(row.get("prompt_tokens", 0)) for row in usage_rows)
    total_completion = sum(int(row.get("completion_tokens", 0)) for row in usage_rows)
    summary = {
        "dry_run": args.dry_run,
        "cases": len(prediction_rows),
        "errors": len(error_rows),
        "tasks": args.tasks,
        "models_requested": args.models,
        "sample_random": args.sample_random,
        "sample_stratified": args.sample_stratified,
        "target_positive_frac": args.target_positive_frac if args.sample_stratified else None,
        "loaded_files": len(loaded_files),
        "missing_files": missing_files,
        "model_prior_rows": len(model_prior_rows),
        "model_prior_hits": model_prior_hits,
        "model_prior_coverage": model_prior_hits / len(model_prior_rows) if model_prior_rows else 0.0,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "mean_tokens_per_case": mean([int(row.get("total_tokens", 0)) for row in usage_rows]) if usage_rows else 0.0,
        "llm_provider": provider,
        "llm_deployment": args.deployment,
        "outputs": {
            "predictions": str(args.out_dir / "equi_agent_live_predictions.csv"),
            "raw_responses": str(args.out_dir / "equi_agent_live_raw_responses.jsonl"),
            "agent_trace": str(args.out_dir / "equi_agent_live_agent_trace.jsonl"),
            "usage": str(args.out_dir / "equi_agent_live_usage.csv"),
            "errors": str(args.out_dir / "equi_agent_live_errors.jsonl"),
        },
    }
    write_json(args.out_dir / "equi_agent_live_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
