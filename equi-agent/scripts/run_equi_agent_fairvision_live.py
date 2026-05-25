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
    "llm_provider",
    "llm_deployment",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
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


def build_live_messages(evidence_packet: dict[str, Any], image_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    system = (
        "You are Equi-Agent, an ophthalmic foundation-model arbitration layer. Your job is benchmark arbitration, "
        "not open-ended clinical consultation. Use only the current task's retinal model outputs as disease evidence. "
        "Use demographic metadata only to select and interpret validation-derived model reliability priors; never use "
        "demographics as direct disease evidence. The field binary_prediction was computed using that model's "
        "validation-selected threshold, so it may differ from probability >= 0.5. Treat prior_unstable=true as "
        "statistical instability; treat low balanced accuracy as weak reliability, not instability. Prefer sensitivity "
        "when a reliable model/subgroup prior shows high false-negative risk. Prefer precision or down-weighting when "
        "a reliable model/subgroup prior shows high false-positive risk and low false-negative risk. Always provide a "
        "forced binary diagnostic prediction and probability for benchmark scoring. Escalation is only an auxiliary "
        "safety/referral flag; it must never replace or abstain from the diagnostic prediction. Return only valid JSON."
    )
    user = {
        "instructions": {
            "internal_agent_protocol": {
                "bio_profiler": (
                    "Summarize metadata only as reliability context for priors. Do not infer disease risk from race, "
                    "ethnicity, sex/gender, or age."
                ),
                "vision_specialist": (
                    "If OCT/SLO images are attached, independently inspect the retinal morphology first, then compare "
                    "against raw probabilities, validation-threshold binary predictions, modality patterns, and vote "
                    "disagreement for the current task only. If no images are attached, state that the visual review is "
                    "probability-only."
                ),
                "equity_auditor": (
                    "Translate false-negative and false-positive priors into sensitivity, precision, neutral, or "
                    "escalation advice. If prior_unstable is false but balanced accuracy is low, call it weak reliability."
                ),
                "orchestrator": (
                    "Anchor final_probability on deterministic_reference.weighted_probability. Adjust by at most 0.10 "
                    "only when the priors strongly justify a sensitivity or precision shift. Choose calibration_action first, "
                    "then apply its threshold: sensitivity_shift uses 0.35, neutral/escalate uses 0.50, and precision_shift "
                    "uses 0.65. final_prediction must be based on final_probability >= the applied threshold."
                ),
                "safety_agent": (
                    "Set escalate_to_human=true for close calls around the applied threshold, severe split votes, statistically "
                    "unstable priors, or severe weak reliability. Do not escalate merely because one or two of nine models disagree. "
                    "If the final probability is far from the applied threshold and votes are not severely split, set escalate_to_human=false. "
                    "Always keep the forced diagnostic prediction for F1 scoring."
                ),
            },
            "task": (
                "Return one forced binary disease decision for the requested task. "
                "Even when escalation is warranted, final_prediction must be 0 or 1 and will be used for F1/AUROC-style evaluation."
            ),
            "threshold": 0.5,
            "dynamic_thresholds": {
                "sensitivity_shift": 0.35,
                "neutral": 0.5,
                "escalate": 0.5,
                "precision_shift": 0.65,
            },
            "benchmark_rule": (
                "Do not abstain. Do not output -1. Do not use escalate_to_human as the diagnosis. "
                "Use it only as a separate safety flag. final_prediction must be threshold-consistent with final_probability "
                "at the applied dynamic threshold."
            ),
            "required_json_schema": {
                "final_probability": "float from 0 to 1",
                "final_prediction": "0 or 1",
                "confidence": "low, medium, or high",
                "primary_model": "model name or ensemble",
                "calibration_action": "sensitivity_shift, precision_shift, neutral, or escalate",
                "escalate_to_human": "boolean",
                "reasoning": "brief clinical-arbitration rationale",
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
        return select_cases(task_predictions, limit, offset)
    all_keys = select_cases(task_predictions, 10**12, 0)
    rng = random.Random(f"{seed}:{task}")
    rng.shuffle(all_keys)
    return all_keys[offset : offset + limit]


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


def threshold_for_action(action: str) -> float:
    if action == "sensitivity_shift":
        return 0.35
    if action == "precision_shift":
        return 0.65
    return 0.50


def normalize_prediction(probability: float, applied_threshold: float) -> int:
    return int(probability >= applied_threshold)


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


def dry_run_response(arbitration: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], str]:
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
        "final_probability": arbitration["final_prob"],
        "final_prediction": arbitration["final_pred"],
        "confidence": "low" if arbitration["safety_decision"] != "ACCEPT" else "medium",
        "primary_model": primary,
        "calibration_action": action,
        "applied_threshold": threshold_for_action(action),
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


def main() -> None:
    args = parse_args()
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
            messages = build_live_messages(evidence_packet, image_payload)

            try:
                if args.dry_run:
                    parsed, usage, raw_text = dry_run_response(arbitration)
                    usage["prompt_tokens"] = sum(
                        estimate_tokens(message_text_for_estimate(m), args.chars_per_token)
                        for m in messages
                    )
                    usage["completion_tokens"] = estimate_tokens(raw_text, args.chars_per_token)
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                else:
                    response = call_llm(client, args.deployment, messages, args.temperature, args.max_output_tokens)
                    raw_text = response_text(response)
                    usage = usage_dict(response)
                    parsed = json_from_text(raw_text)
                    if args.request_sleep_sec > 0:
                        time.sleep(args.request_sleep_sec)

                final_prob = clamp_probability(
                    parsed.get("final_probability"),
                    arbitration["final_prob"],
                    args.max_probability_adjustment,
                )
                calibration_action = normalize_action(parsed.get("calibration_action"))
                applied_threshold = threshold_for_action(calibration_action)
                final_pred = normalize_prediction(final_prob, applied_threshold)
                escalate_to_human, safety_decision = safety_flag_for_case(arbitration, final_prob, applied_threshold)
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
                        "llm_provider": provider,
                        "llm_deployment": args.deployment,
                        **usage,
                    }
                )
                usage_rows.append(
                    {
                        **meta,
                        "llm_provider": provider,
                        "llm_deployment": args.deployment,
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
                        "normalized_response": {
                            "final_probability": final_prob,
                            "final_prediction": final_pred,
                            "applied_threshold": applied_threshold,
                            "calibration_action": calibration_action,
                            "escalate_to_human": escalate_to_human,
                            "safety_decision": safety_decision,
                        },
                        "raw_response": raw_text,
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
        ],
    )
    write_jsonl(args.out_dir / "equi_agent_live_raw_responses.jsonl", raw_rows)
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
            "usage": str(args.out_dir / "equi_agent_live_usage.csv"),
            "errors": str(args.out_dir / "equi_agent_live_errors.jsonl"),
        },
    }
    write_json(args.out_dir / "equi_agent_live_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
