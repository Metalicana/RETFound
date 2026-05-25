from __future__ import annotations

import argparse
import csv
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

from smoke_equi_agent_arbitration import estimate_tokens, fnum, write_csv, write_jsonl


TASK = "progression_forecasting"
DATASET = "harvard_gdp"
CASE_KEY = ("patient_id", "eye_id", "visit_id", "image_id", "task")
DEFAULT_MODELS = ["retfound_oct", "visionfm_oct", "urfound_oct"]
CLASSICAL_MODELS = [
    "rnflt_logreg",
    "bscan_logreg",
    "clinical_logreg",
    "rnflt_clinical_logreg",
    "bscan_clinical_logreg",
    "all_logreg",
]

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
    "mean_probability",
    "weighted_probability",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Equi-Agent arbitration for Harvard-GDP longitudinal glaucoma "
            "progression forecasting. Use --dry-run to validate the pipeline "
            "without API calls."
        )
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument("--metrics-root", type=Path, default=Path("equi-agent/outputs/metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/equi_agent_gdp_progression_live"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--include-classical-baselines",
        action="store_true",
        help="Also include RNFLT, B-scan, clinical, and combined logistic baselines as longitudinal evidence sources.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Use <=0 to process all shared test cases.")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--sample-random", action="store_true")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument(
        "--max-probability-adjustment",
        type=float,
        default=0.15,
        help="Clamp LLM final_probability to deterministic weighted_probability +/- this margin. Use <0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", choices=["auto", "azure", "openai"], default="auto")
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    return parser.parse_args()


def prediction_file(predictions_root: Path, model: str) -> Path:
    return predictions_root / f"gdp_progression_forecasting_{model}.csv"


def key_for(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return tuple(row.get(col, "") for col in CASE_KEY)


def load_predictions(predictions_root: Path, models: list[str]) -> tuple[dict[str, dict[tuple, dict]], list[dict], list[dict]]:
    by_model: dict[str, dict[tuple, dict]] = {}
    loaded_files = []
    missing_files = []
    for model in models:
        path = prediction_file(predictions_root, model)
        if not path.exists():
            missing_files.append({"model": model, "path": str(path), "reason": "missing"})
            continue
        rows = [
            row
            for row in read_csv(path)
            if row.get("dataset") == DATASET and row.get("task") == TASK and row.get("split") == "test"
        ]
        if not rows:
            missing_files.append({"model": model, "path": str(path), "reason": "no test rows"})
            continue
        by_model[model] = {key_for(row): row for row in rows}
        loaded_files.append({"model": model, "path": str(path), "rows": len(rows)})
    return by_model, loaded_files, missing_files


def select_cases(by_model: dict[str, dict[tuple, dict]], limit: int, offset: int, sample_random: bool, seed: int) -> list[tuple]:
    common: set[tuple] | None = None
    for rows_by_key in by_model.values():
        keys = set(rows_by_key)
        common = keys if common is None else common & keys
    if not common:
        return []
    ordered = sorted(common, key=lambda key: (key[0], key[3]))
    if sample_random:
        rng = random.Random(seed)
        rng.shuffle(ordered)
    if limit <= 0:
        limit = len(ordered)
    return ordered[offset : offset + limit]


def case_metadata(rows_by_model: dict[str, dict]) -> dict[str, Any]:
    row = next(iter(rows_by_model.values()))
    return {
        "patient_id": row.get("patient_id", ""),
        "eye_id": row.get("eye_id", ""),
        "visit_id": row.get("visit_id", ""),
        "image_id": row.get("image_id", ""),
        "dataset": row.get("dataset", ""),
        "task": row.get("task", ""),
        "y_true": row.get("y_true", ""),
        "race": row.get("race", ""),
        "ethnicity": row.get("ethnicity", ""),
        "sex_gender": row.get("sex_gender", ""),
        "age": row.get("age", ""),
        "age_group": row.get("age_group", ""),
    }


def aggregate_path(metrics_root: Path, model: str) -> Path:
    return (
        metrics_root
        / f"exp8_gdp_progression_forecasting_{model.replace('_logreg', '')}"
        / f"gdp_progression_forecasting_{model}_aggregate.csv"
    )


def disparity_path(metrics_root: Path, model: str) -> Path:
    return (
        metrics_root
        / f"exp8_gdp_progression_forecasting_{model.replace('_logreg', '')}"
        / f"gdp_progression_forecasting_{model}_disparities.csv"
    )


def load_model_priors(metrics_root: Path, models: list[str]) -> tuple[dict[str, dict[str, float]], list[dict]]:
    priors: dict[str, dict[str, float]] = {}
    loaded = []
    for model in models:
        path = aggregate_path(metrics_root, model)
        if not path.exists():
            continue
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[0]
        priors[model] = {
            "f1": fnum(row.get("f1"), 0.0),
            "balanced_accuracy": fnum(row.get("balanced_accuracy"), 0.5),
            "ece": fnum(row.get("ece"), 0.25),
            "fpr": fnum(row.get("fpr"), 0.0),
            "fnr": fnum(row.get("fnr"), 1.0),
            "sensitivity": fnum(row.get("sensitivity"), 0.0),
            "specificity": fnum(row.get("specificity"), 0.0),
        }
        loaded.append({"model": model, "path": str(path)})
    return priors, loaded


def model_weight(model: str, prior: dict[str, float] | None) -> float:
    if prior is None:
        return 0.5
    balanced = prior.get("balanced_accuracy", 0.5)
    ece = min(prior.get("ece", 0.25), 0.8)
    f1 = prior.get("f1", 0.0)
    return max(0.05, (0.70 * balanced + 0.30 * f1) * (1.0 - ece))


def deterministic_arbitrate(rows_by_model: dict[str, dict], priors: dict[str, dict[str, float]]) -> dict[str, Any]:
    weighted_sum = 0.0
    weight_total = 0.0
    probs = []
    model_rows = []
    positive_votes = 0
    for model, row in sorted(rows_by_model.items()):
        prior = priors.get(model)
        prob = fnum(row.get("y_prob"), 0.0)
        pred = int(fnum(row.get("y_pred"), 0.0) >= 0.5)
        weight = model_weight(model, prior)
        probs.append(prob)
        positive_votes += pred
        weighted_sum += prob * weight
        weight_total += weight
        model_rows.append(
            {
                "model": model,
                "probability": prob,
                "binary_prediction": pred,
                "trust_weight": weight,
                "global_f1": prior.get("f1", 0.0) if prior else 0.0,
                "global_balanced_accuracy": prior.get("balanced_accuracy", 0.5) if prior else 0.5,
                "global_fpr": prior.get("fpr", 0.0) if prior else 0.0,
                "global_fnr": prior.get("fnr", 1.0) if prior else 1.0,
                "global_ece": prior.get("ece", 0.25) if prior else 0.25,
            }
        )
    weighted_probability = weighted_sum / weight_total if weight_total else 0.0
    mean_probability = mean(probs) if probs else 0.0
    disagreement = max(probs) - min(probs) if probs else 0.0
    close_call = abs(weighted_probability - 0.5) < 0.10
    severe_split = 0 < positive_votes < len(rows_by_model)
    weak_reliability = any(row["global_balanced_accuracy"] < 0.55 for row in model_rows)
    safety_decision = "ESCALATE_TO_HUMAN" if close_call or disagreement >= 0.25 or severe_split or weak_reliability else "ACCEPT"
    return {
        "weighted_probability": weighted_probability,
        "mean_probability": mean_probability,
        "weighted_prediction": int(weighted_probability >= 0.5),
        "positive_votes": positive_votes,
        "num_models": len(rows_by_model),
        "disagreement": disagreement,
        "close_call": close_call,
        "safety_decision": safety_decision,
        "model_rows": model_rows,
    }


def build_evidence_packet(meta: dict[str, Any], arbitration: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": {
            "case_id": "|".join(str(meta.get(col, "")) for col in CASE_KEY),
            "dataset": meta.get("dataset", ""),
            "task": "longitudinal glaucoma progression forecasting",
        },
        "patient_metadata_for_reliability_only": {
            "race": meta.get("race", ""),
            "ethnicity": meta.get("ethnicity", ""),
            "sex_gender": meta.get("sex_gender", ""),
            "age": meta.get("age", ""),
            "age_group": meta.get("age_group", ""),
        },
        "evidence_sources": [
            {
                "model": row["model"],
                "probability": round(row["probability"], 6),
                "binary_prediction": row["binary_prediction"],
                "global_trust_weight": round(row["trust_weight"], 6),
                "global_f1": round(row["global_f1"], 6),
                "global_balanced_accuracy": round(row["global_balanced_accuracy"], 6),
                "global_false_positive_rate": round(row["global_fpr"], 6),
                "global_false_negative_rate": round(row["global_fnr"], 6),
                "global_ece": round(row["global_ece"], 6),
            }
            for row in arbitration["model_rows"]
        ],
        "deterministic_reference": {
            "weighted_probability": round(arbitration["weighted_probability"], 6),
            "mean_probability": round(arbitration["mean_probability"], 6),
            "weighted_prediction": arbitration["weighted_prediction"],
            "positive_votes": arbitration["positive_votes"],
            "num_models": arbitration["num_models"],
            "probability_range": round(arbitration["disagreement"], 6),
            "close_call": arbitration["close_call"],
            "safety_decision": arbitration["safety_decision"],
        },
    }


def build_live_messages(evidence_packet: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are Equi-Agent for longitudinal glaucoma progression forecasting. "
        "Your job is benchmark arbitration, not open-ended clinical consultation. "
        "Use only the supplied progression model outputs as disease/progression evidence. "
        "Use demographics only as reliability context; never infer progression risk from race, ethnicity, sex/gender, or age. "
        "The base rate is low, so avoid calling progression positive from demographics or vague concern alone. "
        "However, do not ignore a consistent high-probability or positive-vote signal. "
        "Return only valid JSON."
    )
    user_payload = {
        "instructions": {
            "bio_profiler": "Summarize metadata only as reliability context. Do not use it as direct progression evidence.",
            "temporal_specialist": (
                "Treat OCT foundation probes and optional RNFLT/clinical/logistic sources as longitudinal evidence streams. "
                "Look for consistency across sources rather than one noisy positive."
            ),
            "equity_auditor": (
                "Use global reliability priors as model trust evidence. High FNR supports a sensitivity shift; "
                "high FPR or weak balanced accuracy supports precision or escalation."
            ),
            "orchestrator": (
                "Anchor final_probability on deterministic_reference.weighted_probability. "
                "Adjust by at most 0.15 only when the evidence strongly justifies it. "
                "Choose calibration_action first, then apply its threshold: sensitivity_shift=0.35, neutral/escalate=0.50, precision_shift=0.65. "
                "final_prediction must equal final_probability >= applied threshold."
            ),
            "safety_agent": (
                "Set escalate_to_human=true for close thresholds, severe source disagreement, or weak reliability. "
                "Escalation is separate from the forced benchmark prediction."
            ),
            "required_json_schema": {
                "final_probability": "float from 0 to 1",
                "final_prediction": "0 or 1",
                "confidence": "low, medium, or high",
                "primary_model": "model/source name or ensemble",
                "calibration_action": "sensitivity_shift, precision_shift, neutral, or escalate",
                "escalate_to_human": "boolean",
                "reasoning": "brief arbitration rationale",
            },
        },
        "evidence_packet": evidence_packet,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
    ]


def json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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


def clamp_probability(value: Any, fallback: float, max_adjustment: float) -> float:
    prob = fnum(value, fallback)
    if max_adjustment >= 0:
        prob = min(fallback + max_adjustment, max(fallback - max_adjustment, prob))
    return min(1.0, max(0.0, prob))


def dry_run_response(arbitration: dict[str, Any]) -> tuple[dict[str, Any], str]:
    high_fnr = any(row["global_fnr"] >= 0.80 for row in arbitration["model_rows"])
    high_fpr = any(row["global_fpr"] >= 0.25 for row in arbitration["model_rows"])
    action = "neutral"
    if arbitration["safety_decision"] != "ACCEPT":
        action = "escalate"
    elif high_fnr and arbitration["weighted_probability"] >= 0.25:
        action = "sensitivity_shift"
    elif high_fpr:
        action = "precision_shift"
    primary_model = max(arbitration["model_rows"], key=lambda row: row["trust_weight"])["model"]
    parsed = {
        "final_probability": arbitration["weighted_probability"],
        "final_prediction": int(arbitration["weighted_probability"] >= threshold_for_action(action)),
        "confidence": "low" if action == "escalate" else "medium",
        "primary_model": primary_model,
        "calibration_action": action,
        "escalate_to_human": action == "escalate",
        "reasoning": "Dry-run deterministic GDP progression arbitration.",
    }
    return parsed, json.dumps(parsed, sort_keys=True)


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
        return provider, AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

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


def response_text(response: Any) -> str:
    return response.choices[0].message.content or ""


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def main() -> None:
    args = parse_args()
    models = list(dict.fromkeys(args.models + (CLASSICAL_MODELS if args.include_classical_baselines else [])))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_model, loaded_files, missing_files = load_predictions(args.predictions_root, models)
    priors, loaded_priors = load_model_priors(args.metrics_root, list(by_model))
    keys = select_cases(by_model, args.max_cases, args.seed_offset, args.sample_random, args.random_seed)

    provider = "dry_run"
    client = None
    if not args.dry_run:
        provider, client = make_client(args.provider, args.api_version)

    prediction_rows = []
    usage_rows = []
    raw_rows = []
    error_rows = []
    case_rows = []

    for key in keys:
        rows_by_model = {model: rows[key] for model, rows in by_model.items() if key in rows}
        meta = case_metadata(rows_by_model)
        arbitration = deterministic_arbitrate(rows_by_model, priors)
        evidence_packet = build_evidence_packet(meta, arbitration)
        messages = build_live_messages(evidence_packet)

        try:
            if args.dry_run:
                parsed, raw_text = dry_run_response(arbitration)
                usage = {
                    "prompt_tokens": sum(estimate_tokens(message["content"], args.chars_per_token) for message in messages),
                    "completion_tokens": estimate_tokens(raw_text, args.chars_per_token),
                }
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            else:
                response = call_llm(client, args.deployment, messages, args.temperature, args.max_output_tokens)
                raw_text = response_text(response)
                parsed = json_from_text(raw_text)
                usage = usage_dict(response)
                if args.request_sleep_sec > 0:
                    time.sleep(args.request_sleep_sec)

            action = normalize_action(parsed.get("calibration_action"))
            applied_threshold = threshold_for_action(action)
            final_prob = clamp_probability(parsed.get("final_probability"), arbitration["weighted_probability"], args.max_probability_adjustment)
            final_pred = int(final_prob >= applied_threshold)
            close_to_threshold = abs(final_prob - applied_threshold) < 0.075
            severe_disagreement = arbitration["disagreement"] >= 0.25 or 0 < arbitration["positive_votes"] < arbitration["num_models"]
            escalate_to_human = bool(parsed.get("escalate_to_human")) or close_to_threshold or severe_disagreement
            safety_decision = "ESCALATE_TO_HUMAN" if escalate_to_human else "ACCEPT"

            prediction_rows.append(
                {
                    **meta,
                    "model_name": "equi_agent_gdp_progression_live" if not args.dry_run else "equi_agent_gdp_progression_dry_run",
                    "y_prob": f"{final_prob:.6f}",
                    "y_pred": final_pred,
                    "applied_threshold": f"{applied_threshold:.2f}",
                    "split": "test",
                    "positive_votes": arbitration["positive_votes"],
                    "num_models": arbitration["num_models"],
                    "mean_probability": f"{arbitration['mean_probability']:.6f}",
                    "weighted_probability": f"{arbitration['weighted_probability']:.6f}",
                    "disagreement": f"{arbitration['disagreement']:.6f}",
                    "close_call": close_to_threshold,
                    "safety_decision": safety_decision,
                    "primary_model": parsed.get("primary_model", ""),
                    "confidence": parsed.get("confidence", ""),
                    "calibration_action": action,
                    "escalate_to_human": escalate_to_human,
                    "llm_provider": provider,
                    "llm_deployment": args.deployment,
                    **usage,
                }
            )
            usage_rows.append({**meta, "llm_provider": provider, "llm_deployment": args.deployment, **usage})
            raw_rows.append(
                {
                    **meta,
                    "messages": messages,
                    "evidence_packet": evidence_packet,
                    "parsed_response": parsed,
                    "raw_response": raw_text,
                }
            )
            case_rows.append({**meta, **arbitration, "evidence_packet": evidence_packet})
        except Exception as exc:
            error_rows.append(
                {
                    **meta,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "messages": messages,
                    "evidence_packet": evidence_packet,
                }
            )

    write_csv(args.out_dir / "equi_agent_gdp_progression_predictions.csv", prediction_rows, OUTPUT_FIELDS)
    write_csv(
        args.out_dir / "equi_agent_gdp_progression_usage.csv",
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
    write_jsonl(args.out_dir / "equi_agent_gdp_progression_raw_responses.jsonl", raw_rows)
    write_jsonl(args.out_dir / "equi_agent_gdp_progression_errors.jsonl", error_rows)
    write_jsonl(args.out_dir / "equi_agent_gdp_progression_cases.jsonl", case_rows)
    write_jsonl(args.out_dir / "equi_agent_gdp_progression_loaded_files.jsonl", loaded_files)
    write_jsonl(args.out_dir / "equi_agent_gdp_progression_missing_files.jsonl", missing_files)
    write_jsonl(args.out_dir / "equi_agent_gdp_progression_loaded_priors.jsonl", loaded_priors)

    total_prompt = sum(int(row.get("prompt_tokens", 0)) for row in usage_rows)
    total_completion = sum(int(row.get("completion_tokens", 0)) for row in usage_rows)
    summary = {
        "dry_run": args.dry_run,
        "cases": len(prediction_rows),
        "errors": len(error_rows),
        "task": TASK,
        "models_requested": models,
        "loaded_files": loaded_files,
        "missing_files": missing_files,
        "loaded_priors": loaded_priors,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "mean_tokens_per_case": mean([int(row.get("total_tokens", 0)) for row in usage_rows]) if usage_rows else 0.0,
        "llm_provider": provider,
        "llm_deployment": args.deployment,
        "outputs": {
            "predictions": str(args.out_dir / "equi_agent_gdp_progression_predictions.csv"),
            "raw_responses": str(args.out_dir / "equi_agent_gdp_progression_raw_responses.jsonl"),
            "usage": str(args.out_dir / "equi_agent_gdp_progression_usage.csv"),
            "errors": str(args.out_dir / "equi_agent_gdp_progression_errors.jsonl"),
        },
    }
    write_json(args.out_dir / "equi_agent_gdp_progression_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
