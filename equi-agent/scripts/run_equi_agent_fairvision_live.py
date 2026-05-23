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

from smoke_equi_agent_arbitration import (
    DEFAULT_MODELS,
    TASKS,
    case_metadata,
    estimate_tokens,
    fnum,
    load_predictions,
    load_priors,
    mock_arbitrate,
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
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/equi_agent_live"))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-cases-per-task", type=int, default=3, help="Use <=0 to process all common cases.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset into each task's shared case list.")
    parser.add_argument("--sample-random", action="store_true", help="Randomly sample common cases per task instead of taking sorted cases.")
    parser.add_argument("--random-seed", type=int, default=2026, help="Seed used with --sample-random.")
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
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


def build_live_messages(evidence_packet: dict[str, Any]) -> list[dict[str, str]]:
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
                    "Compare raw probabilities, validation-threshold binary predictions, modality patterns, and vote "
                    "disagreement for the current task only."
                ),
                "equity_auditor": (
                    "Translate false-negative and false-positive priors into sensitivity, precision, neutral, or "
                    "escalation advice. If prior_unstable is false but balanced accuracy is low, call it weak reliability."
                ),
                "orchestrator": (
                    "Anchor final_probability on deterministic_reference.weighted_probability. Adjust by at most 0.10 "
                    "only when the priors strongly justify a sensitivity or precision shift. final_prediction must equal "
                    "1 when final_probability >= 0.5 and must equal 0 when final_probability < 0.5. If a sensitivity "
                    "shift still leaves final_probability below 0.5, the forced benchmark diagnosis remains 0."
                ),
                "safety_agent": (
                    "Set escalate_to_human=true for major disagreement, close calls, statistically unstable priors, or "
                    "severe weak reliability. If deterministic_reference.safety_decision is ACCEPT, there is no disagreement, "
                    "and the case is not a close call, set escalate_to_human=false unless there is an explicit data-quality problem. "
                    "Always keep the forced diagnostic prediction for F1 scoring."
                ),
            },
            "task": (
                "Return one forced binary disease decision for the requested task. "
                "Even when escalation is warranted, final_prediction must be 0 or 1 and will be used for F1/AUROC-style evaluation."
            ),
            "threshold": 0.5,
            "benchmark_rule": (
                "Do not abstain. Do not output -1. Do not use escalate_to_human as the diagnosis. "
                "Use it only as a separate safety flag. final_prediction must be threshold-consistent with final_probability at 0.5."
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
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, sort_keys=True)},
    ]


def select_case_keys(task_predictions: dict[str, dict[tuple, dict]], limit: int, offset: int, sample_random: bool, seed: int, task: str) -> list[tuple]:
    if not sample_random:
        return select_cases(task_predictions, limit, offset)
    all_keys = select_cases(task_predictions, 10**12, 0)
    rng = random.Random(f"{seed}:{task}")
    rng.shuffle(all_keys)
    return all_keys[offset : offset + limit]


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


def normalize_prediction(value: Any, probability: float) -> int:
    return int(probability >= 0.5)


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
        keys = select_case_keys(by_task[task], limit, args.seed_offset, args.sample_random, args.random_seed, task)
        for key in keys:
            rows_by_model = {
                model: rows_by_key[key]
                for model, rows_by_key in by_task[task].items()
                if key in rows_by_key
            }
            meta = case_metadata(rows_by_model)
            arbitration = mock_arbitrate(task, rows_by_model, meta, subgroup_priors, global_priors)
            evidence_packet = build_evidence_packet(meta, arbitration)
            messages = build_live_messages(evidence_packet)

            try:
                if args.dry_run:
                    parsed, usage, raw_text = dry_run_response(arbitration)
                    usage["prompt_tokens"] = sum(estimate_tokens(m["content"], args.chars_per_token) for m in messages)
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
                final_pred = normalize_prediction(parsed.get("final_prediction"), final_prob)
                prediction_rows.append(
                    {
                        **meta,
                        "model_name": "equi_agent_live" if not args.dry_run else "equi_agent_live_dry_run",
                        "y_prob": f"{final_prob:.6f}",
                        "y_pred": final_pred,
                        "split": "test",
                        "positive_votes": arbitration["positive_votes"],
                        "num_models": arbitration["num_models"],
                        "disagreement": arbitration["disagreement"],
                        "close_call": arbitration["close_call"],
                        "safety_decision": arbitration["safety_decision"],
                        "primary_model": parsed.get("primary_model", ""),
                        "confidence": parsed.get("confidence", ""),
                        "calibration_action": parsed.get("calibration_action", ""),
                        "escalate_to_human": parsed.get("escalate_to_human", ""),
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
