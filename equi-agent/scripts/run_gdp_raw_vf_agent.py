from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


EQUI_AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EQUI_AGENT_ROOT))

from FunctionalInterpretationAgent.function_interpreter import (  # noqa: E402
    GDP_TD_COLUMNS,
    compute_md_from_td_values,
    summarize_td_values,
    vf_severity,
)


OUTPUT_FIELDS = [
    "filename",
    "split_column",
    "split_value",
    "label_column",
    "y_true",
    "pred_gl",
    "is_correct",
    "final_probability",
    "confidence",
    "computed_md_from_td",
    "stored_md_audit",
    "md_audit_abs_diff",
    "vf_severity",
    "td_point_count",
    "td_min",
    "td_max",
    "td_mean",
    "td_depressed_at_least_2db_count",
    "td_depressed_at_least_5db_count",
    "td_depressed_at_least_10db_count",
    "td_depressed_at_least_5db_fraction",
    "calibration_action",
    "escalate_to_human",
    "safety_decision",
    "functional_summary",
    "orchestrator_rationale",
    "raw_response",
    "llm_provider",
    "llm_deployment",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Equi-Agent external validation on Harvard-GDP using raw visual-field "
            "total-deviation points. The stored MD column and target label are withheld "
            "from the agent prompt and saved only for audit/scoring."
        )
    )
    parser.add_argument("--data-summary", type=Path, default=REPO_ROOT / "Datasets" / "GDP" / "data_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=EQUI_AGENT_ROOT / "outputs" / "gdp_raw_vf_agent")
    parser.add_argument("--split-column", default="progression_forecasting_use")
    parser.add_argument("--split-value", default="test")
    parser.add_argument("--label-column", default="glaucoma")
    parser.add_argument("--max-cases", type=int, default=0, help="Use <=0 for all matching cases.")
    parser.add_argument("--sample-random", action="store_true")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic local rule without LLM calls.")
    parser.add_argument(
        "--functional-agent-mode",
        choices=["deterministic", "llm"],
        default="llm",
        help="Use deterministic summary or call the Functional Specialist LLM after computing MD from TD.",
    )
    parser.add_argument("--provider", choices=["auto", "azure", "openai"], default="auto")
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "gpt-5.1")
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--chars-per-token", type=float, default=4.0, help="Dry-run usage estimate only.")
    return parser.parse_args()


def resolve_data_summary(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.parent / "ReadMe" / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find GDP data summary at {path} or fallback {fallback}. "
        "Pass --data-summary explicitly if your GDP layout is different."
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def int_label(value: Any) -> int | None:
    number = fnum(value)
    if number is None:
        return None
    if int(number) not in {0, 1}:
        return None
    return int(number)


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(math.ceil(len(text) / max(chars_per_token, 1.0)))


def td_values_from_row(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for col in GDP_TD_COLUMNS:
        number = fnum(row.get(col))
        if number is not None:
            values[col] = number
    return values


def td_grid_6x9(td_values: dict[str, float]) -> list[list[float | None]]:
    grid = []
    for row_idx in range(6):
        current = []
        for col_idx in range(9):
            point = row_idx * 9 + col_idx + 1
            current.append(td_values.get(f"td{point}"))
        grid.append(current)
    return grid


def deterministic_functional_summary(computed_md: float | None, td_summary: dict[str, Any]) -> str:
    severity = vf_severity(computed_md)
    if computed_md is None:
        return (
            "- Computed MD from raw total-deviation points is unavailable.\n"
            "- Functional status is indeterminate because no usable visual-field vector was provided."
        )
    point_count = int(td_summary.get("td_point_count") or 0)
    depressed_5 = int(td_summary.get("td_depressed_at_least_5db_count") or 0)
    depressed_10 = int(td_summary.get("td_depressed_at_least_10db_count") or 0)
    return (
        f"- Computed MD from {point_count} raw total-deviation points is {computed_md:.2f} dB, "
        f"classified as {severity} by the prespecified thresholds.\n"
        f"- The field has {depressed_5} points at or below -5 dB and {depressed_10} points at or below -10 dB.\n"
        "- This summary is derived from raw TD points; stored MD is not used for agent inference."
    )


def clinical_narrative(row: dict[str, str]) -> str:
    age = row.get("age", "unknown")
    gender = row.get("gender", row.get("sex_gender", "unknown"))
    race = row.get("race", "unknown")
    hispanic = row.get("hispanic", row.get("ethnicity", "unknown"))
    return (
        f"Harvard-GDP external validation case with age {age}, gender {gender}, race {race}, "
        f"and Hispanic ethnicity field {hispanic}. Demographic fields are provided only for context "
        "and must not be used as direct disease evidence."
    )


def run_functional_agent(
    row: dict[str, str],
    td_values: dict[str, float],
    computed_md: float | None,
    mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    td_summary = summarize_td_values(list(td_values.values()))
    if dry_run or mode == "deterministic":
        return {
            "agent": "Functional Specialist",
            "full_report": deterministic_functional_summary(computed_md, td_summary),
            "summary": deterministic_functional_summary(computed_md, td_summary),
            "computed_md_from_td": computed_md,
            "md_used": computed_md,
            "severity": vf_severity(computed_md),
            "td_summary": td_summary,
        }

    from FunctionalInterpretationAgent.function_interpreter import FunctionalSpecialist

    state = {
        "metadata": {
            "age": row.get("age"),
            "gender": row.get("gender"),
            "race": row.get("race"),
            "hispanic": row.get("hispanic"),
            "td_values": td_values,
        },
        "clinical_narrative": clinical_narrative(row),
    }
    return FunctionalSpecialist().analyze(state)


def build_evidence_packet(
    row: dict[str, str],
    td_values: dict[str, float],
    computed_md: float | None,
    functional_output: dict[str, Any],
) -> dict[str, Any]:
    td_summary = summarize_td_values(list(td_values.values()))
    point_count = int(td_summary.get("td_point_count") or 0)
    depressed_5 = int(td_summary.get("td_depressed_at_least_5db_count") or 0)
    return {
        "case": {
            "filename": row.get("filename", ""),
            "dataset": "Harvard-GDP",
            "task": "external glaucoma detection from raw visual field",
            "hidden_label_policy": (
                "The target label and stored MD are withheld from this prompt. Stored MD is retained only "
                "for post-run audit outside the agent."
            ),
        },
        "patient_metadata_context_only": {
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
            "race": row.get("race", ""),
            "hispanic": row.get("hispanic", ""),
        },
        "raw_visual_field_total_deviation_db": td_values,
        "raw_visual_field_grid_6x9_with_blind_spots": td_grid_6x9(td_values),
        "computed_functional_features_from_raw_vf": {
            "computed_md_from_td": computed_md,
            "severity": vf_severity(computed_md),
            **td_summary,
            "td_depressed_at_least_5db_fraction": depressed_5 / point_count if point_count else None,
        },
        "functional_specialist_output": {
            "summary": functional_output.get("summary", ""),
            "severity": functional_output.get("severity", vf_severity(computed_md)),
            "computed_md_from_td": functional_output.get("computed_md_from_td", computed_md),
            "td_summary": functional_output.get("td_summary", td_summary),
        },
    }


def build_messages(evidence_packet: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are Equi-Agent running an external Harvard-GDP glaucoma validation from raw visual-field data. "
        "Your job is to classify glaucoma from the supplied total-deviation visual field and functional-agent summary.\n\n"
        "Critical leakage rule: the true glaucoma label and stored MD column are not provided. Do not ask for them, "
        "do not infer that a filename or split encodes them, and do not use demographics as direct disease evidence.\n\n"
        "The Functional Specialist has computed an MD-like score from the raw TD points. Use that computed value and "
        "the full TD vector as functional evidence. Normal is > -2 dB, early loss is -2 to -6 dB, moderate loss is "
        "-6 to -12 dB, and advanced loss is < -12 dB. Substantial clustered depressed points may support glaucoma "
        "even when the mean is borderline; a normal field should usually argue against glaucoma unless there is "
        "strong contradictory evidence in the supplied field pattern.\n\n"
        "Return compact valid JSON only, with no markdown, no comments, and no trailing commas. Required schema: "
        "{\"final_prediction\": 0 or 1, \"final_probability\": float, \"confidence\": \"low|medium|high\", "
        "\"calibration_action\": \"neutral|sensitivity_shift|precision_shift|escalate\", "
        "\"escalate_to_human\": boolean, \"safety_decision\": \"ACCEPT|ESCALATE_TO_HUMAN|INSUFFICIENT_DATA\", "
        "\"reasoning\": \"one concise sentence\", \"functional_findings\": \"one concise sentence\"}."
    )
    user = {
        "instructions": {
            "classification_target": "glaucoma presence",
            "use_raw_visual_field": True,
            "stored_md_available_to_agent": False,
            "label_available_to_agent": False,
            "demographics_direct_disease_evidence": False,
            "output_prediction_required_for_retrospective_scoring": True,
        },
        "evidence_packet": evidence_packet,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, sort_keys=True)},
    ]


def dry_run_decision(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    features = evidence_packet["computed_functional_features_from_raw_vf"]
    md = features.get("computed_md_from_td")
    point_count = int(features.get("td_point_count") or 0)
    depressed_5_fraction = features.get("td_depressed_at_least_5db_fraction") or 0.0
    depressed_10 = int(features.get("td_depressed_at_least_10db_count") or 0)
    if md is None:
        pred = 0
        prob = 0.5
        confidence = "low"
        safety = "INSUFFICIENT_DATA"
        reason = "No usable raw visual-field points were available."
    else:
        functional_signal = (-md / 12.0) + depressed_5_fraction + min(depressed_10 / max(point_count, 1), 0.25)
        prob = max(0.02, min(0.98, functional_signal))
        pred = int(md <= -2.0 or depressed_5_fraction >= 0.15 or depressed_10 >= 3)
        confidence = "high" if abs(md + 2.0) > 2.0 else "medium"
        safety = "ACCEPT" if confidence == "high" else "ESCALATE_TO_HUMAN"
        reason = f"Deterministic dry run used computed MD {md:.2f} dB and depressed-point burden."
    return {
        "final_prediction": pred,
        "final_probability": prob,
        "confidence": confidence,
        "calibration_action": "neutral",
        "escalate_to_human": safety != "ACCEPT",
        "safety_decision": safety,
        "reasoning": reason,
        "functional_findings": evidence_packet["functional_specialist_output"].get("summary", ""),
    }


def get_client(provider: str, deployment: str, api_version: str):
    load_dotenv()
    if provider == "auto":
        provider = "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "openai"
    if provider == "azure":
        from openai import AzureOpenAI

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for Azure runs.")
        return provider, AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY for OpenAI runs.")
    return provider, OpenAI(api_key=api_key)


def completion(client: Any, provider: str, deployment: str, messages: list[dict[str, str]], temperature: float, max_tokens: int):
    kwargs = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
    }
    if provider == "azure":
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    return client.chat.completions.create(**kwargs)


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_prediction(value: Any) -> int:
    number = fnum(value, 0.0)
    return int(number >= 0.5)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("is_correct") in {0, 1}]
    tp = sum(1 for row in valid if row["y_true"] == 1 and row["pred_gl"] == 1)
    tn = sum(1 for row in valid if row["y_true"] == 0 and row["pred_gl"] == 0)
    fp = sum(1 for row in valid if row["y_true"] == 0 and row["pred_gl"] == 1)
    fn = sum(1 for row in valid if row["y_true"] == 1 and row["pred_gl"] == 0)
    accuracy = mean([row["is_correct"] for row in valid]) if valid else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "n": len(rows),
        "valid": len(valid),
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def computed_md_rule_metrics(rows: list[dict[str, Any]], threshold: float = -2.0) -> dict[str, Any]:
    rule_rows = []
    for row in rows:
        y_true = row.get("y_true")
        computed_md = fnum(row.get("computed_md_from_td"))
        if y_true not in {0, 1} or computed_md is None:
            continue
        pred = int(computed_md <= threshold)
        rule_rows.append({"y_true": y_true, "pred_gl": pred, "is_correct": int(pred == y_true)})
    return {
        "rule": f"computed_md_from_td <= {threshold:g}",
        **metrics(rule_rows),
    }


def main() -> None:
    args = parse_args()
    args.data_summary = resolve_data_summary(args.data_summary)
    rows = read_csv(args.data_summary)
    selected = [
        row
        for row in rows
        if row.get(args.split_column) == args.split_value and int_label(row.get(args.label_column)) is not None
    ]
    if args.sample_random:
        rng = random.Random(args.random_seed)
        rng.shuffle(selected)
    if args.max_cases > 0:
        selected = selected[: args.max_cases]

    provider = "dry_run"
    client = None
    if not args.dry_run:
        provider, client = get_client(args.provider, args.deployment, args.api_version)

    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []

    for index, row in enumerate(selected, start=1):
        y_true = int_label(row.get(args.label_column))
        td_values = td_values_from_row(row)
        computed_md = compute_md_from_td_values(list(td_values.values()))
        td_summary = summarize_td_values(list(td_values.values()))
        stored_md = fnum(row.get("md"))
        md_abs_diff = abs(computed_md - stored_md) if computed_md is not None and stored_md is not None else None
        try:
            functional_output = run_functional_agent(
                row,
                td_values,
                computed_md,
                mode=args.functional_agent_mode,
                dry_run=args.dry_run,
            )
            evidence_packet = build_evidence_packet(row, td_values, computed_md, functional_output)
            messages = build_messages(evidence_packet)
            if args.dry_run:
                parsed = dry_run_decision(evidence_packet)
                raw_response = json.dumps(parsed, sort_keys=True)
                usage = {
                    "prompt_tokens": sum(estimate_tokens(message["content"], args.chars_per_token) for message in messages),
                    "completion_tokens": estimate_tokens(raw_response, args.chars_per_token),
                    "total_tokens": 0,
                }
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            else:
                response = completion(
                    client,
                    provider,
                    args.deployment,
                    messages,
                    args.temperature,
                    args.max_output_tokens,
                )
                raw_response = response.choices[0].message.content or ""
                parsed = json_from_text(raw_response)
                usage = usage_dict(response)

            pred_gl = normalize_prediction(parsed.get("final_prediction"))
            is_correct = int(pred_gl == y_true) if y_true is not None else -1
            output_row = {
                "filename": row.get("filename", ""),
                "split_column": args.split_column,
                "split_value": args.split_value,
                "label_column": args.label_column,
                "y_true": y_true,
                "pred_gl": pred_gl,
                "is_correct": is_correct,
                "final_probability": fnum(parsed.get("final_probability"), 0.0),
                "confidence": parsed.get("confidence", ""),
                "computed_md_from_td": computed_md,
                "stored_md_audit": stored_md,
                "md_audit_abs_diff": md_abs_diff,
                "vf_severity": vf_severity(computed_md),
                "td_point_count": td_summary.get("td_point_count"),
                "td_min": td_summary.get("td_min"),
                "td_max": td_summary.get("td_max"),
                "td_mean": td_summary.get("td_mean"),
                "td_depressed_at_least_2db_count": td_summary.get("td_depressed_at_least_2db_count"),
                "td_depressed_at_least_5db_count": td_summary.get("td_depressed_at_least_5db_count"),
                "td_depressed_at_least_10db_count": td_summary.get("td_depressed_at_least_10db_count"),
                "td_depressed_at_least_5db_fraction": (
                    (td_summary.get("td_depressed_at_least_5db_count") or 0)
                    / (td_summary.get("td_point_count") or 1)
                ),
                "calibration_action": parsed.get("calibration_action", ""),
                "escalate_to_human": boolish(parsed.get("escalate_to_human")),
                "safety_decision": parsed.get("safety_decision", ""),
                "functional_summary": functional_output.get("summary", ""),
                "orchestrator_rationale": parsed.get("reasoning", ""),
                "raw_response": raw_response,
                "llm_provider": provider,
                "llm_deployment": args.deployment,
                **usage,
            }
            predictions.append(output_row)
            traces.append(
                {
                    "index": index,
                    "filename": row.get("filename", ""),
                    "evidence_packet": evidence_packet,
                    "parsed": parsed,
                    "raw_response": raw_response,
                    "usage": usage,
                }
            )
            usage_rows.append(
                {
                    "filename": row.get("filename", ""),
                    "llm_provider": provider,
                    "llm_deployment": args.deployment,
                    **usage,
                }
            )
            if args.request_sleep_sec > 0:
                time.sleep(args.request_sleep_sec)
        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "filename": row.get("filename", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    out_predictions = args.out_dir / "gdp_raw_vf_agent_predictions.csv"
    out_trace = args.out_dir / "gdp_raw_vf_agent_trace.jsonl"
    out_errors = args.out_dir / "gdp_raw_vf_agent_errors.jsonl"
    out_usage = args.out_dir / "gdp_raw_vf_agent_usage.csv"
    out_summary = args.out_dir / "gdp_raw_vf_agent_summary.json"

    write_csv(out_predictions, predictions, OUTPUT_FIELDS)
    write_jsonl(out_trace, traces)
    write_jsonl(out_errors, errors)
    write_csv(out_usage, usage_rows, ["filename", "llm_provider", "llm_deployment", "prompt_tokens", "completion_tokens", "total_tokens"])

    metric_summary = metrics(predictions)
    md_rule_summary = computed_md_rule_metrics(predictions)
    summary = {
        **metric_summary,
        "computed_md_rule_baseline": md_rule_summary,
        "dry_run": args.dry_run,
        "functional_agent_mode": args.functional_agent_mode,
        "data_summary": str(args.data_summary),
        "split_column": args.split_column,
        "split_value": args.split_value,
        "label_column": args.label_column,
        "errors": len(errors),
        "llm_provider": provider,
        "llm_deployment": args.deployment,
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in usage_rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in usage_rows),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in usage_rows),
        "outputs": {
            "predictions": str(out_predictions),
            "trace": str(out_trace),
            "errors": str(out_errors),
            "usage": str(out_usage),
            "summary": str(out_summary),
        },
    }
    write_json(out_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
