from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_equi_agent_gdp_progression_live import (
    DATASET,
    OUTPUT_FIELDS,
    TASK,
    build_evidence_packet,
    build_live_messages,
    call_llm,
    case_metadata,
    clamp_probability,
    deterministic_arbitrate,
    dry_run_response,
    json_from_text,
    load_model_priors,
    load_predictions,
    make_client,
    normalize_action,
    parse_case_key_columns,
    response_text,
    short_text,
    threshold_for_action,
    trace_field,
    usage_dict,
    write_json,
)
from smoke_equi_agent_arbitration import estimate_tokens, write_csv, write_jsonl


TARGETS = [
    "md",
    "vfi",
    "td_pointwise",
    "md_fast",
    "md_fast_no_p_cut",
    "td_pointwise_no_p_cut",
]

DEFAULT_MODELS = [
    "gdp_native_rnflt_tds_multitask_efficientnet",
    "retfound_oct",
    "rnflt_logreg",
    "clinical_logreg",
    "bscan_logreg",
    "rnflt_clinical_logreg",
    "bscan_clinical_logreg",
    "all_logreg",
]

MODEL_NAME = "equi_agent_gdp_progression_multitarget_live"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Equi-Agent for all six GDP progression endpoints in one call per patient."
    )
    parser.add_argument(
        "--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions")
    )
    parser.add_argument("--metrics-root", type=Path, default=Path("equi-agent/outputs/metrics"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("equi-agent/outputs/equi_agent_gdp_progression_multitarget_live")
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument(
        "--case-key-columns",
        default="image_id,task",
        help=(
            "Columns used to align model predictions. GDP model exporters serialize "
            "patient_id inconsistently, while image_id is the locked canonical case ID."
        ),
    )
    parser.add_argument(
        "--reference-strategy",
        choices=["weighted", "best_f1", "best_balanced_accuracy", "best_auroc"],
        default="weighted",
    )
    parser.add_argument("--lock-reference-prediction", action="store_true")
    parser.add_argument("--max-probability-adjustment", type=float, default=0.15)
    parser.add_argument("--deployment", default="gpt-5.1")
    parser.add_argument("--provider", choices=["auto", "azure", "openai"], default="auto")
    parser.add_argument("--api-version", default="2024-12-01-preview")
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=5.0)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--require-oof-priors", action="store_true")
    parser.add_argument("--expected-prior-cases", type=int, default=300)
    parser.add_argument("--max-consecutive-errors", type=int, default=0,
                        help="Stop after this many failed cases; 0 disables the circuit breaker")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def prediction_path(out_dir: Path, target: str) -> Path:
    return out_dir / f"predictions_{target}.csv"


def load_existing(
    out_dir: Path,
    targets: list[str],
    case_key_columns: tuple[str, ...],
) -> dict[str, dict[tuple[str, ...], dict[str, str]]]:
    result: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    for target in targets:
        path = prediction_path(out_dir, target)
        rows = read_csv(path) if path.is_file() else []
        result[target] = {
            tuple(row.get(column, "") for column in case_key_columns): row
            for row in rows
        }
    return result


def save_predictions(
    out_dir: Path,
    targets: list[str],
    predictions: dict[str, dict[tuple[str, ...], dict[str, Any]]],
) -> None:
    for target in targets:
        rows = [predictions[target][key] for key in sorted(predictions[target])]
        path = prediction_path(out_dir, target)
        temporary = path.with_suffix(".csv.tmp")
        write_csv(temporary, rows, OUTPUT_FIELDS)
        temporary.replace(path)


def load_target_inputs(args: argparse.Namespace, case_key_columns: tuple[str, ...]):
    target_inputs = {}
    canonical_keys: set[tuple[str, ...]] | None = None
    for target in args.targets:
        prediction_prefix = f"gdp_progression_forecasting_{target}"
        metrics_prefix = f"exp8_gdp_progression_forecasting_{target}"
        by_model, loaded_files, missing_files = load_predictions(
            args.predictions_root,
            prediction_prefix,
            args.models,
            case_key_columns,
        )
        if not by_model:
            raise FileNotFoundError(f"No usable progression prediction files for target={target}")
        priors, loaded_priors = load_model_priors(
            args.metrics_root,
            metrics_prefix,
            prediction_prefix,
            list(by_model),
        )
        if getattr(args, "require_oof_priors", False):
            validate_oof_inputs(args, target, by_model, loaded_files, missing_files, loaded_priors)
        common = set.intersection(*(set(rows) for rows in by_model.values()))
        if len(common) != args.expected_cases:
            counts = {model: len(rows) for model, rows in by_model.items()}
            raise ValueError(
                f"Expected {args.expected_cases} shared test cases for target={target}; "
                f"found {len(common)}; model_rows={counts}"
            )
        canonical_keys = common if canonical_keys is None else canonical_keys & common
        target_inputs[target] = {
            "prediction_prefix": prediction_prefix,
            "metrics_prefix": metrics_prefix,
            "by_model": by_model,
            "priors": priors,
            "loaded_files": loaded_files,
            "missing_files": missing_files,
            "loaded_priors": loaded_priors,
        }
    if canonical_keys is None or len(canonical_keys) != args.expected_cases:
        raise ValueError(
            f"Six endpoint cohorts do not align to {args.expected_cases} shared cases: "
            f"{len(canonical_keys or [])}"
        )
    return target_inputs, sorted(canonical_keys)


def validate_oof_inputs(args, target, by_model, loaded_files, missing_files, loaded_priors):
    if missing_files or set(by_model) != set(args.models):
        raise ValueError(f"Missing requested models for {target}: {missing_files}")
    if {r["model"] for r in loaded_priors} != set(args.models):
        raise ValueError(f"Missing OOF priors for {target}")
    priors_by_model = {r["model"]: read_csv(Path(r["path"])) for r in loaded_priors}
    for entry in loaded_files:
        model = entry["model"]
        raw_rows = read_csv(Path(entry["path"]))
        if len(raw_rows) != args.expected_cases or len(by_model[model]) != args.expected_cases:
            raise ValueError(f"Duplicate, missing or non-test predictions: {target}/{model}")
        prior_rows = priors_by_model[model]
        if len(prior_rows) != 1:
            raise ValueError(f"Expected one OOF aggregate: {target}/{model}")
        prior = prior_rows[0]
        if (prior.get("split") != "oof" or prior.get("prior_source") != "development_oof"
                or prior.get("progression_target") != target or prior.get("model_name") != model
                or prior.get("f1_average") != "binary"
                or float(prior.get("n", 0)) != args.expected_prior_cases or not prior.get("run_fingerprint")):
            raise ValueError(f"Refusing non-OOF, incomplete or mismatched priors: {target}/{model}")
        for metric in ("f1", "auroc", "balanced_accuracy", "ece", "fpr", "fnr", "sensitivity", "specificity"):
            number = float(prior[metric])
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f"Invalid OOF {metric}: {target}/{model}")
        for row in raw_rows:
            probability = float(row["y_prob"])
            if (row.get("progression_target") != target or row.get("model_name") != model
                    or row.get("run_fingerprint") != prior["run_fingerprint"]
                    or float(row["y_true"]) not in (0, 1) or float(row["y_pred"]) not in (0, 1)
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError(f"Invalid prediction/provenance: {target}/{model}")
    shared = set.intersection(*(set(rows) for rows in by_model.values()))
    for key in shared:
        if len({float(rows[key]["y_true"]) for rows in by_model.values()}) != 1:
            raise ValueError(f"Models disagree on ground truth: {target}/{key}")


def strict_resume_guard(args, target_inputs):
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    excluded = {"out_dir", "max_cases", "max_retries", "retry_sleep_sec", "request_sleep_sec", "max_consecutive_errors"}
    configuration = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items() if key not in excluded},
        "inputs": {target: {item["path"]: digest(item["path"])
                            for field in ("loaded_files", "loaded_priors") for item in inputs[field]}
                   for target, inputs in target_inputs.items()},
        "implementation": {name: digest(Path(__file__).with_name(name)) for name in (
            Path(__file__).name, "run_equi_agent_gdp_progression_live.py")},
    }
    path = args.out_dir / "input_fingerprint.json"
    if path.exists():
        if json.loads(path.read_text()) != configuration:
            raise ValueError("Agent inputs/configuration changed; use a new --out-dir, not mixed predictions")
    elif any(args.out_dir.iterdir()):
        raise ValueError("Strict OOF agent requires an empty output directory or its own matching provenance record")
    else:
        write_json(path, configuration)


def build_multitarget_messages(packets: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    first_messages = build_live_messages(next(iter(packets.values())))
    first_user = json.loads(first_messages[1]["content"])
    system = (
        first_messages[0]["content"]
        + "\n\nEvaluate all requested Harvard-GDP endpoints in one response. Treat each endpoint independently; "
        "never transfer a probability, vote, or conclusion from one endpoint to another. Return one JSON object "
        "with a predictions object keyed by the supplied endpoint names. The complete source-level numeric audit "
        "is already preserved in endpoint_evidence_packets, so do not repeat a source_assessments list in the "
        "response. For this multi-endpoint call, this compactness requirement supersedes the earlier request to "
        "repeat every source metric in the output trace. Return only the compact decision summary requested below; "
        "keep every rationale to one short sentence and keep the entire response below 3,000 tokens."
    )
    schema = {
        "final_probability": "float from 0 to 1",
        "final_prediction": "0 or 1",
        "confidence": "low, medium, or high",
        "primary_model": "model/source name or ensemble",
        "calibration_action": "sensitivity_shift, precision_shift, neutral, or escalate",
        "escalate_to_human": "boolean",
        "reasoning": "one short sentence",
        "agent_trace": {
            "progression_evidence_agent": {
                "evidence_summary": "one short sentence with the decisive probabilities or votes",
            },
            "equity_agent": {
                "reliability_concern": (
                    "false_negative_risk, false_positive_risk, calibration_risk, "
                    "minimal_risk, or unstable_priors"
                ),
                "threshold_policy": "sensitivity_shift, precision_shift, neutral, or escalate",
            },
            "orchestrator": "one short sentence explaining the final probability and label",
            "safety_agent": {
                "escalation_reasons": ["zero to two short reasons"],
            },
        },
    }
    compact_instructions = {
        key: value
        for key, value in first_user["instructions"].items()
        if key != "required_json_schema"
    }
    compact_instructions["temporal_specialist"] = (
        "Assess every supplied source internally, but return only one short evidence_summary naming the decisive "
        "probabilities or votes and the primary source. Do not return a per-source list."
    )
    compact_instructions["equity_agent"] = (
        "Apply the supplied source-level FP, FN, calibration, and stability evidence internally. Return only the "
        "dominant reliability_concern and threshold_policy; do not repeat the full reliability table."
    )
    user = {
        "instructions": {
            **compact_instructions,
            "required_json_schema": {
                "predictions": {target: schema for target in packets},
            },
        },
        "endpoint_evidence_packets": packets,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, sort_keys=True)},
    ]


def normalize_multitarget_response(raw: str, targets: list[str]) -> dict[str, dict[str, Any]]:
    parsed = json_from_text(raw)
    predictions = parsed.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("Agent response must contain a predictions object")
    if set(predictions) != set(targets):
        raise ValueError(
            f"Agent response targets differ: expected={sorted(targets)} found={sorted(predictions)}"
        )
    for target, response in predictions.items():
        if not isinstance(response, dict):
            raise ValueError(f"Agent response for {target} must be an object")
        probability = float(response["final_probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid final_probability for {target}: {probability}")
    return predictions


def allocate_usage(usage: dict[str, int], target_index: int, target_count: int) -> dict[str, int]:
    allocated = {}
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        total = int(usage.get(key, 0))
        quotient, remainder = divmod(total, target_count)
        allocated[key] = quotient + int(target_index < remainder)
    return allocated


def normalize_target_row(
    meta: dict[str, Any],
    arbitration: dict[str, Any],
    parsed: dict[str, Any],
    args: argparse.Namespace,
    usage: dict[str, int],
    provider: str,
) -> dict[str, Any]:
    action = normalize_action(parsed.get("calibration_action"))
    recommended_threshold = threshold_for_action(action)
    final_probability = clamp_probability(
        parsed.get("final_probability"),
        arbitration["reference_probability"],
        args.max_probability_adjustment,
    )
    if args.lock_reference_prediction and arbitration["reference_strategy"] != "weighted":
        final_prediction = int(arbitration["reference_prediction"])
        applied_threshold = float(arbitration["reference_threshold"])
        prediction_lock_active = True
    else:
        applied_threshold = recommended_threshold
        final_prediction = int(final_probability >= applied_threshold)
        prediction_lock_active = False
    close_call = abs(final_probability - applied_threshold) < 0.075
    split_vote = 0 < arbitration["positive_votes"] < arbitration["num_models"]
    severe_disagreement = arbitration["disagreement"] >= 0.25 or split_vote
    escalate = bool(parsed.get("escalate_to_human")) or close_call or severe_disagreement
    safety_decision = "ESCALATE_TO_HUMAN" if escalate else "ACCEPT"
    trace = parsed.get("agent_trace", {})
    if not isinstance(trace, dict):
        trace = {}
    return {
        **meta,
        "model_name": MODEL_NAME if not args.dry_run else MODEL_NAME.replace("_live", "_dry_run"),
        "y_prob": f"{final_probability:.6f}",
        "y_pred": final_prediction,
        "applied_threshold": f"{applied_threshold:.2f}",
        "recommended_threshold": f"{recommended_threshold:.2f}",
        "prediction_lock_active": prediction_lock_active,
        "split": "test",
        "positive_votes": arbitration["positive_votes"],
        "num_models": arbitration["num_models"],
        "mean_probability": f"{arbitration['mean_probability']:.6f}",
        "weighted_probability": f"{arbitration['weighted_probability']:.6f}",
        "disagreement": f"{arbitration['disagreement']:.6f}",
        "close_call": close_call,
        "safety_decision": safety_decision,
        "primary_model": parsed.get("primary_model", ""),
        "confidence": parsed.get("confidence", ""),
        "calibration_action": action,
        "escalate_to_human": escalate,
        "progression_evidence_summary": trace_field(trace, "progression_evidence_agent", "evidence_summary"),
        "equity_reliability_concern": trace_field(trace, "equity_agent", "reliability_concern"),
        "equity_threshold_policy": trace_field(trace, "equity_agent", "threshold_policy"),
        "orchestrator_rationale": trace_field(trace, "orchestrator"),
        "safety_reasons": trace_field(trace, "safety_agent", "escalation_reasons"),
        "agent_trace_json": json.dumps(trace, sort_keys=True),
        "llm_provider": provider,
        "llm_deployment": args.deployment,
        **usage,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_key_columns = parse_case_key_columns(args.case_key_columns)
    target_inputs, keys = load_target_inputs(args, case_key_columns)
    full_keys = set(keys)
    if args.require_oof_priors:
        strict_resume_guard(args, target_inputs)
    if args.max_cases > 0:
        keys = keys[: args.max_cases]

    existing = load_existing(args.out_dir, args.targets, case_key_columns)
    if args.require_oof_priors:
        for target, rows in existing.items():
            source = next(iter(target_inputs[target]["by_model"].values()))
            raw_path = prediction_path(args.out_dir, target)
            if raw_path.exists() and len(read_csv(raw_path)) != len(rows):
                raise ValueError(f"Duplicate cached agent predictions: {target}")
            for key, row in rows.items():
                if (key not in full_keys or row.get("split") != "test"
                        or float(row["y_true"]) != float(source[key]["y_true"])
                        or float(row["y_pred"]) not in (0, 1)
                        or not math.isfinite(float(row["y_prob"])) or not 0 <= float(row["y_prob"]) <= 1):
                    raise ValueError(f"Invalid cached agent prediction: {target}/{key}")
    completed = set.intersection(*(set(existing[target]) for target in args.targets))
    pending = [key for key in keys if key not in completed]

    provider = "dry_run"
    client = None
    if not args.dry_run and pending:
        provider, client = make_client(args.provider, args.api_version)

    attempts_path = args.out_dir / "attempts.jsonl"
    errors_path = args.out_dir / "errors.jsonl"
    prompt_path = args.out_dir / "prompt_snapshot.json"
    run_config_path = args.out_dir / "resolved_config.json"
    new_errors = 0
    consecutive_errors = 0

    sample_packets = {}
    sample_key = keys[0]
    for target in args.targets:
        inputs = target_inputs[target]
        rows = {model: values[sample_key] for model, values in inputs["by_model"].items()}
        arbitration = deterministic_arbitrate(rows, inputs["priors"], args.reference_strategy)
        packet = build_evidence_packet(case_metadata(rows), arbitration, case_key_columns)
        packet["progression_target"] = target
        sample_packets[target] = packet
    sample_messages = build_multitarget_messages(sample_packets)
    write_json(prompt_path, {"messages": sample_messages})
    write_json(
        run_config_path,
        {
            "targets": args.targets,
            "models_requested": args.models,
            "case_key_columns": list(case_key_columns),
            "reference_strategy": args.reference_strategy,
            "lock_reference_prediction": args.lock_reference_prediction,
            "deployment": args.deployment,
            "provider": args.provider,
            "dry_run": args.dry_run,
            "calls_required_for_full_cohort": len(full_keys),
            "requested_cases": len(keys),
            "predictions_per_call": len(args.targets),
            "target_sources": {
                target: {
                    "loaded_files": target_inputs[target]["loaded_files"],
                    "missing_files": target_inputs[target]["missing_files"],
                    "loaded_priors": target_inputs[target]["loaded_priors"],
                }
                for target in args.targets
            },
        },
    )

    for position, key in enumerate(pending, start=1):
        packets = {}
        metas = {}
        arbitrations = {}
        for target in args.targets:
            inputs = target_inputs[target]
            rows = {model: values[key] for model, values in inputs["by_model"].items()}
            meta = case_metadata(rows)
            arbitration = deterministic_arbitrate(rows, inputs["priors"], args.reference_strategy)
            packet = build_evidence_packet(meta, arbitration, case_key_columns)
            packet["progression_target"] = target
            packets[target] = packet
            metas[target] = meta
            arbitrations[target] = arbitration
        messages = build_multitarget_messages(packets)
        raw = ""
        last_error: Exception | None = None

        for attempt in range(args.max_retries + 1):
            try:
                if args.dry_run:
                    parsed_by_target = {
                        target: dry_run_response(arbitrations[target])[0]
                        for target in args.targets
                    }
                    raw = json.dumps({"predictions": parsed_by_target}, sort_keys=True)
                    usage = {
                        "prompt_tokens": sum(
                            estimate_tokens(message["content"], args.chars_per_token)
                            for message in messages
                        ),
                        "completion_tokens": estimate_tokens(raw, args.chars_per_token),
                    }
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                else:
                    response = call_llm(
                        client,
                        args.deployment,
                        messages,
                        args.temperature,
                        args.max_output_tokens,
                    )
                    raw = response_text(response)
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    if finish_reason == "length":
                        raise RuntimeError(
                            "LLM response reached the completion-token limit before producing complete JSON"
                        )
                    parsed_by_target = normalize_multitarget_response(raw, args.targets)
                    usage = usage_dict(response)

                case_outputs = {}
                for target_index, target in enumerate(args.targets):
                    case_outputs[target] = normalize_target_row(
                        metas[target],
                        arbitrations[target],
                        parsed_by_target[target],
                        args,
                        allocate_usage(usage, target_index, len(args.targets)),
                        provider,
                    )
                for target in args.targets:
                    existing[target][key] = case_outputs[target]
                save_predictions(args.out_dir, args.targets, existing)
                append_jsonl(
                    attempts_path,
                    {
                        "timestamp_utc": utc_now(),
                        "case_key": key,
                        "attempt": attempt,
                        "messages": messages,
                        "evidence_packets": packets,
                        "raw_response": raw,
                        "parsed_response": {"predictions": parsed_by_target},
                        "usage": usage,
                    },
                )
                print(
                    f"done {position}/{len(pending)} case={key[0]} "
                    + " ".join(
                        f"{target}={existing[target][key]['y_pred']}" for target in args.targets
                    ),
                    flush=True,
                )
                last_error = None
                consecutive_errors = 0
                break
            except Exception as exc:
                last_error = exc
                append_jsonl(
                    attempts_path,
                    {
                        "timestamp_utc": utc_now(),
                        "case_key": key,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_response": raw,
                        "messages": messages,
                        "evidence_packets": packets,
                    },
                )
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep_sec * (attempt + 1))
        if last_error is not None:
            new_errors += 1
            consecutive_errors += 1
            append_jsonl(
                errors_path,
                {
                    "timestamp_utc": utc_now(),
                    "case_key": key,
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                },
            )
            print(f"error {position}/{len(pending)} case={key[0]}: {last_error}", flush=True)
            if args.max_consecutive_errors > 0 and consecutive_errors >= args.max_consecutive_errors:
                print("Stopped after consecutive failures; completed cases are preserved for resume", flush=True)
                break
        if args.request_sleep_sec > 0:
            time.sleep(args.request_sleep_sec)

    completed = set.intersection(*(set(existing[target]) for target in args.targets))
    token_rows = list(existing[args.targets[0]].values())
    summary = {
        "dry_run": args.dry_run,
        "targets": args.targets,
        "models_requested": args.models,
        "case_key_columns": list(case_key_columns),
        "completed_calls": len(completed),
        "missing_calls": len(full_keys - completed),
        "predictions_per_completed_call": len(args.targets),
        "complete_locked_cohort": completed == full_keys,
        "complete_requested_cohort": set(keys).issubset(completed),
        "complete_live_cohort": not args.dry_run and completed == full_keys,
        "new_errors": new_errors,
        "error_types": dict(
            Counter(
                json.loads(line).get("error_type", "")
                for line in errors_path.read_text(encoding="utf-8").splitlines()
            )
        )
        if errors_path.is_file()
        else {},
        "prompt_tokens_allocated_to_first_target": sum(
            int(float(row.get("prompt_tokens", 0) or 0)) for row in token_rows
        ),
        "outputs": {
            "predictions_by_target": {
                target: str(prediction_path(args.out_dir, target)) for target in args.targets
            },
            "attempts": str(attempts_path),
            "errors": str(errors_path),
            "prompt_snapshot": str(prompt_path),
            "resolved_config": str(run_config_path),
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["complete_requested_cohort"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
