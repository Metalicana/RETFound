from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llm_api_config import (
    call_claude_messages,
    config_for_model,
    is_non_retryable_api_error,
    require_shared_api_key,
)
from run_gdp_progression_llm_baseline import (
    EXPECTED_TEST_POSITIVES,
    GDP_TD_COLUMNS,
    OUTPUT_FIELDS,
    PROGRESSION_TARGET_DESCRIPTIONS,
    append_jsonl,
    build_user_prompt as build_single_target_user_prompt,
    case_key,
    compute_metrics,
    default_deployment,
    json_sha256,
    load_rnflt,
    normalize_response as normalize_single_target_response,
    parse_json_object,
    prediction_row,
    read_csv,
    render_rnflt_png,
    repo_root,
    round_numbers,
    td_evidence,
    utc_now,
    validate_and_select_manifest,
    write_csv,
    write_json,
)


TARGETS = list(PROGRESSION_TARGET_DESCRIPTIONS)

SYSTEM_PROMPT = (
    "You are a standalone ophthalmic glaucoma-progression evaluator. Predict all six Harvard-GDP binary "
    "progression endpoints together using only the supplied RNFL thickness map and the 52 visual-field "
    "total-deviation values. This is an independent multi-task LLM baseline: you are not receiving another "
    "model's probability, a known diagnosis, any study labels, or any progression labels. Do not invent "
    "follow-up examinations, longitudinal slopes, p-values, intraocular pressure, symptoms, treatment history, "
    "or demographic risk. Describe only patterns supported by the supplied structural and functional evidence. "
    "Return one JSON object with exactly two top-level fields: predictions and confidence. confidence must be "
    "low, moderate, or high. predictions must contain exactly these six keys: md, vfi, td_pointwise, md_fast, "
    "md_fast_no_p_cut, and td_pointwise_no_p_cut. Each prediction must contain exactly: progression_detected "
    "(integer 0 or 1), progression_probability (number from 0 to 1), and reasoning (brief evidence-based "
    "explanation specific to that endpoint). For every endpoint, progression_detected must equal 1 when its "
    "probability is at least 0.5 and 0 otherwise. You must choose 0 or 1 for every endpoint even when uncertain."
)


def build_user_prompt(
    rnflt_stats: dict[str, Any],
    td_values: dict[str, float],
    td_summary: dict[str, Any],
) -> str:
    endpoint_lines = "\n".join(
        f"- {target}: {description}"
        for target, description in PROGRESSION_TARGET_DESCRIPTIONS.items()
    )
    evidence = build_single_target_user_prompt(
        "td_pointwise_no_p_cut",
        rnflt_stats,
        td_values,
        td_summary,
    )
    evidence = evidence.replace(
        "Study endpoint: pointwise total-deviation-based progression without the p-value cutoff "
        "(Harvard-GDP identifier: td_pointwise_no_p_cut).\n",
        "",
        1,
    )
    return (
        "Predict all six binary Harvard-GDP progression endpoints in one response:\n"
        f"{endpoint_lines}\n"
        f"{evidence}"
    )


def normalize_response(raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    predictions = parsed.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("LLM response field 'predictions' must be a JSON object")
    if set(predictions) != set(TARGETS):
        raise ValueError(
            f"LLM response must contain exactly the six targets; found {sorted(predictions)}"
        )
    confidence = str(parsed.get("confidence", "")).strip().lower()
    if confidence not in {"low", "moderate", "high"}:
        raise ValueError(f"confidence must be low, moderate, or high, found {confidence!r}")

    normalized = {}
    for target in TARGETS:
        target_payload = predictions[target]
        if not isinstance(target_payload, dict):
            raise ValueError(f"Prediction for {target} must be a JSON object")
        single_raw = json.dumps(
            {
                **target_payload,
                "confidence": confidence,
            }
        )
        single = normalize_single_target_response(single_raw)
        normalized[target] = single
    return {
        "confidence": confidence,
        "predictions": normalized,
        "parsed_response": parsed,
    }


class AzureMultiTargetEvaluator:
    def __init__(self, model_name: str, deployment: str, response_format: str):
        from openai import AzureOpenAI

        self.model_name = model_name
        self.deployment = deployment
        self.response_format = response_format
        config = config_for_model(model_name, deployment)
        self.client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=require_shared_api_key(),
            api_version=config.api_version,
        )

    def analyze(self, user_prompt: str, image_png: bytes) -> tuple[str, dict[str, int]]:
        request: dict[str, Any] = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(image_png).decode("ascii")
                            },
                        },
                    ],
                },
            ],
        }
        if self.model_name == "gpt-5.1":
            request["temperature"] = 0.2
        if self.response_format == "json_object":
            request["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request)
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return raw, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(
                getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens
            ),
        }


class ClaudeMultiTargetEvaluator:
    def __init__(self, deployment: str):
        self.deployment = deployment
        self.config = config_for_model("claude-haiku-4.5", deployment)

    def analyze(self, user_prompt: str, image_png: bytes) -> tuple[str, dict[str, int]]:
        response = call_claude_messages(
            self.config,
            {
                "model": self.deployment,
                "max_tokens": 1536,
                "temperature": 0.2,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(image_png).decode("ascii"),
                                },
                            },
                        ],
                    }
                ],
            },
        )
        raw = "\n".join(
            str(block.get("text", ""))
            for block in response.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        return raw, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


def make_evaluator(args: argparse.Namespace):
    if args.model == "claude-haiku-4.5":
        return ClaudeMultiTargetEvaluator(args.deployment)
    return AzureMultiTargetEvaluator(args.model, args.deployment, args.response_format)


def manifest_args(args: argparse.Namespace, target: str) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=args.manifests_root / f"gdp_progression_forecasting_{target}.csv",
        split=args.split,
        progression_target=target,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        allow_cohort_mismatch=args.allow_cohort_mismatch,
        expected_cases=args.expected_cases,
        expected_positives=EXPECTED_TEST_POSITIVES[target],
    )


def evidence_signature(row: dict[str, str]) -> dict[str, Any]:
    return {
        "patient_id": row.get("patient_id", ""),
        "eye_id": row.get("eye_id", ""),
        "visit_id": row.get("visit_id", ""),
        "image_id": row.get("image_id", ""),
        "rnflt_path": row.get("resolved_rnflt_path", ""),
        "rnflt_key": row.get("rnflt_key", ""),
        "td_values": [row.get(column, "") for column in GDP_TD_COLUMNS],
    }


def load_multitarget_cohort(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_target: dict[str, dict[str, dict[str, str]]] = {}
    audits = {}
    for target in TARGETS:
        target_rows, audit = validate_and_select_manifest(manifest_args(args, target))
        rows_by_target[target] = {case_key(row): row for row in target_rows}
        audits[target] = audit

    reference_keys = set(rows_by_target[TARGETS[0]])
    for target in TARGETS[1:]:
        target_keys = set(rows_by_target[target])
        if target_keys != reference_keys:
            raise ValueError(
                f"Target manifests do not contain the same locked cases for {target}: "
                f"missing={sorted(reference_keys - target_keys)[:5]}, "
                f"extra={sorted(target_keys - reference_keys)[:5]}"
            )

    cases = []
    for key in sorted(reference_keys):
        target_rows = {target: rows_by_target[target][key] for target in TARGETS}
        reference_signature = evidence_signature(target_rows[TARGETS[0]])
        for target in TARGETS[1:]:
            if evidence_signature(target_rows[target]) != reference_signature:
                raise ValueError(f"Evidence mismatch across target manifests for case {key}, target={target}")
        cases.append(
            {
                "case_key": key,
                "evidence_row": target_rows[TARGETS[0]],
                "target_rows": target_rows,
            }
        )

    return cases, {
        "cases": len(cases),
        "targets": TARGETS,
        "expected_test_positives": EXPECTED_TEST_POSITIVES,
        "manifest_audits": audits,
        "calls_per_model": len(cases),
        "predictions_per_call": len(TARGETS),
        "input_modalities": ["RNFLT thickness map", "52 visual-field total-deviation values"],
    }


def prediction_path(out_dir: Path, target: str) -> Path:
    return out_dir / f"predictions_{target}.csv"


def load_existing_predictions(out_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    output = {}
    for target in TARGETS:
        path = prediction_path(out_dir, target)
        rows = read_csv(path) if path.exists() else []
        indexed = {case_key(row): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate prediction rows in {path}")
        output[target] = indexed
    return output


def completed_case_keys(
    predictions: dict[str, dict[str, dict[str, str]]],
) -> set[str]:
    key_sets = [set(predictions[target]) for target in TARGETS]
    return set.intersection(*key_sets) if key_sets else set()


def save_predictions(
    out_dir: Path,
    predictions: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for target in TARGETS:
        rows = [predictions[target][key] for key in sorted(predictions[target])]
        write_csv(prediction_path(out_dir, target), rows, OUTPUT_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict all six Harvard-GDP progression endpoints in one LLM call per patient."
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "manifests",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["gpt-5.1", "gpt-5.4", "gpt-5.6-luna", "claude-haiku-4.5"],
        required=True,
    )
    parser.add_argument("--deployment", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--allow-cohort-mismatch", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=5.0)
    parser.add_argument("--request-sleep-sec", type=float, default=0.0)
    parser.add_argument("--response-format", choices=["json_object", "none"], default="json_object")
    parser.add_argument(
        "--path-prefix-from",
        default=os.getenv("GDP_PATH_PREFIX_FROM", "/Users/metalicana/projects_spring_2026/RETFound"),
    )
    parser.add_argument(
        "--path-prefix-to",
        default=os.getenv("GDP_PATH_PREFIX_TO", str(repo_root())),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.deployment is None:
        args.deployment = default_deployment(args.model)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = args.out_dir / "attempts.jsonl"
    errors_path = args.out_dir / "errors.jsonl"
    config_path = args.out_dir / "run_config.json"
    prompt_path = args.out_dir / "prompt_snapshot.json"
    summary_path = args.out_dir / "summary.json"

    cases, cohort_audit = load_multitarget_cohort(args)
    first_row = cases[0]["evidence_row"]
    first_rnflt, first_rnflt_stats = load_rnflt(first_row)
    first_td_values, first_td_summary = td_evidence(first_row)
    first_user_prompt = build_user_prompt(first_rnflt_stats, first_td_values, first_td_summary)

    prompt_snapshot = {
        "model_name": args.model,
        "deployment": args.deployment,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": json_sha256(SYSTEM_PROMPT),
        "first_case_key": cases[0]["case_key"],
        "first_case_user_prompt": first_user_prompt,
        "image_attachment": "Color-rendered RNFLT map; image bytes are not duplicated in this JSON file.",
        "cohort_audit": cohort_audit,
    }
    write_json(prompt_path, prompt_snapshot)

    core_config = {
        "model_name": args.model,
        "deployment": args.deployment,
        "manifest_sha256_by_target": {
            target: cohort_audit["manifest_audits"][target]["manifest_sha256"]
            for target in TARGETS
        },
        "split": args.split,
        "targets": TARGETS,
        "system_prompt_sha256": prompt_snapshot["system_prompt_sha256"],
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("core_config") != core_config:
            raise RuntimeError(
                f"Existing output directory has a different configuration: {config_path}. "
                "Use a new --out-dir."
            )
    write_json(
        config_path,
        {
            "created_or_resumed_utc": utc_now(),
            "core_config": core_config,
            "cohort_audit": cohort_audit,
            "output_fields": OUTPUT_FIELDS,
        },
    )

    if args.dry_run:
        summary = {
            "dry_run": True,
            "model_name": args.model,
            "deployment": args.deployment,
            "cohort_audit": cohort_audit,
            "prompt_snapshot": str(prompt_path),
            "rnflt_first_case_shape": list(first_rnflt.shape),
        }
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    predictions = load_existing_predictions(args.out_dir)
    completed = completed_case_keys(predictions)
    pending_cases = [case for case in cases if case["case_key"] not in completed]
    if args.max_cases > 0:
        pending_cases = pending_cases[: args.max_cases]
    evaluator = make_evaluator(args)

    new_error_count = 0
    for index, case in enumerate(pending_cases, start=1):
        key = case["case_key"]
        row = case["evidence_row"]
        rnflt, rnflt_stats = load_rnflt(row)
        td_values, td_summary = td_evidence(row)
        user_prompt = build_user_prompt(rnflt_stats, td_values, td_summary)
        image_png = render_rnflt_png(rnflt)

        last_error: Exception | None = None
        for attempt in range(args.max_retries + 1):
            started = utc_now()
            try:
                raw, usage = evaluator.analyze(user_prompt, image_png)
                normalized = normalize_response(raw)
                for target in TARGETS:
                    target_normalized = normalized["predictions"][target]
                    predictions[target][key] = prediction_row(
                        case["target_rows"][target],
                        args.model,
                        target_normalized,
                        usage,
                    )
                save_predictions(args.out_dir, predictions)
                append_jsonl(
                    attempts_path,
                    {
                        "timestamp_utc": started,
                        "case_key": key,
                        "attempt": attempt,
                        "model_name": args.model,
                        "deployment": args.deployment,
                        "input": {
                            "modalities": cohort_audit["input_modalities"],
                            "rnflt_path": row["resolved_rnflt_path"],
                            "rnflt_stats": round_numbers(rnflt_stats),
                            "td_values": round_numbers(td_values),
                            "td_summary": round_numbers(td_summary),
                            "system_prompt": SYSTEM_PROMPT,
                            "user_prompt": user_prompt,
                        },
                        "raw_response": raw,
                        "parsed_response": normalized["parsed_response"],
                        "usage": usage,
                    },
                )
                labels = ",".join(
                    f"{target}={normalized['predictions'][target]['y_pred']}"
                    for target in TARGETS
                )
                print(
                    f"done {index}/{len(pending_cases)} case={row['image_id']} {labels}"
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                append_jsonl(
                    attempts_path,
                    {
                        "timestamp_utc": started,
                        "case_key": key,
                        "attempt": attempt,
                        "model_name": args.model,
                        "deployment": args.deployment,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "input": {
                            "modalities": cohort_audit["input_modalities"],
                            "rnflt_path": row["resolved_rnflt_path"],
                            "rnflt_stats": round_numbers(rnflt_stats),
                            "td_values": round_numbers(td_values),
                            "td_summary": round_numbers(td_summary),
                            "system_prompt": SYSTEM_PROMPT,
                            "user_prompt": user_prompt,
                        },
                    },
                )
                if is_non_retryable_api_error(exc):
                    raise RuntimeError(
                        f"Non-retryable API failure for {args.model} deployment "
                        f"{args.deployment}; aborting before the next case: {exc}"
                    ) from exc
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep_sec * (attempt + 1))

        if last_error is not None:
            new_error_count += 1
            append_jsonl(
                errors_path,
                {
                    "timestamp_utc": utc_now(),
                    "case_key": key,
                    "image_id": row.get("image_id", ""),
                    "model_name": args.model,
                    "deployment": args.deployment,
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                },
            )
            print(f"error {index}/{len(pending_cases)} case={row['image_id']}: {last_error}")
        if args.request_sleep_sec > 0:
            time.sleep(args.request_sleep_sec)

    completed = completed_case_keys(predictions)
    metrics_by_target = {
        target: compute_metrics(list(predictions[target].values()))
        for target in TARGETS
    }
    usage_rows = list(predictions[TARGETS[0]].values())
    total_prompt_tokens = sum(int(float(row.get("prompt_tokens", 0) or 0)) for row in usage_rows)
    total_completion_tokens = sum(
        int(float(row.get("completion_tokens", 0) or 0)) for row in usage_rows
    )
    summary = {
        "dry_run": False,
        "model_name": args.model,
        "deployment": args.deployment,
        "cohort_audit": cohort_audit,
        "completed_calls": len(completed),
        "missing_calls": len(cases) - len(completed),
        "predictions_per_completed_call": len(TARGETS),
        "total_endpoint_predictions": len(completed) * len(TARGETS),
        "complete_locked_cohort": len(completed) == len(cases),
        "new_errors": new_error_count,
        "metrics_valid_subset_by_target": metrics_by_target,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "outputs": {
            "predictions_by_target": {
                target: str(prediction_path(args.out_dir, target))
                for target in TARGETS
            },
            "attempts": str(attempts_path),
            "errors": str(errors_path),
            "prompt_snapshot": str(prompt_path),
            "run_config": str(config_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
