from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_api_config import (
    call_claude_messages,
    config_for_model,
    is_non_retryable_api_error,
    require_shared_api_key,
)

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


GDP_TD_COLUMNS = [
    *[f"td{index}" for index in range(1, 25)],
    *[f"td{index}" for index in range(26, 34)],
    *[f"td{index}" for index in range(35, 55)],
]

PROGRESSION_LABEL_COLUMNS = {
    "label_raw",
    "y_true",
    "progression_md",
    "progression_vfi",
    "progression_td_pointwise",
    "progression_md_fast",
    "progression_md_fast_no_p_cut",
    "progression_td_pointwise_no_p_cut",
}

MODEL_CHOICES = ["gpt-5.1", "gpt-5.4", "gpt-5.6-luna", "claude-haiku-4.5"]

PROGRESSION_TARGET_DESCRIPTIONS = {
    "md": "mean-deviation-based progression",
    "vfi": "Visual Field Index-based progression",
    "td_pointwise": "pointwise total-deviation-based progression",
    "md_fast": "rapid mean-deviation-based progression",
    "md_fast_no_p_cut": "rapid mean-deviation-based progression without the p-value cutoff",
    "td_pointwise_no_p_cut": "pointwise total-deviation-based progression without the p-value cutoff",
}

EXPECTED_TEST_POSITIVES = {
    "md": 18,
    "vfi": 19,
    "td_pointwise": 18,
    "md_fast": 4,
    "md_fast_no_p_cut": 6,
    "td_pointwise_no_p_cut": 60,
}

SYSTEM_PROMPT = (
    "You are a standalone ophthalmic glaucoma-progression evaluator. Predict the binary study endpoint using "
    "only the supplied RNFL thickness map and the 52 visual-field total-deviation values. This is an "
    "independent LLM baseline: you are not receiving another model's probability, a known diagnosis, the "
    "study label, or a progression label. Do not invent follow-up examinations, longitudinal slopes, p-values, "
    "intraocular pressure, symptoms, treatment history, or demographic risk. Describe only patterns supported "
    "by the supplied structural and functional evidence. Return one JSON object with exactly these fields: "
    "progression_detected (integer 0 or 1), progression_probability (number from 0 to 1), confidence "
    "(low, moderate, or high), and reasoning (brief evidence-based explanation). The binary decision must equal "
    "1 when progression_probability is at least 0.5 and 0 otherwise. You must choose 0 or 1 even when uncertain."
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
    "metadata_missing_flag",
    "progression_target",
    "confidence",
    "reasoning",
    "rnflt_path",
    "rnflt_key",
    "td_point_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def case_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(column, ""))
        for column in ["patient_id", "eye_id", "visit_id", "image_id", "task"]
    )


def resolve_data_path(raw_path: str, prefix_from: str, prefix_to: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.exists():
        return path

    if prefix_from and prefix_to and raw_path.startswith(prefix_from):
        candidate = Path(prefix_to) / raw_path[len(prefix_from) :].lstrip("/")
        if candidate.exists():
            return candidate
        path = candidate

    marker = "/RETFound/"
    if marker in raw_path:
        candidate = repo_root() / raw_path.split(marker, 1)[1]
        if candidate.exists():
            return candidate
        path = candidate
    return path


def validate_and_select_manifest(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not args.manifest.exists():
        raise FileNotFoundError(f"GDP progression manifest does not exist: {args.manifest}")
    rows = read_csv(args.manifest)
    if not rows:
        raise ValueError(f"GDP progression manifest is empty: {args.manifest}")

    required = {
        "patient_id",
        "eye_id",
        "visit_id",
        "image_id",
        "dataset",
        "task",
        "split",
        "y_true",
        "progression_target",
        "rnflt_path",
        "rnflt_key",
        "race",
        "ethnicity",
        "sex_gender",
        "age",
        "age_group",
        "metadata_missing_flag",
        *GDP_TD_COLUMNS,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"GDP progression manifest is missing columns: {missing}")

    selected = [
        row
        for row in rows
        if str(row.get("split", "")).strip().lower() == args.split
        and str(row.get("progression_target", "")).strip() == args.progression_target
    ]
    if not selected:
        raise ValueError(
            f"No rows found for split={args.split!r}, progression_target={args.progression_target!r}"
        )

    labels = []
    duplicate_keys = []
    seen = set()
    missing_td_cases = []
    missing_rnflt_cases = []
    for row in selected:
        label = optional_float(row.get("y_true"))
        if label not in {0.0, 1.0}:
            raise ValueError(f"Non-binary y_true for case {case_key(row)}: {row.get('y_true')!r}")
        labels.append(int(label))

        key = case_key(row)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)

        missing_td = [column for column in GDP_TD_COLUMNS if optional_float(row.get(column)) is None]
        if missing_td:
            missing_td_cases.append({"case_key": key, "missing_columns": missing_td})

        rnflt_path = resolve_data_path(row["rnflt_path"], args.path_prefix_from, args.path_prefix_to)
        row["resolved_rnflt_path"] = str(rnflt_path)
        if not rnflt_path.is_file():
            missing_rnflt_cases.append({"case_key": key, "path": str(rnflt_path)})

    if duplicate_keys:
        raise ValueError(f"Duplicate GDP case keys: {duplicate_keys[:10]}")
    if missing_td_cases:
        raise ValueError(f"Cases with incomplete 52-point TDS input: {missing_td_cases[:3]}")
    if missing_rnflt_cases:
        raise FileNotFoundError(f"Cases with missing RNFLT files: {missing_rnflt_cases[:3]}")

    positive_count = sum(labels)
    if not args.allow_cohort_mismatch:
        if len(selected) != args.expected_cases or positive_count != args.expected_positives:
            raise ValueError(
                "Locked cohort mismatch: "
                f"found cases={len(selected)}, positives={positive_count}; "
                f"expected cases={args.expected_cases}, positives={args.expected_positives}. "
                "Use --allow-cohort-mismatch only for an intentional non-manuscript run."
            )

    selected.sort(key=case_key)
    audit = {
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "split": args.split,
        "progression_target": args.progression_target,
        "cases": len(selected),
        "positives": positive_count,
        "negatives": len(selected) - positive_count,
        "td_columns": GDP_TD_COLUMNS,
        "td_point_count": len(GDP_TD_COLUMNS),
        "input_modalities": ["RNFLT thickness map", "52 visual-field total-deviation values"],
        "prompt_excluded_columns": sorted(PROGRESSION_LABEL_COLUMNS),
    }
    return selected, audit


def load_rnflt(row: dict[str, str]):
    import numpy as np

    path = Path(row["resolved_rnflt_path"])
    key = str(row.get("rnflt_key") or "rnflt")
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"RNFLT key {key!r} not found in {path}; keys={data.files}")
        array = np.asarray(data[key], dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D RNFLT map, found shape={array.shape}: {path}")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"RNFLT map has no finite values: {path}")
    stats = {
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }
    return array, stats


def render_rnflt_png(rnflt) -> bytes:
    import matplotlib.pyplot as plt
    import numpy as np

    finite = rnflt[np.isfinite(rnflt)]
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = low + 1.0
    figure, axis = plt.subplots(figsize=(6, 6), dpi=140)
    plot = axis.imshow(rnflt, cmap="turbo", vmin=low, vmax=high)
    axis.set_title("RNFL Thickness Map")
    axis.axis("off")
    figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04, label="Thickness (source units)")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def td_evidence(row: dict[str, str]) -> tuple[dict[str, float], dict[str, Any]]:
    import numpy as np

    values = {column: float(row[column]) for column in GDP_TD_COLUMNS}
    vector = np.asarray(list(values.values()), dtype=np.float64)
    summary = {
        "point_count": int(vector.size),
        "minimum_db": float(np.min(vector)),
        "maximum_db": float(np.max(vector)),
        "mean_db": float(np.mean(vector)),
        "median_db": float(np.median(vector)),
        "points_at_or_below_minus_2_db": int(np.sum(vector <= -2)),
        "points_at_or_below_minus_5_db": int(np.sum(vector <= -5)),
        "points_at_or_below_minus_10_db": int(np.sum(vector <= -10)),
    }
    return values, summary


def round_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_numbers(item) for item in value]
    return value


def build_user_prompt(
    progression_target: str,
    rnflt_stats: dict[str, Any],
    td_values: dict[str, float],
    td_summary: dict[str, Any],
) -> str:
    target_description = PROGRESSION_TARGET_DESCRIPTIONS[progression_target]
    return (
        f"Study endpoint: {target_description} (Harvard-GDP identifier: {progression_target}).\n"
        "Attached image: one color-rendered RNFL thickness map.\n"
        f"RNFLT summary (source units): {json.dumps(round_numbers(rnflt_stats), sort_keys=True)}\n"
        f"Visual-field total-deviation summary: {json.dumps(round_numbers(td_summary), sort_keys=True)}\n"
        f"Visual-field total-deviation values in dB: {json.dumps(round_numbers(td_values))}\n"
        "No demographics, longitudinal follow-up measurements, known labels, or model predictions are supplied. "
        "Predict the binary endpoint now."
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must decode to a JSON object")
    return parsed


def normalize_response(raw: str) -> dict[str, Any]:
    parsed = parse_json_object(raw)
    prediction = int(parsed["progression_detected"])
    probability = float(parsed["progression_probability"])
    if prediction not in {0, 1}:
        raise ValueError(f"progression_detected must be 0 or 1, found {prediction!r}")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"progression_probability must be in [0, 1], found {probability!r}")
    threshold_prediction = int(probability >= 0.5)
    if prediction != threshold_prediction:
        raise ValueError(
            "Inconsistent LLM response: "
            f"progression_detected={prediction}, progression_probability={probability}"
        )
    confidence = str(parsed.get("confidence", "")).strip().lower()
    if confidence not in {"low", "moderate", "high"}:
        raise ValueError(f"confidence must be low, moderate, or high, found {confidence!r}")
    return {
        "y_pred": prediction,
        "y_prob": probability,
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "parsed_response": parsed,
    }


class AzureProgressionEvaluator:
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
        content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
                },
            },
        ]
        request: dict[str, Any] = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
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
            "total_tokens": int(getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens),
        }


class ClaudeProgressionEvaluator:
    def __init__(self, deployment: str):
        self.deployment = deployment
        self.config = config_for_model("claude-haiku-4.5", deployment)

    def analyze(self, user_prompt: str, image_png: bytes) -> tuple[str, dict[str, int]]:
        response = call_claude_messages(
            self.config,
            {
                "model": self.deployment,
                "max_tokens": 512,
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
    deployment = args.deployment or args.model
    if args.model == "claude-haiku-4.5":
        return ClaudeProgressionEvaluator(deployment)
    return AzureProgressionEvaluator(args.model, deployment, args.response_format)


def prediction_row(
    manifest_row: dict[str, str],
    model_name: str,
    normalized: dict[str, Any],
    usage: dict[str, int],
) -> dict[str, Any]:
    return {
        **{column: manifest_row.get(column, "") for column in OUTPUT_FIELDS},
        "model_name": model_name,
        "y_true": int(float(manifest_row["y_true"])),
        "y_prob": normalized["y_prob"],
        "y_pred": normalized["y_pred"],
        "progression_target": manifest_row.get("progression_target", ""),
        "confidence": normalized["confidence"],
        "reasoning": normalized["reasoning"],
        "rnflt_path": manifest_row["resolved_rnflt_path"],
        "rnflt_key": manifest_row.get("rnflt_key") or "rnflt",
        "td_point_count": len(GDP_TD_COLUMNS),
        **usage,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tn = sum(1 for row in rows if int(row["y_true"]) == 0 and int(row["y_pred"]) == 0)
    fp = sum(1 for row in rows if int(row["y_true"]) == 0 and int(row["y_pred"]) == 1)
    fn = sum(1 for row in rows if int(row["y_true"]) == 1 and int(row["y_pred"]) == 0)
    tp = sum(1 for row in rows if int(row["y_true"]) == 1 and int(row["y_pred"]) == 1)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": len(rows),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def default_deployment(model: str) -> str:
    if model == "claude-haiku-4.5":
        return os.getenv(
            "CLAUDE_HAIKU45_DEPLOYMENT",
            os.getenv("ANTHROPIC_DEPLOYMENT", "claude-haiku-4-5"),
        )
    deployment_variables = {
        "gpt-5.1": "GPT51_DEPLOYMENT",
        "gpt-5.4": "GPT54_DEPLOYMENT",
        "gpt-5.6-luna": "GPT56_DEPLOYMENT",
    }
    return os.getenv(deployment_variables[model], model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone LLM baseline on one locked Harvard-GDP progression endpoint."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--deployment", default=None)
    parser.add_argument(
        "--progression-target",
        choices=sorted(PROGRESSION_TARGET_DESCRIPTIONS),
        default="td_pointwise_no_p_cut",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--expected-positives", type=int, default=None)
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
    if args.manifest is None:
        args.manifest = (
            repo_root()
            / "equi-agent"
            / "outputs"
            / "manifests"
            / f"gdp_progression_forecasting_{args.progression_target}.csv"
        )
    if args.expected_positives is None:
        args.expected_positives = EXPECTED_TEST_POSITIVES[args.progression_target]
    if args.deployment is None:
        args.deployment = default_deployment(args.model)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / "predictions.csv"
    attempts_path = args.out_dir / "attempts.jsonl"
    errors_path = args.out_dir / "errors.jsonl"
    config_path = args.out_dir / "run_config.json"
    prompt_path = args.out_dir / "prompt_snapshot.json"
    summary_path = args.out_dir / "summary.json"

    manifest_rows, cohort_audit = validate_and_select_manifest(args)
    first_rnflt, first_rnflt_stats = load_rnflt(manifest_rows[0])
    first_td_values, first_td_summary = td_evidence(manifest_rows[0])
    first_user_prompt = build_user_prompt(
        args.progression_target,
        first_rnflt_stats,
        first_td_values,
        first_td_summary,
    )
    prompt_snapshot = {
        "model_name": args.model,
        "deployment": args.deployment,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": json_sha256(SYSTEM_PROMPT),
        "first_case_key": case_key(manifest_rows[0]),
        "first_case_user_prompt": first_user_prompt,
        "image_attachment": "Color-rendered RNFLT map; image bytes are not duplicated in this JSON file.",
        "cohort_audit": cohort_audit,
    }
    write_json(prompt_path, prompt_snapshot)

    core_config = {
        "model_name": args.model,
        "deployment": args.deployment,
        "manifest": str(args.manifest),
        "manifest_sha256": cohort_audit["manifest_sha256"],
        "split": args.split,
        "progression_target": args.progression_target,
        "system_prompt_sha256": prompt_snapshot["system_prompt_sha256"],
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        previous_core = previous.get("core_config", {})
        if previous_core != core_config:
            raise RuntimeError(
                f"Existing output directory has a different run configuration: {config_path}. "
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
            "cohort_audit": cohort_audit,
            "model_name": args.model,
            "deployment": args.deployment,
            "prompt_snapshot": str(prompt_path),
            "rnflt_first_case_shape": list(first_rnflt.shape),
        }
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    existing = read_csv(predictions_path) if predictions_path.exists() else []
    by_key = {case_key(row): row for row in existing}
    if len(by_key) != len(existing):
        raise ValueError(f"Duplicate prediction rows in resume file: {predictions_path}")
    evaluator = make_evaluator(args)

    pending_rows = [row for row in manifest_rows if case_key(row) not in by_key]
    if args.max_cases > 0:
        pending_rows = pending_rows[: args.max_cases]

    new_error_count = 0
    for index, row in enumerate(pending_rows, start=1):
        key = case_key(row)
        rnflt, rnflt_stats = load_rnflt(row)
        td_values, td_summary = td_evidence(row)
        user_prompt = build_user_prompt(
            args.progression_target,
            rnflt_stats,
            td_values,
            td_summary,
        )
        image_png = render_rnflt_png(rnflt)

        last_error: Exception | None = None
        for attempt in range(args.max_retries + 1):
            started = utc_now()
            try:
                raw, usage = evaluator.analyze(user_prompt, image_png)
                normalized = normalize_response(raw)
                result = prediction_row(row, args.model, normalized, usage)
                by_key[key] = result
                ordered = [by_key[item] for item in sorted(by_key)]
                write_csv(predictions_path, ordered, OUTPUT_FIELDS)
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
                print(
                    f"done {index}/{len(pending_rows)} case={row['image_id']} "
                    f"pred={normalized['y_pred']} p={normalized['y_prob']:.3f}"
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
            print(f"error {index}/{len(pending_rows)} case={row['image_id']}: {last_error}")
        if args.request_sleep_sec > 0:
            time.sleep(args.request_sleep_sec)

    final_rows = [by_key[item] for item in sorted(by_key)]
    metrics = compute_metrics(final_rows)
    total_prompt_tokens = sum(int(float(row.get("prompt_tokens", 0) or 0)) for row in final_rows)
    total_completion_tokens = sum(int(float(row.get("completion_tokens", 0) or 0)) for row in final_rows)
    summary = {
        "dry_run": False,
        "model_name": args.model,
        "deployment": args.deployment,
        "cohort_audit": cohort_audit,
        "valid_predictions": len(final_rows),
        "missing_predictions": len(manifest_rows) - len(final_rows),
        "complete_locked_cohort": len(final_rows) == len(manifest_rows),
        "new_errors": new_error_count,
        "metrics_valid_subset": metrics,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "outputs": {
            "predictions": str(predictions_path),
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
