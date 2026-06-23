from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CASE_KEY_FIELDS = ["patient_id", "eye_id", "visit_id", "image_id", "task"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Equi-Agent live/dry-run trace folders for stored prompts, "
            "run parameters, and per-agent trace fields."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("equi-agent/outputs/equi_agent_dynamic_few_shot_allmodels_dryrun1"),
        help="Directory containing equi_agent_live_*.csv/json/jsonl outputs.",
    )
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument(
        "--extract-prompts",
        action="store_true",
        help="Write prompt text samples to equi_agent_live_prompt_samples/.",
    )
    parser.add_argument(
        "--prompt-sample-limit",
        type=int,
        default=3,
        help="Maximum number of cases whose prompts are written with --extract-prompts.",
    )
    parser.add_argument("--snippet-chars", type=int, default=500)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{line_no}: expected JSON object")
            rows.append(obj)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def case_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in CASE_KEY_FIELDS)


def case_id(row: dict[str, Any]) -> str:
    return "|".join(f"{field}={row.get(field, '')}" for field in CASE_KEY_FIELDS)


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text[:180].strip("_") or "case"


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                pieces.append(str(part))
            elif part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                pieces.append("[image_url]")
            else:
                pieces.append(json.dumps(part, sort_keys=True))
        return "\n".join(pieces)
    return json.dumps(content, sort_keys=True)


def compact_snippet(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def message_summary(messages: Any, snippet_chars: int) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {
            "messages_present": False,
            "message_count": 0,
            "roles": [],
            "message_hashes": [],
            "message_lengths": [],
            "message_snippets": [],
        }
    roles: list[str] = []
    hashes: list[str] = []
    lengths: list[int] = []
    snippets: list[str] = []
    has_image_payload = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        roles.append(str(message.get("role", "")))
        text = message_text(message)
        hashes.append(sha12(text))
        lengths.append(len(text))
        snippets.append(compact_snippet(text, snippet_chars))
        content = message.get("content")
        if isinstance(content, list):
            has_image_payload = has_image_payload or any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in content
            )
    return {
        "messages_present": True,
        "message_count": len(messages),
        "roles": roles,
        "message_hashes": hashes,
        "message_lengths": lengths,
        "message_snippets": snippets,
        "has_image_payload": has_image_payload,
        "agent_specific_prompt_messages_present": any(role not in {"system", "user", "assistant"} for role in roles),
    }


def prompt_protocol(messages: Any) -> dict[str, Any]:
    """Extract agent protocol blocks from the stored user prompt when present."""
    if not isinstance(messages, list):
        return {}
    system_text = ""
    user_text = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text = message_text(message)
        if role == "system" and not system_text:
            system_text = text
        elif role == "user" and not user_text:
            user_text = text

    user_obj: dict[str, Any] = {}
    if user_text:
        try:
            parsed = json.loads(user_text)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            user_obj = parsed

    instructions = user_obj.get("instructions")
    if not isinstance(instructions, dict):
        instructions = {}

    internal_agent_protocol = instructions.get("internal_agent_protocol")
    if not isinstance(internal_agent_protocol, dict):
        internal_agent_protocol = {}

    required_schema = instructions.get("required_json_schema")
    if not isinstance(required_schema, dict):
        required_schema = user_obj.get("required_output_schema")
    if not isinstance(required_schema, dict):
        required_schema = {}

    agent_schema = required_schema.get("agent_trace")
    if not isinstance(agent_schema, dict):
        agent_schema = {}

    dynamic_thresholds = instructions.get("dynamic_thresholds")
    if not isinstance(dynamic_thresholds, dict):
        dynamic_thresholds = user_obj.get("threshold_policy")
    if not isinstance(dynamic_thresholds, dict):
        dynamic_thresholds = {}

    evidence_packet = user_obj.get("evidence_packet")
    evidence_keys = sorted(evidence_packet.keys()) if isinstance(evidence_packet, dict) else []

    return {
        "system_prompt_present": bool(system_text),
        "system_prompt_sha12": sha12(system_text) if system_text else "",
        "system_prompt_length": len(system_text),
        "user_json_present": bool(user_obj),
        "internal_agent_protocol_present": bool(internal_agent_protocol),
        "internal_agent_protocol_agents": sorted(internal_agent_protocol),
        "internal_agent_protocol": internal_agent_protocol,
        "required_schema_agents": sorted(agent_schema),
        "dynamic_thresholds": dynamic_thresholds,
        "evidence_packet_keys": evidence_keys,
    }


def agent_trace_from(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("agent_trace")
    if isinstance(trace, dict):
        return trace
    if isinstance(trace, str) and trace.strip():
        try:
            parsed = json.loads(trace)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def summarize_agent_trace(trace: dict[str, Any]) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    agents = sorted(key for key, value in trace.items() if isinstance(value, dict))
    fields: dict[str, list[str]] = {}
    table_lengths: dict[str, int] = {}
    for agent in agents:
        value = trace[agent]
        if not isinstance(value, dict):
            continue
        fields[agent] = sorted(value.keys())
        table = value.get("model_reliability_table")
        if isinstance(table, list):
            table_lengths[agent] = len(table)
    return agents, fields, table_lengths


def unique_nonempty(rows: list[dict[str, str]], column: str) -> list[str]:
    values = sorted({str(row.get(column, "")) for row in rows if str(row.get(column, "")) != ""})
    return values


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def extract_prompt_samples(
    run_dir: Path,
    raw_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, str]]:
    prompt_dir = run_dir / "equi_agent_live_prompt_samples"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, str]] = []
    for row in raw_rows[: max(0, limit)]:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        safe_id = safe_filename(case_id(row))
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = safe_filename(str(message.get("role", "message")))
            path = prompt_dir / f"{safe_id}.{index:02d}.{role}.txt"
            path.write_text(message_text(message))
            written.append(
                {
                    "case_id": case_id(row),
                    "role": str(message.get("role", "")),
                    "path": str(path),
                }
            )
    return written


def print_console_summary(audit: dict[str, Any]) -> None:
    params = audit["run_parameters"]
    prompt = audit["prompt_storage"]
    agents = audit["agent_trace"]
    print(f"run_dir: {audit['run_dir']}")
    print(f"cases: {params.get('cases')}  dry_run: {params.get('dry_run')}  provider: {params.get('llm_provider')}")
    print(f"deployment: {params.get('llm_deployment')}  prompt_variant: {params.get('prompt_variant')}")
    print(
        "messages stored: "
        f"{prompt['raw_rows_with_messages']}/{prompt['raw_rows']}  "
        f"role patterns: {prompt['role_patterns']}"
    )
    print(
        "agent traces: "
        f"{agents['rows_with_agent_trace']}/{agents['trace_rows']}  "
        f"agents: {', '.join(agents['agents_present'])}"
    )
    if audit["parameter_gaps"]:
        print("not fully persisted: " + ", ".join(audit["parameter_gaps"]))
    print(f"audit_json: {audit['outputs']['audit_json']}")
    print(f"audit_csv: {audit['outputs']['audit_csv']}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    summary_path = run_dir / "equi_agent_live_summary.json"
    raw_path = run_dir / "equi_agent_live_raw_responses.jsonl"
    trace_path = run_dir / "equi_agent_live_agent_trace.jsonl"
    predictions_path = run_dir / "equi_agent_live_predictions.csv"
    usage_path = run_dir / "equi_agent_live_usage.csv"

    summary = read_json(summary_path)
    raw_rows = read_jsonl(raw_path)
    trace_rows = read_jsonl(trace_path)
    prediction_rows = read_csv(predictions_path)
    usage_rows = read_csv(usage_path)

    prediction_by_key = {case_key(row): row for row in prediction_rows}
    usage_by_key = {case_key(row): row for row in usage_rows}

    case_audit_rows: list[dict[str, Any]] = []
    role_patterns: Counter[str] = Counter()
    message_hash_patterns: Counter[str] = Counter()
    agent_counts: Counter[str] = Counter()
    raw_agent_field_sets: dict[str, Counter[str]] = defaultdict(Counter)
    raw_model_reliability_table_lengths: dict[str, Counter[int]] = defaultdict(Counter)
    rows_with_messages = 0
    rows_with_raw_response = 0
    rows_with_agent_trace = 0
    rows_with_image_payload = 0
    rows_with_agent_prompt_protocol = 0
    agent_prompt_instruction_hashes: dict[str, Counter[str]] = defaultdict(Counter)
    agent_prompt_instruction_samples: dict[str, str] = {}
    dynamic_threshold_patterns: Counter[str] = Counter()

    for index, raw_row in enumerate(raw_rows):
        key = case_key(raw_row)
        pred_row = prediction_by_key.get(key, {})
        usage_row = usage_by_key.get(key, {})
        msg_summary = message_summary(raw_row.get("messages"), args.snippet_chars)
        prompt_info = prompt_protocol(raw_row.get("messages"))
        trace = agent_trace_from(raw_row)
        agents, fields, table_lengths = summarize_agent_trace(trace)

        if msg_summary["messages_present"]:
            rows_with_messages += 1
        if raw_row.get("raw_response"):
            rows_with_raw_response += 1
        if trace:
            rows_with_agent_trace += 1
        if msg_summary.get("has_image_payload"):
            rows_with_image_payload += 1
        if prompt_info.get("internal_agent_protocol_present"):
            rows_with_agent_prompt_protocol += 1
            protocol = prompt_info.get("internal_agent_protocol", {})
            if isinstance(protocol, dict):
                for agent, instruction in protocol.items():
                    instruction_text = str(instruction)
                    agent_prompt_instruction_hashes[str(agent)][sha12(instruction_text)] += 1
                    agent_prompt_instruction_samples.setdefault(str(agent), instruction_text)
        thresholds = prompt_info.get("dynamic_thresholds")
        if isinstance(thresholds, dict) and thresholds:
            dynamic_threshold_patterns[compact_json(thresholds)] += 1

        role_pattern = ",".join(msg_summary["roles"])
        role_patterns[role_pattern] += 1
        message_hash_patterns[",".join(msg_summary["message_hashes"])] += 1
        for agent in agents:
            agent_counts[agent] += 1
            raw_agent_field_sets[agent][",".join(fields.get(agent, []))] += 1
            if agent in table_lengths:
                raw_model_reliability_table_lengths[agent][table_lengths[agent]] += 1

        case_audit_rows.append(
            {
                "case_index": index,
                "case_id": case_id(raw_row),
                "task": raw_row.get("task", ""),
                "patient_id": raw_row.get("patient_id", ""),
                "y_true": raw_row.get("y_true", ""),
                "llm_provider": raw_row.get("llm_provider", pred_row.get("llm_provider", "")),
                "llm_deployment": raw_row.get("llm_deployment", pred_row.get("llm_deployment", "")),
                "raw_response_present": bool(raw_row.get("raw_response")),
                "messages_present": msg_summary["messages_present"],
                "message_roles": role_pattern,
                "message_hashes": ",".join(msg_summary["message_hashes"]),
                "message_lengths": ",".join(str(length) for length in msg_summary["message_lengths"]),
                "has_image_payload": msg_summary.get("has_image_payload", False),
                "agent_specific_prompt_messages_present": msg_summary.get(
                    "agent_specific_prompt_messages_present",
                    False,
                ),
                "internal_agent_protocol_present": prompt_info.get("internal_agent_protocol_present", False),
                "internal_agent_protocol_agents": ",".join(
                    prompt_info.get("internal_agent_protocol_agents", [])
                ),
                "required_schema_agents": ",".join(prompt_info.get("required_schema_agents", [])),
                "dynamic_thresholds_json": compact_json(prompt_info.get("dynamic_thresholds", {})),
                "agents_present": ",".join(agents),
                "agent_fields_json": compact_json(fields),
                "model_reliability_table_lengths_json": compact_json(table_lengths),
                "final_probability": pred_row.get("y_prob", ""),
                "final_prediction": pred_row.get("y_pred", ""),
                "applied_threshold": pred_row.get("applied_threshold", ""),
                "calibration_action": pred_row.get("calibration_action", ""),
                "few_shot_action_mode": pred_row.get("few_shot_action_mode", ""),
                "escalate_to_human": pred_row.get("escalate_to_human", ""),
                "prompt_tokens": usage_row.get("prompt_tokens", pred_row.get("prompt_tokens", "")),
                "completion_tokens": usage_row.get("completion_tokens", pred_row.get("completion_tokens", "")),
                "total_tokens": usage_row.get("total_tokens", pred_row.get("total_tokens", "")),
            }
        )

    trace_agent_counts: Counter[str] = Counter()
    trace_agent_field_sets: dict[str, Counter[str]] = defaultdict(Counter)
    trace_model_reliability_table_lengths: dict[str, Counter[int]] = defaultdict(Counter)
    trace_rows_with_trace = 0
    for row in trace_rows:
        trace = agent_trace_from(row)
        if trace:
            trace_rows_with_trace += 1
        agents, fields, table_lengths = summarize_agent_trace(trace)
        for agent in agents:
            trace_agent_counts[agent] += 1
            trace_agent_field_sets[agent][",".join(fields.get(agent, []))] += 1
            if agent in table_lengths:
                trace_model_reliability_table_lengths[agent][table_lengths[agent]] += 1

    out_json = args.out_json or (run_dir / "equi_agent_live_trace_audit.json")
    out_csv = args.out_csv or (run_dir / "equi_agent_live_trace_audit_cases.csv")

    prompt_samples: list[dict[str, str]] = []
    if args.extract_prompts:
        prompt_samples = extract_prompt_samples(run_dir, raw_rows, args.prompt_sample_limit)

    role_pattern_dict = dict(role_patterns)
    provider_values = unique_nonempty(prediction_rows, "llm_provider")
    deployment_values = unique_nonempty(prediction_rows, "llm_deployment")
    action_modes = unique_nonempty(prediction_rows, "few_shot_action_mode")
    applied_thresholds = unique_nonempty(prediction_rows, "applied_threshold")
    calibration_actions = dict(Counter(row.get("calibration_action", "") for row in prediction_rows))

    parameter_gaps: list[str] = []
    for field in ["temperature", "max_output_tokens", "max_retries", "request_sleep_sec", "command_line", "git_commit"]:
        if field not in summary:
            parameter_gaps.append(field)
    if not action_modes and summary.get("prompt_variant") == "dynamic_few_shot":
        parameter_gaps.append("few_shot_action_mode")

    audit: dict[str, Any] = {
        "run_dir": str(run_dir),
        "input_files": {
            "summary": str(summary_path),
            "raw_responses": str(raw_path),
            "agent_trace": str(trace_path),
            "predictions": str(predictions_path),
            "usage": str(usage_path),
        },
        "outputs": {
            "audit_json": str(out_json),
            "audit_csv": str(out_csv),
        },
        "run_parameters": {
            "cases": summary.get("cases", len(prediction_rows) or len(raw_rows)),
            "dry_run": summary.get("dry_run"),
            "llm_provider": summary.get("llm_provider") or (provider_values[0] if len(provider_values) == 1 else provider_values),
            "llm_deployment": summary.get("llm_deployment")
            or (deployment_values[0] if len(deployment_values) == 1 else deployment_values),
            "prompt_variant": summary.get("prompt_variant"),
            "few_shot_k": summary.get("few_shot_k"),
            "few_shot_action_mode_values": action_modes,
            "models_requested": summary.get("models_requested", []),
            "tasks": summary.get("tasks", []),
            "loaded_files": summary.get("loaded_files"),
            "validation_files_loaded": summary.get("validation_files_loaded"),
            "model_prior_coverage": summary.get("model_prior_coverage"),
            "prompt_tokens": summary.get("prompt_tokens"),
            "completion_tokens": summary.get("completion_tokens"),
            "total_tokens": summary.get("total_tokens"),
            "applied_thresholds_seen": applied_thresholds,
            "calibration_actions_seen": calibration_actions,
        },
        "runner_default_parameters_not_saved_in_summary": {
            "temperature": 0.0,
            "max_output_tokens": 700,
            "sensitivity_threshold": 0.35,
            "neutral_threshold": 0.50,
            "precision_threshold": 0.65,
            "source": "defaults in scripts/run_equi_agent_fairvision_live.py; actual CLI overrides cannot be proven from this output folder unless separately logged",
        },
        "parameter_gaps": parameter_gaps,
        "prompt_storage": {
            "raw_rows": len(raw_rows),
            "raw_rows_with_messages": rows_with_messages,
            "raw_rows_with_raw_response": rows_with_raw_response,
            "rows_with_image_payload": rows_with_image_payload,
            "rows_with_internal_agent_protocol": rows_with_agent_prompt_protocol,
            "role_patterns": role_pattern_dict,
            "message_hash_patterns": dict(message_hash_patterns),
            "dynamic_threshold_patterns": dict(dynamic_threshold_patterns),
            "prompt_samples_written": prompt_samples,
            "interpretation": (
                "Messages are stored per case in equi_agent_live_raw_responses.jsonl. "
                "For this runner, agent-specific roles are encoded inside the system/user JSON prompt; "
                "they are not separate chat messages or separate API calls."
            ),
        },
        "agent_prompt_protocol": {
            "rows_with_internal_agent_protocol": rows_with_agent_prompt_protocol,
            "agent_instruction_hashes": {
                agent: dict(counter)
                for agent, counter in sorted(agent_prompt_instruction_hashes.items())
            },
            "agent_instruction_samples": agent_prompt_instruction_samples,
            "interpretation": (
                "These are prompt instructions embedded in the stored user JSON. "
                "They are agent-role instructions, not evidence of independent per-agent LLM calls."
            ),
        },
        "agent_trace": {
            "trace_rows": len(trace_rows),
            "rows_with_agent_trace": max(rows_with_agent_trace, trace_rows_with_trace),
            "agents_present": sorted(set(agent_counts) | set(trace_agent_counts)),
            "raw_response_agent_counts": dict(agent_counts),
            "trace_file_agent_counts": dict(trace_agent_counts),
            "raw_response_agent_field_sets": {
                agent: dict(counter)
                for agent, counter in sorted(raw_agent_field_sets.items())
            },
            "trace_file_agent_field_sets": {
                agent: dict(counter)
                for agent, counter in sorted(trace_agent_field_sets.items())
            },
            "raw_response_model_reliability_table_lengths": {
                agent: {str(length): count for length, count in sorted(counter.items())}
                for agent, counter in sorted(raw_model_reliability_table_lengths.items())
            },
            "trace_file_model_reliability_table_lengths": {
                agent: {str(length): count for length, count in sorted(counter.items())}
                for agent, counter in sorted(trace_model_reliability_table_lengths.items())
            },
        },
    }

    write_csv(out_csv, case_audit_rows)
    write_json(out_json, audit)
    print_console_summary(audit)


if __name__ == "__main__":
    main()
