#!/usr/bin/env python3
"""Audit case-level changes from a foundation model to the glaucoma agent.

This script never calls an LLM and never changes predictions. It joins a locked
foundation-model prediction file to an existing external-agent result file,
then reports regressions, rescues, invalid outputs, and evidence-ablation traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_FIELDS = [
    "case_id",
    "split",
    "y_true",
    "baseline_probability",
    "baseline_threshold",
    "baseline_prediction",
    "baseline_correct",
    "agent_normalized_probability_percent",
    "agent_prediction",
    "agent_correct",
    "comparison_category",
    "decision_transition",
    "agent_changed_baseline_label",
    "vertical_cdr",
    "cdr_zone",
    "cfp_impression",
    "cfp_impression_label",
    "counterfactual_full_evidence_label",
    "counterfactual_without_retfound_label",
    "counterfactual_without_visual_label",
    "counterfactual_without_cdr_label",
    "counterfactual_label_flip_scenarios",
    "counterfactual_evidence_sensitive",
    "counterfactual_interpretation",
    "final_disagrees_with_counterfactual_full_evidence",
    "orchestrator_reasoning",
    "agent_error",
    "sex_gender",
    "age",
    "age_group",
    "patient_id",
    "cfp_report",
    "counterfactual_trace_json",
    "agentic_decision",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Locked foundation-model predictions_test.csv containing case_id, y_true, y_prob, threshold, and y_pred.",
    )
    parser.add_argument(
        "--agent",
        type=Path,
        required=True,
        help="External-agent predictions.csv produced by run_external_glaucoma_agent.py.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def integer(value: Any, default: int | None = None) -> int | None:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    return parsed


def number(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def compact_text(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def markdown_text(value: Any, limit: int = 220) -> str:
    return compact_text(value, limit).replace("|", "\\|")


def fenced_text(value: Any) -> str:
    return str(value or "").strip().replace("```", "''' ") or "Unavailable"


def cfp_impression(report: str) -> tuple[str, str]:
    match = re.search(r"\bIMPRESSION\s*:\s*([^\r\n]+)", report or "", flags=re.IGNORECASE)
    impression = compact_text(match.group(1) if match else "", 300)
    lower = impression.lower()
    if "supports glaucoma" in lower:
        label = "1"
    elif "supports normal" in lower:
        label = "0"
    elif "indeterminate" in lower:
        label = "-1"
    else:
        label = ""
    return impression, label


def cdr_zone(value: Any) -> str:
    cdr = number(value)
    if cdr is None:
        return "unavailable"
    if cdr < 0.55:
        return "<0.55"
    if cdr < 0.65:
        return "0.55-0.65"
    return ">=0.65"


def orchestrator_reasoning(decision: str) -> str:
    match = re.search(r"\bReasoning\s*:\s*(.*)", decision or "", flags=re.IGNORECASE | re.DOTALL)
    return compact_text(match.group(1) if match else decision, 1200)


def parse_trace(value: str) -> tuple[dict[str, Any], str]:
    if not str(value or "").strip():
        return {}, ""
    try:
        trace = json.loads(value)
    except json.JSONDecodeError as exc:
        return {}, f"counterfactual_trace_json_error: {exc}"
    if not isinstance(trace, dict):
        return {}, "counterfactual_trace_not_object"
    return trace, ""


def scenario_map(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = trace.get("scenarios", [])
    if not isinstance(scenarios, list):
        return {}
    return {
        str(row.get("name")): row
        for row in scenarios
        if isinstance(row, dict) and str(row.get("name", "")).strip()
    }


def scenario_label(scenarios: dict[str, dict[str, Any]], name: str) -> str:
    value = integer(scenarios.get(name, {}).get("diagnosis"))
    return str(value) if value in {-1, 0, 1} else ""


def select_agent_rows(
    rows: list[dict[str, str]], split: str
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Choose the latest valid retry, otherwise the latest attempted row."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("split", "")).strip().lower() != split.lower():
            continue
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        grouped.setdefault(case_id, []).append(row)

    selected: dict[str, dict[str, str]] = {}
    duplicate_counts: dict[str, int] = {}
    for case_id, attempts in grouped.items():
        if len(attempts) > 1:
            duplicate_counts[case_id] = len(attempts)
        valid = [row for row in attempts if integer(row.get("Pred_GL")) in {0, 1}]
        selected[case_id] = valid[-1] if valid else attempts[-1]
    return selected, duplicate_counts


def unique_baseline_rows(rows: list[dict[str, str]], split: str) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        row_split = str(row.get("split", split)).strip().lower()
        if row_split and row_split != split.lower():
            continue
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Baseline row is missing case_id")
        if case_id in selected:
            raise ValueError(f"Duplicate baseline case_id: {case_id}")
        selected[case_id] = row
    return selected


def comparison_category(y_true: int, baseline_prediction: int, agent_prediction: int | None) -> str:
    if agent_prediction not in {0, 1}:
        return "invalid"
    baseline_correct = baseline_prediction == y_true
    agent_correct = agent_prediction == y_true
    if baseline_correct and agent_correct:
        return "both_correct"
    if baseline_correct and not agent_correct:
        return "regression"
    if not baseline_correct and agent_correct:
        return "rescue"
    return "both_wrong"


def decision_transition(y_true: int, baseline_prediction: int, agent_prediction: int | None) -> str:
    if agent_prediction not in {0, 1}:
        return "invalid_agent_output"
    if baseline_prediction == agent_prediction:
        return "unchanged_correct" if baseline_prediction == y_true else "unchanged_wrong"
    transitions = {
        (1, 1, 0): "lost_true_positive",
        (0, 0, 1): "lost_true_negative",
        (1, 0, 1): "rescued_true_positive",
        (0, 1, 0): "rescued_true_negative",
    }
    return transitions[(y_true, baseline_prediction, agent_prediction)]


def confusion(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    valid = [row for row in rows if integer(row.get(prediction_field)) in {0, 1}]
    tn = fp = fn = tp = 0
    for row in valid:
        truth = int(row["y_true"])
        prediction = int(row[prediction_field])
        tn += truth == 0 and prediction == 0
        fp += truth == 0 and prediction == 1
        fn += truth == 1 and prediction == 0
        tp += truth == 1 and prediction == 1
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": len(valid),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / len(valid) if valid else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def audit_row(baseline: dict[str, str], agent: dict[str, str] | None, split: str) -> dict[str, Any]:
    case_id = str(baseline.get("case_id", "")).strip()
    truth = integer(baseline.get("y_true"))
    baseline_prediction = integer(baseline.get("y_pred"))
    if truth not in {0, 1} or baseline_prediction not in {0, 1}:
        raise ValueError(f"Invalid baseline label/prediction for case {case_id}")

    agent = agent or {}
    agent_truth = integer(agent.get("Ground_Truth"))
    if agent_truth in {0, 1} and agent_truth != truth:
        raise ValueError(
            f"Ground-truth mismatch for {case_id}: baseline={truth}, agent={agent_truth}"
        )
    agent_prediction = integer(agent.get("Pred_GL"))
    if agent_prediction not in {0, 1}:
        agent_prediction = None

    trace, trace_error = parse_trace(agent.get("Counterfactual_Trace", ""))
    scenarios = scenario_map(trace)
    full_label = scenario_label(scenarios, "full_evidence")
    if not full_label:
        value = integer(trace.get("full_evidence_diagnosis"))
        full_label = str(value) if value in {-1, 0, 1} else ""
    impression, impression_label = cfp_impression(agent.get("CFP_Report", ""))
    error_parts = [str(agent.get("error", "")).strip(), trace_error]
    if not agent:
        error_parts.append("missing_agent_row")

    row = {
        "case_id": case_id,
        "split": split,
        "y_true": truth,
        "baseline_probability": baseline.get("y_prob", ""),
        "baseline_threshold": baseline.get("threshold", ""),
        "baseline_prediction": baseline_prediction,
        "baseline_correct": baseline_prediction == truth,
        "agent_normalized_probability_percent": agent.get("Agent_RETFound_Probability_Pct", ""),
        "agent_prediction": "" if agent_prediction is None else agent_prediction,
        "agent_correct": "" if agent_prediction is None else agent_prediction == truth,
        "comparison_category": comparison_category(truth, baseline_prediction, agent_prediction),
        "decision_transition": decision_transition(truth, baseline_prediction, agent_prediction),
        "agent_changed_baseline_label": "" if agent_prediction is None else agent_prediction != baseline_prediction,
        "vertical_cdr": agent.get("Vertical_CDR", ""),
        "cdr_zone": cdr_zone(agent.get("Vertical_CDR", "")),
        "cfp_impression": impression,
        "cfp_impression_label": impression_label,
        "counterfactual_full_evidence_label": full_label,
        "counterfactual_without_retfound_label": scenario_label(scenarios, "without_retfound_probability"),
        "counterfactual_without_visual_label": scenario_label(scenarios, "without_visual_interpretation"),
        "counterfactual_without_cdr_label": scenario_label(scenarios, "without_cdr_tool"),
        "counterfactual_label_flip_scenarios": ";".join(
            str(item) for item in trace.get("label_flip_scenarios", [])
        ) if isinstance(trace.get("label_flip_scenarios", []), list) else "",
        "counterfactual_evidence_sensitive": trace.get("evidence_sensitive", ""),
        "counterfactual_interpretation": compact_text(trace.get("interpretation", ""), 1200),
        "final_disagrees_with_counterfactual_full_evidence": (
            "" if agent_prediction is None or full_label not in {"0", "1"}
            else agent_prediction != int(full_label)
        ),
        "orchestrator_reasoning": orchestrator_reasoning(agent.get("Agentic_Decision", "")),
        "agent_error": "; ".join(item for item in error_parts if item),
        "sex_gender": baseline.get("sex_gender", ""),
        "age": baseline.get("age", ""),
        "age_group": baseline.get("age_group", ""),
        "patient_id": baseline.get("patient_id", ""),
        "cfp_report": agent.get("CFP_Report", ""),
        "counterfactual_trace_json": agent.get("Counterfactual_Trace", ""),
        "agentic_decision": agent.get("Agentic_Decision", ""),
    }
    return row


def metrics_markdown(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['n']} | {metrics['tn']} | {metrics['fp']} | "
        f"{metrics['fn']} | {metrics['tp']} | {metrics['f1']:.3f} | "
        f"{metrics['sensitivity']:.3f} | {metrics['specificity']:.3f} | "
        f"{metrics['balanced_accuracy']:.3f} |"
    )


def case_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Case | Truth | RETFound p/pred | Agent pred | CDR | CFP impression | CF full / no RETFound / no visual / no CDR | Orchestrator reasoning |",
        "|---|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        baseline = f"{number(row['baseline_probability'], 0.0):.3f}/{row['baseline_prediction']}"
        counterfactual = " / ".join(
            str(row.get(field, "")) or "NA"
            for field in (
                "counterfactual_full_evidence_label",
                "counterfactual_without_retfound_label",
                "counterfactual_without_visual_label",
                "counterfactual_without_cdr_label",
            )
        )
        lines.append(
            "| " + " | ".join(
                [
                    markdown_text(row["case_id"], 40),
                    str(row["y_true"]),
                    baseline,
                    str(row.get("agent_prediction", "invalid")) or "invalid",
                    str(row.get("vertical_cdr", "")) or "NA",
                    markdown_text(row.get("cfp_impression", ""), 90) or "NA",
                    counterfactual,
                    markdown_text(row.get("orchestrator_reasoning", ""), 180) or "NA",
                ]
            ) + " |"
        )
    return lines


def detailed_regression_traces(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        trace, _ = parse_trace(str(row.get("counterfactual_trace_json", "")))
        scenarios = scenario_map(trace)
        lines.extend(
            [
                f"### {row['case_id']}",
                "",
                f"- Truth: `{row['y_true']}`",
                f"- RETFound: probability `{row['baseline_probability']}`, threshold `{row['baseline_threshold']}`, prediction `{row['baseline_prediction']}`",
                f"- Agent: normalized RETFound score `{row['agent_normalized_probability_percent']}%`, prediction `{row['agent_prediction']}`",
                f"- Vertical CDR: `{row['vertical_cdr'] or 'unavailable'}`",
                "",
                "**CFP specialist output**",
                "",
                "```text",
                fenced_text(row.get("cfp_report", "")),
                "```",
                "",
                "**Evidence-ablation diagnoses**",
                "",
                "| Scenario | Diagnosis | Confidence | Reasoning |",
                "|---|---:|---|---|",
            ]
        )
        for name in (
            "full_evidence",
            "without_retfound_probability",
            "without_visual_interpretation",
            "without_cdr_tool",
        ):
            scenario = scenarios.get(name, {})
            lines.append(
                f"| `{name}` | {scenario.get('diagnosis', 'NA')} | "
                f"{markdown_text(scenario.get('confidence', ''), 40) or 'NA'} | "
                f"{markdown_text(scenario.get('reasoning', ''), 320) or 'NA'} |"
            )
        lines.extend(
            [
                "",
                "**Final orchestrator output**",
                "",
                "```text",
                fenced_text(row.get("agentic_decision", "")),
                "```",
                "",
            ]
        )
    return lines


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    regressions = [row for row in rows if row["comparison_category"] == "regression"]
    lost_positives = [row for row in rows if row["decision_transition"] == "lost_true_positive"]
    rescues = [row for row in rows if row["comparison_category"] == "rescue"]
    invalid = [row for row in rows if row["comparison_category"] == "invalid"]
    valid = [row for row in rows if row["comparison_category"] != "invalid"]

    lines = [
        "# External Glaucoma Agent Regression Audit",
        "",
        "This is a retrospective decision audit. It does not rerun models, call an LLM, or use test labels to alter predictions.",
        "",
        "## Actual Pipeline",
        "",
        "1. RETFound receives the PAPILA CFP and supplies one numeric glaucoma probability.",
        "2. The vision LLM receives the same CFP image and supplies a free-text CFP report.",
        "3. The CDR tool receives the same CFP and supplies vertical CDR.",
        "4. The so-called counterfactual agent receives the RETFound score, CFP report, and CDR; it produces four leave-one-evidence-out diagnoses. It does not generate demographic counterfactual patients on PAPILA.",
        "5. The final orchestrator receives all three CFP-derived representations plus the evidence-ablation trace and emits the final label.",
        "",
        "## Result",
        "",
        f"- Locked baseline cases: **{len(rows)}**",
        f"- Comparable valid agent outputs: **{len(valid)}**",
        f"- Agent regressions: **{len(regressions)}**",
        f"- Agent rescues: **{len(rescues)}**",
        f"- Lost RETFound true positives: **{len(lost_positives)}**",
        f"- Invalid or missing agent outputs: **{len(invalid)}**",
        "",
        "## Metrics",
        "",
        "| System | N | TN | FP | FN | TP | F1 | Sensitivity | Specificity | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metrics_markdown("RETFound, all locked cases", summary["metrics"]["baseline_all"]),
        metrics_markdown("RETFound, agent-valid subset", summary["metrics"]["baseline_valid_subset"]),
        metrics_markdown("Agent, valid subset", summary["metrics"]["agent_valid_subset"]),
        "",
        "The valid-subset rows are the direct paired comparison. Invalid outputs are not silently counted as correct or incorrect.",
        "",
        "## Decision Transitions",
        "",
    ]
    for name, count in summary["decision_transition_counts"].items():
        lines.append(f"- `{name}`: {count}")

    lines.extend(["", "## Lost True Positives", ""])
    lines.extend(case_table(lost_positives) if lost_positives else ["None."])
    lines.extend(["", "## All Regressions", ""])
    lines.extend(case_table(regressions) if regressions else ["None."])
    lines.extend(["", "## Rescues", ""])
    lines.extend(case_table(rescues) if rescues else ["None."])
    lines.extend(["", "## Detailed Regression Traces", ""])
    lines.extend(detailed_regression_traces(regressions) if regressions else ["None."])
    lines.extend(["", "## Invalid Outputs", ""])
    if invalid:
        lines.extend([
            "| Case | Truth | RETFound prediction | Error |",
            "|---|---:|---:|---|",
        ])
        for row in invalid:
            lines.append(
                f"| {markdown_text(row['case_id'], 40)} | {row['y_true']} | "
                f"{row['baseline_prediction']} | {markdown_text(row['agent_error'], 240) or 'unparseable label'} |"
            )
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "RETFound probability, CFP specialist text, and CDR are all derived from the same photograph on PAPILA. Their agreement is correlated evidence, not three independent votes. The ablation labels diagnose which representation influenced the LLM; they are not additional ground-truth observations.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = arguments()
    baseline_by_case = unique_baseline_rows(read_csv(args.baseline), args.split)
    if not baseline_by_case:
        raise ValueError(f"No baseline rows found for split={args.split!r}")
    agent_by_case, duplicate_attempts = select_agent_rows(read_csv(args.agent), args.split)

    audit_rows = [
        audit_row(baseline, agent_by_case.get(case_id), args.split)
        for case_id, baseline in baseline_by_case.items()
    ]
    regressions = [row for row in audit_rows if row["comparison_category"] == "regression"]
    rescues = [row for row in audit_rows if row["comparison_category"] == "rescue"]
    invalid = [row for row in audit_rows if row["comparison_category"] == "invalid"]
    valid = [row for row in audit_rows if row["comparison_category"] != "invalid"]
    lost_true_positives = [
        row for row in audit_rows if row["decision_transition"] == "lost_true_positive"
    ]

    summary = {
        "baseline_path": str(args.baseline),
        "agent_path": str(args.agent),
        "split": args.split,
        "locked_baseline_cases": len(audit_rows),
        "agent_rows_selected": len(agent_by_case),
        "agent_cases_not_in_baseline": sorted(set(agent_by_case) - set(baseline_by_case)),
        "duplicate_agent_attempt_counts": duplicate_attempts,
        "valid_paired_cases": len(valid),
        "invalid_or_missing_agent_cases": len(invalid),
        "regressions": len(regressions),
        "rescues": len(rescues),
        "lost_true_positives": len(lost_true_positives),
        "comparison_category_counts": dict(Counter(row["comparison_category"] for row in audit_rows)),
        "decision_transition_counts": dict(Counter(row["decision_transition"] for row in audit_rows)),
        "cfp_impression_counts": dict(Counter(row["cfp_impression_label"] or "unparsed" for row in audit_rows)),
        "cdr_zone_counts": dict(Counter(row["cdr_zone"] for row in audit_rows)),
        "metrics": {
            "baseline_all": confusion(audit_rows, "baseline_prediction"),
            "baseline_valid_subset": confusion(valid, "baseline_prediction"),
            "agent_valid_subset": confusion(valid, "agent_prediction"),
        },
        "output_definitions": {
            "regression": "baseline correct and agent wrong",
            "rescue": "baseline wrong and agent correct",
            "lost_true_positive": "y_true=1, baseline_prediction=1, agent_prediction=0",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "case_audit.csv", audit_rows)
    write_csv(args.out_dir / "regressions.csv", regressions)
    write_csv(args.out_dir / "lost_true_positives.csv", lost_true_positives)
    write_csv(args.out_dir / "rescues.csv", rescues)
    write_csv(args.out_dir / "invalid_cases.csv", invalid)
    write_json(args.out_dir / "audit_summary.json", summary)
    report = build_report(summary, audit_rows)
    (args.out_dir / "audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
