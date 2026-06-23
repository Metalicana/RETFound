from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IDENTITY_COLUMNS = ["patient_id", "eye_id", "visit_id", "image_id", "dataset", "task"]
CASE_REVIEW_COLUMNS = [
    *IDENTITY_COLUMNS,
    "y_true",
    "model_name",
    "y_pred",
    "y_prob",
    "accepted",
    "escalate_to_human",
    "explicit_reasons",
    "derived_flags",
    "reason_combination",
    "positive_votes",
    "num_models",
    "vote_rate",
    "disagreement",
    "disagreement_rate",
    "close_call",
    "weighted_reliability",
    "risk_score",
    "escalation_policy_score",
    "conformal_label_set",
    "conformal_set_size",
    "conformal_q",
    "conformal_attribute",
    "conformal_subgroup",
    "confidence",
    "primary_model",
    "few_shot_case_ids",
    "vision_evidence_summary",
    "calibration_action",
    "safety_decision",
    "safety_reasons",
    "equity_reliability_concern",
    "equity_threshold_policy",
    "orchestrator_rationale",
    "llm_provider",
    "llm_deployment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze accepted/escalated failure cases from Equi-Agent or "
            "reliability-selective arbitration prediction CSVs."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--counterfactuals", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        columns = ordered
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def int_label(value: Any) -> int:
    return int(round(finite(value, 0.0)))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "escalate", "escalate_to_human"}


def is_present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def row_escalated(row: dict[str, str]) -> bool:
    if is_present(row.get("escalate_to_human")):
        return boolish(row.get("escalate_to_human"))
    if is_present(row.get("accepted")):
        return not boolish(row.get("accepted"))
    if is_present(row.get("safety_decision")):
        return "escalate" in str(row.get("safety_decision", "")).lower()
    return False


def split_reasons(value: Any) -> list[str]:
    if not is_present(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    for sep in [";", "|", ","]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def explicit_reasons(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    for column in ["escalation_reasons", "safety_reasons"]:
        reasons.extend(split_reasons(row.get(column)))
    return sorted(dict.fromkeys(reasons))


def derived_flags(row: dict[str, str]) -> list[str]:
    flags: list[str] = []
    if boolish(row.get("close_call")):
        flags.append("close_call")
    if boolish(row.get("disagreement")) or finite(row.get("disagreement_rate"), 0.0) >= 0.25:
        flags.append("model_disagreement")
    if finite(row.get("conformal_set_size"), 1.0) > 1.0 or ";" in str(row.get("conformal_label_set", "")):
        flags.append("conformal_ambiguous")
    if is_present(row.get("weighted_reliability")) and finite(row.get("weighted_reliability")) < 0.35:
        flags.append("low_reliability")
    if is_present(row.get("risk_score")) and finite(row.get("risk_score")) >= 0.75:
        flags.append("high_risk_score")
    concern = str(row.get("equity_reliability_concern", "")).strip()
    if concern and concern.lower() not in {"none", "nan", "false"}:
        flags.append(f"equity_{concern}")
    return sorted(dict.fromkeys(flags))


def reason_bundle(row: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    explicit = explicit_reasons(row)
    derived = derived_flags(row)
    if explicit:
        primary = sorted(dict.fromkeys([*explicit, *derived]))
    elif row_escalated(row):
        primary = derived or ["unspecified_escalation"]
    else:
        primary = []
    return explicit, derived, primary


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def metric_block(rows: list[dict[str, str]]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in rows:
        y_true = int_label(row.get("y_true"))
        y_pred = int_label(row.get("y_pred"))
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "n": len(rows),
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(rows)),
        "precision": safe_div(tp, tp + fp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if not math.isnan(sensitivity) and not math.isnan(specificity)
            else math.nan
        ),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "error_rate": safe_div(fp + fn, len(rows)),
    }


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def fmt_float(value: Any) -> Any:
    number = finite(value)
    if math.isnan(number):
        return value
    return f"{number:.6f}"


def summarize_escalation(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    by_task: dict[str, list[dict[str, str]]] = {"overall": rows}
    for row in rows:
        by_task.setdefault(str(row.get("task", "")), []).append(row)
    for task, group in by_task.items():
        accepted = [row for row in group if not row_escalated(row)]
        escalated = [row for row in group if row_escalated(row)]
        output.append(
            {
                "task": task,
                "n": len(group),
                "accepted_n": len(accepted),
                "escalated_n": len(escalated),
                "coverage": safe_div(len(accepted), len(group)),
                "escalation_rate": safe_div(len(escalated), len(group)),
                **prefixed("forced", metric_block(group)),
                **prefixed("accepted", metric_block(accepted)),
                **prefixed("escalated_forced", metric_block(escalated)),
            }
        )
    return output


def summarize_reasons(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reason_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    reason_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    escalated_rows = [row for row in rows if row_escalated(row)]

    for row in escalated_rows:
        _, _, reasons = reason_bundle(row)
        combo = ";".join(reasons) if reasons else "none"
        combo_counts[combo] += 1
        for reason in reasons:
            reason_counts[reason] += 1
            reason_rows[reason].append(row)

    count_rows = [
        {
            "reason_type": "individual",
            "reason": reason,
            "n": count,
            "pct_of_escalated": safe_div(count, len(escalated_rows)),
        }
        for reason, count in reason_counts.most_common()
    ]
    count_rows.extend(
        {
            "reason_type": "combination",
            "reason": reason,
            "n": count,
            "pct_of_escalated": safe_div(count, len(escalated_rows)),
        }
        for reason, count in combo_counts.most_common()
    )

    metric_rows = []
    for reason, reason_group in sorted(reason_rows.items()):
        metric_rows.append(
            {
                "reason": reason,
                "n": len(reason_group),
                "pct_of_escalated": safe_div(len(reason_group), len(escalated_rows)),
                **metric_block(reason_group),
            }
        )
    return count_rows, metric_rows


def summarize_reason_source(rows: list[dict[str, str]], source: str) -> list[dict[str, Any]]:
    if source not in {"explicit", "derived"}:
        raise ValueError(f"Unknown reason source: {source}")
    escalated_rows = [row for row in rows if row_escalated(row)]
    counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    for row in escalated_rows:
        explicit, derived, _ = reason_bundle(row)
        reasons = explicit if source == "explicit" else derived
        combo_counts[";".join(reasons) if reasons else "none"] += 1
        for reason in reasons:
            counts[reason] += 1
    output = [
        {
            "reason_type": "individual",
            "reason": reason,
            "n": count,
            "pct_of_escalated": safe_div(count, len(escalated_rows)),
        }
        for reason, count in counts.most_common()
    ]
    output.extend(
        {
            "reason_type": "combination",
            "reason": reason,
            "n": count,
            "pct_of_escalated": safe_div(count, len(escalated_rows)),
        }
        for reason, count in combo_counts.most_common()
    )
    return output


def add_review_columns(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        explicit, derived, reasons = reason_bundle(row)
        enriched = dict(row)
        enriched["accepted"] = not row_escalated(row)
        enriched["escalate_to_human"] = row_escalated(row)
        enriched["explicit_reasons"] = ";".join(explicit)
        enriched["derived_flags"] = ";".join(derived)
        enriched["reason_combination"] = ";".join(reasons)
        output.append(enriched)
    return output


def summarize_counterfactuals(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    def summarize(scope: str, group: list[dict[str, str]], task: str = "", attribute: str = "") -> dict[str, Any]:
        abs_prob = [abs(finite(row.get("counterfactual_y_prob")) - finite(row.get("base_y_prob"))) for row in group]
        abs_prob = [value for value in abs_prob if not math.isnan(value)]
        abs_risk = [abs(finite(row.get("risk_shift"))) for row in group if is_present(row.get("risk_shift"))]
        abs_reliability = [
            abs(finite(row.get("reliability_shift"))) for row in group if is_present(row.get("reliability_shift"))
        ]
        sorted_prob = sorted(abs_prob)
        p95_index = max(0, min(len(sorted_prob) - 1, math.ceil(0.95 * len(sorted_prob)) - 1)) if sorted_prob else 0
        return {
            "scope": scope,
            "task": task,
            "attribute": attribute,
            "n": len(group),
            "label_flip_rate": safe_div(sum(int_label(row.get("label_flipped")) for row in group), len(group)),
            "escalation_flip_rate": safe_div(sum(int_label(row.get("escalation_flipped")) for row in group), len(group)),
            "mean_abs_probability_shift": safe_div(sum(abs_prob), len(abs_prob)),
            "p95_abs_probability_shift": sorted_prob[p95_index] if sorted_prob else math.nan,
            "max_abs_probability_shift": max(abs_prob) if abs_prob else math.nan,
            "mean_abs_risk_shift": safe_div(sum(abs_risk), len(abs_risk)),
            "mean_abs_reliability_shift": safe_div(sum(abs_reliability), len(abs_reliability)),
            "has_risk_shift": bool(abs_risk),
            "has_reliability_shift": bool(abs_reliability),
        }

    overall = [summarize("overall", rows)]
    by_attribute = []
    grouped_attribute: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_attribute[(str(row.get("task", "")), str(row.get("attribute", "")))].append(row)
    for (task, attribute), group in sorted(grouped_attribute.items()):
        by_attribute.append(summarize("task_attribute", group, task=task, attribute=attribute))
    flipped = [
        row
        for row in rows
        if int_label(row.get("label_flipped")) == 1 or int_label(row.get("escalation_flipped")) == 1
    ]
    return overall, by_attribute, flipped


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.predictions)
    enriched = add_review_columns(rows)
    metrics_rows = summarize_escalation(rows)
    reason_count_rows, reason_metric_rows = summarize_reasons(rows)
    explicit_reason_rows = summarize_reason_source(rows, "explicit")
    derived_flag_rows = summarize_reason_source(rows, "derived")
    failure_rows = [
        row for row in enriched if is_present(row.get("y_true")) and int_label(row.get("y_true")) != int_label(row.get("y_pred"))
    ]
    escalated_rows = [row for row in enriched if row_escalated(row)]

    write_csv(args.out_dir / "escalation_metrics_by_task.csv", metrics_rows)
    write_csv(args.out_dir / "escalation_reason_counts.csv", reason_count_rows)
    write_csv(args.out_dir / "explicit_escalation_reason_counts.csv", explicit_reason_rows)
    write_csv(args.out_dir / "derived_escalation_flag_counts.csv", derived_flag_rows)
    write_csv(args.out_dir / "escalation_reason_metrics.csv", reason_metric_rows)
    write_csv(args.out_dir / "failure_cases.csv", failure_rows)
    write_csv(args.out_dir / "escalated_case_review_table.csv", escalated_rows, CASE_REVIEW_COLUMNS)

    summary: dict[str, Any] = {
        "predictions": str(args.predictions),
        "n_rows": len(rows),
        "outputs": {
            "metrics": str(args.out_dir / "escalation_metrics_by_task.csv"),
            "reason_counts": str(args.out_dir / "escalation_reason_counts.csv"),
            "explicit_reason_counts": str(args.out_dir / "explicit_escalation_reason_counts.csv"),
            "derived_flag_counts": str(args.out_dir / "derived_escalation_flag_counts.csv"),
            "reason_metrics": str(args.out_dir / "escalation_reason_metrics.csv"),
            "failure_cases": str(args.out_dir / "failure_cases.csv"),
            "escalated_case_review_table": str(args.out_dir / "escalated_case_review_table.csv"),
        },
    }

    if args.counterfactuals:
        counterfactual_rows = read_csv(args.counterfactuals)
        cf_overall, cf_by_attribute, cf_flipped = summarize_counterfactuals(counterfactual_rows)
        write_csv(args.out_dir / "counterfactual_trust_summary.csv", cf_overall)
        write_csv(args.out_dir / "counterfactual_trust_by_attribute.csv", cf_by_attribute)
        write_csv(args.out_dir / "counterfactual_flipped_cases.csv", cf_flipped)
        summary["counterfactuals"] = {
            "path": str(args.counterfactuals),
            "n_rows": len(counterfactual_rows),
            "has_risk_shift": any(is_present(row.get("risk_shift")) for row in counterfactual_rows),
            "has_reliability_shift": any(is_present(row.get("reliability_shift")) for row in counterfactual_rows),
            "outputs": {
                "summary": str(args.out_dir / "counterfactual_trust_summary.csv"),
                "by_attribute": str(args.out_dir / "counterfactual_trust_by_attribute.csv"),
                "flipped_cases": str(args.out_dir / "counterfactual_flipped_cases.csv"),
            },
        }

    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
