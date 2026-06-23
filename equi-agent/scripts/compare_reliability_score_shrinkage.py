from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = ["amd", "dr", "glaucoma"]
DEFAULT_MODELS = [
    "retfound_oct",
    "mirage_slo",
    "flair_slo",
    "ret_clip_slo",
    "visionfm_slo",
    "visionfm_oct",
    "retizero_slo",
    "urfound_slo",
    "urfound_oct",
]
MODEL_ALIASES = {
    "retfound_oct": ["fairvision_oct_retfound"],
    "mirage_slo": ["fairvision_slo_mirage"],
}
CASE_COLUMNS = ["patient_id", "eye_id", "visit_id", "image_id", "dataset", "task"]
METADATA_COLUMNS = ["race", "ethnicity", "sex_gender", "age", "age_group", "metadata_missing_flag"]
ATTRIBUTES = [
    "race_x_sex_gender",
    "race_x_age_group",
    "sex_gender_x_age_group",
    "race",
    "ethnicity",
    "sex_gender",
    "age_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing FairVision reliability arbitration against explicit score-level "
            "shrinkage: R = lambda_g * R_local + (1 - lambda_g) * R_global."
        )
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument("--old-predictions", type=Path, default=Path("equi-agent/outputs/fairvision_reliability_selective_arbitration/selective_arbitration_predictions.csv"))
    parser.add_argument("--subgroup-priors", type=Path, default=Path("equi-agent/outputs/metrics/validation_subgroup_priors.csv"))
    parser.add_argument("--global-priors", type=Path, default=Path("equi-agent/outputs/metrics/validation_subgroup_priors_global.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/fairvision_reliability_score_shrinkage_comparison"))
    parser.add_argument("--subgroup-shrinkage-k", type=float, default=50.0)
    parser.add_argument("--conformal-shrinkage-k", type=float, default=50.0)
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.50)
    parser.add_argument("--low-reliability-threshold", type=float, default=0.35)
    parser.add_argument("--disagreement-rate-threshold", type=float, default=0.25)
    parser.add_argument("--close-call-margin", type=float, default=0.08)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
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


def norm(value: Any) -> str:
    if value is None:
        return "missing"
    text = str(value).strip()
    return text.lower() if text and text.lower() != "nan" else "missing"


def int_label(value: Any) -> int:
    return int(round(finite(value, 0.0)))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def case_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(col, "")) for col in CASE_COLUMNS)


def subgroup_value(row: dict[str, Any], attr: str) -> str:
    if attr == "race_x_sex_gender":
        return f"{norm(row.get('race'))} x {norm(row.get('sex_gender'))}"
    if attr == "race_x_age_group":
        return f"{norm(row.get('race'))} x {norm(row.get('age_group'))}"
    if attr == "sex_gender_x_age_group":
        return f"{norm(row.get('sex_gender'))} x {norm(row.get('age_group'))}"
    return norm(row.get(attr))


def add_intersectional(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for col in METADATA_COLUMNS:
        out.setdefault(col, "")
    out["race_x_sex_gender"] = subgroup_value(out, "race_x_sex_gender")
    out["race_x_age_group"] = subgroup_value(out, "race_x_age_group")
    out["sex_gender_x_age_group"] = subgroup_value(out, "sex_gender_x_age_group")
    return out


def prediction_candidates(root: Path, task: str, model: str, split: str) -> list[Path]:
    suffixes = [f"{split}_thresholded", split] if split == "test" else [split, f"{split}_thresholded"]
    stems = [f"fairvision_{task}_{model}", f"fairvision_{model}"] + MODEL_ALIASES.get(model, [])
    return [root / f"{stem}_{suffix}.csv" for stem in stems for suffix in suffixes]


def prediction_file(root: Path, task: str, model: str, split: str) -> Path | None:
    for path in prediction_candidates(root, task, model, split):
        if path.exists():
            return path
    return None


def load_model_predictions(root: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_values = {"test"} if split == "test" else {"val", "validation"}
    for task in TASKS:
        for model in DEFAULT_MODELS:
            path = prediction_file(root, task, model, split)
            if path is None:
                raise SystemExit(f"Missing prediction file for {task}/{model}/{split}")
            for row in read_csv(path):
                if row.get("task") and str(row.get("task")) != task:
                    continue
                if row.get("split") and str(row.get("split")).lower() not in split_values:
                    continue
                row = add_intersectional(row)
                row["task"] = task
                row["split"] = split
                row["model_name"] = model
                row["y_prob"] = finite(row.get("y_prob"), 0.0)
                row["y_true"] = int_label(row.get("y_true"))
                row["y_pred"] = int_label(row.get("y_pred")) if str(row.get("y_pred", "")).strip() else int(row["y_prob"] >= 0.5)
                rows.append(row)
    return restrict_to_common_cases(rows)


def restrict_to_common_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[case_key(row)][str(row["model_name"])] = row
    keep_keys = {key for key, model_rows in grouped.items() if len(model_rows) == len(DEFAULT_MODELS)}
    return [row for row in rows if case_key(row) in keep_keys]


def load_priors(subgroup_path: Path, global_path: Path) -> tuple[dict[tuple[str, str, str, str, str], dict[str, str]], dict[tuple[str, str, str], dict[str, str]]]:
    subgroup: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    global_: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_csv(subgroup_path):
        key = (row.get("dataset", ""), row.get("task", ""), row.get("model_name", ""), row.get("attribute", ""), norm(row.get("subgroup", "")))
        subgroup[key] = row
        if row.get("attribute") == "GLOBAL":
            global_[(key[0], key[1], key[2])] = row
    for row in read_csv(global_path):
        global_[(row.get("dataset", ""), row.get("task", ""), row.get("model_name", ""))] = row
    return subgroup, global_


def metric_values(row: dict[str, Any] | None) -> dict[str, float]:
    return {
        "auroc": finite(row.get("auroc") if row else None, 0.5),
        "f1": finite(row.get("f1") if row else None, 0.0),
        "ece": finite(row.get("ece") if row else None, 0.5),
        "fnr": finite(row.get("fnr") if row else None, 0.5),
        "fpr": finite(row.get("fpr") if row else None, 0.5),
    }


def reliability_score(metrics: dict[str, float]) -> float:
    return (
        0.45 * metrics["auroc"]
        + 0.35 * metrics["f1"]
        - 0.10 * metrics["ece"]
        - 0.07 * metrics["fnr"]
        - 0.03 * metrics["fpr"]
    )


def case_prior(
    meta: dict[str, Any],
    model: str,
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict[str, str]],
    global_lookup: dict[tuple[str, str, str], dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset = str(meta.get("dataset", ""))
    task = str(meta.get("task", ""))
    global_prior = global_lookup.get((dataset, task, model))
    local = None
    attr = "GLOBAL"
    subgroup = "GLOBAL"
    for candidate_attr in ATTRIBUTES:
        candidate_subgroup = subgroup_value(meta, candidate_attr)
        candidate = subgroup_lookup.get((dataset, task, model, candidate_attr, candidate_subgroup))
        if candidate is not None:
            local = candidate
            attr = candidate_attr
            subgroup = candidate_subgroup
            break
    n = finite(local.get("n") if local else None, 0.0)
    lam = n / (n + args.subgroup_shrinkage_k) if n > 0 else 0.0
    r_global = reliability_score(metric_values(global_prior))
    r_local = reliability_score(metric_values(local)) if local is not None else r_global
    return {
        "attribute": attr,
        "subgroup": subgroup,
        "lambda": lam,
        "local_n": n,
        "global_n": finite(global_prior.get("n") if global_prior else None, 0.0),
        "local_unstable": boolish(local.get("unstable") if local else False),
        "local_reliability_score": r_local,
        "global_reliability_score": r_global,
        "reliability_score": lam * r_local + (1.0 - lam) * r_global,
    }


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def arbitrate_case(
    group: list[dict[str, Any]],
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict[str, str]],
    global_lookup: dict[tuple[str, str, str], dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    meta = {col: group[0].get(col, "") for col in CASE_COLUMNS + ["y_true"] + METADATA_COLUMNS + ATTRIBUTES}
    vote_rate = sum(int_label(row["y_pred"]) for row in group) / len(group)
    majority_pred = int(vote_rate >= 0.5)
    disagreement_rate = min(vote_rate, 1.0 - vote_rate)
    logits = []
    probs = []
    scores = []
    for row in group:
        prior = case_prior(meta, str(row["model_name"]), subgroup_lookup, global_lookup, args)
        score = prior["reliability_score"]
        prob = finite(row.get("y_prob"), 0.0)
        pred = int_label(row.get("y_pred"))
        uncertainty = max(0.0, 1.0 - 2.0 * abs(prob - 0.5))
        disagrees = float(pred != majority_pred)
        logits.append(args.beta * score - args.gamma * uncertainty - args.delta * disagrees)
        probs.append(prob)
        scores.append(score)
    weights = softmax(logits)
    final_prob = sum(weight * prob for weight, prob in zip(weights, probs))
    weighted_reliability = sum(weight * score for weight, score in zip(weights, scores))
    margin = abs(final_prob - 0.5)
    risk_score = max(
        1.0 - min(1.0, 2.0 * margin),
        disagreement_rate,
        max(0.0, args.low_reliability_threshold - weighted_reliability),
    )
    return {
        **meta,
        "model_name": "reliability_score_shrunk_weighted",
        "y_prob": final_prob,
        "y_pred": int(final_prob >= 0.5),
        "split": str(group[0].get("split", "")),
        "num_models": len(group),
        "vote_rate": vote_rate,
        "positive_votes": sum(int_label(row["y_pred"]) for row in group),
        "disagreement_rate": disagreement_rate,
        "close_call": margin < args.close_call_margin,
        "weighted_reliability": weighted_reliability,
        "risk_score": risk_score,
    }


def arbitrate_rows(
    rows: list[dict[str, Any]],
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict[str, str]],
    global_lookup: dict[tuple[str, str, str], dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)
    return [arbitrate_case(group, subgroup_lookup, global_lookup, args) for _, group in sorted(grouped.items())]


def conformal_quantile(scores: list[float], alpha: float) -> float:
    clean = sorted(value for value in scores if not math.isnan(value))
    if not clean:
        return 0.5
    rank = int(math.ceil((len(clean) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(clean))
    return clean[rank - 1]


def build_conformal_tables(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, float], dict[tuple[str, str, str], dict[str, float]]]:
    by_task: dict[str, list[float]] = defaultdict(list)
    by_local: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        score = 1.0 - finite(row["y_prob"]) if int_label(row["y_true"]) == 1 else finite(row["y_prob"])
        task = str(row.get("task", ""))
        by_task[task].append(score)
        for attr in ATTRIBUTES:
            by_local[(task, attr, subgroup_value(row, attr))].append(score)
    global_q = {task: conformal_quantile(scores, args.conformal_alpha) for task, scores in by_task.items()}
    local_q = {
        key: {"q": conformal_quantile(scores, args.conformal_alpha), "n": float(len(scores))}
        for key, scores in by_local.items()
    }
    return global_q, local_q


def case_conformal_q(row: dict[str, Any], global_q: dict[str, float], local_q: dict[tuple[str, str, str], dict[str, float]], args: argparse.Namespace) -> tuple[float, str, str, float]:
    task = str(row.get("task", ""))
    base_q = global_q.get(task, 0.5)
    for attr in ATTRIBUTES:
        subgroup = subgroup_value(row, attr)
        local = local_q.get((task, attr, subgroup))
        if local is None:
            continue
        n = finite(local.get("n"), 0.0)
        lam = n / (n + args.conformal_shrinkage_k) if n > 0 else 0.0
        return lam * finite(local.get("q"), base_q) + (1.0 - lam) * base_q, attr, subgroup, lam
    return base_q, "GLOBAL", "GLOBAL", 0.0


def add_escalation(rows: list[dict[str, Any]], global_q: dict[str, float], local_q: dict[tuple[str, str, str], dict[str, float]], args: argparse.Namespace) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        row = dict(row)
        reasons = []
        if finite(row.get("disagreement_rate"), 0.0) >= args.disagreement_rate_threshold:
            reasons.append("model_disagreement")
        if boolish(row.get("close_call")):
            reasons.append("close_call")
        if finite(row.get("weighted_reliability"), 1.0) < args.low_reliability_threshold:
            reasons.append("low_reliability")
        prob = finite(row.get("y_prob"), 0.5)
        q, q_attr, q_subgroup, q_lambda = case_conformal_q(row, global_q, local_q, args)
        labels = []
        if prob <= q:
            labels.append("negative")
        if (1.0 - prob) <= q:
            labels.append("positive")
        if len(labels) != 1:
            reasons.append("conformal_ambiguous")
        row["accepted"] = len(reasons) == 0
        row["escalate_to_human"] = not row["accepted"]
        row["escalation_reasons"] = ";".join(reasons)
        row["conformal_label_set"] = ";".join(labels) if labels else "empty"
        row["conformal_set_size"] = len(labels)
        row["conformal_q"] = q
        row["conformal_attribute"] = q_attr
        row["conformal_subgroup"] = q_subgroup
        row["conformal_lambda"] = q_lambda
        row["escalation_policy_score"] = finite(row["risk_score"]) + float(row["escalate_to_human"])
        output.append(row)
    return output


def safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in rows:
        y = int_label(row.get("y_true"))
        yp = int_label(row.get("y_pred"))
        if y == 1 and yp == 1:
            tp += 1
        elif y == 0 and yp == 0:
            tn += 1
        elif y == 0 and yp == 1:
            fp += 1
        elif y == 1 and yp == 0:
            fn += 1
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(rows)),
        "precision": safe_div(tp, tp + fp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2 if not math.isnan(sens) and not math.isnan(spec) else math.nan,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for row in rows:
        by_task.setdefault(str(row.get("task", "")), []).append(row)
    output = []
    for task, group in by_task.items():
        accepted = [row for row in group if not boolish(row.get("escalate_to_human"))]
        escalated = [row for row in group if boolish(row.get("escalate_to_human"))]
        output.append(
            {
                "task": task,
                "n": len(group),
                "accepted_n": len(accepted),
                "escalated_n": len(escalated),
                "coverage": safe_div(len(accepted), len(group)),
                "escalation_rate": safe_div(len(escalated), len(group)),
                **{f"forced_{k}": v for k, v in metrics(group).items()},
                **{f"accepted_{k}": v for k, v in metrics(accepted).items()},
                **{f"escalated_forced_{k}": v for k, v in metrics(escalated).items()},
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subgroup_lookup, global_lookup = load_priors(args.subgroup_priors, args.global_priors)
    val_base = load_model_predictions(args.predictions_root, "val")
    test_base = load_model_predictions(args.predictions_root, "test")
    val = arbitrate_rows(val_base, subgroup_lookup, global_lookup, args)
    test = arbitrate_rows(test_base, subgroup_lookup, global_lookup, args)
    global_q, local_q = build_conformal_tables(val, args)
    new_rows = add_escalation(test, global_q, local_q, args)

    old_rows = {case_key(row): row for row in read_csv(args.old_predictions)}
    deescalated = []
    newly_escalated = []
    changed_prediction = []
    for row in new_rows:
        old = old_rows.get(case_key(row))
        if old is None:
            continue
        row["old_y_prob"] = old.get("y_prob", "")
        row["old_y_pred"] = old.get("y_pred", "")
        row["old_escalate_to_human"] = old.get("escalate_to_human", "")
        row["old_escalation_reasons"] = old.get("escalation_reasons", "")
        if boolish(old.get("escalate_to_human")) and not boolish(row.get("escalate_to_human")):
            deescalated.append(row)
        if not boolish(old.get("escalate_to_human")) and boolish(row.get("escalate_to_human")):
            newly_escalated.append(row)
        if int_label(old.get("y_pred")) != int_label(row.get("y_pred")):
            changed_prediction.append(row)

    accepted = [row for row in new_rows if not boolish(row.get("escalate_to_human"))]
    low_only_deescalated = [
        row for row in deescalated if row.get("old_escalation_reasons") == "low_reliability"
    ]
    summary = {
        "score_formula": "R = lambda_g * R_local + (1 - lambda_g) * R_global",
        "local_global_formula": "0.45*AUROC + 0.35*F1 - 0.10*ECE - 0.07*FNR - 0.03*FPR",
        "test_cases": len(new_rows),
        "old_escalated": sum(boolish(row.get("escalate_to_human")) for row in old_rows.values()),
        "new_escalated": sum(boolish(row.get("escalate_to_human")) for row in new_rows),
        "deescalated_old_to_new": len(deescalated),
        "newly_escalated_old_to_new": len(newly_escalated),
        "prediction_label_changed": len(changed_prediction),
        "low_reliability_only_deescalated": len(low_only_deescalated),
        "new_accepted_metrics": metrics(accepted),
        "deescalated_metrics": metrics(deescalated),
        "low_reliability_only_deescalated_metrics": metrics(low_only_deescalated),
        "outputs": {
            "new_predictions": str(args.out_dir / "score_shrinkage_predictions.csv"),
            "metrics_by_task": str(args.out_dir / "score_shrinkage_metrics_by_task.csv"),
            "deescalated_cases": str(args.out_dir / "deescalated_cases.csv"),
            "newly_escalated_cases": str(args.out_dir / "newly_escalated_cases.csv"),
            "changed_prediction_cases": str(args.out_dir / "changed_prediction_cases.csv"),
        },
    }
    write_csv(args.out_dir / "score_shrinkage_predictions.csv", new_rows)
    write_csv(args.out_dir / "score_shrinkage_metrics_by_task.csv", summarize(new_rows))
    write_csv(args.out_dir / "deescalated_cases.csv", deescalated)
    write_csv(args.out_dir / "newly_escalated_cases.csv", newly_escalated)
    write_csv(args.out_dir / "changed_prediction_cases.csv", changed_prediction)
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
