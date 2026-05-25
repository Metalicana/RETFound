from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


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

TASKS = ["amd", "dr", "glaucoma"]
KEY_COLUMNS = ["patient_id", "eye_id", "visit_id", "image_id", "task"]
STANDARD_COLUMNS = [
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
]
ATTRIBUTES = [
    "race",
    "ethnicity",
    "sex_gender",
    "age_group",
    "race_x_sex_gender",
    "race_x_age_group",
    "sex_gender_x_age_group",
]
METRIC_FIELDS = [
    "task",
    "model_name",
    "n",
    "n_positive",
    "n_negative",
    "auroc",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "fpr",
    "fnr",
    "ece",
    "brier",
    "worst_group_f1",
    "max_min_fpr_gap",
    "max_min_fnr_gap",
    "tp",
    "tn",
    "fp",
    "fn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Equi-Agent live predictions against deterministic baselines "
            "on the exact same FairVision subset."
        )
    )
    parser.add_argument(
        "--equi-predictions",
        type=Path,
        default=Path("equi-agent/outputs/equi_agent_live_stratified500_tuned_v3/equi_agent_live_predictions.csv"),
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument(
        "--global-priors",
        type=Path,
        default=Path("equi-agent/outputs/metrics/validation_subgroup_priors_global.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/tables/equi_agent_subset_compare"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--min-negative", type=int, default=5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def key_for(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(col, "") for col in KEY_COLUMNS)


def float_value(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def int_label(value: object) -> int:
    return int(round(float_value(value)))


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def model_file(predictions_root: Path, task: str, model: str) -> Path | None:
    direct = predictions_root / f"fairvision_{task}_{model}_test_thresholded.csv"
    if direct.exists():
        return direct
    combined = {
        "retfound_oct": predictions_root / "fairvision_oct_retfound_test_thresholded.csv",
        "mirage_slo": predictions_root / "fairvision_slo_mirage_test_thresholded.csv",
    }.get(model)
    if combined and combined.exists():
        return combined
    fallback = predictions_root / f"fairvision_{model}_test_thresholded.csv"
    if fallback.exists():
        return fallback
    return None


def add_intersections(row: dict[str, object]) -> dict[str, object]:
    row = dict(row)
    row["race_x_sex_gender"] = f"{row.get('race', 'missing')}|{row.get('sex_gender', 'missing')}"
    row["race_x_age_group"] = f"{row.get('race', 'missing')}|{row.get('age_group', 'missing')}"
    row["sex_gender_x_age_group"] = f"{row.get('sex_gender', 'missing')}|{row.get('age_group', 'missing')}"
    return row


def standard_row(source: dict[str, str], model_name: str, y_prob: float, y_pred: int) -> dict[str, object]:
    row = {col: source.get(col, "") for col in STANDARD_COLUMNS}
    row["model_name"] = model_name
    row["y_prob"] = float(y_prob)
    row["y_pred"] = int(y_pred)
    row["y_true"] = int_label(source["y_true"])
    row["metadata_missing_flag"] = bool_value(source.get("metadata_missing_flag", False))
    return add_intersections(row)


def auroc(rows: list[dict[str, object]]) -> float | None:
    pairs = [(float_value(row["y_prob"]), int_label(row["y_true"])) for row in rows]
    pos = sum(label == 1 for _, label in pairs)
    neg = sum(label == 0 for _, label in pairs)
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += sum(avg_rank for _, label in pairs[i:j] if label == 1)
        i = j
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def ece(rows: list[dict[str, object]], bins: int = 10) -> float:
    total = len(rows)
    if total == 0:
        return 0.0
    acc = 0.0
    for bin_index in range(bins):
        lo = bin_index / bins
        hi = (bin_index + 1) / bins
        bucket = [
            row
            for row in rows
            if float_value(row["y_prob"]) >= lo and (float_value(row["y_prob"]) < hi or bin_index == bins - 1)
        ]
        if not bucket:
            continue
        confidence = sum(float_value(row["y_prob"]) for row in bucket) / len(bucket)
        observed = sum(int_label(row["y_true"]) for row in bucket) / len(bucket)
        acc += len(bucket) / total * abs(confidence - observed)
    return acc


def confusion(rows: list[dict[str, object]]) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for row in rows:
        y_true = int_label(row["y_true"])
        y_pred = int_label(row["y_pred"])
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
    return tp, tn, fp, fn


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = safe_div(tp, tp + fp) or 0.0
    recall = safe_div(tp, tp + fn) or 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def subgroup_summary(rows: list[dict[str, object]], min_positive: int, min_negative: int) -> dict[str, float | None]:
    f1s = []
    fpr_gaps = []
    fnr_gaps = []
    for attr in ATTRIBUTES:
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(attr, "missing"))].append(row)
        attr_f1s = []
        fprs = []
        fnrs = []
        for group_rows in groups.values():
            positives = sum(int_label(row["y_true"]) == 1 for row in group_rows)
            negatives = sum(int_label(row["y_true"]) == 0 for row in group_rows)
            if positives < min_positive or negatives < min_negative:
                continue
            tp, tn, fp, fn = confusion(group_rows)
            attr_f1s.append(f1_from_counts(tp, fp, fn))
            fpr = safe_div(fp, fp + tn)
            fnr = safe_div(fn, fn + tp)
            if fpr is not None:
                fprs.append(fpr)
            if fnr is not None:
                fnrs.append(fnr)
        if attr_f1s:
            f1s.append(min(attr_f1s))
        if len(fprs) >= 2:
            fpr_gaps.append(max(fprs) - min(fprs))
        if len(fnrs) >= 2:
            fnr_gaps.append(max(fnrs) - min(fnrs))
    return {
        "worst_group_f1": min(f1s) if f1s else None,
        "max_min_fpr_gap": max(fpr_gaps) if fpr_gaps else None,
        "max_min_fnr_gap": max(fnr_gaps) if fnr_gaps else None,
    }


def metrics(task: str, model_name: str, rows: list[dict[str, object]], min_positive: int, min_negative: int) -> dict[str, object]:
    tp, tn, fp, fn = confusion(rows)
    n = len(rows)
    n_positive = tp + fn
    n_negative = tn + fp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = f1_from_counts(tp, fp, fn)
    brier = sum((float_value(row["y_prob"]) - int_label(row["y_true"])) ** 2 for row in rows) / n if n else None
    subgroup = subgroup_summary(rows, min_positive, min_negative)
    return {
        "task": task,
        "model_name": model_name,
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "auroc": auroc(rows),
        "f1": f1,
        "accuracy": safe_div(tp + tn, n),
        "balanced_accuracy": None if recall is None or specificity is None else (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "ece": ece(rows),
        "brier": brier,
        **subgroup,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_global_priors(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    priors = {}
    if not path.exists():
        return priors
    for row in read_csv(path):
        if row.get("dataset") != "harvard_fairvision":
            continue
        if row.get("attribute") != "GLOBAL" or row.get("subgroup") != "GLOBAL":
            continue
        priors[(row["task"], row["model_name"])] = {
            "auroc": float_value(row.get("auroc")),
            "f1": float_value(row.get("f1")),
            "ece": float_value(row.get("ece")),
        }
    return priors


def weighted_average(candidates: list[dict[str, object]], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return sum(float_value(row["y_prob"]) for row in candidates) / len(candidates)
    return sum(float_value(row["y_prob"]) * weight for row, weight in zip(candidates, weights)) / total


def make_fusion_rows(
    model_name: str,
    subset_rows: list[dict[str, object]],
    rows_by_key_model: dict[tuple[tuple[str, ...], str], dict[str, object]],
    models: list[str],
    weight_fn,
) -> list[dict[str, object]]:
    out = []
    for subset_row in subset_rows:
        key = key_for(subset_row)
        candidates = [rows_by_key_model[(key, model)] for model in models if (key, model) in rows_by_key_model]
        if not candidates:
            continue
        weights = [weight_fn(candidate) for candidate in candidates]
        y_prob = weighted_average(candidates, weights)
        out.append(standard_row(subset_row, model_name, y_prob, int(y_prob >= 0.5)))
    return out


def make_source_selection_rows(
    model_name: str,
    subset_rows: list[dict[str, object]],
    rows_by_key_model: dict[tuple[tuple[str, ...], str], dict[str, object]],
    models: list[str],
    score_fn,
) -> list[dict[str, object]]:
    out = []
    for subset_row in subset_rows:
        key = key_for(subset_row)
        candidates = [rows_by_key_model[(key, model)] for model in models if (key, model) in rows_by_key_model]
        if not candidates:
            continue
        chosen = max(candidates, key=score_fn)
        out.append(standard_row(subset_row, model_name, float_value(chosen["y_prob"]), int_label(chosen["y_pred"])))
    return out


def load_model_rows(args, subset_keys_by_task) -> tuple[dict[str, dict[tuple[str, ...], dict[str, object]]], list[str]]:
    model_rows: dict[str, dict[tuple[str, ...], dict[str, object]]] = {}
    available_models = []
    for model in args.models:
        by_key = {}
        for task in args.tasks:
            path = model_file(args.predictions_root, task, model)
            if path is None:
                continue
            for row in read_csv(path):
                if row.get("task") != task or row.get("split") != "test":
                    continue
                key = key_for(row)
                if key not in subset_keys_by_task[task]:
                    continue
                by_key[key] = add_intersections(row)
        if by_key:
            model_rows[model] = by_key
            available_models.append(model)
    return model_rows, available_models


def macro_average(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        if row["task"] == "macro":
            continue
        by_model[str(row["model_name"])].append(row)
    out = []
    for model_name, rows in by_model.items():
        if len(rows) < 1:
            continue
        macro = {"task": "macro", "model_name": model_name}
        for field in METRIC_FIELDS:
            if field in {"task", "model_name"}:
                continue
            values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float)) and row.get(field) is not None]
            macro[field] = sum(values) / len(values) if values else None
        out.append(macro)
    return out


def main() -> None:
    args = parse_args()
    equi_rows = [
        add_intersections(standard_row(row, "equi_agent_live", float_value(row["y_prob"]), int_label(row["y_pred"])))
        for row in read_csv(args.equi_predictions)
        if row.get("task") in set(args.tasks)
    ]
    subset_rows_by_task = {
        task: [row for row in equi_rows if row.get("task") == task]
        for task in args.tasks
    }
    subset_keys_by_task = {
        task: {key_for(row) for row in rows}
        for task, rows in subset_rows_by_task.items()
    }
    global_priors = load_global_priors(args.global_priors)
    model_rows, available_models = load_model_rows(args, subset_keys_by_task)

    prediction_rows = []
    metric_rows = []
    coverage_rows = []
    rows_by_key_model = {}
    for model, by_key in model_rows.items():
        for key, row in by_key.items():
            rows_by_key_model[(key, model)] = row

    for task in args.tasks:
        subset_rows = subset_rows_by_task[task]
        prediction_rows.extend(subset_rows)
        metric_rows.append(metrics(task, "equi_agent_live", subset_rows, args.min_positive, args.min_negative))

        task_models = []
        for model in available_models:
            rows = []
            for subset_row in subset_rows:
                key = key_for(subset_row)
                if key in model_rows[model]:
                    source = model_rows[model][key]
                    rows.append(standard_row(source, model, float_value(source["y_prob"]), int_label(source["y_pred"])))
            coverage_rows.append({"task": task, "model_name": model, "matched": len(rows), "expected": len(subset_rows)})
            if len(rows) == len(subset_rows):
                task_models.append(model)
                prediction_rows.extend(rows)
                metric_rows.append(metrics(task, model, rows, args.min_positive, args.min_negative))

        task_subset_rows = subset_rows
        task_rows_by_key_model = {
            (key, model): row
            for (key, model), row in rows_by_key_model.items()
            if key in subset_keys_by_task[task] and model in task_models
        }
        priors_for = lambda metric, model: global_priors.get((task, model), {}).get(metric, 0.0)

        fusion_specs = [
            ("mean_probability_all", lambda row: 1.0),
            ("confidence_weighted_all", lambda row: abs(float_value(row["y_prob"]) - 0.5)),
            ("validation_auroc_weighted_all", lambda row: max(priors_for("auroc", str(row["model_name"])), 0.0)),
            ("validation_f1_weighted_all", lambda row: max(priors_for("f1", str(row["model_name"])), 0.0)),
            ("validation_ece_inverse_weighted_all", lambda row: 1.0 / (0.001 + max(priors_for("ece", str(row["model_name"])), 0.0))),
        ]
        for name, weight_fn in fusion_specs:
            rows = make_fusion_rows(name, task_subset_rows, task_rows_by_key_model, task_models, weight_fn)
            prediction_rows.extend(rows)
            metric_rows.append(metrics(task, name, rows, args.min_positive, args.min_negative))

        source_specs = [
            ("dynamic_global_auroc_source", lambda row: priors_for("auroc", str(row["model_name"]))),
            ("dynamic_global_f1_source", lambda row: priors_for("f1", str(row["model_name"]))),
            ("dynamic_global_ece_source", lambda row: -priors_for("ece", str(row["model_name"]))),
        ]
        for name, score_fn in source_specs:
            rows = make_source_selection_rows(name, task_subset_rows, task_rows_by_key_model, task_models, score_fn)
            prediction_rows.extend(rows)
            metric_rows.append(metrics(task, name, rows, args.min_positive, args.min_negative))

        standalone_rows = [row for row in metric_rows if row["task"] == task and row["model_name"] in task_models]
        if standalone_rows:
            best_model = max(standalone_rows, key=lambda row: row.get("f1") or 0.0)["model_name"]
            rows = []
            for subset_row in task_subset_rows:
                source = model_rows[str(best_model)][key_for(subset_row)]
                rows.append(standard_row(source, "oracle_best_subset_single_model", float_value(source["y_prob"]), int_label(source["y_pred"])))
            prediction_rows.extend(rows)
            metric_rows.append(metrics(task, "oracle_best_subset_single_model", rows, args.min_positive, args.min_negative))

    metric_rows.extend(macro_average(metric_rows))
    metric_rows.sort(key=lambda row: (str(row["task"]), -(row.get("f1") or 0.0), str(row["model_name"])))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "subset_comparison_predictions.csv", prediction_rows, STANDARD_COLUMNS)
    write_csv(args.out_dir / "subset_comparison_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "subset_comparison_coverage.csv", coverage_rows, ["task", "model_name", "matched", "expected"])
    write_markdown(args.out_dir / "subset_comparison_metrics.md", metric_rows, [
        "task",
        "model_name",
        "n",
        "auroc",
        "f1",
        "balanced_accuracy",
        "recall",
        "specificity",
        "ece",
        "worst_group_f1",
        "max_min_fpr_gap",
        "max_min_fnr_gap",
    ])

    print(f"available_models={available_models}")
    print(f"wrote={args.out_dir}")
    print("\nMacro rows:")
    for row in [r for r in metric_rows if r["task"] == "macro"][:20]:
        print(
            f"{row['model_name']:36s} "
            f"auroc={fmt(row.get('auroc'))} f1={fmt(row.get('f1'))} "
            f"bal_acc={fmt(row.get('balanced_accuracy'))} ece={fmt(row.get('ece'))}"
        )

    print("\nTop by task:")
    for task in args.tasks:
        print(f"\n{task}")
        for row in [r for r in metric_rows if r["task"] == task][:8]:
            print(
                f"{row['model_name']:36s} "
                f"auroc={fmt(row.get('auroc'))} f1={fmt(row.get('f1'))} "
                f"recall={fmt(row.get('recall'))} spec={fmt(row.get('specificity'))}"
            )


if __name__ == "__main__":
    main()
