from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TASKS = ["amd", "dr", "glaucoma"]
ATTRIBUTES = [
    "GLOBAL",
    "race",
    "ethnicity",
    "sex_gender",
    "age_group",
    "race_x_sex_gender",
    "race_x_age_group",
    "sex_gender_x_age_group",
]
INTERSECTIONAL_PAIRS = [
    ("race", "sex_gender"),
    ("race", "age_group"),
    ("sex_gender", "age_group"),
]
METRIC_FIELDS = [
    "dataset",
    "task",
    "model_name",
    "attribute",
    "subgroup",
    "n",
    "n_positive",
    "n_negative",
    "auroc",
    "auprc",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "brier",
    "nll",
    "ece",
    "tp",
    "tn",
    "fp",
    "fn",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "fpr",
    "fnr",
    "unstable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build validation-only subgroup reliability priors for every available FairVision model. "
            "Also writes EquityAgent JSON calibration files with false-positive and false-negative rates."
        )
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument("--metrics-root", type=Path, default=Path("equi-agent/outputs/metrics"))
    parser.add_argument("--equity-json-dir", type=Path, default=Path("equi-agent/EquityAgent/JSONs"))
    parser.add_argument("--out", type=Path, default=Path("equi-agent/outputs/metrics/validation_subgroup_priors.csv"))
    parser.add_argument(
        "--global-out",
        type=Path,
        default=Path("equi-agent/outputs/metrics/validation_subgroup_priors_global.csv"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("equi-agent/outputs/metrics/validation_subgroup_priors.json"),
    )
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
        return default if math.isnan(number) else number
    except Exception:
        return default


def safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def sigmoid_clip(prob: float) -> float:
    return min(max(prob, 1e-7), 1.0 - 1e-7)


def ece(y_true: list[int], y_prob: list[float], bins: int = 10) -> float:
    if not y_true:
        return math.nan
    total = len(y_true)
    score = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        if idx == 0:
            selected = [pos for pos, prob in enumerate(y_prob) if prob >= lower and prob <= upper]
        else:
            selected = [pos for pos, prob in enumerate(y_prob) if prob > lower and prob <= upper]
        if not selected:
            continue
        avg_prob = sum(y_prob[pos] for pos in selected) / len(selected)
        avg_true = sum(y_true[pos] for pos in selected) / len(selected)
        score += (len(selected) / total) * abs(avg_prob - avg_true)
    return score


def auroc(y_true: list[int], y_prob: list[float]) -> float:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ordered = sorted(zip(y_prob, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    idx = 0
    while idx < len(ordered):
        end = idx
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx)) / 2
        rank_sum += avg_rank * sum(label for _, label in ordered[idx : end + 1])
        rank += end - idx + 1
        idx = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def auprc(y_true: list[int], y_prob: list[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return math.nan
    ordered = sorted(zip(y_prob, y_true), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, label in ordered:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def metrics(rows: list[dict], min_positive: int, min_negative: int) -> dict:
    y_true = [int(fnum(row.get("y_true"), 0.0)) for row in rows]
    y_prob = [fnum(row.get("y_prob"), 0.0) for row in rows]
    y_pred = [int(fnum(row.get("y_pred"), 0.0)) for row in rows]
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    n_positive = sum(y_true)
    n_negative = len(y_true) - n_positive
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    brier = sum((prob - true) ** 2 for true, prob in zip(y_true, y_prob)) / len(rows) if rows else math.nan
    nll = (
        -sum(true * math.log(sigmoid_clip(prob)) + (1 - true) * math.log(1 - sigmoid_clip(prob)) for true, prob in zip(y_true, y_prob))
        / len(rows)
        if rows
        else math.nan
    )
    return {
        "n": len(rows),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "f1": f1,
        "accuracy": safe_div(tp + tn, len(rows)),
        "balanced_accuracy": (sensitivity + specificity) / 2 if not math.isnan(sensitivity) and not math.isnan(specificity) else math.nan,
        "brier": brier,
        "nll": nll,
        "ece": ece(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "unstable": n_positive < min_positive or n_negative < min_negative,
    }


def norm(value: str | None) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() != "nan" else "missing"


def threshold_files(metrics_root: Path) -> dict[tuple[str, str, str], float]:
    thresholds = {}
    paths = list(metrics_root.glob("thresholds_fairvision_*.csv"))
    paths += list((metrics_root / "exp1_standalone_oct").glob("thresholds_*.csv"))
    paths += list((metrics_root / "exp1_standalone_slo").glob("thresholds_*.csv"))
    paths += list((metrics_root / "exp1_static_fusion_mean").glob("thresholds_*.csv"))
    paths += list((metrics_root / "exp1_static_fusion_confidence_weighted").glob("thresholds_*.csv"))
    for path in paths:
        for row in read_csv(path):
            key = (row.get("dataset", ""), row.get("task", ""), row.get("model_name", ""))
            if all(key):
                thresholds[key] = fnum(row.get("threshold"), 0.5)
    return thresholds


def val_prediction_files(predictions_root: Path) -> list[Path]:
    paths = []
    for path in predictions_root.glob("*_val.csv"):
        if "_smoke_" in path.name:
            continue
        paths.append(path)
    return sorted(paths)


def add_intersections(row: dict) -> dict:
    output = dict(row)
    for left, right in INTERSECTIONAL_PAIRS:
        output[f"{left}_x_{right}"] = f"{norm(row.get(left))} x {norm(row.get(right))}"
    return output


def load_validation_rows(predictions_root: Path, metrics_root: Path) -> tuple[list[dict], list[dict]]:
    thresholds = threshold_files(metrics_root)
    rows = []
    loaded = []
    for path in val_prediction_files(predictions_root):
        file_rows = []
        for raw in read_csv(path):
            if raw.get("split") not in {"val", "validation"}:
                continue
            row = add_intersections(raw)
            key = (row.get("dataset", ""), row.get("task", ""), row.get("model_name", ""))
            threshold = thresholds.get(key)
            if threshold is not None:
                row["y_pred"] = str(int(fnum(row.get("y_prob"), 0.0) >= threshold))
                row["threshold"] = threshold
            file_rows.append(row)
        if file_rows:
            loaded.append({"path": str(path), "rows": len(file_rows), "model_name": file_rows[0].get("model_name")})
            rows.extend(file_rows)
    return rows, loaded


def build_prior_rows(rows: list[dict], min_positive: int, min_negative: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("task", ""), row.get("model_name", ""))].append(row)

    subgroup_rows = []
    global_rows = []
    for (dataset, task, model), group_rows in sorted(grouped.items()):
        global_metric = {
            "dataset": dataset,
            "task": task,
            "model_name": model,
            "attribute": "GLOBAL",
            "subgroup": "GLOBAL",
        }
        global_metric.update(metrics(group_rows, min_positive, min_negative))
        global_metric["unstable"] = ""
        subgroup_rows.append(global_metric)
        global_rows.append({key: global_metric[key] for key in METRIC_FIELDS if key != "unstable"})

        for attr in ATTRIBUTES:
            if attr == "GLOBAL":
                continue
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in group_rows:
                groups[norm(row.get(attr))].append(row)
            for subgroup, attr_rows in sorted(groups.items()):
                metric = {
                    "dataset": dataset,
                    "task": task,
                    "model_name": model,
                    "attribute": attr,
                    "subgroup": subgroup,
                }
                metric.update(metrics(attr_rows, min_positive, min_negative))
                subgroup_rows.append(metric)
    return subgroup_rows, global_rows


def clean_json_number(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_equity_jsons(equity_json_dir: Path, subgroup_rows: list[dict]) -> None:
    equity_json_dir.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in subgroup_rows:
        by_model[str(row["model_name"])].append(row)

    combined = {"models": {}}
    for model, rows in sorted(by_model.items()):
        payload = {
            "source": "validation_subgroup_priors",
            "model_name": model,
            "false_positive": defaultdict(lambda: defaultdict(dict)),
            "false_negative": defaultdict(lambda: defaultdict(dict)),
            "counts": defaultdict(lambda: defaultdict(dict)),
        }
        for row in rows:
            task = str(row["task"])
            attr = str(row["attribute"])
            subgroup = str(row["subgroup"])
            payload["false_positive"][task][attr][subgroup] = clean_json_number(row["fpr"])
            payload["false_negative"][task][attr][subgroup] = clean_json_number(row["fnr"])
            payload["counts"][task][attr][subgroup] = {
                "n": row["n"],
                "n_positive": row["n_positive"],
                "n_negative": row["n_negative"],
                "unstable": bool(row.get("unstable")) if attr != "GLOBAL" else False,
            }
        serializable = json.loads(json.dumps(payload))
        combined["models"][model] = serializable
        (equity_json_dir / f"equity_{model}_calibration.json").write_text(
            json.dumps(serializable, indent=2, sort_keys=True)
        )

    (equity_json_dir / "equity_all_models_calibration.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True)
    )


def main() -> None:
    args = parse_args()
    rows, loaded = load_validation_rows(args.predictions_root, args.metrics_root)
    subgroup_rows, global_rows = build_prior_rows(rows, args.min_positive, args.min_negative)
    write_csv(args.out, subgroup_rows, METRIC_FIELDS)
    write_csv(args.global_out, global_rows, [key for key in METRIC_FIELDS if key != "unstable"])
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(subgroup_rows, indent=2, default=clean_json_number))
    write_equity_jsons(args.equity_json_dir, subgroup_rows)

    summary = {
        "validation_rows": len(rows),
        "loaded_files": len(loaded),
        "subgroup_prior_rows": len(subgroup_rows),
        "global_prior_rows": len(global_rows),
        "models": sorted({row["model_name"] for row in rows}),
        "tasks": sorted({row["task"] for row in rows}),
        "out": str(args.out),
        "global_out": str(args.global_out),
        "json_out": str(args.json_out),
        "equity_json_dir": str(args.equity_json_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
