from __future__ import annotations

import argparse
import csv
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
METRIC_FIELDS = [
    "task",
    "model_name",
    "n",
    "tn",
    "fp",
    "fn",
    "tp",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Equi-Agent predictions against standalone foundation models on the exact same subset."
    )
    parser.add_argument("--equi-predictions", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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
    return None


def key_for(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(col, "") for col in KEY_COLUMNS)


def metrics(task: str, model_name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    y_true = [int(float(row["y_true"])) for row in rows]
    y_pred = [int(float(row["y_pred"])) for row in rows]
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    n = len(rows)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "task": task,
        "model_name": model_name,
        "n": n,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
    }


def main() -> None:
    args = parse_args()
    equi_rows = [row for row in read_csv(args.equi_predictions) if row.get("task") in set(args.tasks)]
    subset_keys_by_task = {
        task: {key_for(row) for row in equi_rows if row.get("task") == task}
        for task in args.tasks
    }

    metric_rows = []
    for task in args.tasks:
        task_equi = [row for row in equi_rows if row.get("task") == task]
        metric_rows.append(metrics(task, "equi_agent_live", task_equi))

        for model in args.models:
            path = model_file(args.predictions_root, task, model)
            if path is None:
                continue
            rows = [
                row
                for row in read_csv(path)
                if row.get("task") == task
                and row.get("split") == "test"
                and key_for(row) in subset_keys_by_task[task]
            ]
            if len(rows) != len(task_equi):
                print(f"warning: {task} {model} matched {len(rows)} rows, expected {len(task_equi)}")
            metric_rows.append(metrics(task, model, rows))

    for task in args.tasks:
        print(f"\n{task}")
        task_rows = [row for row in metric_rows if row["task"] == task]
        task_rows.sort(key=lambda row: float(row["f1"]), reverse=True)
        for row in task_rows:
            print(
                f"{row['model_name']:24s} "
                f"f1={row['f1']:.3f} acc={row['accuracy']:.3f} "
                f"recall={row['recall']:.3f} spec={row['specificity']:.3f} "
                f"tp={row['tp']} fp={row['fp']} fn={row['fn']} tn={row['tn']}"
            )

    if args.out is not None:
        write_csv(args.out, metric_rows)
        print(f"\nwrote={args.out}")


if __name__ == "__main__":
    main()
