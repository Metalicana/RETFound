from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def int_label(value: object, default: int | None = None) -> int | None:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(rows: list[dict[str, str]], pred_col: str) -> dict[str, object]:
    valid = []
    for row in rows:
        truth = int_label(row.get("y_true"))
        pred = int_label(row.get(pred_col))
        if truth in {0, 1} and pred in {0, 1}:
            valid.append((truth, pred))

    confusion = Counter(valid)
    correct = sum(1 for truth, pred in valid if truth == pred)
    n = len(valid)
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "tn": confusion[(0, 0)],
        "fp": confusion[(0, 1)],
        "fn": confusion[(1, 0)],
        "tp": confusion[(1, 1)],
    }


def print_metrics(title: str, rows: list[dict[str, str]], pred_col: str) -> None:
    print(f"\n{title} ({pred_col})")
    overall = metrics(rows, pred_col)
    print(
        "overall "
        f"n={overall['n']} acc={overall['accuracy'] if overall['accuracy'] is not None else ''} "
        f"tn={overall['tn']} fp={overall['fp']} fn={overall['fn']} tp={overall['tp']}"
    )

    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task", "")].append(row)
    for task in sorted(by_task):
        item = metrics(by_task[task], pred_col)
        print(
            f"{task:10s} "
            f"n={item['n']:3d} acc={item['accuracy'] if item['accuracy'] is not None else ''} "
            f"tn={item['tn']:3d} fp={item['fp']:3d} fn={item['fn']:3d} tp={item['tp']:3d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare live Equi-Agent predictions against the validation-neighbor "
            "few-shot recommendation columns written by run_equi_agent_fairvision_live.py."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    print(f"file: {args.predictions}")
    print(f"rows: {len(rows)}")

    print_metrics("Agent final labels", rows, "y_pred")
    if rows and "few_shot_recommended_label" in rows[0]:
        print_metrics("Few-shot recommendation labels", rows, "few_shot_recommended_label")

    if rows and "calibration_action" in rows[0]:
        print("\nAgent calibration actions")
        by_task_action: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            by_task_action[row.get("task", "")][row.get("calibration_action", "")] += 1
        for task in sorted(by_task_action):
            print(task, dict(sorted(by_task_action[task].items())))

    if rows and "few_shot_recommended_action" in rows[0]:
        print("\nFew-shot recommended actions")
        by_task_action = defaultdict(Counter)
        for row in rows:
            by_task_action[row.get("task", "")][row.get("few_shot_recommended_action", "")] += 1
        for task in sorted(by_task_action):
            print(task, dict(sorted(by_task_action[task].items())))


if __name__ == "__main__":
    main()
