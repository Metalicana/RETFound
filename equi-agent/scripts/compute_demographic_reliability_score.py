from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PRIORS_JSON = Path("equi-agent/outputs/metrics/validation_subgroup_priors.json")
DEFAULT_SUPPORT_CSV = Path("equi-agent/outputs/fairvision_reliability_selective_arbitration/selective_arbitration_predictions.csv")


TASK_ALIASES = {
    "amd": "amd",
    "dr": "dr",
    "diabetic_retinopathy": "dr",
    "diabetic retinopathy": "dr",
    "glaucoma": "glaucoma",
}
RACE_ALIASES = {
    "caucasian": "white",
    "white": "white",
    "black": "black",
    "asian": "asian",
    "hispanic": "hispanic",
}
GENDER_ALIASES = {
    "gender": "sex_gender",
    "sex": "sex_gender",
    "sex_gender": "sex_gender",
    "male": "male",
    "female": "female",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute demographic reliability risk from model FNR/FPR priors. "
            "Higher score means worse expected reliability."
        )
    )
    parser.add_argument("--priors-json", type=Path, default=DEFAULT_PRIORS_JSON)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--task", required=True, help="amd, dr, or glaucoma")
    parser.add_argument("--model", required=True, help="Exact model_name, e.g. retfound_oct")
    parser.add_argument("--age-group", required=True, help="younger, middle-aged, or older")
    parser.add_argument("--race", required=True, help="white/caucasian, black, asian, etc.")
    parser.add_argument("--gender", required=True, help="male or female")
    parser.add_argument("--k", type=float, default=50.0, help="Intersectional shrinkage parameter.")
    parser.add_argument("--fnr-weight", type=float, default=0.85)
    parser.add_argument("--fpr-weight", type=float, default=0.15)
    parser.add_argument(
        "--normalize-local-lambdas",
        action="store_true",
        help=(
            "Normalize age/race/gender support weights before computing R_local. "
            "Default follows the requested raw weighted sum."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact text report.")
    return parser.parse_args()


def norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def norm_task(value: str) -> str:
    key = norm(value)
    if key not in TASK_ALIASES:
        raise SystemExit(f"Unknown task '{value}'. Expected one of: {sorted(TASK_ALIASES)}")
    return TASK_ALIASES[key]


def norm_race(value: str) -> str:
    key = norm(value)
    return RACE_ALIASES.get(key, key)


def norm_gender(value: str) -> str:
    key = norm(value)
    return GENDER_ALIASES.get(key, key)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"Expected list-of-rows JSON at {path}")
    return [row for row in data if isinstance(row, dict)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def prior_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup = {}
    for row in rows:
        lookup[
            (
                norm(row.get("task")),
                norm(row.get("model_name")),
                norm(row.get("attribute")),
                norm(row.get("subgroup")),
            )
        ] = row
    return lookup


def require_prior(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    task: str,
    model: str,
    attribute: str,
    subgroup: str,
) -> dict[str, Any]:
    key = (norm(task), norm(model), norm(attribute), norm(subgroup))
    row = lookup.get(key)
    if row is None:
        available = sorted(
            {
                candidate[3]
                for candidate in lookup
                if candidate[0] == norm(task) and candidate[1] == norm(model) and candidate[2] == norm(attribute)
            }
        )
        raise SystemExit(
            f"Missing prior for task={task}, model={model}, attribute={attribute}, subgroup={subgroup}. "
            f"Available subgroups for this attribute: {available}"
        )
    return row


def error_score(row: dict[str, Any], fnr_weight: float, fpr_weight: float) -> float:
    return fnr_weight * finite(row.get("fnr"), 0.5) + fpr_weight * finite(row.get("fpr"), 0.5)


def support_counts(rows: list[dict[str, str]], task: str, age_group: str, race: str, gender: str) -> dict[str, Any]:
    task_rows = [row for row in rows if norm(row.get("task")) == task]
    if not task_rows:
        raise SystemExit(f"No support rows found for task={task}")
    n_total = len(task_rows)
    age_n = sum(norm(row.get("age_group")) == age_group for row in task_rows)
    race_n = sum(norm_race(row.get("race", "")) == race for row in task_rows)
    gender_n = sum(norm_gender(row.get("sex_gender", "")) == gender for row in task_rows)
    intersection_n = sum(
        norm(row.get("age_group")) == age_group
        and norm_race(row.get("race", "")) == race
        and norm_gender(row.get("sex_gender", "")) == gender
        for row in task_rows
    )
    return {
        "n_total": n_total,
        "age_n": age_n,
        "race_n": race_n,
        "gender_n": gender_n,
        "intersection_n": intersection_n,
        "lambda_age": age_n / n_total,
        "lambda_race": race_n / n_total,
        "lambda_gender": gender_n / n_total,
    }


def main() -> None:
    args = parse_args()
    task = norm_task(args.task)
    model = norm(args.model)
    age_group = norm(args.age_group)
    race = norm_race(args.race)
    gender = norm_gender(args.gender)

    priors = prior_lookup(load_json_rows(args.priors_json))
    support = support_counts(read_csv(args.support_csv), task, age_group, race, gender)

    global_row = require_prior(priors, task, model, "GLOBAL", "GLOBAL")
    age_row = require_prior(priors, task, model, "age_group", age_group)
    race_row = require_prior(priors, task, model, "race", race)
    gender_row = require_prior(priors, task, model, "sex_gender", gender)

    r_global = error_score(global_row, args.fnr_weight, args.fpr_weight)
    r_age = error_score(age_row, args.fnr_weight, args.fpr_weight)
    r_race = error_score(race_row, args.fnr_weight, args.fpr_weight)
    r_gender = error_score(gender_row, args.fnr_weight, args.fpr_weight)

    lambda_age = support["lambda_age"]
    lambda_race = support["lambda_race"]
    lambda_gender = support["lambda_gender"]
    lambda_sum = lambda_age + lambda_race + lambda_gender
    raw_local = lambda_age * r_age + lambda_race * r_race + lambda_gender * r_gender
    normalized_local = raw_local / lambda_sum if lambda_sum else math.nan
    r_local = normalized_local if args.normalize_local_lambdas else raw_local

    intersection_n = support["intersection_n"]
    shrinkage_lambda = intersection_n / (intersection_n + args.k) if intersection_n > 0 and args.k >= 0 else 0.0
    r_final = shrinkage_lambda * r_local + (1.0 - shrinkage_lambda) * r_global

    result = {
        "task": task,
        "model_name": model,
        "age_group": age_group,
        "race": race,
        "gender": gender,
        "formula": {
            "component_score": f"{args.fnr_weight}*FNR + {args.fpr_weight}*FPR",
            "r_local": "lambda_age*R_age + lambda_race*R_race + lambda_gender*R_gender",
            "r_final": "lambda*R_local + (1-lambda)*R_global",
            "local_lambdas_normalized": bool(args.normalize_local_lambdas),
        },
        "scores": {
            "R_global": r_global,
            "R_age": r_age,
            "R_race": r_race,
            "R_gender": r_gender,
            "R_local": r_local,
            "R_local_raw": raw_local,
            "R_local_normalized": normalized_local,
            "R_final": r_final,
        },
        "support": {
            **support,
            "lambda_sum": lambda_sum,
            "intersection_lambda": shrinkage_lambda,
            "k": args.k,
            "support_source": str(args.support_csv),
        },
        "prior_rows": {
            "global": {
                "n": global_row.get("n"),
                "fnr": global_row.get("fnr"),
                "fpr": global_row.get("fpr"),
            },
            "age": {
                "n": age_row.get("n"),
                "fnr": age_row.get("fnr"),
                "fpr": age_row.get("fpr"),
                "unstable": age_row.get("unstable"),
            },
            "race": {
                "n": race_row.get("n"),
                "fnr": race_row.get("fnr"),
                "fpr": race_row.get("fpr"),
                "unstable": race_row.get("unstable"),
            },
            "gender": {
                "n": gender_row.get("n"),
                "fnr": gender_row.get("fnr"),
                "fpr": gender_row.get("fpr"),
                "unstable": gender_row.get("unstable"),
            },
        },
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"task={task} model={model} age_group={age_group} race={race} gender={gender}")
    print(f"R_global={r_global:.6f}")
    print(f"R_age={r_age:.6f}  lambda_age={lambda_age:.6f}  age_n={support['age_n']}/{support['n_total']}")
    print(f"R_race={r_race:.6f}  lambda_race={lambda_race:.6f}  race_n={support['race_n']}/{support['n_total']}")
    print(f"R_gender={r_gender:.6f}  lambda_gender={lambda_gender:.6f}  gender_n={support['gender_n']}/{support['n_total']}")
    print(f"R_local_raw={raw_local:.6f}  lambda_sum={lambda_sum:.6f}")
    print(f"R_local_normalized={normalized_local:.6f}")
    print(f"intersection_n={intersection_n}  k={args.k:g}  lambda={shrinkage_lambda:.6f}")
    print(f"R_final={r_final:.6f}")
    if not args.normalize_local_lambdas:
        print("note=R_final uses raw unnormalized local lambdas, exactly as requested.")


if __name__ == "__main__":
    main()
