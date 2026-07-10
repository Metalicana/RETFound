from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PRIORS_JSON = Path("equi-agent/outputs/metrics/validation_subgroup_priors.json")
DEFAULT_SUPPORT_CSV = Path("equi-agent/outputs/fairvision_reliability_selective_arbitration/selective_arbitration_predictions.csv")
DEFAULT_SUBGROUP_RELIABILITY_CSV = Path("equi-agent/outputs/metrics/demographic_reliability_subgroup_model_scores.csv")
DEFAULT_FNR_WEIGHT = 0.35
DEFAULT_FPR_WEIGHT = 0.25
DEFAULT_ECE_WEIGHT = 0.15
DEFAULT_AUROC_WEIGHT = 0.15
DEFAULT_F1_WEIGHT = 0.10


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
            "Compute demographic reliability risk from model performance priors. "
            "Higher score means worse expected reliability."
        )
    )
    parser.add_argument("--priors-json", type=Path, default=DEFAULT_PRIORS_JSON)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument(
        "--subgroup-reliability-csv",
        type=Path,
        help=(
            "Optional precomputed task x age_group x race x gender x model score table. "
            "When supplied, the script looks up the requested row instead of recomputing from priors."
        ),
    )
    parser.add_argument("--score-column", default="final_R_bad")
    parser.add_argument("--task", required=True, help="amd, dr, or glaucoma")
    parser.add_argument("--model", required=True, help="Exact model_name, e.g. retfound_oct")
    parser.add_argument("--age-group", required=True, help="younger, middle-aged, or older")
    parser.add_argument("--race", required=True, help="white/caucasian, black, asian, etc.")
    parser.add_argument("--gender", required=True, help="male or female")
    parser.add_argument("--k", type=float, default=50.0, help="Intersectional shrinkage parameter.")
    parser.add_argument("--fnr-weight", type=float, default=DEFAULT_FNR_WEIGHT)
    parser.add_argument("--fpr-weight", type=float, default=DEFAULT_FPR_WEIGHT)
    parser.add_argument("--ece-weight", type=float, default=DEFAULT_ECE_WEIGHT)
    parser.add_argument("--auroc-weight", type=float, default=DEFAULT_AUROC_WEIGHT)
    parser.add_argument("--f1-weight", type=float, default=DEFAULT_F1_WEIGHT)
    parser.add_argument(
        "--raw-local-lambdas",
        action="store_true",
        help="Use the unnormalized age/race/gender weighted sum for R_local.",
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


def row_gender(row: dict[str, Any]) -> str:
    return norm_gender(str(row.get("gender", row.get("sex_gender", ""))))


def lookup_precomputed_score(
    rows: list[dict[str, str]],
    task: str,
    model: str,
    age_group: str,
    race: str,
    gender: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if norm(row.get("task")) == task
        and norm(row.get("model_name")) == model
        and norm(row.get("age_group")) == age_group
        and norm_race(str(row.get("race", ""))) == race
        and row_gender(row) == gender
    ]
    if matches:
        return matches[0]

    available_models = sorted({norm(row.get("model_name")) for row in rows if norm(row.get("task")) == task})
    available_subgroups = sorted(
        {
            (
                norm(row.get("age_group")),
                norm_race(str(row.get("race", ""))),
                row_gender(row),
            )
            for row in rows
            if norm(row.get("task")) == task and norm(row.get("model_name")) == model
        }
    )
    raise SystemExit(
        "No precomputed subgroup reliability row found for "
        f"task={task}, model={model}, age_group={age_group}, race={race}, gender={gender}.\n"
        f"Available models for task={task}: {available_models}\n"
        f"Available subgroups for this task/model: {available_subgroups[:20]}"
    )


def precomputed_result(
    row: dict[str, str],
    args: argparse.Namespace,
    task: str,
    model: str,
    age_group: str,
    race: str,
    gender: str,
) -> dict[str, Any]:
    if args.score_column not in row:
        raise SystemExit(
            f"Score column '{args.score_column}' not found in {args.subgroup_reliability_csv}. "
            f"Available columns: {list(row.keys())}"
        )
    final_r_bad = finite(row.get(args.score_column))
    if math.isnan(final_r_bad):
        raise SystemExit(
            f"Score column '{args.score_column}' is not numeric for "
            f"task={task}, model={model}, age_group={age_group}, race={race}, gender={gender}"
        )
    return {
        "task": task,
        "model_name": model,
        "age_group": age_group,
        "race": race,
        "gender": gender,
        "source": str(args.subgroup_reliability_csv),
        "score_column": args.score_column,
        "score_direction": row.get("score_direction", "lower_is_better"),
        "score_formula": row.get("score_formula", ""),
        "scores": {
            "final_R_bad": final_r_bad,
            "arbitration_reliability_score": 1.0 - final_r_bad,
            "R_global_bad": finite(row.get("R_global_bad")),
            "R_local_bad": finite(row.get("R_local_bad")),
            "R_age_bad": finite(row.get("R_age_bad")),
            "R_race_bad": finite(row.get("R_race_bad")),
            "R_gender_bad": finite(row.get("R_gender_bad")),
        },
        "rank": {
            "rank": finite(row.get("rank")),
            "is_winner": int(finite(row.get("is_winner"), 0)),
            "is_runner_up": int(finite(row.get("is_runner_up"), 0)),
            "model_set": row.get("model_set", ""),
        },
        "support": {
            "n_total": finite(row.get("n_total")),
            "age_n": finite(row.get("age_n")),
            "race_n": finite(row.get("race_n")),
            "gender_n": finite(row.get("gender_n")),
            "intersection_n": finite(row.get("intersection_n")),
            "lambda_age": finite(row.get("lambda_age")),
            "lambda_race": finite(row.get("lambda_race")),
            "lambda_gender": finite(row.get("lambda_gender")),
            "lambda_sum": finite(row.get("lambda_sum")),
            "intersection_lambda": finite(row.get("intersection_lambda")),
        },
    }


def print_precomputed_result(result: dict[str, Any]) -> None:
    scores = result["scores"]
    support = result["support"]
    rank = result["rank"]
#    print(
#        "task={task} model={model_name} age_group={age_group} race={race} gender={gender}".format(**result)
#    )
#    print(f"source={result['source']}")
#    print(f"formula={result['score_formula']}")
#    print(f"direction={result['score_direction']}; lower final_R_bad is better")
#    print(f"arbitration_reliability_score=1-final_R_bad={scores['arbitration_reliability_score']:.6f}")
#    print(f"rank={rank['rank']:.0f}  is_winner={rank['is_winner']}  is_runner_up={rank['is_runner_up']}  model_set={rank['model_set']}")
#    print(f"R_global_bad={scores['R_global_bad']:.6f}")
#    print(f"R_local_bad={scores['R_local_bad']:.6f}")
#    print(f"R_age_bad={scores['R_age_bad']:.6f}  age_n={support['age_n']:.0f}/{support['n_total']:.0f}")
#    print(f"R_race_bad={scores['R_race_bad']:.6f}  race_n={support['race_n']:.0f}/{support['n_total']:.0f}")
#    print(f"R_gender_bad={scores['R_gender_bad']:.6f}  gender_n={support['gender_n']:.0f}/{support['n_total']:.0f}")
#    print(f"intersection_n={support['intersection_n']:.0f}  lambda={support['intersection_lambda']:.6f}")
#    print(f"final_R_bad={scores['final_R_bad']:.6f}")

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


def metric_value(row: dict[str, Any], fallback_row: dict[str, Any] | None, metric: str, default: float) -> float:
    value = finite(row.get(metric))
    if not math.isnan(value):
        return value
    if fallback_row is not None and fallback_row is not row:
        fallback = finite(fallback_row.get(metric))
        if not math.isnan(fallback):
            return fallback
    return default


def risk_components(row: dict[str, Any], fallback_row: dict[str, Any] | None = None) -> dict[str, float]:
    f1 = metric_value(row, fallback_row, "f1", 0.0)
    auroc = metric_value(row, fallback_row, "auroc", 0.5)
    return {
        "fnr": metric_value(row, fallback_row, "fnr", 0.5),
        "fpr": metric_value(row, fallback_row, "fpr", 0.5),
        "ece": metric_value(row, fallback_row, "ece", 0.5),
        "one_minus_auroc": 1.0 - auroc,
        "one_minus_f1": 1.0 - f1,
    }


def risk_score(
    row: dict[str, Any],
    fallback_row: dict[str, Any] | None,
    fnr_weight: float,
    fpr_weight: float,
    ece_weight: float,
    auroc_weight: float,
    f1_weight: float,
) -> float:
    components = risk_components(row, fallback_row)
    return (
        fnr_weight * components["fnr"]
        + fpr_weight * components["fpr"]
        + ece_weight * components["ece"]
        + auroc_weight * components["one_minus_auroc"]
        + f1_weight * components["one_minus_f1"]
    )


def formula_string(args: argparse.Namespace) -> str:
    return (
        f"{args.fnr_weight}*FNR + {args.fpr_weight}*FPR + {args.ece_weight}*ECE + "
        f"{args.auroc_weight}*(1-AUROC) + {args.f1_weight}*(1-F1)"
    )


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


def main(args=None) -> None:
    if args is None:
      args = parse_args()
    task = norm_task(args.task)
    model = norm(args.model)
    age_group = norm(args.age_group)
    race = norm_race(args.race)
    gender = norm_gender(args.gender)

    if args.subgroup_reliability_csv is not None:
        row = lookup_precomputed_score(
            read_csv(args.subgroup_reliability_csv),
            task,
            model,
            age_group,
            race,
            gender,
        )
        result = precomputed_result(row, args, task, model, age_group, race, gender)
        val = result["scores"]["final_R_bad"]
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_precomputed_result(result)
        return round(val,4)

    priors = prior_lookup(load_json_rows(args.priors_json))
    support = support_counts(read_csv(args.support_csv), task, age_group, race, gender)

    global_row = require_prior(priors, task, model, "GLOBAL", "GLOBAL")
    age_row = require_prior(priors, task, model, "age_group", age_group)
    race_row = require_prior(priors, task, model, "race", race)
    gender_row = require_prior(priors, task, model, "sex_gender", gender)

    r_global = risk_score(
        global_row,
        None,
        args.fnr_weight,
        args.fpr_weight,
        args.ece_weight,
        args.auroc_weight,
        args.f1_weight,
    )
    r_age = risk_score(
        age_row,
        global_row,
        args.fnr_weight,
        args.fpr_weight,
        args.ece_weight,
        args.auroc_weight,
        args.f1_weight,
    )
    r_race = risk_score(
        race_row,
        global_row,
        args.fnr_weight,
        args.fpr_weight,
        args.ece_weight,
        args.auroc_weight,
        args.f1_weight,
    )
    r_gender = risk_score(
        gender_row,
        global_row,
        args.fnr_weight,
        args.fpr_weight,
        args.ece_weight,
        args.auroc_weight,
        args.f1_weight,
    )

    lambda_age = support["lambda_age"]
    lambda_race = support["lambda_race"]
    lambda_gender = support["lambda_gender"]
    lambda_sum = lambda_age + lambda_race + lambda_gender
    raw_local = lambda_age * r_age + lambda_race * r_race + lambda_gender * r_gender
    normalized_local = raw_local / lambda_sum if lambda_sum else math.nan
    r_local = raw_local if args.raw_local_lambdas else normalized_local

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
            "component_score": formula_string(args),
            "direction": "higher is worse; lower is better",
            "missing_component_rule": "undefined subgroup F1/AUROC/ECE/FNR/FPR fall back to the model's global metric for the same task",
            "r_local": (
                "lambda-normalized weighted mean of R_age, R_race, and R_gender"
                if not args.raw_local_lambdas
                else "lambda_age*R_age + lambda_race*R_race + lambda_gender*R_gender"
            ),
            "r_final": "lambda*R_local + (1-lambda)*R_global",
            "local_lambdas_normalized": not bool(args.raw_local_lambdas),
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
                "ece": global_row.get("ece"),
                "auroc": global_row.get("auroc"),
                "f1": global_row.get("f1"),
            },
            "age": {
                "n": age_row.get("n"),
                "fnr": age_row.get("fnr"),
                "fpr": age_row.get("fpr"),
                "ece": age_row.get("ece"),
                "auroc": age_row.get("auroc"),
                "f1": age_row.get("f1"),
                "unstable": age_row.get("unstable"),
            },
            "race": {
                "n": race_row.get("n"),
                "fnr": race_row.get("fnr"),
                "fpr": race_row.get("fpr"),
                "ece": race_row.get("ece"),
                "auroc": race_row.get("auroc"),
                "f1": race_row.get("f1"),
                "unstable": race_row.get("unstable"),
            },
            "gender": {
                "n": gender_row.get("n"),
                "fnr": gender_row.get("fnr"),
                "fpr": gender_row.get("fpr"),
                "ece": gender_row.get("ece"),
                "auroc": gender_row.get("auroc"),
                "f1": gender_row.get("f1"),
                "unstable": gender_row.get("unstable"),
            },
        },
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"task={task} model={model} age_group={age_group} race={race} gender={gender}")
    print(f"formula={formula_string(args)}")
    print("direction=higher risk is worse; lower is better")
    print(f"R_global={r_global:.6f}")
    print(f"R_age={r_age:.6f}  lambda_age={lambda_age:.6f}  age_n={support['age_n']}/{support['n_total']}")
    print(f"R_race={r_race:.6f}  lambda_race={lambda_race:.6f}  race_n={support['race_n']}/{support['n_total']}")
    print(f"R_gender={r_gender:.6f}  lambda_gender={lambda_gender:.6f}  gender_n={support['gender_n']}/{support['n_total']}")
    print(f"R_local_raw={raw_local:.6f}  lambda_sum={lambda_sum:.6f}")
    print(f"R_local_normalized={normalized_local:.6f}")
    print(f"intersection_n={intersection_n}  k={args.k:g}  lambda={shrinkage_lambda:.6f}")
    print(f"R_final={r_final:.6f}")
    if args.raw_local_lambdas:
        print("note=R_final uses raw unnormalized local lambdas.")
    else:
        print("note=R_final uses normalized local lambdas by default.")

if __name__ == "__main__":
    main()
