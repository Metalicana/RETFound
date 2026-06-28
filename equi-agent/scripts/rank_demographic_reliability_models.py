from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from compute_demographic_reliability_score import (
    DEFAULT_PRIORS_JSON,
    DEFAULT_SUPPORT_CSV,
    DEFAULT_AUROC_WEIGHT,
    DEFAULT_ECE_WEIGHT,
    DEFAULT_F1_WEIGHT,
    DEFAULT_FNR_WEIGHT,
    DEFAULT_FPR_WEIGHT,
    finite,
    formula_string,
    load_json_rows,
    norm,
    norm_gender,
    norm_race,
    norm_task,
    prior_lookup,
    read_csv,
    require_prior,
    risk_score,
)


FOUNDATION_MODELS = {
    "flair_slo",
    "mirage_slo",
    "ret_clip_slo",
    "retfound_oct",
    "retizero_slo",
    "urfound_oct",
    "urfound_slo",
    "visionfm_oct",
    "visionfm_slo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank models by demographic reliability risk for every observed "
            "task x age_group x race x gender subgroup. Lower score is better."
        )
    )
    parser.add_argument("--priors-json", type=Path, default=DEFAULT_PRIORS_JSON)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/metrics/demographic_reliability_rankings"))
    parser.add_argument("--model-set", choices=["all", "foundation"], default="foundation")
    parser.add_argument("--task", help="Optional task filter: amd, dr, or glaucoma.")
    parser.add_argument("--k", type=float, default=50.0)
    parser.add_argument("--fnr-weight", type=float, default=DEFAULT_FNR_WEIGHT)
    parser.add_argument("--fpr-weight", type=float, default=DEFAULT_FPR_WEIGHT)
    parser.add_argument("--ece-weight", type=float, default=DEFAULT_ECE_WEIGHT)
    parser.add_argument("--auroc-weight", type=float, default=DEFAULT_AUROC_WEIGHT)
    parser.add_argument("--f1-weight", type=float, default=DEFAULT_F1_WEIGHT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def available_models(prior_rows: list[dict[str, Any]], model_set: str) -> list[str]:
    models = sorted(
        {
            norm(row.get("model_name"))
            for row in prior_rows
            if norm(row.get("attribute")) == "global" and norm(row.get("subgroup")) == "global"
        }
    )
    if model_set == "foundation":
        models = [model for model in models if model in FOUNDATION_MODELS]
    return models


def support_tables(support_rows: list[dict[str, str]], task_filter: str | None) -> dict[str, Any]:
    rows = []
    for row in support_rows:
        task = norm(row.get("task"))
        if task_filter is not None and task != task_filter:
            continue
        age_group = norm(row.get("age_group"))
        race = norm_race(row.get("race", ""))
        gender = norm_gender(row.get("sex_gender", ""))
        if not task or not age_group or not race or not gender:
            continue
        rows.append({**row, "_task": task, "_age_group": age_group, "_race": race, "_gender": gender})

    n_total = Counter(row["_task"] for row in rows)
    age_counts = Counter((row["_task"], row["_age_group"]) for row in rows)
    race_counts = Counter((row["_task"], row["_race"]) for row in rows)
    gender_counts = Counter((row["_task"], row["_gender"]) for row in rows)
    combo_counts = Counter((row["_task"], row["_age_group"], row["_race"], row["_gender"]) for row in rows)
    return {
        "n_total": n_total,
        "age_counts": age_counts,
        "race_counts": race_counts,
        "gender_counts": gender_counts,
        "combo_counts": combo_counts,
    }


def score_model_for_combo(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    support: dict[str, Any],
    task: str,
    model: str,
    age_group: str,
    race: str,
    gender: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    global_row = require_prior(lookup, task, model, "GLOBAL", "GLOBAL")
    age_row = require_prior(lookup, task, model, "age_group", age_group)
    race_row = require_prior(lookup, task, model, "race", race)
    gender_row = require_prior(lookup, task, model, "sex_gender", gender)

    n_total = support["n_total"][task]
    age_n = support["age_counts"][(task, age_group)]
    race_n = support["race_counts"][(task, race)]
    gender_n = support["gender_counts"][(task, gender)]
    intersection_n = support["combo_counts"][(task, age_group, race, gender)]

    lambda_age = age_n / n_total
    lambda_race = race_n / n_total
    lambda_gender = gender_n / n_total
    lambda_sum = lambda_age + lambda_race + lambda_gender
    intersection_lambda = intersection_n / (intersection_n + args.k) if intersection_n > 0 and args.k >= 0 else 0.0

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
    r_local_raw = lambda_age * r_age + lambda_race * r_race + lambda_gender * r_gender
    r_local = r_local_raw / lambda_sum if lambda_sum else math.nan
    r_final = intersection_lambda * r_local + (1.0 - intersection_lambda) * r_global

    return {
        "task": task,
        "age_group": age_group,
        "race": race,
        "gender": gender,
        "model_name": model,
        "score": r_final,
        "score_formula": formula_string(args),
        "R_global": r_global,
        "R_age": r_age,
        "R_race": r_race,
        "R_gender": r_gender,
        "R_local": r_local,
        "R_local_raw": r_local_raw,
        "lambda_age": lambda_age,
        "lambda_race": lambda_race,
        "lambda_gender": lambda_gender,
        "lambda_sum": lambda_sum,
        "intersection_lambda": intersection_lambda,
        "n_total": n_total,
        "age_n": age_n,
        "race_n": race_n,
        "gender_n": gender_n,
        "intersection_n": intersection_n,
        "global_fnr": global_row.get("fnr"),
        "global_fpr": global_row.get("fpr"),
        "global_ece": global_row.get("ece"),
        "global_auroc": global_row.get("auroc"),
        "global_f1": global_row.get("f1"),
        "age_fnr": age_row.get("fnr"),
        "age_fpr": age_row.get("fpr"),
        "age_ece": age_row.get("ece"),
        "age_auroc": age_row.get("auroc"),
        "age_f1": age_row.get("f1"),
        "race_fnr": race_row.get("fnr"),
        "race_fpr": race_row.get("fpr"),
        "race_ece": race_row.get("ece"),
        "race_auroc": race_row.get("auroc"),
        "race_f1": race_row.get("f1"),
        "gender_fnr": gender_row.get("fnr"),
        "gender_fpr": gender_row.get("fpr"),
        "gender_ece": gender_row.get("ece"),
        "gender_auroc": gender_row.get("auroc"),
        "gender_f1": gender_row.get("f1"),
    }


def fmt_score(value: Any) -> str:
    number = finite(value)
    return "" if math.isnan(number) else f"{number:.4f}"


def main() -> None:
    args = parse_args()
    task_filter = norm_task(args.task) if args.task else None
    prior_rows = load_json_rows(args.priors_json)
    lookup = prior_lookup(prior_rows)
    models = available_models(prior_rows, args.model_set)
    if not models:
        raise SystemExit(f"No models found for --model-set {args.model_set}")

    support = support_tables(read_csv(args.support_csv), task_filter)
    combo_keys = sorted(support["combo_counts"])

    long_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    winner_counts: Counter[tuple[str, str]] = Counter()
    runner_up_counts: Counter[tuple[str, str]] = Counter()

    for task, age_group, race, gender in combo_keys:
        model_scores = [
            score_model_for_combo(lookup, support, task, model, age_group, race, gender, args)
            for model in models
        ]
        model_scores.sort(key=lambda row: (finite(row["score"]), row["model_name"]))
        for rank, row in enumerate(model_scores, start=1):
            long_rows.append({"rank": rank, **row})

        winner = model_scores[0]
        runner_up = model_scores[1] if len(model_scores) > 1 else None
        winner_counts[(task, winner["model_name"])] += 1
        if runner_up is not None:
            runner_up_counts[(task, runner_up["model_name"])] += 1
        ranking_rows.append(
            {
                "task": task,
                "age_group": age_group,
                "race": race,
                "gender": gender,
                "intersection_n": winner["intersection_n"],
                "n_total": winner["n_total"],
                "intersection_lambda": winner["intersection_lambda"],
                "score_formula": formula_string(args),
                "winner_model": winner["model_name"],
                "winner_score": winner["score"],
                "runner_up_model": runner_up["model_name"] if runner_up else "",
                "runner_up_score": runner_up["score"] if runner_up else "",
                "score_gap": (runner_up["score"] - winner["score"]) if runner_up else "",
                "ranked_models": ";".join(f"{row['model_name']}={fmt_score(row['score'])}" for row in model_scores),
            }
        )

    count_rows = []
    for task in sorted({key[0] for key in winner_counts}):
        for model in models:
            count_rows.append(
                {
                    "task": task,
                    "model_name": model,
                    "winner_count": winner_counts[(task, model)],
                    "runner_up_count": runner_up_counts[(task, model)],
                }
            )

    suffix = args.model_set
    ranking_path = args.out_dir / f"demographic_reliability_winners_{suffix}.csv"
    long_path = args.out_dir / f"demographic_reliability_scores_long_{suffix}.csv"
    counts_path = args.out_dir / f"demographic_reliability_winner_counts_{suffix}.csv"
    write_csv(ranking_path, ranking_rows)
    write_csv(long_path, long_rows)
    write_csv(counts_path, count_rows)

    print(f"model_set={args.model_set}")
    print(f"formula={formula_string(args)}")
    print("direction=lower score is better")
    print(f"models={len(models)}: {', '.join(models)}")
    print(f"subgroup_combos={len(ranking_rows)}")
    print(f"wrote={ranking_path}")
    print(f"wrote={long_path}")
    print(f"wrote={counts_path}")
    print("\nWinner counts")
    for row in count_rows:
        if row["winner_count"] or row["runner_up_count"]:
            print(
                f"{row['task']:8s} {row['model_name']:20s} "
                f"wins={row['winner_count']:2d} runner_up={row['runner_up_count']:2d}"
            )
    print("\nFirst 12 subgroup winners")
    for row in ranking_rows[:12]:
        print(
            f"{row['task']:8s} {row['age_group']:11s} {row['race']:6s} {row['gender']:6s} "
            f"n={row['intersection_n']:3d} winner={row['winner_model']} "
            f"{fmt_score(row['winner_score'])} runner={row['runner_up_model']} "
            f"{fmt_score(row['runner_up_score'])}"
        )


if __name__ == "__main__":
    main()
