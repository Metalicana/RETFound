from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


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
CASE_KEY = ("patient_id", "eye_id", "visit_id", "image_id", "task")
OUTPUT_TOKEN_BUDGET = {
    "bio_profiler": 120,
    "vision_specialist": 180,
    "equity_auditor": 220,
    "orchestrator": 220,
    "safety_agent": 140,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run Equi-Agent arbitration over existing FairVision prediction CSVs. "
            "Writes mock arbitration outputs plus prompt/token/cost estimates."
        )
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument("--metrics-root", type=Path, default=Path("equi-agent/outputs/metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/agent_smoke"))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-cases-per-task", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset into the shared case list.")
    parser.add_argument(
        "--input-usd-per-1m",
        type=float,
        default=0.0,
        help="Azure/OpenAI input-token price per 1M tokens. Defaults to 0; set your deployment price.",
    )
    parser.add_argument(
        "--output-usd-per-1m",
        type=float,
        default=0.0,
        help="Azure/OpenAI output-token price per 1M tokens. Defaults to 0; set your deployment price.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Heuristic tokenizer estimate. Use 4 chars/token unless you later swap in tiktoken.",
    )
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def key_for(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return tuple(row.get(col, "") for col in CASE_KEY)


def load_predictions(predictions_root: Path, tasks: list[str], models: list[str]):
    by_task: dict[str, dict[str, dict[tuple, dict]]] = {task: {} for task in tasks}
    loaded_files = []
    missing_files = []
    for task in tasks:
        for model in models:
            path = model_file(predictions_root, task, model)
            if path is None:
                missing_files.append({"task": task, "model": model})
                continue
            rows = [row for row in read_csv(path) if row.get("task") == task and row.get("split") == "test"]
            if not rows:
                missing_files.append({"task": task, "model": model, "path": str(path), "reason": "no rows"})
                continue
            by_task[task][model] = {key_for(row): row for row in rows}
            loaded_files.append({"task": task, "model": model, "path": str(path), "rows": len(rows)})
    return by_task, loaded_files, missing_files


def load_priors(metrics_root: Path) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    subgroup_path = metrics_root / "validation_subgroup_priors.csv"
    global_path = metrics_root / "validation_subgroup_priors_global.csv"
    subgroup = {}
    global_prior = {}
    if subgroup_path.exists():
        for row in read_csv(subgroup_path):
            subgroup[
                (
                    row.get("task", ""),
                    row.get("model_name", ""),
                    row.get("attribute", ""),
                    normalize(row.get("subgroup", "")),
                )
            ] = row
    if global_path.exists():
        for row in read_csv(global_path):
            global_prior[(row.get("task", ""), row.get("model_name", ""))] = row
    return subgroup, global_prior


def normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def fnum(value: str | float | None, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def select_cases(task_predictions: dict[str, dict[tuple, dict]], limit: int, offset: int) -> list[tuple]:
    if not task_predictions:
        return []
    common = None
    for rows_by_key in task_predictions.values():
        keys = set(rows_by_key)
        common = keys if common is None else common & keys
    if not common:
        return []
    ordered = sorted(common, key=lambda key: (key[0], key[3]))
    return ordered[offset : offset + limit]


def case_metadata(rows_by_model: dict[str, dict]) -> dict:
    row = next(iter(rows_by_model.values()))
    return {
        "patient_id": row.get("patient_id", ""),
        "eye_id": row.get("eye_id", ""),
        "visit_id": row.get("visit_id", ""),
        "image_id": row.get("image_id", ""),
        "dataset": row.get("dataset", ""),
        "task": row.get("task", ""),
        "y_true": row.get("y_true", ""),
        "race": row.get("race", ""),
        "ethnicity": row.get("ethnicity", ""),
        "sex_gender": row.get("sex_gender", ""),
        "age": row.get("age", ""),
        "age_group": row.get("age_group", ""),
    }


def prior_for_case(
    task: str,
    model: str,
    meta: dict,
    subgroup_priors: dict[tuple, dict],
    global_priors: dict[tuple, dict],
) -> dict | None:
    for attr in ["race_x_age_group", "race_x_sex_gender", "sex_gender_x_age_group", "race", "age_group", "sex_gender", "ethnicity"]:
        if "_x_" in attr:
            parts = attr.split("_x_")
            subgroup = " x ".join(normalize(meta.get(part, "")) for part in parts)
        else:
            subgroup = normalize(meta.get(attr, ""))
        row = subgroup_priors.get((task, model, attr, subgroup))
        if row and normalize(row.get("unstable")) not in {"true", "1"}:
            return row
    return global_priors.get((task, model))


def model_weight(task: str, model: str, prior: dict | None) -> float:
    if not prior:
        return 0.5
    balanced = fnum(prior.get("balanced_accuracy"), 0.5)
    ece = fnum(prior.get("ece"), 0.25)
    return max(0.05, balanced * (1.0 - min(ece, 0.8)))


def mock_arbitrate(task: str, rows_by_model: dict[str, dict], meta: dict, subgroup_priors, global_priors) -> dict:
    weighted_sum = 0.0
    weight_total = 0.0
    model_rows = []
    positive_votes = 0
    for model, row in sorted(rows_by_model.items()):
        prior = prior_for_case(task, model, meta, subgroup_priors, global_priors)
        weight = model_weight(task, model, prior)
        prob = fnum(row.get("y_prob"), 0.0)
        pred = int(fnum(row.get("y_pred"), 0.0) >= 0.5)
        weighted_sum += prob * weight
        weight_total += weight
        positive_votes += pred
        model_rows.append(
            {
                "model": model,
                "prob": prob,
                "pred": pred,
                "weight": weight,
                "prior_balanced_accuracy": fnum(prior.get("balanced_accuracy") if prior else None, 0.5),
                "prior_fpr": fnum(prior.get("fpr") if prior else None, 0.0),
                "prior_fnr": fnum(prior.get("fnr") if prior else None, 0.0),
            }
        )
    final_prob = weighted_sum / weight_total if weight_total else 0.0
    final_pred = int(final_prob >= 0.5)
    disagreement = positive_votes not in {0, len(rows_by_model)}
    close_call = abs(final_prob - 0.5) < 0.1
    safety_decision = "ESCALATE_TO_HUMAN" if disagreement or close_call else "ACCEPT"
    return {
        "final_prob": final_prob,
        "final_pred": final_pred,
        "positive_votes": positive_votes,
        "num_models": len(rows_by_model),
        "disagreement": disagreement,
        "close_call": close_call,
        "safety_decision": safety_decision,
        "model_rows": model_rows,
    }


def build_prompts(meta: dict, arbitration: dict) -> dict[str, str]:
    model_lines = "\n".join(
        f"- {row['model']}: prob={row['prob']:.3f}, pred={row['pred']}, trust_weight={row['weight']:.3f}, "
        f"prior_bal_acc={row['prior_balanced_accuracy']:.3f}, prior_fpr={row['prior_fpr']:.3f}, prior_fnr={row['prior_fnr']:.3f}"
        for row in arbitration["model_rows"]
    )
    bio = (
        "You are BioProfiler. Convert patient metadata into a concise clinical context.\n"
        f"Patient: race={meta['race']}, ethnicity={meta['ethnicity']}, sex/gender={meta['sex_gender']}, "
        f"age={meta['age']}, age_group={meta['age_group']}, task={meta['task']}."
    )
    vision = (
        "You are Vision Specialist. Review model probabilities and summarize structural evidence reliability.\n"
        f"Model outputs:\n{model_lines}"
    )
    equity = (
        "You are Equity Auditor. Translate validation-derived subgroup priors into sensitivity or precision advice.\n"
        f"Patient context: {json.dumps(meta, sort_keys=True)}\nModel/prior summary:\n{model_lines}"
    )
    orchestrator = (
        "You are Final Orchestrator. Produce a binary disease decision from model evidence, equity priors, and disagreement.\n"
        f"Task={meta['task']}. Mock weighted probability={arbitration['final_prob']:.3f}. "
        f"Positive votes={arbitration['positive_votes']}/{arbitration['num_models']}."
    )
    safety = (
        "You are Safety Agent. Audit uncertainty, model disagreement, and subgroup reliability before clinical use.\n"
        f"Decision={arbitration['final_pred']}, probability={arbitration['final_prob']:.3f}, "
        f"disagreement={arbitration['disagreement']}, close_call={arbitration['close_call']}."
    )
    return {
        "bio_profiler": bio,
        "vision_specialist": vision,
        "equity_auditor": equity,
        "orchestrator": orchestrator,
        "safety_agent": safety,
    }


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, math.ceil(len(text) / chars_per_token))


def main() -> None:
    args = parse_args()
    by_task, loaded_files, missing_files = load_predictions(args.predictions_root, args.tasks, args.models)
    subgroup_priors, global_priors = load_priors(args.metrics_root)

    prediction_rows = []
    prompt_rows = []
    cost_rows = []
    case_rows = []

    for task in args.tasks:
        keys = select_cases(by_task[task], args.max_cases_per_task, args.seed_offset)
        for key in keys:
            rows_by_model = {
                model: rows_by_key[key]
                for model, rows_by_key in by_task[task].items()
                if key in rows_by_key
            }
            meta = case_metadata(rows_by_model)
            arbitration = mock_arbitrate(task, rows_by_model, meta, subgroup_priors, global_priors)
            prompts = build_prompts(meta, arbitration)
            input_tokens = {agent: estimate_tokens(prompt, args.chars_per_token) for agent, prompt in prompts.items()}
            output_tokens = OUTPUT_TOKEN_BUDGET.copy()
            total_input_tokens = sum(input_tokens.values())
            total_output_tokens = sum(output_tokens.values())
            estimated_cost = (
                total_input_tokens * args.input_usd_per_1m / 1_000_000
                + total_output_tokens * args.output_usd_per_1m / 1_000_000
            )

            prediction_rows.append(
                {
                    **meta,
                    "model_name": "equi_agent_smoke_mock",
                    "y_prob": f"{arbitration['final_prob']:.6f}",
                    "y_pred": arbitration["final_pred"],
                    "split": "test",
                    "positive_votes": arbitration["positive_votes"],
                    "num_models": arbitration["num_models"],
                    "disagreement": arbitration["disagreement"],
                    "close_call": arbitration["close_call"],
                    "safety_decision": arbitration["safety_decision"],
                    "estimated_input_tokens": total_input_tokens,
                    "estimated_output_tokens": total_output_tokens,
                    "estimated_cost_usd": f"{estimated_cost:.8f}",
                }
            )
            case_rows.append({**meta, **arbitration})
            for agent, prompt in prompts.items():
                prompt_rows.append(
                    {
                        **meta,
                        "agent": agent,
                        "input_tokens_est": input_tokens[agent],
                        "output_tokens_budget": output_tokens[agent],
                        "prompt": prompt,
                    }
                )
            cost_rows.append(
                {
                    **meta,
                    "estimated_input_tokens": total_input_tokens,
                    "estimated_output_tokens": total_output_tokens,
                    "estimated_total_tokens": total_input_tokens + total_output_tokens,
                    "estimated_cost_usd": f"{estimated_cost:.8f}",
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "equi_agent_smoke_predictions.csv",
        prediction_rows,
        [
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
            "positive_votes",
            "num_models",
            "disagreement",
            "close_call",
            "safety_decision",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "estimated_cost_usd",
        ],
    )
    write_csv(
        args.out_dir / "equi_agent_smoke_costs.csv",
        cost_rows,
        [
            "patient_id",
            "eye_id",
            "visit_id",
            "image_id",
            "dataset",
            "task",
            "y_true",
            "race",
            "ethnicity",
            "sex_gender",
            "age",
            "age_group",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "estimated_total_tokens",
            "estimated_cost_usd",
        ],
    )
    write_jsonl(args.out_dir / "equi_agent_smoke_prompts.jsonl", prompt_rows)
    write_jsonl(args.out_dir / "equi_agent_smoke_cases.jsonl", case_rows)
    write_jsonl(args.out_dir / "equi_agent_smoke_loaded_files.jsonl", loaded_files)
    write_jsonl(args.out_dir / "equi_agent_smoke_missing_files.jsonl", missing_files)

    total_in = sum(int(row["estimated_input_tokens"]) for row in cost_rows)
    total_out = sum(int(row["estimated_output_tokens"]) for row in cost_rows)
    total_cost = sum(float(row["estimated_cost_usd"]) for row in cost_rows)
    summary = {
        "cases": len(prediction_rows),
        "tasks": args.tasks,
        "models_requested": args.models,
        "loaded_files": len(loaded_files),
        "missing_files": missing_files,
        "estimated_input_tokens": total_in,
        "estimated_output_tokens": total_out,
        "estimated_total_tokens": total_in + total_out,
        "estimated_total_cost_usd": total_cost,
        "estimated_cost_per_case_usd": total_cost / len(prediction_rows) if prediction_rows else 0.0,
        "mean_input_tokens_per_case": mean([int(row["estimated_input_tokens"]) for row in cost_rows]) if cost_rows else 0.0,
        "mean_output_tokens_per_case": mean([int(row["estimated_output_tokens"]) for row in cost_rows]) if cost_rows else 0.0,
        "price_note": "Cost uses --input-usd-per-1m and --output-usd-per-1m; defaults are 0. Set your Azure deployment prices.",
    }
    (args.out_dir / "equi_agent_smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
