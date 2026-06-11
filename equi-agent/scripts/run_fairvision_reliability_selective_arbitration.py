from __future__ import annotations

import argparse
import json
import math
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
COUNTERFACTUAL_ATTRIBUTES = ["race", "ethnicity", "sex_gender", "age_group"]
RELIABILITY_FIELDS = ["auroc", "f1", "ece", "fnr", "fpr"]
MODEL_ALIASES = {
    "retfound_oct": ["fairvision_oct_retfound"],
    "mirage_slo": ["fairvision_slo_mirage"],
}
pd = None
np = None


def require_runtime_libs() -> None:
    global pd, np
    if pd is None or np is None:
        import numpy as _np
        import pandas as _pd

        np = _np
        pd = _pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic reliability-conditioned selective arbitration on FairVision "
            "prediction CSVs. Outputs forced predictions, escalation flags, ablations, "
            "conformal thresholds, and risk-coverage curves."
        )
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("equi-agent/outputs/predictions"))
    parser.add_argument("--metrics-root", type=Path, default=Path("equi-agent/outputs/metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/fairvision_reliability_selective_arbitration"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--subgroup-priors",
        type=Path,
        default=Path("equi-agent/outputs/metrics/validation_subgroup_priors.csv"),
    )
    parser.add_argument(
        "--global-priors",
        type=Path,
        default=Path("equi-agent/outputs/metrics/validation_subgroup_priors_global.csv"),
    )
    parser.add_argument("--subgroup-shrinkage-k", type=float, default=50.0)
    parser.add_argument("--conformal-shrinkage-k", type=float, default=50.0)
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=4.0, help="Reliability score softmax multiplier.")
    parser.add_argument("--gamma", type=float, default=0.75, help="Per-model uncertainty penalty.")
    parser.add_argument("--delta", type=float, default=0.50, help="Per-model vote-disagreement penalty.")
    parser.add_argument("--weight-auroc", type=float, default=0.45)
    parser.add_argument("--weight-f1", type=float, default=0.35)
    parser.add_argument("--weight-ece", type=float, default=0.10)
    parser.add_argument("--weight-fnr", type=float, default=0.07)
    parser.add_argument("--weight-fpr", type=float, default=0.03)
    parser.add_argument("--close-call-margin", type=float, default=0.08)
    parser.add_argument("--disagreement-rate-threshold", type=float, default=0.25)
    parser.add_argument("--low-reliability-threshold", type=float, default=0.35)
    parser.add_argument("--min-group-positive", type=int, default=5)
    parser.add_argument("--min-group-negative", type=int, default=5)
    parser.add_argument(
        "--coverage-grid",
        nargs="+",
        type=float,
        default=[1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50],
    )
    parser.add_argument("--run-counterfactuals", action="store_true")
    parser.add_argument("--counterfactual-max-values", type=int, default=12)
    return parser.parse_args()


def prediction_candidates(root: Path, task: str, model: str, split: str) -> list[Path]:
    suffixes = [f"{split}_thresholded", split] if split == "test" else [split, f"{split}_thresholded"]
    stems = [f"fairvision_{task}_{model}", f"fairvision_{model}"] + MODEL_ALIASES.get(model, [])
    return [root / f"{stem}_{suffix}.csv" for stem in stems for suffix in suffixes]


def prediction_file(root: Path, task: str, model: str, split: str) -> Path | None:
    for path in prediction_candidates(root, task, model, split):
        if path.exists():
            return path
    return None


def norm(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        if math.isnan(value):
            return "missing"
    except TypeError:
        pass
    text = str(value).strip()
    return text.lower() if text and text.lower() != "nan" else "missing"


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def int_label(value: Any) -> int:
    return int(round(finite(value, 0.0)))


def add_intersectional_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col != "metadata_missing_flag" else False
    out["race_x_sex_gender"] = out.apply(lambda r: f"{norm(r.get('race'))} x {norm(r.get('sex_gender'))}", axis=1)
    out["race_x_age_group"] = out.apply(lambda r: f"{norm(r.get('race'))} x {norm(r.get('age_group'))}", axis=1)
    out["sex_gender_x_age_group"] = out.apply(lambda r: f"{norm(r.get('sex_gender'))} x {norm(r.get('age_group'))}", axis=1)
    return out


def load_predictions(
    root: Path,
    tasks: list[str],
    models: list[str],
    split: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    frames = []
    loaded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    loaded_models: dict[str, list[str]] = {task: [] for task in tasks}

    split_values = {"test"} if split == "test" else {"val", "validation"}
    for task in tasks:
        for model in models:
            path = prediction_file(root, task, model, split)
            if path is None:
                missing.append({"task": task, "model": model, "split": split})
                continue
            df = pd.read_csv(path)
            if "task" in df.columns:
                df = df[df["task"].astype(str) == task]
            if "split" in df.columns:
                df = df[df["split"].astype(str).str.lower().isin(split_values)]
            if df.empty:
                missing.append({"task": task, "model": model, "split": split, "path": str(path), "reason": "no rows"})
                continue
            if "model_name" not in df.columns:
                df["model_name"] = model
            if "y_pred" not in df.columns:
                df["y_pred"] = (pd.to_numeric(df["y_prob"], errors="coerce").fillna(0.0) >= 0.5).astype(int)
            df["model_name"] = model
            df["task"] = task
            df["split"] = split
            frames.append(df)
            loaded_models[task].append(model)
            loaded.append({"task": task, "model": model, "split": split, "path": str(path), "rows": int(len(df))})

    if not frames:
        return pd.DataFrame(), loaded, missing, loaded_models

    out = pd.concat(frames, ignore_index=True)
    out = add_intersectional_columns(out)
    for col in ["y_true", "y_prob", "y_pred"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["y_true", "y_prob"])
    out["y_true"] = out["y_true"].astype(int)
    out["y_pred"] = out["y_pred"].fillna((out["y_prob"] >= 0.5).astype(int)).astype(int)
    out = out.drop_duplicates(CASE_COLUMNS + ["model_name"], keep="first")
    return out, loaded, missing, loaded_models


def restrict_to_common_cases(df: pd.DataFrame, loaded_models: dict[str, list[str]]) -> pd.DataFrame:
    if df.empty:
        return df
    keep_parts = []
    for task, task_df in df.groupby("task", dropna=False):
        expected = len(loaded_models.get(str(task), []))
        if expected == 0:
            continue
        counts = task_df.groupby(CASE_COLUMNS, dropna=False)["model_name"].nunique().reset_index(name="model_count")
        common_keys = counts[counts["model_count"] == expected][CASE_COLUMNS]
        keep_parts.append(task_df.merge(common_keys, on=CASE_COLUMNS, how="inner"))
    if not keep_parts:
        return df.iloc[0:0].copy()
    return pd.concat(keep_parts, ignore_index=True)


def load_prior_tables(subgroup_path: Path, global_path: Path) -> tuple[dict[tuple[str, str, str, str, str], dict], dict[tuple[str, str, str], dict]]:
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict] = {}
    global_lookup: dict[tuple[str, str, str], dict] = {}

    if subgroup_path.exists():
        subgroup_df = pd.read_csv(subgroup_path)
        for row in subgroup_df.to_dict("records"):
            key = (
                str(row.get("dataset", "")),
                str(row.get("task", "")),
                str(row.get("model_name", "")),
                str(row.get("attribute", "")),
                norm(row.get("subgroup", "")),
            )
            subgroup_lookup[key] = row
            if str(row.get("attribute", "")) == "GLOBAL":
                global_lookup[(key[0], key[1], key[2])] = row

    if global_path.exists():
        global_df = pd.read_csv(global_path)
        for row in global_df.to_dict("records"):
            global_lookup[(str(row.get("dataset", "")), str(row.get("task", "")), str(row.get("model_name", "")))] = row

    return subgroup_lookup, global_lookup


def subgroup_value(row: pd.Series | dict[str, Any], attr: str) -> str:
    if attr in row:
        return norm(row.get(attr))
    if attr == "race_x_sex_gender":
        return f"{norm(row.get('race'))} x {norm(row.get('sex_gender'))}"
    if attr == "race_x_age_group":
        return f"{norm(row.get('race'))} x {norm(row.get('age_group'))}"
    if attr == "sex_gender_x_age_group":
        return f"{norm(row.get('sex_gender'))} x {norm(row.get('age_group'))}"
    return norm(row.get(attr))


def blend_metric(local: dict | None, global_prior: dict | None, field: str, lam: float, default: float) -> float:
    global_value = finite(global_prior.get(field) if global_prior else None, default)
    local_value = finite(local.get(field) if local else None, global_value)
    return lam * local_value + (1.0 - lam) * global_value


def case_prior(
    row: pd.Series | dict[str, Any],
    model: str,
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict],
    global_lookup: dict[tuple[str, str, str], dict],
    shrinkage_k: float,
    mode: str,
) -> dict[str, Any]:
    dataset = str(row.get("dataset", ""))
    task = str(row.get("task", ""))
    global_prior = global_lookup.get((dataset, task, model))
    if mode == "global":
        local = None
        attr = "GLOBAL"
        subgroup = "GLOBAL"
        lam = 0.0
    else:
        local = None
        attr = "GLOBAL"
        subgroup = "GLOBAL"
        for candidate_attr in ATTRIBUTES:
            candidate_subgroup = subgroup_value(row, candidate_attr)
            candidate = subgroup_lookup.get((dataset, task, model, candidate_attr, candidate_subgroup))
            if candidate is not None:
                local = candidate
                attr = candidate_attr
                subgroup = candidate_subgroup
                break
        n = finite(local.get("n") if local else None, 0.0)
        lam = n / (n + shrinkage_k) if n > 0 and shrinkage_k >= 0 else 0.0

    metrics = {
        "auroc": blend_metric(local, global_prior, "auroc", lam, 0.5),
        "f1": blend_metric(local, global_prior, "f1", lam, 0.0),
        "ece": blend_metric(local, global_prior, "ece", lam, 0.5),
        "fnr": blend_metric(local, global_prior, "fnr", lam, 0.5),
        "fpr": blend_metric(local, global_prior, "fpr", lam, 0.5),
    }
    return {
        "attribute": attr,
        "subgroup": subgroup,
        "lambda": lam,
        "local_n": finite(local.get("n") if local else None, 0.0),
        "global_n": finite(global_prior.get("n") if global_prior else None, 0.0),
        **metrics,
    }


def reliability_score(prior: dict[str, Any], args: argparse.Namespace) -> float:
    return (
        args.weight_auroc * finite(prior.get("auroc"), 0.5)
        + args.weight_f1 * finite(prior.get("f1"), 0.0)
        - args.weight_ece * finite(prior.get("ece"), 0.5)
        - args.weight_fnr * finite(prior.get("fnr"), 0.5)
        - args.weight_fpr * finite(prior.get("fpr"), 0.5)
    )


def base_metadata(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0].to_dict()
    return {col: first.get(col, "") for col in CASE_COLUMNS + ["y_true"] + METADATA_COLUMNS + ATTRIBUTES}


def arbitrate_case(
    group: pd.DataFrame,
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict],
    global_lookup: dict[tuple[str, str, str], dict],
    args: argparse.Namespace,
    reliability_mode: str,
    metadata_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = base_metadata(group)
    if metadata_override:
        meta.update(metadata_override)
        meta = add_intersectional_columns(pd.DataFrame([meta])).iloc[0].to_dict()

    vote_rate = float(group["y_pred"].mean()) if len(group) else 0.0
    majority_pred = int(vote_rate >= 0.5)
    disagreement_rate = min(vote_rate, 1.0 - vote_rate)
    model_rows = []
    logits = []
    probs = []
    scores = []

    for row in group.to_dict("records"):
        model = str(row["model_name"])
        prior = case_prior(meta, model, subgroup_lookup, global_lookup, args.subgroup_shrinkage_k, reliability_mode)
        score = reliability_score(prior, args)
        prob = finite(row.get("y_prob"), 0.0)
        pred = int_label(row.get("y_pred"))
        uncertainty = max(0.0, 1.0 - 2.0 * abs(prob - 0.5))
        disagrees = float(pred != majority_pred)
        logit = args.beta * score - args.gamma * uncertainty - args.delta * disagrees
        logits.append(logit)
        probs.append(prob)
        scores.append(score)
        model_rows.append(
            {
                **{col: meta.get(col, "") for col in CASE_COLUMNS},
                "model_name": model,
                "source_y_prob": prob,
                "source_y_pred": pred,
                "vote_rate": vote_rate,
                "majority_pred": majority_pred,
                "model_disagrees_with_majority": bool(disagrees),
                "model_uncertainty": uncertainty,
                "reliability_mode": reliability_mode,
                "prior_attribute": prior["attribute"],
                "prior_subgroup": prior["subgroup"],
                "prior_lambda": prior["lambda"],
                "prior_local_n": prior["local_n"],
                "prior_global_n": prior["global_n"],
                "prior_auroc": prior["auroc"],
                "prior_f1": prior["f1"],
                "prior_ece": prior["ece"],
                "prior_fnr": prior["fnr"],
                "prior_fpr": prior["fpr"],
                "reliability_score": score,
                "weight_logit": logit,
            }
        )

    if logits:
        max_logit = max(logits)
        weights = np.exp(np.asarray(logits) - max_logit)
        weights = weights / weights.sum()
        final_prob = float(np.sum(weights * np.asarray(probs)))
        weighted_reliability = float(np.sum(weights * np.asarray(scores)))
    else:
        weights = np.asarray([])
        final_prob = 0.5
        weighted_reliability = 0.0

    for weight, model_row in zip(weights, model_rows):
        model_row["arbitration_weight"] = float(weight)

    margin = abs(final_prob - 0.5)
    risk_score = max(
        1.0 - min(1.0, 2.0 * margin),
        disagreement_rate,
        max(0.0, args.low_reliability_threshold - weighted_reliability),
    )
    pred = {
        **meta,
        "model_name": f"reliability_{reliability_mode}_weighted",
        "y_prob": final_prob,
        "y_pred": int(final_prob >= 0.5),
        "split": str(group.iloc[0].get("split", "")),
        "num_models": int(len(group)),
        "vote_rate": vote_rate,
        "positive_votes": int(group["y_pred"].sum()),
        "disagreement_rate": disagreement_rate,
        "close_call": bool(margin < args.close_call_margin),
        "weighted_reliability": weighted_reliability,
        "risk_score": risk_score,
    }
    return pred, model_rows


def arbitrate_frame(
    predictions: pd.DataFrame,
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict],
    global_lookup: dict[tuple[str, str, str], dict],
    args: argparse.Namespace,
    reliability_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    weight_rows = []
    for _, group in predictions.groupby(CASE_COLUMNS, dropna=False, sort=False):
        pred, model_rows = arbitrate_case(group, subgroup_lookup, global_lookup, args, reliability_mode)
        pred_rows.append(pred)
        weight_rows.extend(model_rows)
    return pd.DataFrame(pred_rows), pd.DataFrame(weight_rows)


def static_mean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby(CASE_COLUMNS, dropna=False, sort=False):
        meta = base_metadata(group)
        prob = float(group["y_prob"].mean())
        vote_rate = float(group["y_pred"].mean())
        rows.append(
            {
                **meta,
                "model_name": "static_mean_probability",
                "y_prob": prob,
                "y_pred": int(prob >= 0.5),
                "split": str(group.iloc[0].get("split", "")),
                "num_models": int(len(group)),
                "vote_rate": vote_rate,
                "positive_votes": int(group["y_pred"].sum()),
                "disagreement_rate": min(vote_rate, 1.0 - vote_rate),
                "close_call": bool(abs(prob - 0.5) < 0.08),
                "weighted_reliability": math.nan,
                "risk_score": 1.0 - min(1.0, 2.0 * abs(prob - 0.5)),
            }
        )
    return pd.DataFrame(rows)


def best_single_global_predictions(
    predictions: pd.DataFrame,
    global_lookup: dict[tuple[str, str, str], dict],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    chosen_by_task: dict[str, str] = {}
    for task in sorted(predictions["task"].dropna().astype(str).unique()):
        task_models = sorted(predictions.loc[predictions["task"] == task, "model_name"].astype(str).unique())
        best_model = task_models[0]
        best_score = -float("inf")
        for model in task_models:
            prior = global_lookup.get(("harvard_fairvision", task, model))
            if prior is None:
                matches = [row for key, row in global_lookup.items() if key[1] == task and key[2] == model]
                prior = matches[0] if matches else {}
            score = reliability_score(
                {
                    "auroc": finite(prior.get("auroc"), 0.5),
                    "f1": finite(prior.get("f1"), 0.0),
                    "ece": finite(prior.get("ece"), 0.5),
                    "fnr": finite(prior.get("fnr"), 0.5),
                    "fpr": finite(prior.get("fpr"), 0.5),
                },
                args,
            )
            if score > best_score:
                best_score = score
                best_model = model
        chosen_by_task[task] = best_model

    for task, model in chosen_by_task.items():
        source = predictions[(predictions["task"] == task) & (predictions["model_name"] == model)].copy()
        source["model_name"] = "best_single_global_reliability"
        source["chosen_source_model"] = model
        source["accepted"] = True
        source["escalate_to_human"] = False
        source["escalation_reasons"] = ""
        rows.append(source)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def conformal_quantile(scores: list[float], alpha: float) -> float:
    clean = sorted(score for score in scores if not math.isnan(score))
    if not clean:
        return 0.5
    rank = int(math.ceil((len(clean) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(clean))
    return float(clean[rank - 1])


def build_conformal_tables(
    val_predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[tuple[str, str, str], dict[str, float]], pd.DataFrame]:
    work = val_predictions.copy()
    work["_nonconformity"] = np.where(work["y_true"].astype(int) == 1, 1.0 - work["y_prob"], work["y_prob"])
    global_q: dict[str, float] = {}
    local_q: dict[tuple[str, str, str], dict[str, float]] = {}
    rows = []

    for task, group in work.groupby("task", dropna=False):
        scores = group["_nonconformity"].astype(float).tolist()
        q = conformal_quantile(scores, args.conformal_alpha)
        global_q[str(task)] = q
        rows.append({"task": task, "attribute": "GLOBAL", "subgroup": "GLOBAL", "n": len(group), "q": q})

        for attr in ATTRIBUTES:
            for subgroup, sub in group.groupby(attr, dropna=False):
                sub_scores = sub["_nonconformity"].astype(float).tolist()
                sub_q = conformal_quantile(sub_scores, args.conformal_alpha)
                key = (str(task), attr, norm(subgroup))
                local_q[key] = {"q": sub_q, "n": float(len(sub))}
                rows.append({"task": task, "attribute": attr, "subgroup": norm(subgroup), "n": len(sub), "q": sub_q})

    return global_q, local_q, pd.DataFrame(rows)


def case_conformal_q(
    row: dict[str, Any] | pd.Series,
    global_q: dict[str, float],
    local_q: dict[tuple[str, str, str], dict[str, float]],
    args: argparse.Namespace,
) -> tuple[float, str, str, float, float]:
    task = str(row.get("task", ""))
    base_q = global_q.get(task, 0.5)
    for attr in ATTRIBUTES:
        subgroup = subgroup_value(row, attr)
        local = local_q.get((task, attr, subgroup))
        if local is None:
            continue
        n = finite(local.get("n"), 0.0)
        lam = n / (n + args.conformal_shrinkage_k) if n > 0 and args.conformal_shrinkage_k >= 0 else 0.0
        q = lam * finite(local.get("q"), base_q) + (1.0 - lam) * base_q
        return q, attr, subgroup, lam, n
    return base_q, "GLOBAL", "GLOBAL", 0.0, 0.0


def add_escalation_columns(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
    global_q: dict[str, float] | None = None,
    local_q: dict[tuple[str, str, str], dict[str, float]] | None = None,
    use_disagreement: bool = True,
    use_conformal: bool = False,
) -> pd.DataFrame:
    out = predictions.copy()
    accepted = []
    reasons = []
    conformal_labels = []
    conformal_set_sizes = []
    conformal_qs = []
    conformal_attrs = []
    conformal_subgroups = []
    conformal_lambdas = []

    for row in out.to_dict("records"):
        row_reasons = []
        prob = finite(row.get("y_prob"), 0.5)
        if use_disagreement and finite(row.get("disagreement_rate"), 0.0) >= args.disagreement_rate_threshold:
            row_reasons.append("model_disagreement")
        if use_disagreement and bool(row.get("close_call")):
            row_reasons.append("close_call")
        if use_disagreement and finite(row.get("weighted_reliability"), 1.0) < args.low_reliability_threshold:
            row_reasons.append("low_reliability")

        q = math.nan
        q_attr = ""
        q_subgroup = ""
        q_lambda = math.nan
        label = ""
        set_size = 1
        if use_conformal and global_q is not None and local_q is not None:
            q, q_attr, q_subgroup, q_lambda, _ = case_conformal_q(row, global_q, local_q, args)
            include_negative = prob <= q
            include_positive = (1.0 - prob) <= q
            labels = []
            if include_negative:
                labels.append("negative")
            if include_positive:
                labels.append("positive")
            set_size = len(labels)
            label = ";".join(labels) if labels else "empty"
            if set_size != 1:
                row_reasons.append("conformal_ambiguous")

        conformal_qs.append(q)
        conformal_attrs.append(q_attr)
        conformal_subgroups.append(q_subgroup)
        conformal_lambdas.append(q_lambda)
        conformal_labels.append(label)
        conformal_set_sizes.append(set_size)
        reasons.append(";".join(row_reasons))
        accepted.append(len(row_reasons) == 0)

    out["accepted"] = accepted
    out["escalate_to_human"] = ~out["accepted"]
    out["escalation_reasons"] = reasons
    out["conformal_label_set"] = conformal_labels
    out["conformal_set_size"] = conformal_set_sizes
    out["conformal_q"] = conformal_qs
    out["conformal_attribute"] = conformal_attrs
    out["conformal_subgroup"] = conformal_subgroups
    out["conformal_lambda"] = conformal_lambdas
    out["risk_score"] = out["risk_score"].astype(float) + out["escalate_to_human"].astype(float)
    return out


def confusion_counts(df: pd.DataFrame) -> tuple[int, int, int, int]:
    y_true = df["y_true"].astype(int)
    y_pred = df["y_pred"].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else math.nan


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    return safe_div(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def ece(df: pd.DataFrame, bins: int = 10) -> float:
    if df.empty:
        return math.nan
    total = len(df)
    score = 0.0
    probs = df["y_prob"].astype(float)
    labels = df["y_true"].astype(float)
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        if idx == 0:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs > lower) & (probs <= upper)
        if not bool(mask.any()):
            continue
        score += float(mask.mean()) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return score


def auroc(df: pd.DataFrame) -> float:
    if df.empty or df["y_true"].nunique() < 2:
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(df["y_true"].astype(int), df["y_prob"].astype(float)))
    except Exception:
        return math.nan


def group_worst_f1(df: pd.DataFrame, min_positive: int, min_negative: int) -> float:
    scores = []
    for attr in ATTRIBUTES:
        if attr not in df.columns:
            continue
        for _, group in df.groupby(attr, dropna=False):
            positives = int((group["y_true"].astype(int) == 1).sum())
            negatives = int((group["y_true"].astype(int) == 0).sum())
            if positives < min_positive or negatives < min_negative:
                continue
            tp, _, fp, fn = confusion_counts(group)
            scores.append(f1_from_counts(tp, fp, fn))
    return min(scores) if scores else math.nan


def metric_block(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "n_positive": 0,
            "n_negative": 0,
            "accuracy": math.nan,
            "f1": math.nan,
            "sensitivity": math.nan,
            "specificity": math.nan,
            "precision": math.nan,
            "fpr": math.nan,
            "fnr": math.nan,
            "balanced_accuracy": math.nan,
            "auroc": math.nan,
            "ece": math.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    tp, tn, fp, fn = confusion_counts(df)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "n": int(len(df)),
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "accuracy": safe_div(tp + tn, len(df)),
        "f1": f1_from_counts(tp, fp, fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": safe_div(tp, tp + fp),
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "balanced_accuracy": (sensitivity + specificity) / 2 if not math.isnan(sensitivity) and not math.isnan(specificity) else math.nan,
        "auroc": auroc(df),
        "ece": ece(df),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def summarize_method(df: pd.DataFrame, method: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for task_name, group in [("overall", df)] + [(str(task), g) for task, g in df.groupby("task", dropna=False)]:
        accepted = group[group.get("accepted", True).astype(bool)] if "accepted" in group.columns else group
        escalated = group[~group.get("accepted", pd.Series(True, index=group.index)).astype(bool)] if "accepted" in group.columns else group.iloc[0:0]
        all_metrics = metric_block(group)
        accepted_metrics = metric_block(accepted)
        row = {
            "method": method,
            "task": task_name,
            **{f"forced_{key}": value for key, value in all_metrics.items()},
            "coverage": safe_div(len(accepted), len(group)),
            "escalation_rate": safe_div(len(escalated), len(group)),
            "accepted_error_rate": (
                float((accepted["y_true"].astype(int) != accepted["y_pred"].astype(int)).mean()) if not accepted.empty else math.nan
            ),
            "escalated_prevalence": float(escalated["y_true"].astype(int).mean()) if not escalated.empty else math.nan,
            "worst_group_f1": group_worst_f1(group, args.min_group_positive, args.min_group_negative),
            "worst_group_accepted_f1": group_worst_f1(accepted, args.min_group_positive, args.min_group_negative),
            **{f"accepted_{key}": value for key, value in accepted_metrics.items()},
        }
        rows.append(row)
    return rows


def risk_coverage_curve(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values("risk_score", ascending=True).reset_index(drop=True)
    n = len(ranked)
    for coverage in args.coverage_grid:
        keep_n = max(1, min(n, int(round(n * coverage)))) if n else 0
        accepted = ranked.iloc[:keep_n].copy()
        accepted["accepted"] = True
        metrics = metric_block(accepted)
        rows.append(
            {
                "coverage": safe_div(keep_n, n),
                "escalation_rate": 1.0 - safe_div(keep_n, n),
                **metrics,
                "worst_group_accepted_f1": group_worst_f1(accepted, args.min_group_positive, args.min_group_negative),
            }
        )
    return pd.DataFrame(rows)


def run_counterfactuals(
    test_predictions: pd.DataFrame,
    subgroup_lookup: dict[tuple[str, str, str, str, str], dict],
    global_lookup: dict[tuple[str, str, str], dict],
    global_q: dict[str, float],
    local_q: dict[tuple[str, str, str], dict[str, float]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    values_by_attr = {
        attr: sorted(test_predictions[attr].dropna().astype(str).unique())[: args.counterfactual_max_values]
        for attr in COUNTERFACTUAL_ATTRIBUTES
        if attr in test_predictions.columns
    }
    rows = []
    for _, group in test_predictions.groupby(CASE_COLUMNS, dropna=False, sort=False):
        base_pred, _ = arbitrate_case(group, subgroup_lookup, global_lookup, args, "shrunk")
        base_full = add_escalation_columns(pd.DataFrame([base_pred]), args, global_q, local_q, use_disagreement=True, use_conformal=True)
        base_row = base_full.iloc[0].to_dict()
        for attr, values in values_by_attr.items():
            current = str(base_row.get(attr, ""))
            for value in values:
                if str(value) == current:
                    continue
                cf_pred, _ = arbitrate_case(group, subgroup_lookup, global_lookup, args, "shrunk", {attr: value})
                cf_full = add_escalation_columns(pd.DataFrame([cf_pred]), args, global_q, local_q, use_disagreement=True, use_conformal=True)
                cf_row = cf_full.iloc[0].to_dict()
                rows.append(
                    {
                        **{col: base_row.get(col, "") for col in CASE_COLUMNS},
                        "attribute": attr,
                        "original_value": current,
                        "counterfactual_value": value,
                        "base_y_prob": base_row["y_prob"],
                        "counterfactual_y_prob": cf_row["y_prob"],
                        "base_y_pred": base_row["y_pred"],
                        "counterfactual_y_pred": cf_row["y_pred"],
                        "base_escalate": base_row["escalate_to_human"],
                        "counterfactual_escalate": cf_row["escalate_to_human"],
                        "label_flipped": int(base_row["y_pred"] != cf_row["y_pred"]),
                        "escalation_flipped": int(base_row["escalate_to_human"] != cf_row["escalate_to_human"]),
                    }
                )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    require_runtime_libs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subgroup_lookup, global_lookup = load_prior_tables(args.subgroup_priors, args.global_priors)
    test_raw, test_loaded, test_missing, test_models = load_predictions(args.predictions_root, args.tasks, args.models, "test")
    val_raw, val_loaded, val_missing, val_models = load_predictions(args.predictions_root, args.tasks, args.models, "val")
    test = restrict_to_common_cases(test_raw, test_models)
    val = restrict_to_common_cases(val_raw, val_models)
    if test.empty:
        raise SystemExit("No common test cases found across the loaded FairVision model prediction files.")

    global_test, global_weights = arbitrate_frame(test, subgroup_lookup, global_lookup, args, "global")
    shrunk_test, shrunk_weights = arbitrate_frame(test, subgroup_lookup, global_lookup, args, "shrunk")
    static_test = static_mean_predictions(test)
    best_single = best_single_global_predictions(test, global_lookup, args)

    if val.empty:
        global_q: dict[str, float] = {}
        local_q: dict[tuple[str, str, str], dict[str, float]] = {}
        conformal_df = pd.DataFrame()
    else:
        shrunk_val, _ = arbitrate_frame(val, subgroup_lookup, global_lookup, args, "shrunk")
        global_q, local_q, conformal_df = build_conformal_tables(shrunk_val, args)

    global_no_escalation = add_escalation_columns(global_test, args, use_disagreement=False, use_conformal=False)
    shrunk_no_escalation = add_escalation_columns(shrunk_test, args, use_disagreement=False, use_conformal=False)
    shrunk_disagreement = add_escalation_columns(shrunk_test, args, use_disagreement=True, use_conformal=False)
    full = add_escalation_columns(shrunk_test, args, global_q, local_q, use_disagreement=True, use_conformal=bool(global_q))
    static_test = add_escalation_columns(static_test, args, use_disagreement=False, use_conformal=False)

    if not best_single.empty:
        best_single = add_intersectional_columns(best_single)

    metrics_rows = []
    for method, frame in [
        ("best_single_global_reliability", best_single),
        ("static_mean_probability", static_test),
        ("global_reliability_weighted", global_no_escalation),
        ("shrunk_subgroup_reliability_weighted", shrunk_no_escalation),
        ("shrunk_plus_disagreement_escalation", shrunk_disagreement),
        ("full_reliability_conformal_escalation", full),
    ]:
        if not frame.empty:
            metrics_rows.extend(summarize_method(frame, method, args))

    base_metrics = []
    for model, model_df in test.groupby("model_name", dropna=False):
        base = add_intersectional_columns(model_df.copy())
        base["accepted"] = True
        base["escalate_to_human"] = False
        base_metrics.extend(summarize_method(base, f"base_{model}", args))

    full.to_csv(args.out_dir / "selective_arbitration_predictions.csv", index=False)
    pd.concat([global_weights, shrunk_weights], ignore_index=True).to_csv(args.out_dir / "selective_arbitration_model_weights.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(args.out_dir / "selective_arbitration_ablation_metrics.csv", index=False)
    pd.DataFrame(base_metrics).to_csv(args.out_dir / "base_model_metrics_on_common_cases.csv", index=False)
    risk_coverage_curve(full, args).to_csv(args.out_dir / "risk_coverage_curve.csv", index=False)
    conformal_df.to_csv(args.out_dir / "conformal_thresholds.csv", index=False)

    counterfactual_summary: dict[str, Any] = {"run": False}
    if args.run_counterfactuals and global_q:
        counterfactuals = run_counterfactuals(test, subgroup_lookup, global_lookup, global_q, local_q, args)
        counterfactuals.to_csv(args.out_dir / "metadata_counterfactuals.csv", index=False)
        if not counterfactuals.empty:
            counterfactual_summary = {
                "run": True,
                "rows": int(len(counterfactuals)),
                "label_flip_rate": float(counterfactuals["label_flipped"].mean()),
                "escalation_flip_rate": float(counterfactuals["escalation_flipped"].mean()),
            }

    summary = {
        "test_common_cases": int(len(full)),
        "validation_common_cases": int(len(val.groupby(CASE_COLUMNS))) if not val.empty else 0,
        "tasks": args.tasks,
        "models_requested": args.models,
        "test_loaded_files": test_loaded,
        "test_missing_files": test_missing,
        "validation_loaded_files": val_loaded,
        "validation_missing_files": val_missing,
        "subgroup_priors": str(args.subgroup_priors),
        "global_priors": str(args.global_priors),
        "subgroup_shrinkage_k": args.subgroup_shrinkage_k,
        "conformal_alpha": args.conformal_alpha,
        "outputs": {
            "predictions": str(args.out_dir / "selective_arbitration_predictions.csv"),
            "weights": str(args.out_dir / "selective_arbitration_model_weights.csv"),
            "ablation_metrics": str(args.out_dir / "selective_arbitration_ablation_metrics.csv"),
            "base_model_metrics": str(args.out_dir / "base_model_metrics_on_common_cases.csv"),
            "risk_coverage": str(args.out_dir / "risk_coverage_curve.csv"),
            "conformal_thresholds": str(args.out_dir / "conformal_thresholds.csv"),
        },
        "counterfactuals": counterfactual_summary,
    }
    write_json(args.out_dir / "selective_arbitration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
