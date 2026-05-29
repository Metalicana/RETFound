from __future__ import annotations

import argparse
from pathlib import Path


METHODS = {
    "exp2_retfound_oct": {
        "method": "RETFound OCT",
        "type": "Foundation model",
        "table3_model": "RETFound OCT",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_mirage_slo": {
        "method": "MIRAGE SLO",
        "type": "Foundation model",
        "table3_model": "MIRAGE SLO",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_flair_slo": {
        "method": "FLAIR SLO",
        "type": "Foundation model",
        "table3_model": "FLAIR SLO",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_ret_clip_slo": {
        "method": "RET-CLIP SLO",
        "type": "Foundation model",
        "table3_model": "RET-CLIP SLO",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_visionfm_slo": {
        "method": "VisionFM SLO",
        "type": "Foundation model",
        "table3_model": "VisionFM SLO",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_visionfm_oct": {
        "method": "VisionFM OCT",
        "type": "Foundation model",
        "table3_model": "VisionFM OCT",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_retizero_slo": {
        "method": "RetiZero SLO",
        "type": "Foundation model",
        "table3_model": "RetiZero SLO",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_urfound_slo": {
        "method": "UrFound SLO",
        "type": "Foundation model",
        "table3_model": "UrFound SLO",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp2_urfound_oct": {
        "method": "UrFound OCT",
        "type": "Foundation model",
        "table3_model": "UrFound OCT",
        "table6": False,
        "confidence": "No",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp1_static_fusion_mean_thresholded": {
        "method": "Mean probability ensemble",
        "type": "Static ensemble",
        "table6": True,
        "confidence": "Yes",
        "subgroup_priors": "No",
        "disagreement": "No",
        "metadata": "No",
    },
    "exp1_static_fusion_confidence_weighted_thresholded": {
        "method": "Confidence-weighted ensemble",
        "type": "Static ensemble",
        "table6": True,
        "confidence": "Yes",
        "subgroup_priors": "No",
        "disagreement": "Weak",
        "metadata": "No",
    },
    "exp1_dynamic_prior_auroc": {
        "method": "Rule-based arbitration, subgroup AUROC prior",
        "type": "Subgroup rule-based arbitration",
        "table6": True,
        "confidence": "Yes",
        "subgroup_priors": "Yes",
        "disagreement": "Yes",
        "metadata": "Yes",
    },
    "exp1_dynamic_global_prior_auroc": {
        "method": "Rule-based arbitration, global AUROC prior",
        "type": "Rule-based arbitration",
        "table6": True,
        "confidence": "Yes",
        "subgroup_priors": "No",
        "disagreement": "Yes",
        "metadata": "No",
    },
    "exp1_dynamic_global_prior_ece": {
        "method": "Rule-based arbitration, global ECE prior",
        "type": "Rule-based calibration",
        "table6": True,
        "confidence": "Yes",
        "subgroup_priors": "No",
        "disagreement": "Yes",
        "metadata": "No",
    },
}

GDP_METHODS = {
    "exp8_gdp_progression_forecasting_td_pointwise_no_p_cut_gdp_native_rnflt_tds_efficientnet": {
        "method": "GDP-native RNFLT+TDS EfficientNet",
        "input": "RNFLT + visual-field TDS",
        "temporal_modeling": "Native GDP progression workflow",
        "fairness_component": "No",
    },
    "exp8_gdp_progression_forecasting_equi_agent_longitudinal": {
        "method": "Equi-Agent progression arbitration",
        "input": "Native helper + priors",
        "temporal_modeling": "Agentic arbitration",
        "fairness_component": "Yes",
    },
    "exp8_gdp_progression_rnflt": {
        "method": "RNFLT logistic baseline",
        "input": "RNFLT",
        "temporal_modeling": "No",
        "fairness_component": "No",
    },
    "exp8_gdp_progression_bscan": {
        "method": "B-scan logistic baseline",
        "input": "OCT B-scan",
        "temporal_modeling": "No",
        "fairness_component": "No",
    },
    "exp8_gdp_progression_clinical": {
        "method": "Clinical tabular logistic baseline",
        "input": "Metadata + visual field",
        "temporal_modeling": "No",
        "fairness_component": "No",
    },
    "exp8_gdp_progression_rnflt_clinical": {
        "method": "RNFLT + clinical logistic baseline",
        "input": "RNFLT + metadata + visual field",
        "temporal_modeling": "No",
        "fairness_component": "No",
    },
    "exp8_gdp_progression_bscan_clinical": {
        "method": "B-scan + clinical logistic baseline",
        "input": "OCT B-scan + metadata + visual field",
        "temporal_modeling": "No",
        "fairness_component": "No",
    },
    "exp8_gdp_progression_all": {
        "method": "OCT + RNFLT + clinical logistic baseline",
        "input": "OCT + RNFLT + metadata + visual field",
        "temporal_modeling": "No",
        "fairness_component": "No",
    },
}

TASK_LABELS = {"amd": "AMD", "dr": "DR", "glaucoma": "Glaucoma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manuscript-ready tables from evaluate_predictions.py outputs."
    )
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimals for manuscript-ready markdown/latex tables.",
    )
    return parser.parse_args()


def require_runtime_libs():
    import pandas as pd

    return pd


def read_metric_outputs(metrics_root: Path, pd):
    aggregate_frames = []
    disparity_frames = []

    for path in sorted(metrics_root.glob("**/*_aggregate.csv")):
        source_dir = path.parent.name
        df = pd.read_csv(path)
        df["source_dir"] = source_dir
        df["aggregate_path"] = str(path)
        aggregate_frames.append(df)

    for path in sorted(metrics_root.glob("**/*_disparities.csv")):
        source_dir = path.parent.name
        df = pd.read_csv(path)
        df["source_dir"] = source_dir
        df["disparities_path"] = str(path)
        disparity_frames.append(df)

    aggregate = pd.concat(aggregate_frames, ignore_index=True) if aggregate_frames else pd.DataFrame()
    disparities = pd.concat(disparity_frames, ignore_index=True) if disparity_frames else pd.DataFrame()
    return aggregate, disparities


def add_method_metadata(df, pd):
    if df.empty:
        return df
    rows = []
    for row in df.to_dict("records"):
        meta = METHODS.get(row["source_dir"]) or GDP_METHODS.get(row["source_dir"])
        if meta is None:
            continue
        rows.append({**row, **meta})
    return pd.DataFrame(rows)


def summarize_disparities(disparities, pd):
    if disparities.empty:
        return pd.DataFrame(
            columns=[
                "source_dir",
                "dataset",
                "task",
                "model_name",
                "split",
                "worst_group_f1",
                "delta_fpr",
                "delta_fnr",
                "subgroup_ece",
            ]
        )

    group_cols = ["source_dir", "dataset", "task", "model_name", "split"]
    rows = []
    for keys, group in disparities.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))
        rows.append(
            {
                **key,
                "worst_group_f1": group["worst_group_f1"].min(skipna=True),
                "worst_group_auroc": group["worst_group_auroc"].min(skipna=True),
                "delta_fpr": group["max_min_fpr_gap"].max(skipna=True),
                "delta_fnr": group["max_min_fnr_gap"].max(skipna=True),
                "subgroup_ece": group["subgroup_ece_max"].max(skipna=True)
                if "subgroup_ece_max" in group.columns
                else group["subgroup_ece_mean"].max(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def merge_aggregate_disparity(aggregate, disparity_summary):
    if aggregate.empty:
        return aggregate
    keys = ["source_dir", "dataset", "task", "model_name", "split"]
    return aggregate.merge(disparity_summary, on=keys, how="left")


def sensitivity(row) -> float:
    return 1.0 - float(row["fnr"])


def specificity(row) -> float:
    return 1.0 - float(row["fpr"])


def build_table3(df, pd):
    rows = []
    subset = df[
        df["source_dir"].isin(
            [
                "exp2_retfound_oct",
                "exp2_mirage_slo",
                "exp2_flair_slo",
                "exp2_ret_clip_slo",
                "exp2_visionfm_slo",
                "exp2_visionfm_oct",
                "exp2_retizero_slo",
                "exp2_urfound_slo",
                "exp2_urfound_oct",
            ]
        )
    ]
    for row in subset.to_dict("records"):
        rows.append(
            {
                "Model": row["table3_model"],
                "Dataset": "Harvard-FairVision",
                "Disease": TASK_LABELS.get(row["task"], row["task"]),
                "AUROC": row["auroc"],
                "F1": row["f1"],
                "Sens.": sensitivity(row),
                "Spec.": specificity(row),
                "Worst-group F1": row.get("worst_group_f1"),
                "Delta FPR": row.get("delta_fpr"),
                "Delta FNR": row.get("delta_fnr"),
                "Subgroup ECE": row.get("subgroup_ece"),
            }
        )
    return pd.DataFrame(rows)


def macro_summary(df, pd):
    rows = []
    for source_dir, group in df.groupby("source_dir", dropna=False):
        meta = METHODS.get(source_dir)
        if meta is None:
            continue
        rows.append(
            {
                "source_dir": source_dir,
                "Method": meta["method"],
                "Type": meta["type"],
                "AUROC": group["auroc"].mean(skipna=True),
                "F1": group["f1"].mean(skipna=True),
                "Balanced Acc.": group["balanced_accuracy"].mean(skipna=True),
                "Sens.": (1.0 - group["fnr"]).mean(skipna=True),
                "Spec.": (1.0 - group["fpr"]).mean(skipna=True),
                "Worst-group F1": group["worst_group_f1"].min(skipna=True),
                "Delta FPR": group["delta_fpr"].max(skipna=True),
                "Delta FNR": group["delta_fnr"].max(skipna=True),
                "ECE": group["ece"].mean(skipna=True),
                "Subgroup ECE": group["subgroup_ece"].max(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def gdp_summary(df, pd):
    rows = []
    subset = df[
        (df["source_dir"].isin(GDP_METHODS))
        & (df["dataset"] == "harvard_gdp")
        & (df["task"] == "progression_forecasting")
    ]
    for source_dir, group in subset.groupby("source_dir", dropna=False):
        meta = GDP_METHODS[source_dir]
        rows.append(
            {
                "source_dir": source_dir,
                "Method": meta["method"],
                "Input": meta["input"],
                "Temporal modeling": meta["temporal_modeling"],
                "Fairness component": meta["fairness_component"],
                "AUROC": group["auroc"].mean(skipna=True),
                "F1": group["f1"].mean(skipna=True),
                "Sens.": (1.0 - group["fnr"]).mean(skipna=True),
                "Spec.": (1.0 - group["fpr"]).mean(skipna=True),
                "Worst-group F1": group["worst_group_f1"].min(skipna=True),
                "Delta FPR": group["delta_fpr"].max(skipna=True),
                "Delta FNR": group["delta_fnr"].max(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def build_table4(df, pd):
    table = macro_summary(df, pd)
    order = [
        "exp2_retfound_oct",
        "exp2_mirage_slo",
        "exp2_flair_slo",
        "exp2_ret_clip_slo",
        "exp2_visionfm_slo",
        "exp2_visionfm_oct",
        "exp2_retizero_slo",
        "exp2_urfound_slo",
        "exp2_urfound_oct",
        "exp1_static_fusion_mean_thresholded",
        "exp1_static_fusion_confidence_weighted_thresholded",
        "exp1_dynamic_global_prior_auroc",
        "exp1_dynamic_global_prior_ece",
        "exp1_dynamic_prior_auroc",
    ]
    table = table[table["source_dir"].isin(order)].copy()
    table["order"] = table["source_dir"].apply(lambda x: order.index(x))
    return table.sort_values("order").drop(columns=["order", "source_dir"])


def build_table6(df, pd):
    table = macro_summary(df, pd)
    rows = []
    for row in table.to_dict("records"):
        meta = METHODS[row["source_dir"]]
        if not meta["table6"]:
            continue
        rows.append(
            {
                "Method": meta["method"],
                "Uses confidence?": meta["confidence"],
                "Uses subgroup priors?": meta["subgroup_priors"],
                "Uses disagreement?": meta["disagreement"],
                "Uses metadata?": meta["metadata"],
                "AUROC": row["AUROC"],
                "F1": row["F1"],
                "Worst-group F1": row["Worst-group F1"],
                "Delta FPR": row["Delta FPR"],
                "Delta FNR": row["Delta FNR"],
                "ECE": row["ECE"],
            }
        )
    out = pd.DataFrame(rows)
    order = {meta["method"]: idx for idx, meta in enumerate(METHODS.values())}
    out["order"] = out["Method"].map(order).fillna(999)
    return out.sort_values("order").drop(columns=["order"])


def build_table10(df, pd):
    table = macro_summary(df, pd)
    wanted = {
        "exp2_retfound_oct": "Best standalone foundation model",
        "exp2_mirage_slo": "MIRAGE SLO",
        "exp2_flair_slo": "FLAIR SLO",
        "exp2_ret_clip_slo": "RET-CLIP SLO",
        "exp2_visionfm_slo": "VisionFM SLO",
        "exp2_visionfm_oct": "VisionFM OCT",
        "exp2_retizero_slo": "RetiZero SLO",
        "exp2_urfound_slo": "UrFound SLO",
        "exp2_urfound_oct": "UrFound OCT",
        "exp1_static_fusion_mean_thresholded": "Mean probability ensemble",
        "exp1_static_fusion_confidence_weighted_thresholded": "Confidence-weighted ensemble",
        "exp1_dynamic_global_prior_auroc": "Dynamic global prior by AUROC",
        "exp1_dynamic_global_prior_ece": "Dynamic global prior by ECE",
    }
    rows = []
    for row in table.to_dict("records"):
        if row["source_dir"] not in wanted:
            continue
        rows.append(
            {
                "Method": wanted[row["source_dir"]],
                "AUROC": row["AUROC"],
                "F1": row["F1"],
                "Global ECE": row["ECE"],
                "Subgroup ECE": row["Subgroup ECE"],
                "Brier": None,
                "NLL": None,
                "Coverage at target risk": None,
                "Worst-group coverage": None,
            }
        )
        if row["source_dir"] == "exp2_retfound_oct":
            rows.append(
                {
                    "Method": "RETFound OCT",
                    "AUROC": row["AUROC"],
                    "F1": row["F1"],
                    "Global ECE": row["ECE"],
                    "Subgroup ECE": row["Subgroup ECE"],
                    "Brier": None,
                    "NLL": None,
                    "Coverage at target risk": None,
                    "Worst-group coverage": None,
                }
            )
    return pd.DataFrame(rows)


def build_table11(df, pd):
    table = gdp_summary(df, pd)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "Method",
                "Input",
                "Temporal modeling",
                "Fairness component",
                "AUROC",
                "F1",
                "Sens.",
                "Spec.",
                "Worst-group F1",
                "Delta FPR",
                "Delta FNR",
            ]
        )
    order = list(GDP_METHODS.keys())
    table["order"] = table["source_dir"].apply(lambda x: order.index(x) if x in order else 999)
    return table.sort_values("order").drop(columns=["order", "source_dir"])


def format_value(value, digits: int) -> str:
    try:
        if value is None:
            return r"\tbd"
        if value != value:
            return r"\tbd"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        if value is None or str(value) == "nan":
            return r"\tbd"
        return str(value)


def display_table(df, digits: int):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in {"f", "i"}:
            out[col] = out[col].apply(lambda x: format_value(x, digits))
        else:
            out[col] = out[col].fillna(r"\tbd")
    return out


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\tbd", r"\tbd")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_table_artifacts(df, out_dir: Path, name: str, digits: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)

    pretty = display_table(df, digits)
    try:
        pretty.to_markdown(out_dir / f"{name}.md", index=False)
    except ImportError:
        write_basic_markdown_table(pretty, out_dir / f"{name}.md")

    with (out_dir / f"{name}.tex").open("w") as f:
        col_spec = "l" * len(pretty.columns)
        f.write(f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(latex_escape(col) for col in pretty.columns) + r" \\" + "\n")
        f.write("\\midrule\n")
        for row in pretty.to_dict("records"):
            f.write(" & ".join(latex_escape(row[col]) for col in pretty.columns) + r" \\" + "\n")
        f.write("\\botrule\n")
        f.write("\\end{tabular}\n")


def write_basic_markdown_table(df, path: Path) -> None:
    columns = [str(col) for col in df.columns]
    rows = df.astype(str).to_dict("records")
    with path.open("w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[col] for col in df.columns) + " |\n")


def main() -> None:
    args = parse_args()
    pd = require_runtime_libs()
    aggregate, disparities = read_metric_outputs(args.metrics_root, pd)
    if aggregate.empty:
        raise FileNotFoundError(f"No *_aggregate.csv files found under {args.metrics_root}")

    aggregate = add_method_metadata(aggregate, pd)
    disparity_summary = summarize_disparities(disparities, pd)
    df = merge_aggregate_disparity(aggregate, disparity_summary)

    tables = {
        "exp1_table3_standalone": build_table3(df, pd),
        "exp2_table4_main_comparison": build_table4(df, pd),
        "exp4_table6_dynamic_vs_static": build_table6(df, pd),
        "exp7_table10_calibration": build_table10(df, pd),
        "exp8_table11_longitudinal_glaucoma": build_table11(df, pd),
    }

    for name, table in tables.items():
        write_table_artifacts(table, args.out_dir, name, args.digits)
        print(f"wrote={args.out_dir / (name + '.csv')} rows={len(table)}")
        print(f"wrote={args.out_dir / (name + '.md')}")
        print(f"wrote={args.out_dir / (name + '.tex')}")


if __name__ == "__main__":
    main()
