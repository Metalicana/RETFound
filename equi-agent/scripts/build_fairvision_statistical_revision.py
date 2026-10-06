"""Decision item 1: FairVision reporting from saved predictions, with no inference.

Writes a separate revision bundle. Historical results and manuscript sources are
never overwritten. The October 5 audit fixes the source selection, not its scores.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest, bootstrap

import audit_manuscript_twenty_questions as audit

ROOT = audit.ROOT
U = audit.uncertainty
PRIOR = ROOT / "equi-agent/outputs/audits/manuscript_20_questions_20261005"
DEFAULT_OUT = ROOT / "equi-agent/outputs/audits/fairvision_statistics_revision_20261005"
SEED = 20261005
COMPARISONS = (
    ("glaucoma", "RETFound"), ("amd", "visionfm_oct"), ("dr", "RETFound"),
    ("glaucoma", "GPT-5.1"), ("amd", "GPT-5.1"), ("dr", "GPT-5.1"),
    ("amd", "urfound_oct"),
)
DISPLAY = {
    "visionfm_oct": "VisionFM OCT", "urfound_oct": "URFound OCT",
    "visionfm_slo": "VisionFM SLO", "urfound_slo": "URFound SLO",
    "flair_slo": "FLAIR SLO", "ret_clip_slo": "RET-CLIP SLO", "retizero_slo": "RetiZero SLO",
}
TABLE_METRICS = ("f1_macro", "worst_group_macro_f1", "sensitivity", "specificity", "balanced_accuracy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_saved(out):
    previous = audit.read_json(PRIOR / "provenance.json")["source_sha256"]
    old_metrics = audit.read(PRIOR / "fairvision_metrics.csv")
    manifest = audit.RECOVERY / "manifest_recovered.csv"
    paths = {r["source"] for r in old_metrics} | {str(manifest.relative_to(ROOT))}
    for name in sorted(paths):
        require(name in previous and sha(ROOT / name) == previous[name],
                f"Historical source changed or unrecorded: {name}")
    locked = {task: {} for task in audit.TASKS}
    for row in audit.read(manifest):
        task, key = row["Task_Folder"].lower(), audit.case_id(row)
        require(key not in locked[task], f"Duplicate manifest ID: {task}/{key}")
        truth = audit.binary(row["Ground_Truth"])
        require(truth is not None, f"Invalid reference label: {task}/{key}")
        locked[task][key] = dict(case_id=key, truth=truth, age=float(row["Age"]),
                                 **U.runner.demographic(row))
    require(all(len(rows) == 250 for rows in locked.values()), "Expected 250 locked cases per task")
    methods = audit.load_methods(locked, out)
    current = audit.read(out / "fairvision_metrics.csv")
    lookup = {(r["task"], r["method"]): r for r in old_metrics}
    require(set(lookup) == {(r["task"], r["method"]) for r in current}, "Method inventory changed")
    for row in current:
        prior = lookup[row["task"], row["method"]]
        require(row["source"] == prior["source"], "Source selection changed")
        for field in ("n", "positive", "negative", "tp", "tn", "fp", "fn", *TABLE_METRICS):
            require(abs(float(row[field]) - float(prior[field])) < 1e-12,
                    f"Audit reproduction failed: {row['task']}/{row['method']}/{field}")
    return methods, current


def group_masks(cases):
    return [np.array([c[attr] == group for c in cases])
            for attr in U.ATTRIBUTES
            for group in sorted({c[attr] for c in cases} - U.UNKNOWN)]


def metric_statistic(cases):
    truth = np.array([U.runner.binary(c["truth"]) for c in cases])
    pred = np.array([U.runner.binary(c["prediction"]) for c in cases])
    masks = group_masks(cases)
    require(len(masks) > 0, "No known subgroups")

    def statistic(positive_ids, negative_ids, axis=-1):
        ids = np.concatenate((positive_ids, negative_ids), axis=axis).astype(int)
        t, p = truth[ids], pred[ids]
        scores = U.metric_array(t, p[None, ...])[0]
        group_scores = []
        for mask in masks:
            members = mask[ids]
            # Sentinel labels exclude nonmembers from all four confusion counts.
            grouped = U.metric_array(np.where(members, t, -2), np.where(members, p, -2)[None, ...])[0, 0]
            group_scores.append(np.where(members.sum(axis=-1) > 0, grouped, np.nan))
        # Keep the same group set in every draw. An absent group makes the
        # minimum undefined; never drop that group or silently discard a draw.
        worst = np.min(group_scores, axis=0)
        return np.concatenate((scores, np.expand_dims(worst, axis=0)), axis=0)

    return truth, statistic


def method_intervals(cases, resamples=10000, seed=SEED):
    require(resamples >= 100, "At least 100 replicates required")
    require(len({c["case_id"] for c in cases}) == len(cases), "Duplicate case ID")
    cases = sorted(cases, key=lambda c: c["case_id"])
    truth, statistic = metric_statistic(cases)
    pos, neg = np.flatnonzero(truth == 1), np.flatnonzero(truth == 0)
    require(len(pos) > 0 and len(neg) > 0, "Both reference classes required")
    rng_key = "rng" if "rng" in inspect.signature(bootstrap).parameters else "random_state"
    result = bootstrap((pos, neg), statistic, vectorized=True, paired=False,
                       method="percentile", batch=250, n_resamples=resamples,
                       confidence_level=.95, **{rng_key: np.random.default_rng(seed)})
    estimates = statistic(pos, neg)
    rows = []
    for i, metric in enumerate((*U.METRICS, "worst_group_macro_f1")):
        invalid = int((~np.isfinite(result.bootstrap_distribution[i])).sum())
        rows.append(dict(metric=metric, estimate=float(estimates[i]),
            ci_lower=float(result.confidence_interval.low[i]) if not invalid else None,
            ci_upper=float(result.confidence_interval.high[i]) if not invalid else None,
            confidence=.95, n=len(cases), n_bootstrap=resamples, seed=seed,
            undefined_replicates=invalid,
            interval_status="available; exploratory percentile interval" if not invalid else
                "NOT FOUND: fixed group absent in bootstrap draws; no draws discarded"))
    return rows


def align(agent, baseline):
    a, b = (U.indexed(rows, "case_id") for rows in (agent, baseline))
    keys = sorted(a.keys() & b.keys())
    require(bool(keys), "No paired cases")
    for key in keys:
        require(a[key]["truth"] == b[key]["truth"], f"Paired label mismatch: {key}")
        require(all(a[key][attr] == b[key][attr] for attr in U.ATTRIBUTES), f"Paired metadata mismatch: {key}")
    return [a[k] for k in keys], [a[k]["prediction"] for k in keys], [b[k]["prediction"] for k in keys]


def comparison_intervals(methods, resamples=10000):
    rows, tests = [], []
    family_n = len(COMPARISONS)
    for task, comparator in COMPARISONS:
        cases, a, b = align(methods[task]["RetinAgent"], methods[task][comparator])
        scope = "complete locked cohort" if len(cases) == 250 else "valid-response intersection only; NOT full cohort"
        pred = {"RetinAgent": a, comparator: b}
        ci, test, _, _, _ = U.analyse(cases, pred, "RetinAgent", resamples=resamples, seed=SEED)
        wider, _, _, _, _ = U.analyse(cases, pred, "RetinAgent", resamples=resamples,
                                     seed=SEED, confidence=1-.05/family_n)
        adjusted = {r["metric"]: r for r in wider if r["kind"] == "difference"}
        for row in ci:
            if row["kind"] == "difference":
                row = {**row, "ci_bonferroni_lower": adjusted[row["metric"]]["ci_lower"],
                       "ci_bonferroni_upper": adjusted[row["metric"]]["ci_upper"],
                       "bonferroni_marginal_confidence": 1-.05/family_n}
            rows.append(dict(task=task, analysis_comparator=comparator, scope=scope, seed=SEED, **row))
        # The earlier helper's within-call adjustment covers one comparator.
        # Replace it with the explicitly declared seven-comparison family.
        test[0].pop("p_holm_comparators")
        tests.append(dict(task=task, scope=scope, n=len(cases), family_size=family_n, **test[0]))
    for row, p in zip(tests, U.holm([r["p_value"] for r in tests])):
        row["p_holm_seven_comparisons"] = p
    return rows, tests


def cell(row, digits=3):
    if row["ci_lower"] is None:
        return f"{row['estimate']:.{digits}f} [NOT FOUND]"
    return f"{row['estimate']:.{digits}f} [{row['ci_lower']:.{digits}f}, {row['ci_upper']:.{digits}f}]"


def exact_intervals(summary):
    rows = []
    for metric, success, total in (("sensitivity", "tp", "positive"), ("specificity", "tn", "negative")):
        n, k = int(summary[total]), int(summary[success])
        require(n > 0, f"Undefined {metric}")
        interval = binomtest(k, n).proportion_ci(confidence_level=.95, method="exact")
        rows.append(dict(task=summary["task"], method=summary["method"], source=summary["source"],
            scope=summary["status"], metric=metric, estimate=k/n, successes=k, denominator=n,
            ci_lower=float(interval.low), ci_upper=float(interval.high), confidence=.95,
            interval_method="Clopper-Pearson exact binomial"))
    return rows


def tex_interval(row):
    bounds = (f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
              if row["ci_lower"] is not None else "NOT FOUND")
    # Braces prevent a leading '[' from becoming the optional spacing of '\\'.
    return r"\shortstack{" + f"{row['estimate']:.3f}" + r"\\{" + bounds + "}}"


def tables(out, summaries, intervals, differences, tests, exact):
    lookup = {(r["task"], r["method"], r["metric"]): r for r in intervals}
    lookup.update({(r["task"], r["method"], r["metric"]): r for r in exact})
    table = [r"% Generated from saved predictions; provisional source reconciliation, not new experiments.",
             r"\begin{table}[p]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Recomputed FairVision results. $n$ is the number of valid saved predictions, not the intended cohort size (250 per task). F1 and worst-group F1 use fixed-label macro-averaging over classes 0 and 1. Worst groups are race, sex and age ($<50$, 50--69, $\geq70$), excluding unknown values, without a support cutoff. Intervals are supplied separately. AMD RetinAgent and glaucoma MIRAGE are incomplete; their rows are not full-cohort estimates. AMD foundation-source versions differ from the previous draft and remain unresolved. No best-value highlighting or superiority claim is made.}",
             r"\label{tab:fairvision_recomputed}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Method & $n$ & Macro-F1 & Worst macro-F1 & Sensitivity & Specificity & Balanced acc. \\"]
    ci_tables = [r"% Requires booktabs. Three separate panels avoid an oversized table."]
    for task in audit.TASKS:
        task_name = dict(glaucoma="Glaucoma", amd="AMD", dr="DR")[task]
        table += [r"\midrule", r"\multicolumn{7}{l}{\textbf{" + task_name + r"}} \\"]
        ci_tables += [r"\begin{table}[p]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2pt}",
            r"\caption{" + task_name + r": marginal 95\% intervals. Sensitivity and specificity use exact binomial intervals; F1 and balanced accuracy use exploratory class-stratified percentile bootstrap intervals (10,000 replicates), conditional on fixed predictions and observed class counts. $n$ counts valid responses. Worst-group minima are reselected over the fixed group set in every draw; small-group and boundary intervals can be unstable. Unknown groups are excluded. Missing predictions are not imputed."
            + (" AMD source-version discrepancies remain unresolved." if task == "amd" else "") + "}",
            r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r"Method & $n$ & Macro-F1 & Worst macro-F1 & Sensitivity & Specificity & Balanced acc. \\", r"\midrule"]
        for row in [r for r in summaries if r["task"] == task]:
            name = DISPLAY.get(row["method"], row["method"])
            table.append(" & ".join([name, row["n"], *[f"{float(row[k]):.3f}" for k in TABLE_METRICS]]) + r" \\")
            cells = []
            for metric in TABLE_METRICS:
                r = lookup[task, row["method"], metric]
                cells.append(tex_interval(r))
            ci_tables.append(" & ".join([name, row["n"], *cells]) + r" \\")
        ci_tables += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "table1_recomputed.tex").write_text("\n".join(table) + "\n")
    (out / "table1_intervals.tex").write_text("\n".join(ci_tables) + "\n")
    paired = [r"\begin{table}[p]", r"\centering\scriptsize",
        r"\caption{RetinAgent minus comparator macro-F1. Intervals are marginal 95\% paired class-stratified percentile intervals. Exact two-sided McNemar tests compare error rates, not F1; Holm correction covers all seven displayed comparisons, including the AMD sensitivity comparison with the manuscript-named URFound. AMD rows use only 210 valid agent responses and cannot support full-cohort claims. Comparators were identified retrospectively; these are exploratory analyses.}",
        r"\begin{tabular}{@{}llrcrr@{}}", r"\toprule",
        r"Task & Comparator & $n$ & $\Delta$ macro-F1 [95\% CI] & Exact $p$ & Holm $p$ \\", r"\midrule"]
    for test in tests:
        d = next(r for r in differences if r["task"] == test["task"] and r["kind"] == "difference"
                 and r["analysis_comparator"] == test["comparator"] and r["metric"] == "f1_macro")
        paired.append(" & ".join([test["task"], DISPLAY.get(test["comparator"], test["comparator"]),
            str(test["n"]), cell(d, 4), f"{test['p_value']:.4g}", f"{test['p_holm_seven_comparisons']:.4g}"]) + r" \\")
    paired += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "paired_comparisons.tex").write_text("\n".join(paired) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    require(not out.exists() or not any(out.iterdir()), "Output must be new or empty; do not overwrite an audit")
    out.mkdir(parents=True, exist_ok=True)
    audit.SOURCES.clear()
    methods, summaries = load_saved(out)
    intervals = []
    for row in summaries:
        intervals.extend(dict(task=row["task"], method=row["method"], source=row["source"],
                              scope=row["status"], expected_n=250, **r)
                         for r in method_intervals(methods[row["task"]][row["method"]]))
    paired, tests = comparison_intervals(methods)
    exact = [r for summary in summaries for r in exact_intervals(summary)]
    audit.write(out, "method_intervals.csv", intervals)
    audit.write(out, "binomial_intervals.csv", exact)
    audit.write(out, "paired_intervals.csv", paired)
    audit.write(out, "paired_tests.csv", tests)
    tables(out, summaries, intervals, paired, tests, exact)
    # Input and output hashes are separate: derived files must not masquerade
    # as independent historical evidence.
    sources = {k: v for k, v in audit.SOURCES.items() if not (ROOT / k).resolve().is_relative_to(out)}
    for path in (Path(__file__), Path(audit.__file__), Path(U.__file__), Path(audit.metrics.__file__),
                 Path(U.runner.__file__), Path(U.runner.audit.__file__)):
        sources[str(path.resolve().relative_to(ROOT))] = sha(path)
    provenance = dict(api_calls=0, model_inference_runs=0, training_runs=0, bootstrap_seed=SEED,
        bootstrap_replicates=10000, bootstrap_method="class-stratified paired percentile; fixed predictions",
        source_selection="Unchanged from October 5 twenty-question audit; original prediction hashes verified",
        family=[dict(task=t, comparator=c) for t, c in COMPARISONS],
        multiplicity="Holm for seven error-rate tests; additional Bonferroni macro-F1 difference intervals",
        table_intervals="Clopper-Pearson sensitivity/specificity; percentile bootstrap F1/balanced accuracy",
        limitations=["Retrospective comparator selection; not a prespecified confirmatory family",
            "AMD missing labels and source-version discrepancies remain unresolved",
            "One recovered glaucoma reference has not been verified against the original NPZ",
            "Case is the resampling unit; patient clustering is not verified",
            "No estimates of LLM run-to-run variation; no fresh test cases",
            "Small-group/minimum/bootstrap boundary intervals can have poor coverage"],
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        source_sha256=sources,
        output_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(summaries)} method rows, {len(intervals)} intervals, {len(tests)} paired tests to {out}")
    print("API requests: 0. Historical predictions and manuscripts unchanged.")


if __name__ == "__main__":
    main()
