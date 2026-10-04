"""Offline, paired uncertainty analysis for recovered or ablation glaucoma results.

No inference, threshold selection, label repair or API calls. Bootstrap entire
paired cases within each truth class, conditional on the balanced cohort design.
The NPZ is the sampling unit: independence across patients must be verified
separately if multiple NPZ files can belong to the same patient.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest, bootstrap

import run_fairvision_ablation as runner

ROOT = runner.ROOT
METRICS = ("f1_macro", "f1_weighted", "f1_positive", "sensitivity", "specificity", "balanced_accuracy", "accuracy")
LABELS = dict(zip(runner.VARIANTS, ("Simple multimodal ensemble", "Reliability-weighted fusion",
                                  "Agents without reliability priors", "RetinAgent (paired full control)")))
ATTRIBUTES = ("race", "sex_gender", "age_group")
UNKNOWN = {"", "unknown", "missing", "nan", "-1"}


def read_json(path):
    return json.loads(path.read_text())


def indexed(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        runner.require(isinstance(key, str) and key.strip() and key not in result, f"Missing/duplicate ID: {key!r}")
        result[key] = row
    return result


def load_paper(root, expected):
    files = [root / name for name in ("manifest_recovered.csv", "agent_predictions_recovered.csv",
                                      "retfound_predictions_recovered.csv", "recovery_report.json")]
    locked = indexed([r for r in runner.read_csv(files[0]) if r["Task_Folder"].lower() == "glaucoma"], "filename")
    agent = indexed(runner.read_csv(files[1]), "Filename")
    baseline = indexed(runner.read_csv(files[2]), "Filename")
    runner.require(len(locked) == expected and set(locked) == set(agent) == set(baseline),
                   "Paper files must contain exactly the same complete locked glaucoma cohort")
    cases, predictions = [], {"RETFound": [], "RetinAgent": []}
    for key in sorted(locked):
        meta, a, b = locked[key], agent[key], baseline[key]
        truth = runner.binary(meta["Ground_Truth"])
        runner.require(truth == runner.binary(a["Ground_Truth"]) == runner.binary(b["Ground_Truth"]),
                       f"Truth mismatch: {key}")
        runner.require(a["Task_Folder"].lower() == b["Disease"].lower() == "glaucoma"
                       and b["Model"] == "RETFound" and b["Modality"] == "OCT", f"Task/model mismatch: {key}")
        demographics = runner.demographic(meta)
        runner.require(demographics == runner.demographic(a) == runner.demographic(b), f"Metadata mismatch: {key}")
        cases.append(dict(case_id=key, truth=truth, **demographics))
        predictions["RETFound"].append(runner.binary(b["Prediction"]))
        predictions["RetinAgent"].append(runner.binary(a["Pred_GL"]))
    recovery = read_json(files[3])
    notes = ["Recovery-stage note: " + note for note in recovery.get("limitations", [])]
    notes.append("Unlike the recovery-stage report, this audit recalculates subgroup metrics for these two methods only.")
    notes.append(f"Recovered row: {recovery.get('case_id')}; original NPZ independently verified: {recovery.get('npz_verified')}.")
    notes.append("Historical results only; not the new full/no-priors ablation and not a new confirmatory test set.")
    return cases, predictions, "RetinAgent", files, notes


def load_ablation(root, expected):
    config, prepared, offline = runner.load_prepared(root)
    cohort = [c for c in prepared if c["task"] == "glaucoma"]
    mapping = indexed(cohort, "case_id")
    runner.require(len(mapping) == expected, "Wrong ablation glaucoma cohort size")
    cohort = [mapping[key] for key in sorted(mapping)]
    files = [root / name for name in ("config.json", "prepared_cases.json", "offline_predictions.json", "live_receipt.json")]
    runner.require(read_json(files[-1])["run"] == config["fingerprint"], "Ablation receipt mismatch")
    repair = root / runner.CDR_REPAIR / "receipt.json"
    if repair.exists():
        record = read_json(repair)
        runner.require(record["status"] == "complete" and record["run_fingerprint"] == config["fingerprint"],
                       "Ablation repair is incomplete or belongs to another run")
        files.append(repair)
    predictions, evidence_ids = {}, {}
    for variant in runner.VARIANTS:
        if variant in runner.VARIANTS[:2]:
            rows = indexed([r for r in offline if r["task"] == "glaucoma" and r["variant"] == variant], "case_id")
        else:
            folder = root / "agent/glaucoma" / variant
            paths = sorted(folder.glob("*.json"))
            runner.require(len(paths) == expected, f"Incomplete {variant}: {len(paths)}/{expected}; wait for the full cohort")
            rows = indexed([read_json(p) for p in paths], "case_id")
            for key, row in rows.items():
                runner.require(row["fingerprint"] == config["fingerprint"] and row["task"] == "glaucoma"
                               and row["variant"] == variant, "Mismatched ablation decision provenance")
                shared = row["shared_evidence_sha256"]
                runner.require(shared and shared == evidence_ids.setdefault(key, shared), "Arms received different evidence")
            files.extend(paths)
        runner.require(set(rows) == set(mapping), f"Ablation case IDs differ for {variant}")
        predictions[LABELS[variant]] = [runner.binary(rows[c["case_id"]]["prediction"]) for c in cohort]
    cases = [dict(case_id=c["case_id"], truth=runner.binary(c["truth"]),
                  **{a: c[a] for a in ATTRIBUTES}) for c in cohort]
    notes = ["Full versus no-priors is the component ablation; fusion comparisons use a different model pool.",
             "Only completed glaucoma cases from this frozen run are used; historical agent scores are not inserted."]
    return cases, predictions, LABELS["retinagent_full"], files, notes


def metric_array(truth, predictions):
    """Last axis is cases; preserve model and optional bootstrap-batch dimensions."""
    tp = np.sum((truth == 1) & (predictions == 1), axis=-1)
    tn = np.sum((truth == 0) & (predictions == 0), axis=-1)
    fp = np.sum((truth == 0) & (predictions == 1), axis=-1)
    fn = np.sum((truth == 1) & (predictions == 0), axis=-1)
    pos, neg = tp + fn, tn + fp

    def divide(a, b, undefined=0.):
        return np.divide(a, b, out=np.full(np.shape(a), undefined, dtype=float), where=b != 0)

    f1p, f1n = divide(2*tp, 2*tp+fp+fn), divide(2*tn, 2*tn+fp+fn)
    se, sp = divide(tp, pos, np.nan), divide(tn, neg, np.nan)
    return np.stack(((f1p+f1n)/2, divide(pos*f1p+neg*f1n, pos+neg), f1p,
                     se, sp, (se+sp)/2, divide(tp+tn, pos+neg)), axis=1)


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, (len(values)-rank)*values[idx])
        adjusted[idx] = min(1., running)
    return adjusted.tolist()


def analyse(cases, predictions, primary, resamples=10000, seed=20261004, confidence=.95):
    runner.require(resamples >= 100 and 0 < confidence < 1, "Invalid bootstrap settings")
    indexed(cases, "case_id")
    names = list(predictions)
    runner.require(primary in names and len(names) >= 2, "Need primary method and at least one comparator")
    truth = np.array([runner.binary(c["truth"]) for c in cases])
    runner.require(all(len(p) == len(cases) for p in predictions.values()), "Prediction count mismatch")
    predicted = np.array([[runner.binary(p) for p in predictions[name]] for name in names])
    pos, neg = np.flatnonzero(truth == 1), np.flatnonzero(truth == 0)
    runner.require(len(pos) and len(neg), "Both truth classes are required")
    main = names.index(primary)
    other = [i for i in range(len(names)) if i != main]

    def statistic(positive_ids, negative_ids, axis=-1):
        # Independently sample the two class strata, but use each sampled case's
        # entire vector of model predictions. Pairing between methods is retained.
        indices = np.concatenate((positive_ids, negative_ids), axis=axis).astype(int)
        values = metric_array(truth[indices], predicted[:, indices])
        deltas = np.stack([values[main] - values[i] for i in other])
        return np.concatenate((values, deltas), axis=0)

    rng_key = "rng" if "rng" in inspect.signature(bootstrap).parameters else "random_state"
    result = bootstrap((pos, neg), statistic, vectorized=True, paired=False, method="percentile", batch=250,
                       n_resamples=resamples, confidence_level=confidence,
                       **{rng_key: np.random.default_rng(seed)})
    estimates = statistic(pos, neg)
    lo, hi = result.confidence_interval
    runner.require(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), "Undefined bootstrap interval; no replicas discarded")
    intervals = []
    for index in range(len(names) + len(other)):
        is_delta = index >= len(names)
        comparator = names[other[index-len(names)]] if is_delta else ""
        for j, metric in enumerate(METRICS):
            intervals.append(dict(kind="difference" if is_delta else "method", method=primary if is_delta else names[index],
                                  comparator=comparator, metric=metric, estimate=float(estimates[index, j]),
                                  ci_lower=float(lo[index, j]), ci_upper=float(hi[index, j]), confidence=confidence,
                                  n=len(cases), n_bootstrap=resamples))
    tests, disagreements = [], []
    for index in other:
        a, b = predicted[main] == truth, predicted[index] == truth
        corrections, regressions = int(np.sum(a & ~b)), int(np.sum(~a & b))
        discordant = corrections + regressions
        tests.append(dict(method=primary, comparator=names[index], corrections=corrections, regressions=regressions,
                          both_correct=int(np.sum(a & b)), both_wrong=int(np.sum(~a & ~b)),
                          test="exact two-sided McNemar (error rates, not F1)",
                          p_value=float(binomtest(corrections, discordant, .5).pvalue) if discordant else 1.))
        for i in np.flatnonzero(a != b):
            disagreements.append(dict(case_id=cases[i]["case_id"], comparator=names[index], truth=int(truth[i]),
                                      primary_prediction=int(predicted[main, i]), comparator_prediction=int(predicted[index, i]),
                                      outcome="correction" if a[i] else "regression"))
    for row, p in zip(tests, holm([r["p_value"] for r in tests])):
        row["p_holm_comparators"] = p
    subgroups, worst = [], []
    for name, prediction in zip(names, predicted):
        scores = []
        for attribute in ATTRIBUTES:
            for group in sorted({c[attribute] for c in cases} - UNKNOWN):
                members = [i for i, c in enumerate(cases) if c[attribute] == group]
                rows = [dict(truth=int(truth[i]), prediction=int(prediction[i])) for i in members]
                metrics = runner.audit.metrics(rows)
                subgroups.append(dict(method=name, attribute=attribute, subgroup=group, **metrics))
                scores.append((metrics["f1_macro"], attribute, group))
        runner.require(scores, "No subgroup metadata for supplementary audit")
        minimum = min(s[0] for s in scores)
        worst.append(dict(method=name, worst_group_macro_f1=minimum,
                          groups="; ".join(f"{a}={g}" for score, a, g in scores if score == minimum),
                          interval_status="Descriptive only; no CI for the minimum supplied by this audit"))
    return intervals, tests, disagreements, subgroups, worst


def report(intervals, tests, source, n, positives, resamples, confidence):
    level = f"{confidence*100:g}%"
    lines = [f"# FairVision Glaucoma Uncertainty: {source}", "", f"N={n}; positives={positives}; negatives={n-positives}.",
             f"{level} class-stratified paired percentile bootstrap intervals; {resamples:,} replicates.",
             "Intervals condition on fixed predictions and class counts; they do not measure LLM run-to-run variability.",
             "Sampling unit is one NPZ case, assuming independent patients; patient clustering is not verified by these files.", "",
             "| Method | Macro-F1 | Sensitivity | Specificity | Balanced accuracy |", "|---|---:|---:|---:|---:|"]
    display = ("f1_macro", "sensitivity", "specificity", "balanced_accuracy")
    for method in dict.fromkeys(r["method"] for r in intervals if r["kind"] == "method"):
        lookup = {r["metric"]: r for r in intervals if r["kind"] == "method" and r["method"] == method}
        cells = [f"{lookup[k]['estimate']:.4f} [{lookup[k]['ci_lower']:.4f}, {lookup[k]['ci_upper']:.4f}]" for k in display]
        lines.append("| " + " | ".join([method, *cells]) + " |")
    lines += ["", "## Paired Differences", "", "Positive differences favour the primary agent. Intervals are marginal, not multiplicity-adjusted.",
              "| Comparator | Macro-F1 difference | Balanced accuracy difference | Corrections / regressions | Exact McNemar p | Holm p |",
              "|---|---:|---:|---:|---:|---:|"]
    for test in tests:
        lookup = {r["metric"]: r for r in intervals if r["kind"] == "difference" and r["comparator"] == test["comparator"]}
        cells = [f"{lookup[k]['estimate']:+.4f} [{lookup[k]['ci_lower']:+.4f}, {lookup[k]['ci_upper']:+.4f}]"
                 for k in ("f1_macro", "balanced_accuracy")]
        lines.append("| " + " | ".join([test["comparator"], *cells,
            f"{test['corrections']} / {test['regressions']}", f"{test['p_value']:.4g}", f"{test['p_holm_comparators']:.4g}"]) + " |")
    lines += ["", "McNemar tests paired error rates, not macro-F1. Holm adjustment covers the displayed comparator error tests only.",
              "A difference interval containing zero does not establish superiority or equivalence.",
              "Weighted and positive-class F1 are separately labelled in intervals.csv; historical weighted F1 is not renamed macro-F1.",
              "Subgroup support, confusion counts and worst-group macro-F1 are retained in supplementary CSVs, without post-hoc exclusions.",
              "Subgroups: race, sex and age (<50, 50-69, >=70); unknown values excluded; no support cutoff; fixed-label [0,1] macro-F1.",
              "No AUROC is inferred from binary labels. No methods, thresholds or prompts are selected using this audit.",
              "These analyses do not undo previous inspection of test results.", "",
              "Methods: [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); "
              "[exact McNemar test](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("paper", "ablation"), default="paper")
    parser.add_argument("--paper-root", type=Path, default=ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_case_recovery")
    parser.add_argument("--run-root", type=Path, default=ROOT / "OphthalmicAgent/outputs/fairvision_ablation_v1")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--expected-cases", type=int, default=250)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--confidence", type=float, default=.95)
    args = parser.parse_args()
    loader, root = (load_paper, args.paper_root) if args.source == "paper" else (load_ablation, args.run_root)
    cases, predictions, primary, files, notes = loader(root.resolve(), args.expected_cases)
    outputs = analyse(cases, predictions, primary, args.resamples, args.seed, args.confidence)
    out = (args.out_dir or ROOT / f"equi-agent/outputs/audits/fairvision_glaucoma_uncertainty_{args.source}_v1").resolve()
    names = ("intervals.csv", "paired_tests.csv", "discordant_cases.csv", "subgroups.csv", "worst_groups.csv")
    all_names = (*names, "report.md", "provenance.json", "paired_cases.csv")
    for name in all_names:
        runner.require(not (out / name).is_symlink() and (out / name).resolve() not in {p.resolve() for p in files},
                       "Refusing to overwrite an input or a symlink")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(names, outputs):
        runner.audit.write_csv(out / name, rows)
    paired = [{**c, **{name: int(p[i]) for name, p in predictions.items()}} for i, c in enumerate(cases)]
    runner.audit.write_csv(out / "paired_cases.csv", paired)
    text = report(outputs[0], outputs[1], args.source, len(cases), sum(c["truth"] for c in cases), args.resamples, args.confidence)
    text += "\n## Source Limitations\n\n" + "\n".join("- " + n for n in notes) + "\n"
    (out / "report.md").write_text(text)
    provenance = dict(source=args.source, primary=primary, expected_cases=args.expected_cases,
                      n_bootstrap=args.resamples, confidence=args.confidence, seed=args.seed,
                      numpy_version=np.__version__, scipy_version=scipy.__version__,
                      method="class-stratified paired percentile bootstrap; fixed-label [0,1] macro-F1",
                      sampling_unit="NPZ case; patient independence assumed, not verified", subgroup_ci=False,
                      api_calls=0, source_limitations=notes,
                      sources={str(p.resolve()): runner.sha(p) for p in files},
                      code={str(p.relative_to(ROOT)): runner.sha(p) for p in [Path(__file__), *runner.code_paths()]})
    runner.write_json(out / "provenance.json", provenance)
    print(text)
    print(f"wrote={out}; no API calls")


if __name__ == "__main__":
    main()
