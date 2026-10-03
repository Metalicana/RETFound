"""Offline score/ranking sensitivity, not an LLM performance ablation or weight search."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import rank_demographic_reliability_models as ranking
from compute_demographic_reliability_score import (
    DEFAULT_PRIORS_JSON, DEFAULT_SUPPORT_CSV, load_json_rows, norm,
    norm_gender, norm_race, prior_lookup, read_csv,
)

CANONICAL = dict(fnr=.35, fpr=.25, ece=.15, auroc=.15, f1=.10)
ROOT = Path(__file__).resolve().parents[2]


def scenarios():
    result = [("canonical", CANONICAL, 50.0, False),
              ("equal_weights", dict.fromkeys(CANONICAL, .2), 50.0, False),
              ("historical_fnr_fpr", dict(fnr=.85, fpr=.15, ece=0, auroc=0, f1=0), 50.0, False)]
    for name in CANONICAL:
        for label, factor in (("omit", 0), ("minus20pct", .8), ("plus20pct", 1.2)):
            weights = {k: v * (factor if k == name else 1) for k, v in CANONICAL.items()}
            total = sum(weights.values())
            result.append((f"{label}_{name}", {k: v / total for k, v in weights.items()}, 50.0, False))
    result.extend([("shrinkage_k25", CANONICAL, 25.0, False),
                   ("shrinkage_k100", CANONICAL, 100.0, False),
                   ("global_only", CANONICAL, 50.0, True)])
    return result


def winners(scores, tolerance=1e-10):
    lowest = min(scores.values())
    return {model for model, value in scores.items() if abs(value - lowest) <= tolerance}


def source(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def score(lookup, support, models_by_task, scenario):
    name, weights, k, global_only = scenario
    args = argparse.Namespace(k=k, **{f"{key}_weight": value for key, value in weights.items()})
    result = {}
    for combo in sorted(support["combo_counts"]):
        task, age, race, sex = combo
        result[combo] = {}
        for model in models_by_task[task]:
            row = ranking.score_model_for_combo(lookup, support, task, model, age, race, sex, args)
            value = row["R_global"] if global_only else row["score"]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid score: {name}/{combo}/{model}: {value}")
            result[combo][model] = value
    return result


def compare(canonical, alternative, support):
    by_task = defaultdict(list)
    details = []
    for combo, scores in canonical.items():
        other = alternative[combo]
        old, new = winners(scores), winners(other)
        drifts = [abs(scores[m] - other[m]) for m in scores]
        row = dict(task=combo[0], age_group=combo[1], race=combo[2], gender=combo[3],
                   n_support=support["combo_counts"][combo],
                   canonical_winners=";".join(sorted(old)), alternative_winners=";".join(sorted(new)),
                   winner_set_changed=old != new, winner_sets_disjoint=not bool(old & new),
                   mean_absolute_trust_change=sum(drifts) / len(drifts),
                   max_absolute_trust_change=max(drifts),
                   retfound_absolute_trust_change=abs(scores["retfound_oct"] - other["retfound_oct"])
                   if "retfound_oct" in scores else None)
        details.append(row)
        by_task[combo[0]].append(row)
    summaries = []
    for task, rows in sorted(by_task.items()):
        n = sum(r["n_support"] for r in rows)
        summaries.append(dict(task=task, combinations=len(rows), support_rows=n,
            changed_winner_sets=sum(r["winner_set_changed"] for r in rows),
            disjoint_winner_sets=sum(r["winner_sets_disjoint"] for r in rows),
            support_weighted_winner_change=sum(r["n_support"] * r["winner_set_changed"] for r in rows) / n,
            mean_absolute_trust_change=sum(r["n_support"] * r["mean_absolute_trust_change"] for r in rows) / n,
            max_absolute_trust_change=max(r["max_absolute_trust_change"] for r in rows),
            retfound_max_absolute_trust_change=max((r["retfound_absolute_trust_change"] for r in rows
                                                   if r["retfound_absolute_trust_change"] is not None), default=None)))
    return summaries, details


def verify_precomputed(canonical, path):
    saved = {}
    for row in read_csv(path):
        key = (norm(row["task"]), norm(row["age_group"]), norm_race(row["race"]), norm_gender(row["gender"]))
        model = norm(row["model_name"])
        if key in canonical and model in canonical[key]:
            if (key, model) in saved:
                raise ValueError(f"Duplicate precomputed score: {key}/{model}")
            saved[key, model] = float(row["final_R_bad"])
    differences = []
    for combo, scores in canonical.items():
        for model, value in scores.items():
            if (combo, model) not in saved:
                raise ValueError(f"Missing precomputed score: {combo}/{model}")
            delta = abs(saved[combo, model] - value)
            if not math.isfinite(delta) or delta > 1e-9:
                raise ValueError(f"Canonical reconstruction differs from saved table: {combo}/{model}: {delta}")
            differences.append(delta)
    return dict(rows=len(differences), max_absolute_difference=max(differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors-json", type=Path, default=ROOT / DEFAULT_PRIORS_JSON)
    parser.add_argument("--support-csv", type=Path, default=ROOT / DEFAULT_SUPPORT_CSV)
    parser.add_argument("--precomputed-csv", type=Path, default=ROOT / "OphthalmicAgent/_extras/CSVs/demographic_reliability_subgroup_model_scores.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "equi-agent/outputs/audits/reliability_weight_sensitivity")
    args = parser.parse_args()
    priors = load_json_rows(args.priors_json)
    lookup = prior_lookup(priors)
    if len(lookup) != len(priors):
        raise ValueError("Duplicate normalized prior keys; do not silently overwrite")
    for row in priors:
        for metric in CANONICAL:
            value = row.get(metric)
            if value is not None and not math.isnan(float(value)) and not 0 <= float(value) <= 1:
                raise ValueError(f"Invalid prior metric {metric}: {value}")
    # Whitelist covariates. This audit never uses held-out labels or predictions to choose weights.
    support_rows = [{k: r.get(k, "") for k in ("task", "age_group", "race", "sex_gender", "split")}
                    for r in read_csv(args.support_csv)]
    support = ranking.support_tables(support_rows, None)
    if not support["combo_counts"] or sum(support["n_total"].values()) != len(support_rows):
        raise ValueError("Empty or incomplete demographic support rows")
    models = {task: ranking.available_models([r for r in priors if norm(r.get("task")) == task], "foundation")
              for task in support["n_total"]}
    if any(not values for values in models.values()):
        raise ValueError("Missing foundation priors for a support task")
    baseline = score(lookup, support, models, scenarios()[0])
    verification = verify_precomputed(baseline, args.precomputed_csv)
    summaries, details, scores = [], [], []
    for scenario in scenarios():
        name = scenario[0]
        alternative = score(lookup, support, models, scenario)
        summary, detail = compare(baseline, alternative, support)
        summaries.extend({"scenario": name, **r} for r in summary)
        details.extend({"scenario": name, **r} for r in detail)
        for combo, values in alternative.items():
            scores.extend(dict(scenario=name, task=combo[0], age_group=combo[1], race=combo[2], gender=combo[3],
                               model_name=m, risk=v, trust=1-v) for m, v in values.items())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranking.write_csv(args.out_dir / "summary.csv", summaries)
    ranking.write_csv(args.out_dir / "subgroup_changes.csv", details)
    ranking.write_csv(args.out_dir / "scores.csv", scores)
    metadata = dict(analysis="offline score/ranking sensitivity only", calls_to_llm=0,
        weights_selected_by_this_analysis=False, downstream_agent_performance_tested=False,
        source_files=[source(p) for p in (args.priors_json, args.support_csv, args.precomputed_csv)],
        source_code=[source(Path(__file__)), source(Path(ranking.__file__)),
                     source(Path(ranking.risk_score.__code__.co_filename))],
        canonical_reconstruction=verification, models=models,
        support_split_counts=dict(Counter(r["split"] for r in support_rows)),
        prior_global_sample_counts=[{k: r.get(k) for k in ("task", "model_name", "n", "split")}
                                    for r in priors if norm(r.get("attribute")) == "global"],
        scenarios=[dict(name=s[0], weights=s[1], k=s[2], global_only=s[3]) for s in scenarios()],
        limitations=["Post-hoc audit, not prospective coefficient selection or clinical utility validation.",
                     "Historical support counts are reproduced, including test covariates where supplied; performance priors are a separate input.",
                     "No test labels are used; labels are not necessary for score/ranking sensitivity.",
                     "Stable rankings do not imply stable agent decisions; downstream ablations require new matched runs.",
                     "The live anchor-only agent need not select among all ranked models."])
    (args.out_dir / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    lines = ["# Reliability Weight Sensitivity", "", "Offline score/ranking analysis only. No API calls. No downstream F1 or robustness claim.",
             f"Canonical reconstruction matched {verification['rows']} saved foundation scores.",
             f"Support split counts: {metadata['support_split_counts']}. Only demographic covariates are used.", "",
             "| Scenario | Task | Changed winner sets | Support-weighted change | Max trust change | RETFound max trust change |",
             "|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        anchor = r["retfound_max_absolute_trust_change"]
        lines.append(f"| {r['scenario']} | {r['task']} | {r['changed_winner_sets']}/{r['combinations']} | "
                     f"{r['support_weighted_winner_change']:.3f} | {r['max_absolute_trust_change']:.4f} | "
                     + (f"{anchor:.4f}" if anchor is not None else "N/A") + " |")
    lines.extend(["", *[f"- {note}" for note in metadata["limitations"]]])
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Matched {verification['rows']} canonical scores; {len(scenarios())} scenarios; no API calls.\nwrote={args.out_dir}")


if __name__ == "__main__":
    main()
