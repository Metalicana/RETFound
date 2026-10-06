FAIRVISION STATISTICAL REVISION: DECISION ITEM 1
=============================================
Target: AI_Agent_For_Eye_Disease-12.pdf, supplied 2026-10-05.
Scope: FairVision statistics, metric definitions, response denominators, claims.
This is not a completed response to all 12 simulated review comments.

The full source of the supplied PDF is not in this checkout. Following the
existing manuscript/revision_20261003 convention, fairvision_replacements.tex
contains insertion-ready replacements with page and section anchors. It does
not silently overwrite the PDF or claim to be a compiled revised manuscript.

Files
-----
Narrative: fairvision_replacements.tex (this directory).
Generated tables, CSVs and hashes:
  equi-agent/outputs/audits/fairvision_statistics_revision_20261005/

table1_recomputed.tex: 37 available method/task rows, actual valid n, corrected
  macro-F1 and worst-group macro-F1, no best-value highlighting.
table1_intervals.tex: three task panels with marginal 95% intervals for all
  Table 1 metrics, including the recomputed worst-group minimum. Sensitivity
  and specificity use exact binomial intervals; other columns use bootstrap.
paired_comparisons.tex: seven comparisons with paired macro-F1 differences,
  marginal intervals, exact McNemar p values and seven-comparison Holm p values.
method_intervals.csv: eight metrics per row (296 intervals), seed and support.
binomial_intervals.csv: 74 exact Clopper-Pearson sensitivity/specificity
  intervals, avoiding zero-width percentile intervals at perfect accuracy.
paired_intervals.csv: method and difference intervals; includes Bonferroni
  bounds at 99.2857% individual confidence for the seven macro-F1 contrasts.
fairvision_metrics.csv: per-method source file, confusion counts, valid n.
fairvision_aligned_predictions.csv: case-level labels and predictions.
fairvision_missing.csv: missing/invalid predictions, not imputed.
fairvision_subgroups.csv: group support, class counts and metrics.
provenance.json: prediction-source hashes verified against the prior audit,
  analysis/input/output hashes, software versions and stated limitations.

Findings that change the prose
-----------------------------
Glaucoma versus RETFound, n=250:
  Delta macro-F1 +0.0228, marginal 95% CI [-0.0274, 0.0743].
  Exact McNemar p=0.5327; Holm p=1.0000 (seven-comparison family).
DR versus RETFound, n=250:
  Delta macro-F1 +0.0201, marginal 95% CI [-0.0081, 0.0486].
  Exact McNemar p=0.2668; Holm p=1.0000.
Neither comparison establishes superiority or equivalence. Positive point
differences can be described, but not as demonstrated performance improvements.
The complete glaucoma/DR comparisons with GPT-5.1 have positive difference
intervals, including the additional Bonferroni bounds, and smaller error rates
after Holm adjustment. This does not isolate individual agent contributions.

AMD, n=210/250 valid responses:
  Macro-F1 0.8850, weighted F1 0.8858, worst-group macro-F1 0.4375.
  Delta versus VisionFM OCT -0.0092, CI [-0.0579, 0.0429].
  Delta versus URFound OCT +0.0279, CI [-0.0245, 0.0805].
Both comparisons use the same 210 cases, not full-250 baseline metrics.
The invalid-response exclusion is not a selective-escalation policy.

Do not treat the corrected table as a resolution of AMD source provenance.
Available full-250 VisionFM OCT / URFound OCT predictions give macro-F1
0.8680 / 0.8474, not the draft's 0.8480 / 0.8637. The source list is frozen to
the prior audit; no run is selected by whichever result looks better.

Methods and limitations
-----------------------
10,000 class-stratified paired percentile replicates, seed 20261005. Same
case draws for both methods. Fixed predictions, no test-set resampling of
model training, no LLM repeats. Patient clustering cannot be verified.
Table sensitivity/specificity intervals are exact binomial, not bootstrap;
paired metric differences remain bootstrap-based. Both are exported.
All seven error-rate tests share one retrospective Holm family, including
the incomplete AMD comparisons. The earlier six-comparison audit is not
overwritten; its adjusted p values must not be mixed with this new family.
Macro-F1 difference intervals are marginal unless explicitly Bonferroni.
The Bonferroni family is seven contrasts for one metric, not all endpoints
and not every previous baseline search. McNemar does not test macro-F1.
Worst-group minima are reselected in every draw. Known groups are fixed;
unknown values are excluded. Missing groups invalidate the interval rather
than disappearing. No such undefined draws occurred in the saved 10,000-
replicate analysis. Small-group/minimum/boundary percentile intervals may
have poor coverage; these are exploratory, not clinical guarantees.
Single-class groups are retained with fixed-label macro-F1; perfect prediction
of a single observed class yields 0.5, not 1.0.

Still open; not marked fixed by this bundle
-----------------------------------------
- Full-250 AMD agent performance: NOT FOUND; 40 predictions missing/invalid.
- AMD baseline source-version reconciliation: unresolved.
- Historical glaucoma sampling procedure/seed and original NPZ reference
  verification: unresolved.
- GDP detection attribution and its paired inference: not resolved here.
- GDP progression, external-cohort and ablation uncertainty: not supplied by
  this FairVision-only bundle. Existing partial/exploratory caveats remain.
- External full-cohort comparisons: do not keep the current 'improved' or
  'approaching' claims on unmatched denominators pending their own revision.
- All other review items, including live escalation, counterfactual attribution,
  learned combination, fairness gaps and repeat variability: separate work.

Reproduce (local saved outputs only)
-----------------------------------
Use an environment with numpy and scipy. The analysis refuses to overwrite
a nonempty output directory. For a new reproduction directory:

python equi-agent/scripts/build_fairvision_statistical_revision.py \
  --out-dir /tmp/retinagent_fairvision_statistics_reproduction

python -m unittest discover -s equi-agent/tests \
  -p test_fairvision_statistical_revision.py -v

Verification: 14 revision tests plus 28 existing twenty-question audit,
glaucoma uncertainty and result-gap tests passed (42 total), using
/private/tmp/ophthalmic-progression-tests/bin/python. Tests check sklearn F1,
paired ID alignment, exact binomial/McNemar values, Holm adjustment, interval
reproducibility, missing-group handling, partial-cohort retention, source
hashes, and LaTeX interval-cell syntax. No LaTeX compiler is installed here;
the manuscript and tables have not been compile/layout-verified.

No model inference, training, API requests, remote access, or Git writes.
The future 250-request budget is untouched. The old 20k-call runner is not
used by this analysis and has NOT thereby become a 250-call runner.
