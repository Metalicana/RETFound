# FairVision Escalation and Accepted-Case Performance

## Original Fixed Policy Result

[fixed_policy_results.tex](fixed_policy_results.tex) contains the methods,
per-task table and interpretation for the original fixed policy: **69.14%
escalated and accepted positive-class F1 0.9020**. Its 2,777 accepted cases
contain only glaucoma and AMD; all 3,000 DR cases were escalated. The saved
acceptance flags and confusion counts were checked against the case-level
prediction file. This is a fixed-policy result, not the score sweep below.

## Score Sweep Figure

[PDF](historical_fusion_escalation_macro_f1.pdf) |
[SVG](historical_fusion_escalation_macro_f1.svg) |
[PNG](historical_fusion_escalation_macro_f1.png)

**Historical deterministic fusion, not the live RetinAgent result.** The source
contains 3,000 held-out cases per task at their observed prevalence. It is not
the 250-case-per-task OphthalmicAgent cohort used for the main diagnostic table.
Do not present this figure as selective performance of that live system.

The three panels show macro-F1 among accepted cases against the percentage
referred to clinician review. Acceptance means `risk_score <= threshold`;
the saved diagnosis for each case is unchanged. Every distinct saved score
is swept, retaining tied scores together. Lines connect feasible operating
points without smoothing; interpolated positions are not additional measured
thresholds. No threshold is optimized or recommended using test labels.

Macro-F1 is recomputed from both class F1 scores with fixed labels `[0, 1]`.
The source `risk_coverage_curve.csv` reports positive-class F1, so it is not
used as if it contained macro-F1. Empty accepted sets have undefined F1.
Single-class subsets remain in the CSV but are not plotted; dashed segments
mark fewer than 20 accepted cases in either reference class. This is a display
warning, not an eligibility rule used to select a favorable operating point.

The saved risk score combines probability uncertainty, model disagreement and
low reliability. The original joint conformal/escalation policy is not applied
on top of this sweep. The original model pool and reliability formula are
recorded in [provenance.json](provenance.json); they differ from the live agent.
Scores are descriptive point estimates. Higher accepted-case F1 alone does not
establish practical workload reduction or improved outcomes for referred cases.
Accepted class composition changes with the threshold, which also affects F1.

## Reproduce

Run from the repository root with matplotlib installed:

```bash
MPLCONFIGDIR=/tmp/retinagent-matplotlib python \
  equi-agent/scripts/plot_fairvision_escalation_macro_f1.py
python -m unittest discover -s equi-agent/tests -p test_fairvision_escalation_macro_f1.py
```

[threshold_curve.csv](threshold_curve.csv) contains every swept threshold,
accepted and escalated counts, accepted class counts, confusion counts and F1.
[threshold_checkpoints.csv](threshold_checkpoints.csv) provides six fixed score
thresholds for inspection, not selected operating points.
[caption.tex](caption.tex) identifies the experiment explicitly.
Source files are read only; no API calls, training or patient-level exports occur.

## Live Agent Figure

The corresponding live-agent curve requires the final prediction and a saved,
well-defined continuous deferral score for every case on each locked cohort.
The currently verified live outputs are insufficient for that sweep. Binary
escalation flags can support a single selective operating point, not a curve.
Unlinked counterfactual confidences should not be substituted for the final
agent's deferral score.
