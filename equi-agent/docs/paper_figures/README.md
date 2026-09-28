# RetinAgent Paper Figures

## Five New Motivation Options

Open **[motivation_review.pdf](motivation_options/motivation_review.pdf)** for
the five new alternatives, or [contact_sheet.png](motivation_options/contact_sheet.png)
to compare them at a glance. The [editable draw.io file](motivation_options/motivation_options.drawio)
has five corresponding tabs. Each option also has its own PDF, SVG and PNG.

1. **Case corrections:** real OCT/SLO examples and recorded incorrect-to-correct decisions.
2. **Model disagreement:** nearly identical model scores, opposite correct models.
3. **Confidence and reliability:** grouped validation-bin bars, with observed support.
4. **Evidence ablation:** one case, three recorded evidence scenarios.
5. **Clinical handoff:** recorded case inputs within a proposed clinician-review workflow.

Start with **1** for an image-led agent motivation, or **2** for the multiple-model
reliability problem. [Captions and source notes](motivation_options/README.md)
separate observed results, illustrative cases, and proposed clinical workflow.
The older `retinagent_paper_figures.pdf` is not this review pack.

## Preserved Author-Edited Figure

[clinical_decisions/clinical_decisions.drawio](clinical_decisions/clinical_decisions.drawio)
and its [PDF](clinical_decisions/clinical_decisions.pdf) remain unchanged.
This preserves the author's edited
OCT/SLO case panels and shows recorded model errors followed by correct agent
decisions. Panel c shows the recorded evidence-ablation response for case b:
**inconclusive without OCT/SLO reports; non-glaucoma with full evidence**.
These are reported scenarios within one LLM call, not independently rerun
interventions or proof of a causal mechanism.
See [caption and provenance](clinical_decisions/caption.md).

The denominator is **249 evaluable cases**, not 250. One agent-file row has
nonbinary truth and is absent from the baseline file. All remaining IDs and
labels agree. The 23 corrected errors and 18 introduced errors are included in
the saved counts. The full paired outcomes remain in the caption's results
notes and source data: **45 to 35 missed glaucoma cases; 17 to 22 false alarms**.
This trade-off belongs in experimental results and limitations, not as the
motivation panel. The selected case illustrations do not demonstrate broad
superiority or measured benefit to patients.
Anatomical descriptions are AI-generated reports, not expert findings.

The manuscript-reported RETFound worst-group F1 remains **0.6344**. Reported
F1 values are attributed separately in the figure source data; they are not
recalculated or plotted in this motivation figure.

Superseded motivation drafts, exports and their generators have been removed.
Experiment outputs, downloaded source images, the architecture diagram and
the historical audit figures are retained.

Use `figure_02_architecture.drawio` next for the conceptual architecture.
The remaining tabs in `retinagent_paper_figures.drawio` are older quantitative
layouts, not completed result figures. All elements are native editable draw.io
shapes and text. The recovered audit figures below are not the motivation figure.

## Recovered CECSL Results (2026-09-27)

The transferred archive has been inspected separately from the repo's original
outputs. Open [recovered/recovered_experiments.drawio](recovered/recovered_experiments.drawio)
for three **data-backed historical/exploratory** figures:

1. Reliability-rule comparison and positive-vote bonus sensitivity.
2. Deterministic risk-coverage curves with retained class composition.
3. Paired PAPILA, Drishti-GS and GAMMA changes, with bootstrap intervals.

See the [audit](recovered/audit.md) and [aggregate source data](recovered/source_data.json).
The earlier audit could not reproduce the manuscript score from the other live
glaucoma run. The clinical-decision figure instead uses the existing OphthalmicAgent
predictions paired with its raw RETFound baseline, explicitly accounting for the invalid
ground-truth row. The other run identities
and the justification for fixed reliability coefficients remain unresolved.
No prompts or experiment outputs were changed.

Rebuild from a separate extracted archive (standard library only):

```bash
python equi-agent/scripts/build_recovered_experiment_figures.py \
  --inputs-root /path/to/separate/extracted/archive
python -m unittest discover -s equi-agent/tests -p test_recovered_experiment_figures.py
```

## Current Scope

| Figure | Scientific question | Status |
| --- | --- | --- |
| 1 | Why might a model score alone be insufficient for a case assessment? | `clinical_decisions/clinical_decisions.drawio`: two recorded case examples and a clearly labelled evidence-ablation scenario; paired outcomes retained in supporting results |
| 2 | How does reliability inform evidence arbitration and clinical handoff? | Complete conceptual diagram; confirm against the final run configuration |
| 3 | What is the risk versus accepted coverage trade-off? | Axes and comparisons planned; no fabricated curves |
| 4 | Where does arbitration help or hurt on external datasets? | Paired-comparison layout only; keep negative results |

Figure 2 is conceptual, not evidence that every experiment invokes every source
or module. It does not imply that the system independently makes clinical
decisions, that subgroup membership is direct disease evidence, or that
demographic invariance establishes fairness. Evidence-ablation reasoning is
shown as conditional on the run. No demographic-counterfactual agent is added.
The diagram uses RetinAgent for consistent paper naming without renaming code.

## Important Provenance Finding

The existing local file
`equi-agent/outputs/fairvision_reliability_selective_arbitration/selective_arbitration_summary.json`
contains the quoted approximately 0.4% label-flip and 5.8% escalation-flip rates.
This is the deterministic selective-arbitration implementation, not an LLM run.
It uses 9,000 task-case rows (3,000 test rows per task), which are not the same
as the paper's previously discussed locked 250-case slices. Do not relabel this
curve or these counterfactual statistics as live RetinAgent results.

Its recorded reliability weights also differ from the 0.35 FNR / 0.25 FPR /
0.15 ECE / 0.15 AUROC-complement / 0.10 F1-complement formulation in the review.
Figure 2 deliberately omits numeric coefficients until the final method and
run-specific configuration are confirmed.

The older `equi-agent/manuscript/` directory is absent in the current checkout.
The current figures live under `docs/paper_figures`. No manuscript or
experimental prediction file was changed by the figure cleanup.

## First CECSL Transfer: Inventory Only

The first path-only check needs no code sync. Run on CECSL:

```bash
cd ~/RETFound
find equi-agent/outputs OphthalmicAgent/outputs -type f \
  \( -name '*summary*.json' -o -name '*predictions*.csv' \
     -o -name '*risk*coverage*.csv' -o -name '*resolved*config*.json' \) \
  -print | sort > /tmp/retinagent_result_inventory.txt
```

Download from your Mac:

```bash
scp ab575577@10.171.42.25:/tmp/retinagent_result_inventory.txt /tmp/
```

For column names and selected summary metadata as well, after you push the
code and pull it on CECSL, run there:

```bash
cd ~/RETFound
python equi-agent/scripts/audit_paper_figure_inputs.py
```

Download from your Mac, while connected to the network used to reach CECSL:

```bash
scp ab575577@10.171.42.25:/tmp/retinagent_figure_audit.zip /tmp/
```

The inventory contains file paths, sizes, timestamps, CSV column names and a
small whitelist of run-summary fields. It does not copy patient rows, image
files, model weights, raw LLM responses, environment files or API credentials.
It does not run inference or alter experiment outputs. Network access and the
SSH address must match your current CECSL connection.

Once we identify the final run directories from that inventory, request only
the needed prediction/metric files. Do not transfer every output directory.

## Required Data for the Quantitative Figures

- Figure 1: per-model predictions on identical cases, ground truth, task,
  modality, subgroup columns and fixed validation-selected thresholds. Subgroup
  counts and paired errors are required; aggregate F1 cannot establish
  complementary failure.
- Figure 3: the same cohort for each method, probability/prediction, a continuous
  uncertainty or reliability ranking score, escalation flag, validation policy,
  missing/invalid outputs and run configuration. A binary escalation flag gives
  one operating point, not an entire curve. A curve from one deterministic
  algorithm cannot stand in for an unrun agent or ablation.
- Figure 4: paired baseline and agent outputs for all eligible cases, exact
  metric averaging, patient grouping, dataset/adaptation provenance and invalid
  output accounting. Include PAPILA regardless of the sign of its result.

Risk is the error fraction among accepted cases. Coverage is the accepted
fraction of the locked eligible cohort. Report accepted-case class counts and
F1 separately; accepted-case F1 is not itself the risk. Select operational
thresholds on validation, never to maximize the test plot. Use patient-level
paired bootstrap intervals where case data and grouping permit them.

The optional future human-study figure needs actual reader-study data. It is
not included here and clinician benefit is not claimed.

## Rebuild

```bash
python equi-agent/scripts/build_paper_drawio.py
```

This regenerates the two draw.io files, overwriting manual edits to those
generated files. Save a differently named working copy before manual editing.
Open `figure_02_architecture.drawio` for the architecture alone. PDF and PNG
exports are generated with the installed draw.io app, not image generation.

## Draft Figure 2 Caption

**RetinAgent as a reliability layer for clinician-facing decision support.**
Available retinal imaging, foundation-model outputs and clinical or structural
findings form the case evidence. Validation-derived task- and subgroup-specific
performance estimates qualify source reliability, with sparse subgroup
estimates shrunk toward global estimates. Patient demographic context informs
reliability lookup rather than serving as direct diagnostic evidence. Evidence
arbitration combines findings with this trust context, examines conflicts and,
where enabled, performs evidence-ablation reasoning. The system provides a
provisional assessment with supporting evidence or an escalation with reasons;
the ophthalmologist retains final responsibility in both branches. This is a
conceptual workflow; available modalities, tools and arbitration policies are
specified separately for each experiment.
