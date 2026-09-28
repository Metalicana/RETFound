# Five Motivation Figure Options

Start with **motivation_review.pdf**, a five-page review pack. All pages are
183 mm wide. **motivation_options.drawio** contains the same five editable tabs.
Individual PDF, PNG, SVG and draw.io files use numbered matching names.
The existing author-edited clinical_decisions figure is not overwritten.

## 1. Case Corrections

**Recommended for the main image-led motivation.** Two retrospective FairVision
examples show incorrect RETFound decisions followed by correct final RetinAgent
decisions. Case data_07057.npz is glaucoma: RETFound probability 0.1938, saved
threshold 0.5, agent label 1. Case data_07062.npz is non-glaucoma: probability
0.7269, saved threshold 0.5, agent label 0. Image summaries are AI-reported;
vCDR values are automated estimates, not expert annotations. The two cases were
selected to illustrate opposite error directions, not to estimate benefit.

## 2. Model Disagreement

**Recommended for motivating multiple-model reliability.** Two held-out cases
have similar opposing RETFound/VisionFM scores but different correct sources.
In data_08423.npz (glaucoma), RETFound predicts negative and VisionFM positive.
In data_09154.npz (non-glaucoma), RETFound predicts negative and VisionFM positive.
OCT is the model input; SLO is shown only as anatomical context. In the full
3,000-case matched glaucoma test benchmark, only RETFound is correct on 418
cases and only VisionFM on 370; both are correct on 1,767 and both wrong on 445.
These are saved thresholded decisions, not predictions rethresholded for the
figure. This benchmark is separate from the 249-case agent cohort. It motivates
case-specific trust but does not demonstrate that the agent chooses the right
FM, achieves oracle performance, or was tested on these illustrated cases.

## 3. Confidence and Reliability

For each model, validation cases scoring in [0.9, 1.0] are grouped separately.
Outlined bars show mean predicted glaucoma probability; filled bars show
the observed glaucoma fraction in that same model's bin. RETFound: 37/38,
URFound: 159/179, VisionFM: 259/307. Bin frequencies are descriptive, have
different sample sizes and case composition, and are not a paired model ranking.
There is no fitted curve or inferential interval. This is validation evidence,
not independent evidence of improvement by a learned reliability selector.

## 4. Evidence Ablation

In case data_07062.npz, the stored response gives non-glaucoma for full evidence,
inconclusive without the OCT/SLO reports, and non-glaucoma without the FM score.
Other available inputs, including demographics, are held conceptually unchanged
and omitted from the diagram for readability. These scenarios were generated
within one response; they are not independently rerun ablations, ground-truth
causal effects, or proof of the final orchestrator's decision mechanism. The
full-evidence label agrees with the final agent CSV. The saved AI visual report
also notes image artifacts and uncertain subtle findings. Its anatomical
interpretation has not been clinically adjudicated.

## 5. Clinical Handoff

The observed inputs and agent decision for data_07057.npz anchor a proposed
clinical workflow: review the original images, reconcile an FM score with
reported structural evidence, then present a source-linked assessment for
ophthalmologist review. The arrangement is a conceptual case brief, not an
existing tested interface. The clinician handoff (dashed arrow) is proposed;
no clinician accuracy, adoption, time-saving, or patient benefit is established.

## Shared Image and Evidence Notes

Images are genuine downloaded FairVision arrays, displayed as stored: central
OCT slice oct_bscans[100, :, :] and slo_fundus, grayscale 0-255. No denoising,
contrast enhancement, lesion annotations, generated medical imagery, or
anatomically targeted slice selection is used. The report can refer to imaging
evidence beyond what the single displayed slice supports. All source hashes,
case labels, predictions and recorded ablation outcomes are checked on build.
The inference prompts, existing experiment files, and manuscript metrics are
not changed. No numbers are inferred from a rendered table.

## Full Paired Agent Results (Not a Motivation Claim)

The illustrative examples must not replace the full results. Among 249 matched
evaluable cases, baseline TN/FP/FN/TP are 108/17/45/79; agent values are
103/22/35/89. There are 23 corrected errors and 18 introduced errors: 17 missed
positives and 6 false positives corrected, but 7 missed positives and 11 false
positives introduced. Thus false alarms increase overall even though case b
illustrates an individual false-positive correction. One raw row (data_07199)
has nonbinary truth and is excluded from both. The source data retains all
these outcomes, not only the illustrated favorable transitions. The
author-reported RETFound worst-group F1 remains 0.6344 and is not recomputed.

## Rebuild

```bash
python equi-agent/scripts/build_motivation_options.py \
  --inputs-root /path/to/extracted/retinagent_figure_inputs \
  --image-root /path/to/downloaded/npz_files
python -m unittest discover -s equi-agent/tests -p test_motivation_options.py
```

The source_data.json file preserves the independent validation, foundation-test,
and paired-agent cohorts. Keep those distinctions when reusing any panel.
Rebuilding replaces generated options. After manual draw.io edits, use a new
--out-dir for regeneration so your edited variants are retained.
