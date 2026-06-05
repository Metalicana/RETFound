# Equi-Agent Project Update

Last updated: 2026-06-04

This file is intended as a handoff note for future Codex/agent sessions. It summarizes the current state of the Equi-Agent retinal foundation-model project, the major experiments already run, the important negative results, the current strongest claims, and the exact next commands to continue.

## Project Goal

The project is developing **Equi-Agent**, a reliability-aware arbitration framework for ophthalmic foundation-model outputs. The intended paper is not just another foundation-model leaderboard. The scientific goal is to show that retinal foundation models can have strong average performance while remaining unreliable across diseases, modalities, or patient subgroups, and that an agentic arbitration layer can audit reliability, expose uncertainty, apply validation-derived reliability priors, and escalate uncertain cases.

The key tasks so far:

- FairVision cross-sectional diagnosis:
  - AMD
  - diabetic retinopathy
  - glaucoma
- Harvard-GDP:
  - glaucoma detection
  - progression forecasting
  - raw visual-field total-deviation/MD label-circularity audit
- External glaucoma validation:
  - currently started with REFUGE2 via KaggleHub
  - focus is non-MD structural glaucoma evidence using optic disc/cup masks

## Important Reframing

Prompt tuning alone is no longer the main path.

Multiple FairVision live Equi-Agent prompt variants were tested. The pattern is stable: LLM arbitration can be made auditable and parseable, but it does not reliably beat validation-selected foundation-model thresholds on raw F1. The useful direction is now:

1. Use foundation models and deterministic validation thresholds as strong diagnostic baselines.
2. Use Equi-Agent as reliability audit, calibration policy, explanation, and escalation layer.
3. For glaucoma, add explicit structural evidence such as vertical cup-to-disc ratio from masks or segmentation, rather than trying to prompt-tune the LLM into inferring structure indirectly.
4. Validate glaucoma on a non-MD-labeled external optic-disc dataset.

This matters because FairVision/GDP glaucoma labels appear strongly related to MD/visual-field information. We should not base the main glaucoma claim on hidden MD leakage.

## Key Files

Main manuscript:

- `equi-agent/manuscript/equi_agent_current_results.tex`

FairVision live agent runner:

- `equi-agent/scripts/run_equi_agent_fairvision_live.py`
- `equi-agent/scripts/run_fairvision_equi_agent_full.sh`

GDP raw visual-field runner:

- `equi-agent/scripts/run_gdp_raw_vf_agent.py`
- `equi-agent/FunctionalInterpretationAgent/function_interpreter.py`

FairVision helper / comparison scripts:

- `equi-agent/scripts/summarize_equi_agent_live.py`
- `equi-agent/scripts/compare_equi_agent_subset.py`
- `equi-agent/scripts/summarize_few_shot_recommendations.py`

REFUGE2 / external glaucoma scripts:

- `equi-agent/scripts/inspect_external_glaucoma_dataset.py`
- `equi-agent/scripts/build_refuge2_manifest.py`
- `equi-agent/scripts/summarize_refuge2_manifest.py`
- `equi-agent/scripts/evaluate_refuge2_cdr_baseline.py`
- `equi-agent/scripts/run_refuge2_structural_glaucoma_validation.py`

## Major Code Changes Already Made

### FairVision Dynamic Few-Shot Agent

`run_equi_agent_fairvision_live.py` now includes a dynamic few-shot prompt mode:

- `PROMPT_VARIANT=dynamic_few_shot`
- retrieves nearest validation cases by model probabilities, threshold margins, vote pattern, and weighted probability
- writes `few_shot_case_ids`
- supports `few_shot_recommendation`
- supports `FEW_SHOT_ACTION_MODE`:
  - `llm`
  - `stabilized`
  - `recommendation`

The intent was to test Medprompt-style nearest validation examples without showing the current test label.

Outcome: it fixed parse errors, but did not solve diagnostic performance.

### GDP Raw Visual Field Agent

`FunctionalInterpretationAgent/function_interpreter.py` now includes:

- raw TD column parsing
- computed MD-like score from TD points
- visual-field severity helper
- TD summary helpers

`run_gdp_raw_vf_agent.py` now:

- reads GDP raw total-deviation values
- computes `computed_md_from_td`
- hides stored MD and label from prompt-time evidence
- audits stored MD separately
- can run deterministic functional-agent mode
- writes predictions, trace, usage, errors, and summary

Outcome: computed MD from raw VF basically recovers GDP glaucoma labels perfectly in the tested split. This is a label-circularity / dataset-insufficiency finding, not a main diagnostic SOTA claim.

### REFUGE2 External Structural Glaucoma Lane

Added REFUGE2 tooling:

- inspect dataset structure
- build manifest
- infer train labels from `g####` / `n####` filename prefix
- pair `images/` with `mask/`
- infer mask background as largest pixel class
- compute:
  - cup area
  - disc area
  - area cup-to-disc ratio
  - vertical cup-to-disc ratio
- evaluate repeated stratified holdout structural baselines
- write structural validation prediction CSV and summary JSON

Important correction: the first mask interpretation was wrong. It treated nonzero pixels as foreground and produced vCDR around `0.15` with no separation. The fixed version infers background as the largest pixel class and gives clinically plausible vCDR values.

## FairVision Results So Far

### Full/large baseline results from manuscript

Completed FairVision diagnostic comparison has RETFound OCT as the strongest foundation-model baseline:

- RETFound OCT macro F1: `0.678`
- balanced accuracy: `0.797`
- ECE: `0.054`
- but large subgroup gaps:
  - delta FPR around `0.490`
  - delta FNR around `0.502`

Static ensembles and simple routing did not remove subgroup gaps.

### Exact-case 50/task no-MD comparison

A no-MD FairVision exact subset gave:

Agent, 50 per disease:

- overall: `0.78`
- AMD: `0.80`
- DR: `0.84`
- glaucoma: `0.70`

RETFound OCT on same subset:

- overall: `0.78`
- AMD: `0.80`
- DR: `0.84`
- glaucoma: `0.70`

MIRAGE SLO on same subset:

- overall: `0.7467`
- AMD: `0.78`
- DR: `0.72`
- glaucoma: `0.74`

Interpretation: no-MD agent does not beat RETFound overall. It can tie or improve specific task behavior depending on subset, but not robustly enough for a raw-F1 dominance claim.

### MD-enabled FairVision 250/task run

Run:

- `outputs/ophthalmic_agent_250each_md_enabled_v1.csv`

Results:

- rows: `750`
- valid: `746`
- overall: `0.8324`
- AMD: `0.6734`
- DR: `0.8795`
- glaucoma: `0.9438`

Glaucoma confusion:

- TN `111`
- FP `14`
- FN `0`
- TP `124`

Interpretation: very strong glaucoma, but likely due to MD/visual-field label alignment. Report this separately as an MD/VF-enabled run and label-circularity-sensitive. Do not mix it with image-only claims.

### Dynamic Few-Shot Prompt Tuning

Initial dynamic few-shot run crashed:

- `equi_agent_dynamic_few_shot_live_strat10_v1`
- completed only `2/30`
- errors `28`

Prompt was too large and JSON parsing failed. Compact dynamic few-shot prompt fixed this.

Compact dynamic few-shot v2:

- `equi_agent_dynamic_few_shot_live_strat10_v2`
- cases `30`
- errors `0`
- but poor performance:
  - AMD F1 `0.571`
  - DR F1 `0.333`
  - glaucoma F1 `0.500`
  - macro/equi-agent F1 `0.468`

Recommendation-controlled dynamic few-shot:

- `equi_agent_dynamic_few_shot_reco_action_live_strat10_v1`
- cases `30`
- errors `0`
- calibration action counts:
  - AMD neutral `8`, precision `2`
  - DR neutral `9`, precision `1`
  - glaucoma neutral `8`, sensitivity `2`
- macro rows:
  - Equi-Agent F1 `0.583`
  - RETFound OCT F1 `0.774`
  - dynamic global AUROC source F1 `0.721`
  - MIRAGE SLO F1 `0.709`

Glaucoma improved in this tiny stratified subset:

- Equi-Agent glaucoma F1 `0.667`
- recall `0.800`
- specificity `0.400`

But macro still poor. Conclusion: dynamic few-shot retrieval is not the missing ingredient. Stop prompt-tuning as the main path.

## Harvard-GDP Results So Far

### GDP Progression Manuscript Result

Current manuscript table includes:

- GDP-native RNFLT+TDS EfficientNet:
  - AUROC `0.816`
  - F1 `0.600`
  - sensitivity `0.500`
  - specificity `0.929`
  - ECE `0.066`
  - worst-group F1 `0.564`

- Equi-Agent live progression arbitration:
  - AUROC `0.816`
  - F1 `0.684`
  - sensitivity `0.650`
  - specificity `0.893`
  - ECE `0.064`
  - worst-group F1 `0.619`
  - escalated `51/200`

This remains a strong current result for progression arbitration.

### GDP Raw VF Glaucoma Detection Audit

`run_gdp_raw_vf_agent.py` was run on 200 GDP cases with raw TD values:

Output path:

- `equi-agent/outputs/gdp_raw_vf_agent_live200_v1/gdp_raw_vf_agent_predictions.csv`

Results:

- rows: `200`
- valid: `200`
- accuracy: `1.0`
- confusion:
  - TN `108`
  - FP `0`
  - FN `0`
  - TP `92`

MD audit:

- computed MD from TD mean: about `-3.347`
- stored MD mean: about `-3.271`
- mean absolute difference: about `0.249 dB`

Interpretation: raw visual-field total-deviation values essentially recover the glaucoma label. This is not an agent genius result. It is a dataset-label circularity / label-definition issue. It is useful as an audit, but should not be framed as standalone SOTA diagnosis.

## REFUGE2 External Structural Glaucoma Lane

### Dataset Download

REFUGE2 was downloaded via `kagglehub` on the cluster:

```python
import kagglehub
path = kagglehub.dataset_download("victorlemosml/refuge2")
print(path)
```

Cluster path:

```text
/home/ab575577/.cache/kagglehub/datasets/victorlemosml/refuge2/versions/1
```

Symlink command:

```bash
cd ~/RETFound
mkdir -p Datasets/ExternalGlaucoma
ln -sfn /home/ab575577/.cache/kagglehub/datasets/victorlemosml/refuge2/versions/1 \
  Datasets/ExternalGlaucoma/REFUGE2
```

### Dataset Structure

Inspector found:

- total images/masks:
  - `.jpg`: `1200`
  - `.bmp`: `800`
  - `.png`: `400`
- parent folders:
  - `REFUGE2/train/images`: `400`
  - `REFUGE2/train/mask`: `400`
  - `REFUGE2/val/images`: `400`
  - `REFUGE2/val/mask`: `400`
  - `REFUGE2/test/images`: `400`
  - `REFUGE2/test/mask`: `400`

Classification labels:

- train filenames use:
  - `g####` for glaucoma
  - `n####` for normal
- val filenames use `V####`
- test filenames use `T####`
- no val/test label file was found in this Kaggle copy

Therefore:

- REFUGE2 train is labeled: `40 glaucoma`, `360 normal`
- REFUGE2 val/test classification labels are hidden/unavailable in this mirror
- do not claim official REFUGE2 test performance from this mirror
- call it "external structural validation on labeled REFUGE2 train with repeated stratified holdout"

### Manifest

Build manifest:

```bash
cd ~/RETFound

python equi-agent/scripts/build_refuge2_manifest.py \
  --root Datasets/ExternalGlaucoma/REFUGE2 \
  --out-csv equi-agent/outputs/manifests/refuge2_manifest_v2.csv
```

Summary:

```bash
python equi-agent/scripts/summarize_refuge2_manifest.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv
```

Corrected structural features:

Train label counts:

- normal: `360`
- glaucoma: `40`

Mask feature rule:

- `background_largest_cup_smaller_substantial_foreground`: `1200`

vCDR by train label:

- normal mean: `0.4624`
- normal median: `0.4518`
- normal max: `0.6657`
- glaucoma mean: `0.7461`
- glaucoma median: `0.6723`
- glaucoma max: `1.0`

Area CDR by train label:

- normal mean: `0.2393`
- normal median: `0.2293`
- glaucoma mean: `0.3643`
- glaucoma median: `0.3575`

Interpretation: vertical cup-to-disc ratio separates glaucoma strongly on labeled REFUGE2 train.

### CDR Baseline

Script:

```bash
equi-agent/scripts/evaluate_refuge2_cdr_baseline.py
```

Run balanced-accuracy objective:

```bash
cd ~/RETFound

python equi-agent/scripts/evaluate_refuge2_cdr_baseline.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv \
  --split train \
  --test-frac 0.25 \
  --seed 2026 \
  --objective balanced_accuracy \
  --repeats 50
```

Vertical CDR repeated holdout result:

- threshold mean: `0.549455`
- threshold sd: `0.007039`
- accuracy mean: `0.8868`
- precision mean: `0.4752`
- recall mean: `0.8360`
- specificity mean: `0.8924`
- F1 mean: `0.5992`
- balanced accuracy mean: `0.8642`
- balanced accuracy median: `0.8667`

Area CDR repeated holdout result:

- balanced accuracy mean: `0.7962`
- F1 mean: `0.4712`

Run F1 objective:

```bash
python equi-agent/scripts/evaluate_refuge2_cdr_baseline.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv \
  --split train \
  --test-frac 0.25 \
  --seed 2026 \
  --objective f1 \
  --repeats 50
```

Vertical CDR repeated holdout result:

- threshold mean: `0.623774`
- threshold sd: `0.007385`
- accuracy mean: `0.9504`
- precision mean: `0.8465`
- recall mean: `0.6300`
- specificity mean: `0.9860`
- F1 mean: `0.7114`
- balanced accuracy mean: `0.8080`

Area CDR repeated holdout result:

- F1 mean: `0.5540`
- accuracy mean: `0.9028`
- precision mean: `0.5367`
- recall mean: `0.6140`
- specificity mean: `0.9349`

Interpretation:

- vCDR is the better structural glaucoma feature.
- `vCDR >= ~0.55` is a sensitivity-oriented threshold.
- `vCDR >= ~0.62` is a precision-oriented threshold.
- This supports adding a Structural Glaucoma Agent.

### Structural Validation Runner

Script:

```bash
equi-agent/scripts/run_refuge2_structural_glaucoma_validation.py
```

Run:

```bash
cd ~/RETFound

python equi-agent/scripts/run_refuge2_structural_glaucoma_validation.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv \
  --split train \
  --test-frac 0.25 \
  --seed 2026 \
  --repeats 50 \
  --out-dir equi-agent/outputs/refuge2_structural_glaucoma_validation_v1
```

Inspect:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("equi-agent/outputs/refuge2_structural_glaucoma_validation_v1/refuge2_structural_glaucoma_summary.json")
s = json.loads(p.read_text())

print("label_counts:", s["label_counts"])
print("thresholds:", s["thresholds"])

print("\nEvaluation metrics:")
for policy, m in s["evaluation_metrics"].items():
    print(policy, m)

print("\nRepeated summary:")
for policy, m in s["repeated_summary"].items():
    print("\n", policy)
    for k in ["f1", "balanced_accuracy", "recall", "specificity", "precision", "threshold"]:
        print(k, m[k])
PY
```

Prediction CSV:

```bash
head -5 equi-agent/outputs/refuge2_structural_glaucoma_validation_v1/refuge2_structural_glaucoma_predictions.csv
```

This runner should be the next thing to run if it has not already been run.

## Proposed Structural Glaucoma Agent Logic

Use mask-derived vertical CDR as explicit evidence:

```text
vCDR < 0.55:
  structural evidence weak/negative

0.55 <= vCDR < 0.62:
  suspicious/borderline structural glaucoma evidence
  escalation recommended

vCDR >= 0.62:
  strong structural glaucoma evidence
```

Suggested agent framing:

- Vision/Foundation Model Agent:
  - provides model probabilities
- Structural Glaucoma Agent:
  - provides vCDR, area CDR, structural threshold zone, confidence
- Equity/Reliability Agent:
  - uses validation-derived FPR/FNR priors to decide trust in foundation model outputs
- Orchestrator:
  - combines model evidence and structural evidence
- Safety Agent:
  - escalates discordance, borderline CDR, missing masks, poor image quality

This is more defensible than prompt-only glaucoma tuning.

## Current Scientific Claims to Keep

Strong:

- Foundation models have substantial subgroup reliability gaps even when average performance is good.
- Static fusion and simple reliability routing do not eliminate gaps.
- Equi-Agent improves GDP progression thresholded F1 over the native helper in the current completed run.
- MD/VF information can dominate glaucoma labels in FairVision/GDP-like settings; this must be audited.
- REFUGE2 mask-derived vCDR provides a non-MD structural glaucoma signal.

Cautious / caveated:

- FairVision MD-enabled glaucoma performance is high, but likely label-circularity-sensitive.
- GDP raw VF gives 100 percent glaucoma label recovery, but that is dataset-label alignment, not standalone agent intelligence.
- REFUGE2 Kaggle copy does not expose val/test classification labels. Current REFUGE2 result is train-labeled stratified holdout, not official test.
- Prompt tuning did not make Equi-Agent beat RETFound/MIRAGE/simple baselines on FairVision raw F1.

Avoid claiming:

- "Equi-Agent is SOTA on FairVision diagnosis" unless a later valid run proves this.
- "GDP raw VF 100 percent proves agent diagnostic superiority."
- "Official REFUGE2 test performance" from the KaggleHub mirror currently used.

## Manuscript Update Suggestions

The manuscript currently says Equi-Agent is an arbitration framework and includes FairVision and GDP results. Future edits should:

1. Add a subsection on label-circularity / functional evidence:
   - FairVision/GDP glaucoma can be strongly tied to MD/VF.
   - Report MD-enabled and no-MD runs separately.

2. Add a REFUGE2 structural validation subsection:
   - Describe REFUGE2 as external optic-disc dataset.
   - State this Kaggle mirror exposes labels only for train.
   - Report repeated stratified holdout on labeled train.
   - Emphasize mask-derived vCDR as explicit structural evidence independent of MD.

3. Reframe Equi-Agent:
   - not "LLM beats every model"
   - rather "reliability-aware arbitration plus structural/functional evidence and escalation"

Suggested sentence:

> On labeled REFUGE2 training images with expert optic-disc/cup masks, vertical cup-to-disc ratio provided a stable non-MD structural glaucoma signal under repeated stratified holdout, supporting a Structural Glaucoma Agent that can arbitrate optic-nerve evidence separately from visual-field-derived labels.

## What To Do Next

1. Run the structural validation runner if not already done:

```bash
python equi-agent/scripts/run_refuge2_structural_glaucoma_validation.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv \
  --split train \
  --test-frac 0.25 \
  --seed 2026 \
  --repeats 50 \
  --out-dir equi-agent/outputs/refuge2_structural_glaucoma_validation_v1
```

2. Paste the JSON summary or inspect it with the command above.

3. Add a manuscript table row for REFUGE2 structural baseline:

Potential columns:

- dataset
- label source
- evidence source
- policy
- threshold
- F1
- balanced accuracy
- sensitivity
- specificity
- repeated holdout mean +/- SD

4. Decide whether to search/download another external glaucoma dataset with full exposed classification labels:

- ACRIMA
- RIM-ONE DL
- DRISHTI-GS

ACRIMA may be easiest for classification-only if downloaded manually or via KaggleHub. RIM-ONE/DRISHTI are valuable if masks are included.

5. Build a generalized external glaucoma manifest parser if another dataset is downloaded.

## Exact Commands Worth Keeping

FairVision dynamic few-shot recommendation run:

```bash
OUT_DIR=equi-agent/outputs/equi_agent_dynamic_few_shot_reco_action_live_strat10_v1 \
PROMPT_VARIANT=dynamic_few_shot \
FEW_SHOT_ACTION_MODE=recommendation \
FEW_SHOT_K=2 \
MAX_CASES_PER_TASK=10 \
SAMPLE_STRATIFIED=1 \
TARGET_POSITIVE_FRAC=0.50 \
RUN_COMPARE=1 \
MAX_OUTPUT_TOKENS=1200 \
bash equi-agent/scripts/run_fairvision_equi_agent_full.sh
```

FairVision few-shot recommendation scorer:

```bash
python equi-agent/scripts/summarize_few_shot_recommendations.py \
  --predictions equi-agent/outputs/equi_agent_dynamic_few_shot_reco_action_live_strat10_v1/equi_agent_live_predictions.csv
```

GDP raw VF agent:

```bash
python equi-agent/scripts/run_gdp_raw_vf_agent.py \
  --max-cases 200 \
  --functional-agent-mode deterministic \
  --out-dir equi-agent/outputs/gdp_raw_vf_agent_live200_v1
```

REFUGE2 manifest:

```bash
python equi-agent/scripts/build_refuge2_manifest.py \
  --root Datasets/ExternalGlaucoma/REFUGE2 \
  --out-csv equi-agent/outputs/manifests/refuge2_manifest_v2.csv
```

REFUGE2 summary:

```bash
python equi-agent/scripts/summarize_refuge2_manifest.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv
```

REFUGE2 CDR baseline:

```bash
python equi-agent/scripts/evaluate_refuge2_cdr_baseline.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv \
  --split train \
  --test-frac 0.25 \
  --seed 2026 \
  --objective balanced_accuracy \
  --repeats 50
```

REFUGE2 structural validation:

```bash
python equi-agent/scripts/run_refuge2_structural_glaucoma_validation.py \
  --manifest equi-agent/outputs/manifests/refuge2_manifest_v2.csv \
  --split train \
  --test-frac 0.25 \
  --seed 2026 \
  --repeats 50 \
  --out-dir equi-agent/outputs/refuge2_structural_glaucoma_validation_v1
```

## Known Pitfalls

- Do not use the first REFUGE2 manifest (`refuge2_manifest.csv`) for CDR claims. It used the wrong mask interpretation and produced invalid vCDR values.
- Use `refuge2_manifest_v2.csv`.
- Do not call REFUGE2 val/test labeled unless labels are found.
- Do not claim official REFUGE2 challenge test performance.
- Do not collapse MD-enabled and no-MD FairVision glaucoma runs into one result.
- Do not treat safety escalation as removing the forced benchmark label. The current runners keep forced labels and escalation separately.
- Do not keep spending on FairVision prompt tuning unless there is a very specific hypothesis. It has repeatedly failed to beat simpler baselines on raw F1.

## Current Bottom Line

The project now has two credible directions:

1. **Reliability auditing and progression arbitration**:
   - GDP progression Equi-Agent improved F1 over the native helper in the current completed endpoint.
   - This supports Equi-Agent as an arbitration layer.

2. **Structural glaucoma evidence**:
   - REFUGE2 mask-derived vCDR gives a strong non-MD structural signal.
   - This supports adding a Structural Glaucoma Agent and strengthens the argument that glaucoma arbitration should not depend on MD leakage.

The weak direction is prompt-only FairVision diagnosis. Treat it as a documented negative experiment and move on.
