# Equi-Agent Experiment Ledger

Last reconstructed: 2026-06-23

This ledger separates results to keep as manuscript evidence from runs that are useful only as diagnostics, prompt-development history, or caveats.

## Results To Keep In The Manuscript

### 1. FairVision Foundation-Model Benchmark

Purpose: establish that retinal foundation models have task-, modality-, calibration-, and subgroup-dependent reliability gaps.

Inputs:

- Harvard-FairVision AMD, DR, glaucoma
- Train/validation/test split per task: 6,000 / 1,000 / 3,000
- Models: RETFound OCT, MIRAGE SLO, FLAIR SLO, RET-CLIP SLO, VisionFM SLO/OCT, RetiZero SLO, UrFound SLO/OCT

Kept results:

- RETFound OCT macro F1: 0.678
- RETFound OCT balanced accuracy: 0.797
- RETFound OCT ECE: 0.054
- RETFound OCT subgroup gaps: delta FPR about 0.490, delta FNR about 0.502
- Static fusion and simple routing did not remove subgroup gaps.

Prompt used: none. These are deterministic model/probe/fusion baselines.

Manuscript role: main reliability-gap motivation.

## 2. FairVision Reliability-Conditioned Selective Arbitration

Purpose: evaluate the locked reliability scoring/escalation tool inside Equi-Agent.

Output folder:

- `equi-agent/outputs/fairvision_reliability_selective_arbitration/`

Inputs:

- 9,000 held-out FairVision test cases
- 3,000 validation cases
- all 9 model streams available for each task
- validation subgroup priors:
  - `equi-agent/outputs/metrics/validation_subgroup_priors.csv`
  - `equi-agent/outputs/metrics/validation_subgroup_priors_global.csv`

Scoring formula:

```text
score = 0.45*AUROC + 0.35*F1 - 0.10*ECE - 0.07*FNR - 0.03*FPR
```

Shrinkage:

```text
lambda = n_group / (n_group + 50)
shrunk_metric = lambda*subgroup_metric + (1-lambda)*global_metric
```

Softmax arbitration:

```text
weight_j proportional to exp(4*score_j - 0.75*uncertainty_j - 0.50*vote_disagreement_j)
```

Selective escalation:

- close-call margin: 0.08
- disagreement-rate threshold: 0.25
- low-reliability threshold: 0.35
- conformal alpha: 0.10

Kept overall result:

- forced F1: 0.729
- forced balanced accuracy: 0.807
- forced ECE: 0.101
- coverage: 0.309
- escalation rate: 0.691
- accepted-case F1: 0.902
- accepted-case sensitivity: 0.890
- accepted-case specificity: 0.914
- accepted-case error rate: 0.098
- worst-group accepted-case F1: 0.615

Disease notes:

- AMD: accepted-case F1 0.936 at 42.2% coverage
- glaucoma: accepted-case F1 0.873 at 50.3% coverage
- DR: current rule escalates 100% of cases; do not claim DR selective deployment readiness.

Metadata counterfactuals:

- rows: 54,000
- label flip rate: 0.00374, about 0.4%
- escalation flip rate: 0.0581, about 5.8%

Prompt used: none for the full deterministic run. This is the auditable scoring/escalation tool that feeds the agentic Orchestrator/Safety Agent.

Manuscript role: main new technical result, but framed as selective triage rather than full automation.

## 3. Harvard-GDP Progression Arbitration

Purpose: show live Equi-Agent arbitration improves a longitudinal progression helper.

Kept baseline:

- GDP-native RNFLT+TDS EfficientNet:
  - AUROC 0.816
  - F1 0.600
  - sensitivity 0.500
  - specificity 0.929
  - ECE 0.066
  - worst-group F1 0.564

Kept Equi-Agent result:

- AUROC 0.816
- F1 0.684
- sensitivity 0.650
- specificity 0.893
- ECE 0.064
- worst-group F1 0.619
- escalated 51 / 200

Prompt used: live GDP progression Equi-Agent pipeline, not the isolated FairVision deterministic tool. Agentic components include Functional Specialist, Equity Auditor, Orchestrator, and Safety Agent.

Manuscript role: strongest live agentic result.

## 4. GDP Raw Visual-Field Audit

Purpose: detect label circularity / visual-field leakage.

Output:

- `equi-agent/outputs/gdp_raw_vf_agent_live200_v1/gdp_raw_vf_agent_predictions.csv`

Kept result:

- rows: 200
- valid: 200
- accuracy: 1.000
- TN 108, FP 0, FN 0, TP 92
- computed TD-derived MD mean: about -3.347
- stored MD mean: about -3.271
- mean absolute difference: about 0.249 dB

Prompt used: deterministic functional-agent mode over raw total-deviation visual-field values; stored MD and label hidden from prompt-time evidence.

Manuscript role: audit/caveat only. This is not diagnostic superiority; it shows GDP glaucoma labels are highly recoverable from raw visual-field data.

## 5. REFUGE2 Structural Glaucoma / CDR Lane

Purpose: establish non-MD structural glaucoma evidence from optic-disc/cup masks.

Dataset:

- KaggleHub REFUGE2 mirror
- labeled train split only: 400 images, 40 glaucoma and 360 normal
- val/test classification labels unavailable in this mirror

Manifest:

- `equi-agent/outputs/manifests/refuge2_manifest_v2.csv`

Structural features:

- mask-derived vertical cup-to-disc ratio
- mask-derived area cup-to-disc ratio

Kept descriptive result:

- normal vCDR mean 0.4624, median 0.4518
- glaucoma vCDR mean 0.7461, median 0.6723
- normal area CDR mean 0.2393
- glaucoma area CDR mean 0.3643

Repeated holdout, balanced-accuracy objective:

- vCDR threshold mean 0.549455
- accuracy 0.8868
- recall 0.8360
- specificity 0.8924
- F1 0.5992
- balanced accuracy 0.8642

Repeated holdout, F1 objective:

- vCDR threshold mean 0.623774
- accuracy 0.9504
- precision 0.8465
- recall 0.6300
- specificity 0.9860
- F1 0.7114
- balanced accuracy 0.8080

Prompt used: none. This is mask-derived structural validation.

Manuscript role: structural glaucoma support, not official REFUGE2 test performance.

## Prompt/Agent Experiments That Are Mostly Negative Or Supporting

### 6. FairVision Exact-Case 50/Task No-MD Agent Comparison

Purpose: compare live/no-MD Equi-Agent to RETFound/MIRAGE on exactly matched cases.

Kept numbers:

- Equi-Agent overall: 0.78
- Equi-Agent AMD: 0.80
- Equi-Agent DR: 0.84
- Equi-Agent glaucoma: 0.70
- RETFound OCT same subset overall: 0.78
- MIRAGE SLO same subset overall: 0.7467

Prompt used: no-MD FairVision agent prompt family, exact variant not fully recovered in local outputs. Treat this as a subset diagnostic result, not a main claim.

Interpretation: no-MD agent ties RETFound overall; it does not establish raw-F1 dominance.

### 7. MD-Enabled FairVision 250/Task Legacy OphthalmicAgent Run

Output:

- `outputs/ophthalmic_agent_250each_md_enabled_v1.csv`

Kept numbers:

- rows 750, valid 746
- overall 0.8324
- AMD 0.6734
- DR 0.8795
- glaucoma 0.9438
- glaucoma confusion: TN 111, FP 14, FN 0, TP 124

Prompt used: legacy OphthalmicAgent-style MD/VF-enabled pipeline.

Interpretation: strong but label-circularity-sensitive. Do not mix with image-only or no-MD claims.

### 8. Dynamic Few-Shot Prompt Runs

Prompt variant:

- `PROMPT_VARIANT=dynamic_few_shot`

Core prompt behavior:

- retrieves nearest validation examples by model probabilities, threshold margins, vote pattern, and weighted probability
- current test label hidden
- examples include validation true labels to calibrate trust
- instructs the Orchestrator not to copy labels blindly

Modes:

- `FEW_SHOT_ACTION_MODE=llm`
- `FEW_SHOT_ACTION_MODE=stabilized`
- `FEW_SHOT_ACTION_MODE=recommendation`

Important runs:

- `equi_agent_dynamic_few_shot_live_strat10_v1`
  - completed 2 / 30
  - errors 28
  - prompt too large / parsing failure

- `equi_agent_dynamic_few_shot_live_strat10_v2`
  - cases 30
  - errors 0
  - AMD F1 0.571
  - DR F1 0.333
  - glaucoma F1 0.500
  - macro F1 0.468

- `equi_agent_dynamic_few_shot_reco_action_live_strat10_v1`
  - cases 30
  - errors 0
  - Equi-Agent macro F1 0.583
  - RETFound OCT F1 0.774
  - dynamic global AUROC source F1 0.721
  - MIRAGE SLO F1 0.709
  - glaucoma F1 0.667, recall 0.800, specificity 0.400

Current local dry-run outputs:

- `equi-agent/outputs/equi_agent_dynamic_few_shot_allmodels_dryrun1/`
- `equi-agent/outputs/equi_agent_dynamic_few_shot_compact_dryrun1/`
- `equi-agent/outputs/equi_agent_dynamic_few_shot_reco_dryrun1/`

Local dry-runs are only 3 cases and should not be used as performance evidence.

Interpretation: dynamic few-shot improved parseability and traces, but did not solve FairVision performance. Keep as prompt-development history.

## Prompt Variants Implemented In `run_equi_agent_fairvision_live.py`

### `current`

Default trust-calibration policy.

### `visual_first`

When images are attached, morphology review happens before reading model votes. Emphasizes drusen/RPE/fluid for AMD, vascular lesions for DR, and cup/RNFL-compatible evidence for glaucoma.

### `f1_rescue`

Optimizes forced benchmark F1 while preserving threshold consistency. Designed to avoid missing positives, especially rare-positive DR.

### `diagnosis_tuned`

Balances thresholded diagnosis performance. Adds stricter evidence requirements for glaucoma positives and more nuanced handling of DR/AMD positives.

### `diagnosis_tuned_v2`

Stricter glaucoma close-call handling. Avoids both broad weak-positive glaucoma calls and over-suppression of borderline glaucoma.

### `ophthalmic_agent_style`

Mimics the legacy OphthalmicAgent hierarchy. Requires explicit model-by-model FP/FN audit in the Equity Agent and disease-specific orchestration.

### `dynamic_few_shot`

Adds similar validation cases as calibration evidence. Includes validation labels but hides the current test label. Meant to calibrate trust, not copy nearest-neighbor labels.

## Current Bottom Line

Keep these as paper evidence:

1. FairVision foundation-model benchmark and subgroup reliability gaps.
2. FairVision reliability-conditioned selective arbitration as the main technical result.
3. GDP live Equi-Agent progression arbitration.
4. GDP raw VF audit as label-circularity warning.
5. REFUGE2 vCDR structural validation as non-MD glaucoma evidence.

Do not make these main claims:

1. Prompt tuning alone solves FairVision.
2. Equi-Agent is SOTA on forced FairVision diagnosis.
3. MD-enabled FairVision glaucoma proves image-only glaucoma reasoning.
4. GDP raw VF 100% accuracy proves agent intelligence.
5. REFUGE2 Kaggle mirror gives official test performance.
