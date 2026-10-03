# Manuscript Results and Methods Closure Audit

This audit separates verified results, user-supplied CECSL results awaiting artifact verification, and unresolved claims in the draft supplied on 3 October 2026. No model training or paid API experiment was run for this audit. The original draft and commented historical sections have not been overwritten. `verified_replacements.tex` contains targeted replacement passages, not a complete manuscript.

## What Is Resolved

| Item | Finding | Action |
|---|---|---|
| GDP progression `0.684 / 0.600` | Latest full v2 report is positive F1 `0.6441 / 0.5905`, macro-F1 `0.7476 / 0.7224`, worst-group positive F1 `0.6102 / 0.5818` (agent/helper). | Replace in abstract, introduction, Results 2.4, table and Discussion consistently. Latest values are from the supplied CECSL report; audit its saved artifacts before submission. |
| General progression improvement | Four endpoints still have zero true positives. TD endpoints improve, with extra false positives. | Report all six; do not claim superiority across endpoints or all metrics. |
| Progression input description | Baseline RNFLT and 52 TD values, not supplied longitudinal visits. | Distinguish label construction over time from baseline-only inference. |
| FairVision full splits | 6,000 train / 1,000 validation / 3,000 test per task. | Fill split placeholders; locked agent slices are separate. Sampling seed still needs its construction receipt. |
| ECE | Ten equal-width bins on [0,1], sample-weighted calibration gap. | Remove bracket around 10. |
| Patient reliability | Support-weighted average of marginal age/race/sex risks, then intersection-count shrinkage, k=50. | Replace the single-subgroup equation; it is not minimum risk or direct intersectional performance. |
| Coefficient sensitivity | 21 offline settings; 486 canonical scores reproduced exactly within 1e-9. | Add supplementary score/ranking analysis. Do not call this downstream agent-performance robustness. |

The revised GDP prompt was informed by inspecting earlier failures on the same 200 cases and checked on five selected prior errors. OOF reliability corrects the prior-data problem, but does not undo this post-hoc test-cohort exposure. Report the v2 evaluation as exploratory; do not describe the prompt as independently test-blinded.

## Open Methods Questions

| Question | Evidence and answer | Remaining work |
|---|---|---|
| GDP split and native method | Public progression subset: 500 patients; current helper uses 300 development / 200 test. Five OOF folds use 240 / 60; final supervised helper fits all 300 at fixed epoch count and threshold 0.5. | Do not call this a reproduction of the complete semi-supervised pseudo-supervisor method. Confirm all six CECSL receipts, using the supplied audit. |
| Endpoint | Released `progression[5]` is `td_pointwise_no_p_cut`. The paper describes TD progression at at least three locations with slope <= -1 dB; released names distinguish no-p-cut. Labels are read, not recomputed. | The release does not supply inference-time follow-up. Do not invent a fixed forecast horizon or unverified significance cutoff. |
| Unparseable outputs | Staged GDP: up to three total attempts, then incomplete case; no negative imputation; complete 200-case cohort required. Review flags do not remove predictions. | FairVision and detection runners differ. `evaluate_gdp_agentic.py` and `evaluate_gdp_llm_baseline.py` record failures as -1 and summaries filter invalid predictions. Verify each reported run's actual failure count; no universal retry rule is justified. |
| Temperature and model | Staged GDP requests GPT-5.1, temperature 0.2, max completion tokens 8000. | Live FairVision `Orchestrator/new.py` currently sets `gpt-5.6-luna` and does not send temperature. Do not assert all agents were GPT-5.1 or that a backbone was held fixed without historical request records. |
| Escalation | Staged GDP `review_required` is an LLM judgement. FairVision code also has <=10% / >=90% helper-confidence bypasses; these are not clinician-escalation thresholds. | Match historical run code and saved flags to each result. The 69.1% rate is a different deterministic experiment. |
| Counterfactual N | Saved deterministic study: 9,000 case-task rows, 54,000 single-attribute substitutions including ethnicity, not five random profiles. Staged GDP: five evidence-ablation scenarios in one call per case. | Neither establishes the live FairVision demographic perturbation count. Recover its original trace/configuration and state coverage of the confidence bypass. |
| Equity Agent | Reliability score retrieval is active in `main_new.py`; separate EquityAgent construction and `analyze_patients` call are commented out. | Do not describe an executed independent LLM Equity reviewer unless the actual run trace proves it. |
| Worst-group definition | Locked foundation collector uses support-weighted F1 over age, gender and race, not macro-F1 and not ethnicity. Staged GDP uses positive F1 with >=20 positives and >=20 negatives and >=2 eligible groups per attribute. | Export subgroup support and both definitions for every canonical run; choose a consistent declared reporting contract before updating all rows. |
| Age groups | Evaluation/prior manifests use <50 / 50-69 / >=70. FairVision lookup caller maps ages at 40 and 60 to the same textual names. Staged GDP recomputes its own OOF priors at 40 and 60. | This is a FairVision historical lookup mismatch, not simply two harmless evaluation conventions. Verify archived run configuration, disclose if present, and evaluate a corrected implementation separately. |

Sources: [GDP paper section 3.2](https://arxiv.org/html/2308.13411v1#S3.SS2), [official release index mapping](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP#dataset), local `OphthalmicAgent/Progression/{evidence,workflow,reporting}.py`, `equi-agent/scripts/estimate_gdp_native_oof.py`, `build_all_model_validation_priors.py`, `collect_fairvision_yusra_foundation_results.py`, `build_manifests.py`, and `OphthalmicAgent/{main_new.py,Orchestrator/new.py}`. Current code establishes implementation behaviour, not necessarily settings used by an older run.

## GDP Detection Is Separate

The detection comparison `0.789 / 0.695` is not superseded by the new progression run. The agent detection file is unavailable locally; its row still needs a paired-cohort audit.

Local `equi-agent/outputs/predictions/gdp_glaucoma_detection_retfound_oct.csv` has 400 rows: TP=118, FN=61, TN=164, FP=57. This reproduces positive F1 0.6667 and balanced accuracy 0.7006, with sensitivity **0.6592** and specificity **0.7421**. The draft incorrectly puts sensitivity into worst-group F1, specificity into sensitivity, and 0.8771 into specificity. Do not infer worst-group F1 from these misaligned cells. This file's macro-F1 is 0.7010 and weighted F1 is 0.7047; switching the column definition changes the F1 values.

The current GDP LLM evaluator also passes an OCT montage and RNFLT image, so the blanket statement that all GDP LLM baselines received OCT only needs the historical run records. No corrected agent/GPT-5.1 detection number is invented here.

## AMD and FairVision Metrics

No local final 250-case AMD agent prediction file was found. The available early AMD agent files are partial, use different cases, or include invalid labels/predictions. Do not substitute any of them for the reported final run. Sensitivity 0.8860 is not a valid whole-number fraction of 125 positives; identify the denominator from the final CSV instead of assuming an accepted-case subset.

The draft's global macro-F1 and subgroup macro-F1 descriptions do not match the existing weighted-F1 collector. On the recovered glaucoma cohort (250 cases):

| Method | Macro-F1 | Weighted F1 | Worst macro-F1, race/sex/age | Worst weighted F1, race/sex/age |
|---|---:|---:|---:|---:|
| RETFound | 0.7484 | 0.7486 | 0.5341 | 0.6394 |
| Agent | 0.7712 | 0.7713 | 0.5729 | 0.6380 |

Including ethnicity changes the agent's worst weighted F1 to 0.6190. Thus neither adding ethnicity nor relabelling weighted F1 as macro-F1 is a cosmetic change. The worst subgroup can also change. On balanced AMD/DR cohorts, overall macro and weighted F1 coincide; subgroup scores generally do not. These are alternative definitions shown for auditing, not a recommendation to select whichever gives a more favourable comparison.

Source: `equi-agent/outputs/audits/fairvision_glaucoma_case_recovery/{agent_predictions_recovered,retfound_predictions_recovered,manifest_recovered}.csv`. The recovery provenance documents the formerly missing glaucoma ground-truth row; no prediction was assigned from its truth.

## Reliability Sensitivity Findings

The executable audit is `equi-agent/scripts/audit_reliability_weight_sensitivity.py`. Outputs are under `equi-agent/outputs/audits/reliability_weight_sensitivity/`, with input/code hashes in `provenance.json` and every score in `scores.csv`.

| Perturbation | Glaucoma changed winners | AMD changed winners | DR changed winners |
|---|---:|---:|---:|
| Any one coefficient +/-20%, renormalized | 0/18 for every setting | 0-2/18 | 0-12/18 |
| Equal weights | 0/18 | 3/18 | 15/18 |
| Historical 0.85 FNR + 0.15 FPR | 18/18 | 18/18 | 18/18 |
| Global-only score | 0/18 | 7/18 | 7/18 |

This analysis keeps the historical score implementation and test-covariate support counts unchanged. It uses validation performance priors and does not optimize against test labels. Since the live agent can use a fixed RETFound anchor, ranking alternative encoders is not the same as testing its final decisions. Absolute anchor trust changes are reported separately for that reason.

Remaining downstream experiment: lock the scenarios, prompts, cases, model deployment and decoding settings before any calls; compare canonical weights, equal weights and global-only reliability on matched cases. Change all derived risks/trust consistently and use separate cache namespaces. Score complete cohorts under one F1/subgroup contract, including failures and review flags. Development-only selection and an untouched confirmation cohort are needed for a confirmatory robustness claim. A label-only recomputation cannot replace these calls. No new API run has been started.

## Counterfactual and Escalation Attribution

`equi-agent/outputs/fairvision_reliability_selective_arbitration/selective_arbitration_summary.json` records a different composite: 0.45 AUROC + 0.35 F1 - 0.10 ECE - 0.07 FNR - 0.03 FPR. It records 9,000 test cases and 54,000 deterministic perturbations, label change 0.00374074 and escalation change 0.05811111. The full setting accepted 2,777 cases and escalated 6,223 (69.144%). Its accepted **positive-class** F1 is 0.9020, not a live-agent accepted macro-F1.

Its escalation settings are disagreement >=0.25, close-call margin 0.08, weighted reliability <0.35, or a non-singleton conformal set with alpha=0.1. These parameters must not be transplanted into the live-agent Methods. Keep this as a separately named deterministic analysis, or rerun a matched live-agent analysis. Remove the live-agent attribution of 0.4%, 5.8% and 0.902 from the abstract/Discussion pending that distinction.

## CECSL Audit

After the user pushes the local files and pulls them on CECSL:

```bash
cd ~/RETFound
conda activate retfound
python equi-agent/scripts/audit_manuscript_result_gaps.py \
  --progression-run OphthalmicAgent/outputs/gdp_progression_staged_v2_smoke5 \
  --out-dir /tmp/retinagent_manuscript_audit
```

This scans case-level result candidates, checks manifest coverage and labels where manifests are present, reports duplicate and invalid rows, computes all F1 definitions, exports subgroup support, and verifies the complete staged progression run and its frozen evidence. It does not pick the highest-scoring candidate, impute predictions, run inference, or change results. Its ZIP contains aggregate metrics, paths and hashes, not patient images, raw explanations, or API keys.

Download from the Mac terminal:

```bash
scp ab575577@10.171.42.25:/tmp/retinagent_manuscript_audit.zip /tmp/
```

Next decisions after the audit: identify canonical AMD and GDP detection runs; verify live request settings; settle F1 and subgroup definitions; then apply one consistent update across the full manuscript. The missing multimodal/agent ablations remain missing experiments, not gaps resolved by an offline coefficient audit.
