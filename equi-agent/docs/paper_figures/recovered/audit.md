# Recovered CECSL Figure Inputs

This audit reads a separate extraction of the 2026-09-27 transfer. No model, LLM, prompt, original output or manuscript was changed. All figures in this directory are historical/exploratory drafts, not replacements for verified final manuscript figures.

## What Is Recovered

- All 17 reliability-search candidates reproduce from their 1,000-case validation winner CSVs, including F1, accuracy and balanced accuracy.
- Three tasks have deterministic risk-coverage results on 3,000 test cases each.
- PAPILA, Drishti-GS and GAMMA have fully paired earlier baseline/agent outputs.

## Main-Figure Blockers

- The recovered live 250-case glaucoma run has weighted F1 **0.6960** (binary F1 0.6935), not the previously supplied paper value **0.7704**. Do not substitute or relabel this run.
- Its configured case list is `equi-agent/outputs/manifests/fairvision_intersectional_balanced_250/all_manifest.csv`.
- `OphthalmicAgent/outputs/glaucoma_counterfactual_250/predictions.csv` has 250 rows and 1 nonbinary ground-truth value(s). It is excluded from all result plots, not silently relabelled.
- Live AMD/DR manuscript runs and matched final GDP/REFUGE agent outputs are not established by this transfer. Existing aggregates alone do not establish their cohorts.
- The three paired external runs below do not reproduce all subsequently updated manuscript rows. They are historical comparisons, not final reported estimates.

## Candidate Run Status

| Directory | Dry run | Cases | Errors |
| --- | --- | ---: | ---: |
| fairvision_balanced_250 | True | 750 | 0 |
| fairvision_live_balanced_250_reproducible_v1 | True | 0 | 0 |
| fairvision_live_glaucoma_250_score_cf_cdr_v1 | False | 250 | 0 |
| fairvision_live_intersectional_balanced_250_reproducible_v1 | True | 3 | 0 |

## Reliability Evidence

The saved report describes case-grouped out-of-fold evaluation on validation data. These are successive exploratory searches on the same development set; they do not provide an independent confirmation of the selected bonus, nested threshold tuning, or the paper's five fixed risk coefficients. OOF fold assignments are not recovered. The 0.974 any-model-correct oracle is not a deployable selector and is not plotted.

| Rule | Positive-class F1 | Accuracy |
| --- | ---: | ---: |
| hierarchical_demo_bin_f1_bonus_0p03 | 0.7512 | 0.7510 |
| hierarchical_demo_bin_f1_bonus_0p02 | 0.7449 | 0.7500 |
| hierarchical_demo_bin_f1_bonus_0p01 | 0.7379 | 0.7500 |
| hierarchical_demo_bin_accuracy | 0.7335 | 0.7500 |
| hierarchical_demo_bin_oct_margin_0p05 | 0.7241 | 0.7500 |
| probability_bin_f1_bonus_0p01 | 0.7447 | 0.7470 |
| hierarchical_demo_bin_oct_margin_0p03 | 0.7217 | 0.7470 |
| probability_bin_accuracy | 0.7369 | 0.7430 |
| tree_correctness | 0.7263 | 0.7430 |
| probability_bin_f1_bonus_0p02 | 0.7416 | 0.7400 |
| probability_bin_f1_bonus_0p03 | 0.7431 | 0.7380 |
| hierarchical_demo_bin_f1_bonus_0p05 | 0.7493 | 0.7350 |
| logistic_correctness | 0.7227 | 0.7360 |
| probability_bin_f1_bonus_0p05 | 0.7410 | 0.7280 |
| directional_demographic_accuracy | 0.7255 | 0.7260 |
| global_accuracy | 0.6817 | 0.7180 |
| demographic_accuracy | 0.6800 | 0.7130 |

## Selective-Performance Scope

The curve is the saved deterministic ensemble's increasing `risk_score` ranking. Its full-cohort confusion counts match the predictions. Each point's coverage, error rate and class support are checked against its saved counts; tied-score membership is not reconstructed. These are descriptive curves without uncertainty bands or matched comparator curves, not evidence of LLM-agent superiority.

Saved reliability weights: `{"auroc": 0.45, "ece": 0.1, "f1": 0.35, "fnr": 0.07, "fpr": 0.03}`.
This differs from the paper's FNR/FPR/ECE/AUROC/F1 risk score. The hard escalation policy and the continuous coverage sweep are different analyses.

## Historical External Comparisons

Last recorded test attempt per case, with exact baseline/agent ID and label agreement required. No invalid final prediction is dropped or replaced with an earlier valid prediction. All three cohorts here have complete valid final outputs. Retries were performed in the original runs; this audit does not establish their selection protocol. F1 conventions are explicitly separated.

| Dataset | Cases | Extra attempts | Baseline binary F1 | Agent binary F1 | Baseline weighted F1 | Agent weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PAPILA | 81 | 2 | 0.5333 | 0.3500 | 0.7673 | 0.7060 |
| Drishti-GS | 51 | 1 | 0.6441 | 0.6667 | 0.5999 | 0.6528 |
| GAMMA | 20 | 13 | 0.8696 | 0.8696 | 0.8465 | 0.8465 |

The paired bootstrap estimates uncertainty conditional on these fixed predictions; it does not include training-seed, threshold-selection or LLM variability. The GAMMA interval is exactly zero because both methods predicted identical labels, not because the experiment establishes equivalence.

## Files And Reproduction

- `recovered_experiments.drawio`: three fully editable result pages.
- `01_reliability_development.drawio`, `02_selective_diagnostics.drawio`, `03_paired_external.drawio`: individual figures.
- `source_data.json`: only aggregate metrics, provenance paths and source hashes. No case IDs, clinical narratives or images are exported.

```bash
python equi-agent/scripts/build_recovered_experiment_figures.py \
  --inputs-root /path/to/separate/extracted/archive
```

Generated figures are overwritten on rebuild; use a differently named copy for manual edits.

Source files checked: 40. See `source_data.json` for SHA-256 values.
