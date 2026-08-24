# RetinAgent / Equi-Agent Project Handoff

Last updated: 2026-08-22

This is the primary context file for continuing this project from a new GPT/Codex
account. Read this file first, then inspect the current Git state and the files named
below. Do not assume that generated results on CECSL exist in the local Mac checkout.

## 1. What The Project Is

RetinAgent (the manuscript name; much of the code is still named Equi-Agent) is a
clinician-facing ophthalmic decision-support system. It combines retinal foundation
model outputs, validation-derived reliability information, structural or functional
evidence, and an LLM-based arbitration layer. It is intended to help an
ophthalmologist rapidly understand a case and identify disagreement or uncertainty;
it is not positioned as an autonomous replacement for the ophthalmologist.

The main experimental areas are:

- FairVision30K binary diagnosis: glaucoma, AMD, and diabetic retinopathy (DR).
- Harvard GDP glaucoma detection.
- Harvard GDP glaucoma progression prediction.
- External glaucoma validation: Drishti-GS1, REFUGE2, PAPILA, and GAMMA.
- Reliability and fairness: subgroup performance, worst-group F1, calibration,
  disagreement, and escalation.

The main paper metrics are F1, worst-group F1, sensitivity, specificity, and balanced
accuracy. AUROC, ECE, confusion counts, coverage, and invalid-output rate should be
retained in source artifacts and supporting analyses.

## 2. How We Work

### Local Mac: code editing and review

- Local repository: `/Users/metalicana/projects_spring_2026/RETFound`
- Use the local checkout to inspect and edit code, write documentation, and run tests
  that do not require the cluster datasets or GPUs.
- Do not infer that a data-dependent script is broken merely because its data or
  generated outputs are absent locally.
- Do not run cluster-only analyses locally and then present the absence of data as an
  experimental finding.
- The user controls commits and pushes. Prepare changes, test them, and show the diff;
  do not push unless explicitly asked.

### CECSL: data, model weights, GPUs, and experiments

- Host seen in the shell: `CECSL4622128797`
- Cluster repository: `/home/ab575577/RETFound` (normally `~/RETFound`)
- The user pushes code from the local machine and pulls it on CECSL before running.
- Generated outputs and large data are generally ignored by Git and remain only on
  CECSL unless explicitly copied back.
- CECSL has two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, each with about 98 GB VRAM.
- Never assume which GPU is available. Check `nvidia-smi` and follow the user's current
  instruction. The most recent GDP suite was intended for GPU 1.
- The cluster does not have `rg`; use `grep`, `find`, or Python there.

Typical synchronization flow:

```bash
# Local Mac: the user performs commit and push after reviewing changes.
git status
git diff

# CECSL:
cd ~/RETFound
git status
git pull --ff-only
```

Do not use destructive Git operations by default. The user previously requested that
incoming collaborator changes dominate and explicitly used `git reset --hard
origin/main`, but that was a one-off instruction. Preserve local changes unless the
user explicitly repeats that request.

At this handoff snapshot, local `main` is at `0366528` and reports one commit behind
`origin/main`. The first action in a new session should be to re-check `git status` and
`git branch -vv`; do not assume this snapshot remains current.

### Copying cluster artifacts back for local inspection

Because `equi-agent/outputs/` and most large data are ignored, use an archive or `scp`
when local inspection is required. Example:

```bash
# On CECSL
cd ~/RETFound
tar -czf /tmp/experiment_artifacts.tgz \
  equi-agent/outputs/path/to/run \
  equi-agent/outputs/metrics/path/to/metrics

# On the Mac
scp ab575577@CECSL4622128797:/tmp/experiment_artifacts.tgz /private/tmp/
```

Ask the user for the resulting artifact or pasted output when direct cluster access is
not available. Do not repeatedly attempt to discover CECSL-only results on the Mac.

## 3. User Preferences And Hard Boundaries

- Be direct and concise. Give the result, the reason, and the exact next command.
- When information is missing, ask for the specific output needed instead of writing
  speculative analysis code or running irrelevant local commands.
- The user executes cluster commands. Commands must be copy-pasteable and must not
  contain duplicated lines, truncated variable assignments, or shell snippets that
  call `exit` in a way that logs the user out.
- Use one GPU only unless explicitly asked otherwise. Respect the requested GPU.
- Use each foundation model's dedicated conda environment. Do not install all model
  dependencies into `retfound`.
- Do not name collaborators in generated manuscript prose or casual status reports.
  Say `collaborator`, `reference protocol`, or `locked-slice protocol` as appropriate.
- Do not report selective/`accepted`-case metrics in the main disease tables. Main
  tables use forced predictions on the complete locked cohort. If selective results
  are shown elsewhere, report coverage.
- Do not fabricate missing values. Use `--` for experiments that remain unrun and
  `N/A` only when a metric is scientifically unavailable.
- For LaTeX replacements, provide each row in a fenced text/LaTeX block.
- Prompt changes are tightly controlled. Do not modify an existing agent prompt
  without explicit permission. If a materially different prompt is needed, implement
  it as a distinct agent or clearly named variant after approval.
- When discussing prompts, show one agent at a time and identify the exact code
  location. Every live run must save inspectable prompt snapshots/reports, raw
  responses, traces, usage, errors, and resolved configuration where supported.
- The external glaucoma counterfactual is evidence-ablation reasoning, not a
  demographic counterfactual. Do not silently reintroduce demographic
  counterfactuals there.
- Do not overstate clinical claims. The system supports clinician workflow and triage;
  it does not replace diagnosis by an ophthalmologist.

## 4. Repository Map

Primary files:

- `equi-agent/manuscript/equi_agent_current_results.tex`: current working result tables.
- `equi-agent/manuscript/disease_tables_draft.tex`: broader Chapter 4 draft tables.
- `equi-agent/manuscript/experiment_todo.md`: tracked experiment TODO, but some items
  are stale relative to later CECSL runs.
- `update.md`: older handoff (last updated 2026-06-04); useful history, not current truth.
- `equi-agent/scripts/run_equi_agent_fairvision_live.py`: FairVision live arbitration.
- `equi-agent/scripts/run_gdp_progression_everything.sh`: resumable six-endpoint GDP
  progression suite.
- `equi-agent/scripts/collect_gdp_progression_complete_results.py`: strict final GDP
  artifact auditor and Markdown/LaTeX collector.
- `OphthalmicAgent/scripts/run_external_glaucoma_agent.py`: PAPILA/GAMMA/Drishti
  external agent runner.
- `OphthalmicAgent/scripts/audit_external_glaucoma_regressions.py`: paired baseline vs
  agent regression/rescue audit.
- `equi-agent/scripts/benchmark_papila_foundation_models.py`: external foundation
  feature/probe benchmark used for PAPILA, GAMMA, and Drishti workflows.
- `equi-agent/scripts/benchmark_refuge2_foundation_models.py`: REFUGE2 benchmark.

Important warning: historical conversation work included a much larger FairVision live
runner, simplified Equity/Orchestrator prompts, subgroup reliability scores, and a
five-call demographic counterfactual implementation. The current tracked
`run_equi_agent_fairvision_live.py` is about 1,883 lines and does not visibly contain all
of those later additions. Incoming Git changes appear to have replaced or reverted some
historical code. Do not claim those features are currently implemented without checking
the latest branch or recovering the relevant commit/artifact.

## 5. Data And Weight Locations On CECSL

### FairVision

Root:

```text
~/RETFound/Datasets/FairVision/
```

Expected structure includes:

```text
HarvardFairVision30k/{AMD,DR,Glaucoma}/ReadMe/data_summary_*.csv
Training/data_*.npz
Validation/data_*.npz
Test/data_*.npz
Training/slo_fundus_*.jpg
Validation/slo_fundus_*.jpg
Test/slo_fundus_*.jpg
```

The rebuilt manifests are under `equi-agent/outputs/manifests/` and contain 10,000
cases per disease: 6,000 train, 1,000 validation, and 3,000 test. If a manifest contains
Mac paths such as `/Users/metalicana/...`, rebuild it on CECSL or use the supported path
prefix remapping. An `NPZ not found` error with a Mac path on CECSL is a path/provenance
failure, not missing clinical data.

### Harvard GDP

Root:

```text
~/RETFound/Datasets/GDP/
  Bscan/
  RNFLT/
  ReadMe/
```

There are six progression label definitions over the same underlying cohort, not six
independent datasets:

```text
md
vfi
td_pointwise
md_fast
md_fast_no_p_cut
td_pointwise_no_p_cut
```

Each generated manifest has 500 rows. The locked test cohort used by the complete suite
has 200 cases. Expected test positives are respectively `18, 19, 18, 4, 6, 60`.
The multi-target LLM implementation deliberately predicts all six endpoints in one call
per patient to avoid six redundant calls over the same evidence.

### External datasets

- PAPILA: `OphthalmicAgent/data_papila/`; raw data and generated manifest exist on
  CECSL. Locked binary cohort: 407 eyes; test: 81 eyes, 15 positive. Gender mapping was
  verified from 244 bilateral patient pairs: `0 = male`, `1 = female`.
- GAMMA: `OphthalmicAgent/data_gamma/`; 100 cases, balanced labels, split 60/20/20.
  GAMMA contains CFP and OCT. In the reported benchmark RETFound uses OCT, the other
  foundation models use CFP, and Ours combines the OCT prediction with CFP-derived
  evidence.
- Drishti-GS1: `OphthalmicAgent/data_drishti/`; official IDs 001-050 are development,
  IDs 051-101 are the locked 51-image test set. Demographic metadata are unavailable.
- REFUGE2 Kaggle archive was downloaded on CECSL to
  `/home/ab575577/.cache/kagglehub/datasets/victorlemosml/refuge2/versions/1`.
  A generated manifest was stored at
  `equi-agent/outputs/manifests/refuge_kaggle_archive_manifest.csv`. The downloaded
  archive exposed labels for the 400-image train split but not the official validation
  or test images. Do not describe a derived split as the official labeled test set.

### Weights seen on CECSL

```text
equi-agent/weights/RETFound_mae_natureOCT.pth
equi-agent/VisionAgent/MIRAGE/MIRAGE-Base.pth
Foundation_Models/RET-CLIP-main/pretrained/ret_clip_vit_b_16.pt
Foundation_Models/RetiZero-main/pretrained/RetiZero.pth
Foundation_Models/UrFound-main/pretrained/urfound_mm.pth
OphthalmicAgent/weights/cfp_glaucoma_best.pth
OphthalmicAgent/weights/cfp_model.pth
OphthalmicAgent/weights/oct_model_best.pth
OphthalmicAgent/weights/slo_model_best.pth
```

Large weights and outputs are ignored by Git and may not exist on the Mac.

## 6. Conda Environments On CECSL

Known environments:

```text
retfound
equi-flair
equi-retclip
equi-retizero
equi-urfound
equi-visionfm
```

Use the model-specific environment for model inference/training. The GDP master runner
currently uses `PYTHON_ENV=retfound` for its common runtime. The most recent blocker was
that `retfound` lacked the Anthropic SDK needed by the Claude baseline.

Fix and verify without printing credentials:

```bash
cd ~/RETFound
conda activate retfound
python -m pip install "anthropic==0.120.2"

python - <<'PY'
import os
from dotenv import load_dotenv
from anthropic import AnthropicFoundry

load_dotenv()
print("AnthropicFoundry import: OK")
print("Claude endpoint configured:", bool(
    os.getenv("ANTHROPIC_FOUNDRY_BASE_URL") or
    os.getenv("AZURE_AI_ANTHROPIC_ENDPOINT")
))
print("Claude key configured:", bool(
    os.getenv("AZURE_API_KEY") or
    os.getenv("AZURE_OPENAI_API_KEY")
))
PY
```

## 7. Current Experimental State

### FairVision glaucoma

The current tracked manuscript has a populated FairVision glaucoma table with
RETFound, VisionFM, URFound, FLAIR, MIRAGE, RET-CLIP, RetiZero, three LLM baselines,
and Ours. Treat `equi-agent/manuscript/equi_agent_current_results.tex` as the current
draft, but preserve source artifacts and regenerate rows rather than manually trusting
the table.

Main scientific lesson: prompt tuning alone did not reliably beat strong
validation-thresholded foundation baselines. Reliability auditing, calibration,
evidence organization, and escalation are the defensible value proposition. For
glaucoma, structural evidence such as vCDR should be explicit. CFP report and CDR from
the same image are correlated representations, not independent votes.

### FairVision AMD and DR foundation rows produced on CECSL

These rows were reported by the locked 250-case collector but remain blank in the
current local manuscript. Recover the prediction/metric artifacts before final use.

| Model | Modality | AMD F1 / worst / sens / spec / bal | DR F1 / worst / sens / spec / bal |
|---|---|---|---|
| VisionFM | OCT | .8480 / .7802 / .8400 / .8560 / .8480 | .8116 / .7653 / .7680 / .8560 / .8120 |
| VisionFM | SLO | .7679 / .7096 / .7520 / .7840 / .7680 | .7083 / .6389 / .6000 / .8240 / .7120 |
| URFound | OCT | .8637 / .8327 / .8160 / .9120 / .8640 | .7855 / .7453 / .6800 / .8960 / .7880 |
| URFound | SLO | .7358 / .5962 / .7600 / .7120 / .7360 | .7119 / .4236 / .7280 / .6960 / .7120 |
| FLAIR | SLO | .5646 / .4365 / .7680 / .3920 / .5800 | .6898 / .5718 / .7760 / .6080 / .6920 |
| RET-CLIP | SLO | .7118 / .6507 / .7360 / .6880 / .7120 | .7391 / .6147 / .8000 / .6800 / .7400 |
| RetiZero | SLO | .7559 / .6933 / .7360 / .7760 / .7560 | .7269 / .6207 / .7920 / .6640 / .7280 |

The collector was `equi-agent/scripts/collect_fairvision_yusra_foundation_results.py`;
despite the historical filename, refer to it generically in conversation and prose.

### PAPILA

Locked test cohort: 81 eyes. Current recorded rows:

| Model | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| RETFound | .5333 | .3636 | .8000 | .7273 | .7636 |
| MIRAGE | .3333 | .0870 | .7333 | .3939 | .5636 |
| RET-CLIP | .4242 | .0000 | .4667 | .8333 | .6500 |
| RetiZero | .4516 | .2222 | .4667 | .8636 | .6652 |
| URFound | .3529 | .0000 | .4000 | .8030 | .6015 |
| Ours | .3500 | .0000 | .4667 | .7273 | .5970 |

The original agent run had 79 valid outputs and two invalid outputs; later collection
selected the latest valid attempt per case. The paired audit showed that arbitration
lost more correct RETFound cases than it rescued. Do not tune the existing prompt on
this tiny test set. Report the negative result honestly or keep PAPILA as external
robustness evidence.

### GAMMA

Locked test cohort: 20 cases. Current recorded rows:

| Model | Modality | F1 | Sensitivity | Specificity | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| RETFound | OCT | .8696 | 1.0000 | .7000 | .8500 |
| MIRAGE | CFP | .7059 | .6000 | .9000 | .7500 |
| RET-CLIP | CFP | .8889 | .8000 | 1.0000 | .9000 |
| RetiZero | CFP | .9474 | .9000 | 1.0000 | .9500 |
| URFound | CFP | .8889 | .8000 | 1.0000 | .9000 |
| Ours | OCT + CFP evidence | .8696 | 1.0000 | .7000 | .8500 |

The agent made no changes relative to RETFound: 17 cases were unchanged-correct and
three unchanged-wrong. That is a valid no-harm result, not an improvement claim.

### Drishti-GS1

An older cluster collector produced RETFound F1 `.6441` and Ours `.6667`. The current
tracked manuscript instead contains RETFound `.5370` and Ours `.7905`, plus LLM rows,
apparently from newer incoming work. Treat the tracked table as the latest draft but
audit its exact prediction files before publication. Do not combine values from the
two protocols.

Very high URFound/RetiZero results on Drishti are plausible but require an explicit
pretraining-overlap/protocol note. These benchmarks use frozen features with a trained
probe, not necessarily zero-shot inference.

### REFUGE2

The current tracked manuscript has RETFound and LLM/Ours rows but leaves RET-CLIP,
RetiZero, and URFound blank. A CECSL reference/MLP run later printed approximately:

| Model | F1 | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|
| RET-CLIP | .4836 | 1.0000 | .3417 | .6708 |
| RetiZero | .3630 | 1.0000 | .2333 | .6167 |
| URFound | .0768 | 1.0000 | .0333 | .5167 |

Do not insert these automatically. The script/protocol names changed between
`yusra_mlp`, `reference_mlp`, and `robust_cv`, and the downloaded archive lacked
official validation/test labels. First recover each `summary.json`, the source
manifest, split definition, threshold-selection policy, and generated benchmark
Markdown. RetiZero and URFound also require pretraining-overlap disclosure.

### Harvard GDP glaucoma detection

The current tracked manuscript is populated for RETFound, VisionFM, URFound, GPT-5.1,
GPT-5.6-luna, Claude Haiku 4.5, and Ours. Verify all rows share the locked 400-case
protocol before final submission. Historical GDP implementations differed, which is
why some VisionFM/URFound values looked unexpectedly high.

### Harvard GDP progression

The old manuscript table reports only `td_pointwise_no_p_cut`:

- RETFound linear probe: F1 `.4030`, worst-group F1 `.1480`, sensitivity `.4170`,
  specificity `.7210`, balanced accuracy `.5690`.
- GDP-native helper: F1 `.6000`, worst-group F1 `.5640`, sensitivity `.5000`,
  specificity `.9290`, balanced accuracy `.7145`.
- Ours: F1 `.6840`, worst-group F1 `.6190`, sensitivity `.6500`, specificity `.8930`,
  balanced accuracy `.7715`.

Those historical native/Ours numbers lacked restored exact source artifacts, so the
new suite was built to regenerate a complete, audited six-endpoint matrix.

The complete suite includes six classical feature baselines, RETFound, a six-output
GDP-native model, GPT-5.1, GPT-5.6-luna, Claude Haiku 4.5, and multi-target Ours for all
six endpoints. It resumes by skipping artifacts that pass strict row/positive-count
checks.

Most recent status: the suite did not run any experiment. It failed immediately in
`preflight` because `anthropic` was missing from `retfound`. After installing and
verifying the SDK and credentials, launch on GPU 1:

```bash
cd ~/RETFound
conda activate retfound

CUDA_VISIBLE_DEVICES=1 GPU=1 \
  bash equi-agent/scripts/run_gdp_progression_everything.sh --launch
```

Monitor:

```bash
bash equi-agent/scripts/run_gdp_progression_everything.sh --status
tail -f equi-agent/outputs/gdp_progression_everything_v1/master.log
```

Final outputs should be:

```text
equi-agent/outputs/gdp_progression_everything_v1/complete_results/
  completion_status.json
  gdp_progression_complete_results.md
  gdp_progression_complete_tables.tex
```

Do not call the suite complete unless `completion_status.json` confirms every method
and endpoint. An existing directory or PID file is not evidence that a run completed.

## 8. Reliability Methodology History

The original subgroup score `final_R_bad` was not meant to be discarded. It is a
validation-derived prior, with lower values interpreted as better and trust often
represented as `1 - final_R_bad`.

A later reliability-boundary experiment tested model selection using probability bins
and demographics on 1,000 glaucoma cases / 9,000 model-case rows:

- Probability-bin selector: accuracy `.743`, F1 `.73695`.
- Hierarchical demographic + probability-bin selector: accuracy `.750`, F1 `.73348`.
- F1-aware hierarchical bonus (`0.03`): accuracy `.751`, balanced accuracy `.75139`,
  F1 `.75125`.
- Any-model oracle accuracy was `.974`, but that is only an unattainable upper bound.

Interpretation: probability calibration region is more predictive of model correctness
than a single global reliability score; demographic conditioning adds a small gain;
an F1-aware tie/bonus can improve positive-class balance. However, the historical
`learn_foundation_reliability_boundaries.py` is not present in the current checkout.
Recover it before claiming this formula as implemented methodology.

Also remember that model-specific binary predictions may use validation-selected
thresholds other than `0.5`. Therefore a probability in the 0.40-0.50 bin can still be
a positive prediction and contribute false positives.

## 9. Agent And Prompt Guardrails

- The Equity Agent concept should receive each foundation model's probability and its
  reliability information, then select/recommend the most reliable source. Avoid
  elaborate disease-specific prompt rules that turn it into a hidden hand-coded
  classifier.
- The orchestrator should combine agent outputs and make a final decision; avoid
  theatrical language such as `source-list judge` and avoid pretending it is merely an
  averaging engine.
- The external glaucoma evidence counterfactual makes diagnoses under full evidence
  and leave-one-evidence-out scenarios (without RETFound probability, without visual
  interpretation, and without CDR). It measures evidence dependence. It does not alter
  race, gender, or age.
- Historical FairVision work explored five demographic combinations and five LLM calls
  to measure label stability (`5-0` and `4-1` stable; `3-2` unstable). Because the
  current tracked runner may no longer contain that implementation, verify before use.
- For any live run, inspect not only predictions but also errors, raw responses, prompt
  snapshots/reports, traces, token usage, and image attachment status.

Common failure interpretation:

- `prediction_rows 0` plus zero tokens means no successful live calls; inspect the
  error summary.
- JSON parse errors can invalidate every case even if the raw text looks plausible.
- `image_error` with a Mac path on CECSL means stale path provenance.
- `images_attached: false` means the vision agent did not actually receive the image.
- A CDR row with vCDR but missing area CDR is incomplete structural evidence and must
  not be described as two independent structural measurements.

## 10. Clinical Human Evaluation Direction

The intended human study is a retrospective multi-reader multi-case (MRMC) evaluation
for an Orlando VA collaboration:

- Compare clinician-only interpretation with clinician + RetinAgent assistance.
- Use a randomized crossover design with case-order randomization and a washout period
  to limit recall.
- Include multiple ophthalmologists/readers and a locked, adjudicated case set.
- Primary endpoint: paired change in diagnostic accuracy or balanced accuracy.
- Secondary endpoints: sensitivity, specificity, F1, reading time, confidence,
  escalation/referral decisions, decision changes, and agreement with adjudicated
  reference standard.
- Capture whether the system changed a correct answer to incorrect and vice versa.
- Add usability/trust measures, but do not use a preference survey alone as evidence of
  clinical benefit or adoption.
- Analyze with reader- and case-aware methods (mixed effects or standard MRMC methods)
  and paired bootstrap confidence intervals.
- Frame the tool as decision support. Clinical governance, IRB/privacy review, data-use
  agreements, and clinician responsibility remain explicit.

## 11. Immediate Next Actions

1. On CECSL, install/verify the Anthropic SDK and Foundry credentials in `retfound`.
2. Relaunch `run_gdp_progression_everything.sh` on GPU 1 and monitor its status file.
3. When complete, inspect `completion_status.json` and copy the generated Markdown and
   LaTeX tables into the manuscript only after cohort/provenance checks pass.
4. Recover the CECSL artifacts for the already-produced AMD/DR locked-slice rows and
   fill the current manuscript gaps.
5. Reconcile REFUGE2 protocol provenance before inserting the three foundation rows.
6. Restore PAPILA/GAMMA source summaries and prediction CSVs for final archival
   provenance.
7. Add confusion counts, subgroup support, confidence intervals, and paired statistical
   comparisons before submission.

## 12. New-Account Startup Prompt

Use this at the beginning of the new account/session:

> We are continuing the RetinAgent/Equi-Agent ophthalmic AI project. First read
> `NEW_ACCOUNT_PROJECT_HANDOFF.md`, `equi-agent/manuscript/equi_agent_current_results.tex`,
> and `equi-agent/manuscript/experiment_todo.md`. Re-check Git state because the handoff
> may be behind `origin/main`. The local Mac checkout is for code editing; data, weights,
> environments, GPUs, and most outputs are on CECSL at `~/RETFound`. I execute cluster
> commands and control Git pushes. Do not modify existing agent prompts without explicit
> permission, do not run cluster-only analyses locally, do not fabricate missing table
> values, and keep commands concise and copy-pasteable. Our immediate task is to finish
> the audited six-endpoint Harvard GDP progression suite, then fill only the genuinely
> missing manuscript rows.
