FROZEN RETINAGENT RERUN
======================

This is a new experiment, not a repair or overwrite of historical results.
There is no SSH or automatic remote execution. Only --stage run makes API
calls. The shell launcher calls prepare followed by run; launch it on CECSL
only when ready to incur those calls. A fresh output directory is required.

Frozen design
-------------
cohorts.csv contains 250 cases per FairVision task, 125 positive/125 negative.
Selection seed: 20261006. Each task's eligible pool contains 2750 test cases
after excluding all 250 locally identified prior live-agent attempts, including
invalid responses. selection_receipt.json contains source hashes and exclusions.
Selection does not use model predictions or correctness. Preparation rejects
overlap with additional prior agentic result CSVs present on CECSL. It does not
prove that no other unrecorded remote experiment inspected these cases.

FairVision reliability and evaluation ages: <50, 50-69, >=70. The four historical
FairVision entrypoints now use the same mapping. GDP's separate age convention
has not been changed. Reliability for the live anchor is recomputed from 1000
validation images per task using the actual OCT checkpoint; validation support
counts, not test support counts, control shrinkage. Coefficients remain
(FNR,FPR,ECE,1-AUROC,1-F1)=(.35,.25,.15,.15,.10), k=50.

The four FairVision arms use identical selected case IDs:
1. Simple multimodel OCT+SLO probability average, validation-selected threshold.
2. Reliability-weighted multimodel fusion, validation-selected threshold.
3. Task-specific OphthalmicAgent without supplied reliability estimates.
4. Paired full task-specific OphthalmicAgent with supplied reliability estimates.
The first two arms are comparison baselines, not identical-evidence component
removals: they use multiple saved encoder probes; the live arms use the original
RETFound OCT anchor plus OCT/SLO image reports. The paired live arms share image
reports, patient narrative, tool probability and CDR. Both rerun the five-scenario
evidence-ablation stage and final orchestrator. Their order varies deterministically
by case ID. Historical scores are never substituted into this table.

Task definitions are explicit in the frozen request adapter: glaucoma present;
any AMD; vision-threatening DR (severe NPDR/PDR, with no/mild/moderate DR negative).
The underlying task-specific clinical prompts are retained. The output contract
requires diagnosis 0/1, reasoning, overview, escalation_required (JSON boolean)
and escalation_reason. Review is a separate, qualitative agent judgment for poor
quality, inadequate evidence, uncertainty or unresolved disagreement. There is
no invented numeric escalation threshold. A review flag never drops the forced
diagnosis from full-cohort metrics. After three invalid/failed requests the run
stops, retaining successful caches. No failure is imputed as negative.

Model and generation lock
-------------------------
Every API stage is forced through CachedClient: Azure deployment gpt-5.1,
API version 2024-12-01-preview, temperature 0.2, top_p 1.0,
max_completion_tokens 8192; seed omitted. The user confirmed that the Azure
deployment is frozen. model_identity.json records the returned model string;
a changed returned string stops further calls. API request timestamps, response
IDs, usage and content are saved per call, with embedded images represented by
hashes in saved requests. This is not a guarantee of deterministic LLM output.

source_lock.json freezes the source, prompts, cohort, protocol and tests. All
must match committed Git content before preparation/inference. The prompt export
is a source string inventory, including inactive legacy strings, not a claim
that every exported template executes. api/ contains the actual executed request
templates. confirmation.json records the Git commit used for preparation.
Input image and checkpoint SHA256 hashes are bound at CECSL preflight because
those files are not on this workstation. environment.json records package
versions; changing them prevents silent resume. CDR's Hugging Face model ID and
resolved revision are recorded and checked for consistency across cohorts/resume.

Live demographic audit
----------------------
For each of the 750 full-agent cases, enumerate six single-attribute changes:
two alternative races, one alternative sex, two alternative age bands (values
35/60/75), and one alternative ethnicity. These are not five random profiles.
Each profile reruns Bio-Profiler, the five-scenario evidence-ablation call and
final orchestrator with recomputed reliability. Image-only reports, OCT score
and CDR remain fixed. Ethnicity does not enter the reliability lookup, so its
perturbations audit the narrative path. The original and changed forced labels
and explicit escalation flags yield task/attribute/direction change rates.
At nonzero temperature, these differences can include LLM sampling variability;
they are not isolated causal demographic effects or a repeatability experiment.

External cohorts
----------------
All 51 Drishti-GS1 test images (34 positive) and all 400 REFUGE2 test images
(40 positive) receive new CFP report, evidence-ablation and final-agent calls.
Sources are the complete saved RETFound CFP probability CSVs, not partial agent
responses. The agent receives RAW probabilities, not scores normalized using
test labels. The external RETFound comparison uses a prespecified 0.5 threshold.
This differs from historical normalized-score runs and may change performance.
No unverified subgroup/global trust score is manufactured when none is supplied.
These external cohorts were previously evaluated: completeness does not make
them untouched confirmatory test sets.

Cost and resume
---------------
Expected successful API calls, before retries, from a completely empty cache:
FairVision paired arms: 750 * (3 shared + 2 per arm) = 5250.
Demographic audit: 750 * 6 * 3 = 13500.
External cohorts: 451 * 3 = 1353.
Total: 20103. Local validation, fusion and CDR add no LLM calls.
Default request-attempt cap: 22000. SDK internal retries are disabled. The cap
counts attempts, not dollars or tokens; monetary cost depends on actual usage.
No escalation rate or improvement over a helper is promised.
Re-running the same command resumes cached successful stages, including a
partially completed case. Do not delete caches or change the frozen sources.
No historical API caches are used. Never use this new runner on an old run root.
Changes to shared ablation code mean older frozen ablation runs should remain
on their original commit, not silently resume under this new protocol.

CECSL commands (after pushing/pulling this commit)
------------------------------------------------
cd ~/RETFound
conda activate retfound
SCRIPT=OphthalmicAgent/scripts/run_frozen_confirmation.py
RUN="$HOME/RETFound/OphthalmicAgent/outputs/fairvision_confirmation_v1"

# Offline preparation and complete image/checkpoint/label preflight:
python "$SCRIPT" --stage prepare --run-root "$RUN" \
  --data-root "$HOME/RETFound/Datasets/FairVision"
python "$SCRIPT" --stage preflight --run-root "$RUN"

# Only after successful preflight; starts the entire paid experiment:
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  bash OphthalmicAgent/scripts/run_frozen_confirmation.sh "$RUN" \
  --device cuda:0 --max-api-calls 22000 > "$RUN/run.log" 2>&1 < /dev/null &
echo $! > "$RUN/run.pid"

# Check progress or resume with the same launcher after a stopped run:
tail -n 30 "$RUN/run.log"

prepare accepts --predictions-root, --oct-weights and --external-root when
nondefault paths are needed. Defaults: equi-agent/outputs/predictions;
OphthalmicAgent/weights/oct_model_8_slices_not_center.pth; OphthalmicAgent.
All nine saved model-probe validation/test CSV pairs must be present. Preparation
checks their IDs, labels and demographics against the cohort. Preflight verifies
3750 FairVision NPZ files (3000 validation + 750 test), target reference labels,
451 external images and the OCT checkpoint before any API call.

Outputs
-------
results.csv / table.md: four arms, three tasks, macro-F1 and subgroup metrics.
subgroups.csv: every group, confusion counts, fixed-label [0,1] macro-F1;
race, sex and age, unknown excluded, no post-hoc support exclusions.
live_escalation.csv: flagged/unflagged fractions, forced-label macro-F1 and
2000 class-stratified bootstrap 95% percentile intervals, seed 20261006.
Intervals condition on saved labels and selected group class counts, assume
independent NPZ cases and do not measure LLM repeatability. Empty groups have
undefined F1. Single-class groups retain fixed [0,1] macro-F1 semantics.
live_demographic_counterfactuals.csv / live_demographic_directions.csv:
paired outcomes for all 4500 profiles, with directional rate denominators.
external_metrics.csv: all-51/all-400 confusion counts and forced metrics.
completion.json: complete=true only after every required output is validated.
There are no partial-cohort final tables. collection runs automatically at
completion; --stage collect is also an offline, strict completeness check.

Fresh live-agent cases do not erase prior inspection of full-test model/fusion
results. Report this limitation and do not revise prompts on these new errors
and then relabel the same cases as a new independent confirmation.

Local no-network verification
-----------------------------
python -m unittest discover -s OphthalmicAgent/tests \
  -p 'test_frozen_confirmation.py' -v
python -m unittest discover -s OphthalmicAgent/tests \
  -p 'test_fairvision_ablation.py' -v
Tests use synthetic API responses. They require numpy, pandas and scipy but
not Azure credentials, GPU images or paid calls.
