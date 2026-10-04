# FairVision Component Comparison and Reliability Ablation

Entry point: `OphthalmicAgent/scripts/run_fairvision_ablation.py`.

## What This Tests

| Row | Inputs and decision |
|---|---|
| Simple multimodal ensemble | Unweighted mean of the saved probabilities from nine FM/modality configurations; validation-selected macro-F1 threshold. |
| Reliability-weighted fusion | Same probabilities, weighted by `1 - risk` for each patient's marginal demographic groups, followed by a validation-selected macro-F1 threshold. No LLM calls. |
| Agents without reliability priors | Existing task-specific OphthalmicAgent Bio-Profiler, OCT/SLO reports, RETFound probability and CDR; no numeric reliability information in either the evidence audit or final orchestrator. |
| RetinAgent paired full control | Identical shared upstream evidence, plus validation-derived reliability of the exact checkpoint used by the agent. Independent evidence-audit and final calls. |

The first two rows are **comparison baselines**, not literal component removals
from the live agent: the current FairVision agent has one RETFound OCT predictor
and multimodal image reports, not nine probability-producing tools. The last
two rows directly test removing reliability. They cannot establish the
contribution of every other agent component.

The default nine configurations are RETFound OCT, VisionFM OCT/SLO, UrFound
OCT/SLO, MIRAGE SLO, FLAIR SLO, RET-CLIP SLO and RetiZero SLO. Thresholds for
fusion are selected on 1,000 validation cases/task, over the fixed grid
0.01--0.99; ties prefer the threshold nearest 0.5, then the larger value.
Do not choose a pool, threshold or reporting definition after seeing test gains.

## Locked Evaluation

- Same 250 test cases per task, identified by `(task, image_id)`. The recovered
  manifest has 124/126 glaucoma positives/negatives, and 125/125 for AMD and DR.
- Macro-F1 always includes both binary labels. Worst-group F1 is the minimum
  **macro-F1** over recorded race, sex and age groups (<50, 50--69, >=70), with
  no support cutoff. Ethnicity is not included in this contract. Change the
  draft caption claiming ethnicity accordingly; do not mix with older weighted
  subgroup F1. Subgroup counts and confusion matrices are exported.
- Mean worst-group F1 is the unweighted mean of the three task minima, not a
  pooled-patient minimum. Both live arms must complete all 250 cases for a task
  before either is scored. No missing prediction is imputed or dropped.
- With both labels included, a subgroup containing only one class has macro-F1
  at most 0.5. This applies to the younger AMD subgroup in these offline rows.
  Interpret its support explicitly rather than treating that floor as proof of
  disparate error rates, or comparing it with the historical weighted-F1 minimum.
- Existing full-agent numbers are **not** inserted. In particular, the newly
  available `results/fairvision/fairvision_amd_agentic.csv` has 40 invalid
  predictions; its reported weighted-F1 0.8858 describes 210 valid rows.

## Reliability and Paired Calls

Priors and support counts use validation data only. Risk coefficients remain
0.35 FNR, 0.25 FPR, 0.15 ECE, 0.15 (1-AUROC), 0.10 (1-positive-class F1).
ECE uses the existing ten-bin implementation. Age/race/sex marginal risks are
support-weighted and normalized, then shrunk toward global risk with
`n_intersection / (n_intersection + 50)`. Undefined subgroup metrics use the
global metric. These are heuristic coefficients, not newly validated optima.

The agent's checkpoint is evaluated on the 3,000 validation images **without
API calls or training** before the live comparison. Its own 0.5 decision rule
is retained for these priors; probabilities are rounded as in the existing
Vision Agent. Saved fusion-probe priors are not substituted for this checkpoint.
Checkpoint and image hashes are checked on resume.

This is a corrected paired experiment, not an exact reproduction of historical
scores: subgroup support now comes from validation rather than test covariates;
age boundaries are consistently 50/70; the shared glaucoma-only counterfactual
prompt is adapted to AMD/DR; and final decisions use a strict JSON output
contract rather than silently falling back to negative. The original
task-specific specialist and orchestrator modules remain unchanged. Both arms
get the same formatting/clarification changes and shared upstream evidence.

GPT-5.1, Azure API version `2024-12-01-preview`. Bio-Profiler and final
orchestrator temperature 0.3, image-report temperature 0.2 when unspecified,
counterfactual audit temperature 0. Generation budgets are 4,096 tokens except
the five-scenario audit (8,192). These are output limits, not guaranteed usage.
Requests, usage and raw accepted responses are cached; image payloads in logs
are replaced with hashes. Failed, refused or truncated calls are retried up to
three total attempts, then halt the run. Resume reuses successful calls.

Both arms receive demographics; only numerical reliability is withheld. A
counterfactual trace generated with reliability is never reused by the arm
without it. The five scenarios are evidence-source ablations, not a demographic
perturbation experiment. Case order is deterministic, and the first arm varies
by a hash of case ID. No stochastic repeats are claimed.

## Run on CECSL

After syncing code, either prepare directly from the existing cluster files:

```bash
cd ~/RETFound
conda activate retfound
SCRIPT=OphthalmicAgent/scripts/run_fairvision_ablation.py
RUN="$HOME/RETFound/OphthalmicAgent/outputs/fairvision_ablation_v1"
python "$SCRIPT" --stage prepare --run-root "$RUN" \
  --data-root "$HOME/RETFound/Datasets/FairVision"
```

Or transfer the prepared bundle from the Mac (avoids assuming the recovered
manifest or every prediction CSV is already present on CECSL):

```bash
# Mac terminal
scp /tmp/fairvision_ablation_inputs.tar.gz ab575577@10.171.42.25:/tmp/
```

```bash
# CECSL terminal, fresh RUN directory only
cd ~/RETFound
conda activate retfound
SCRIPT=OphthalmicAgent/scripts/run_fairvision_ablation.py
RUN="$HOME/RETFound/OphthalmicAgent/outputs/fairvision_ablation_v1"
mkdir -p "$RUN"
tar --keep-old-files -xzf /tmp/fairvision_ablation_inputs.tar.gz -C "$RUN"
python "$SCRIPT" --stage configure --run-root "$RUN" \
  --data-root "$HOME/RETFound/Datasets/FairVision"
```

`--data-root` accepts the OphthalmicAgent directory with `data/` underneath it,
or the FairVision dataset root. Supported dataset layouts include
`Glaucoma/Test/file.npz`, `Test/Glaucoma/file.npz`, `Test/file.npz`, and
`HarvardFairVision30k/Glaucoma/Test/file.npz`, with the corresponding validation
and AMD/DR folders. Task and split are preserved; the resolver never searches
recursively by bare filename or substitutes a file from another split. Symlink
aliases are accepted; multiple distinct matches are rejected as ambiguous.

The default checkpoint is
`OphthalmicAgent/weights/oct_model_8_slices_not_center.pth`; override
`--oct-weights` during prepare/configure if needed. Configure checks all 3,750
image paths before changing the saved configuration. Smoke checks the images
and checkpoint before its first paid call. The environment must support the
existing OCT/SLO agents and the Hugging Face CDR checkpoint. No checkpoint is
trained by this runner.

For a read-only path audit (no API clients or imaging dependencies):

```bash
python "$SCRIPT" --stage paths --run-root "$RUN"
```

The original October 3 bundle only supported the legacy OphthalmicAgent layout.
If it stopped with `Missing 3750 images`, sync this code fix and reuse the already
extracted bundle without another transfer or extraction:

```bash
python "$SCRIPT" --stage configure --run-root "$RUN" \
  --upgrade-path-layout \
  --data-root "$HOME/RETFound/Datasets/FairVision"
```

The upgrade recognizes only the two original path-handling source hashes. It
archives the old config, keeps the prepared cohort, labels, predictions and
thresholds unchanged, and refuses to proceed if prompts or other source files
have changed, any inference artifacts exist, or another process holds the run
lock. It is not a general bypass for stale caches. macOS tar extended-attribute
warnings are unrelated to missing dataset images.

First run one paired case per task. This needs **21 successful API calls**
before retries; validation inference is cached for the full run:

```bash
CUDA_VISIBLE_DEVICES=1 python -u "$SCRIPT" --stage smoke --run-root "$RUN" --device cuda:0
```

Only after smoke passes, launch/resume all three tasks:

```bash
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  python "$SCRIPT" --stage run --run-root "$RUN" --device cuda:0 \
  >> "$RUN/run.log" 2>&1 < /dev/null &
echo $! > "$RUN/run.pid"
tail -n 40 "$RUN/run.log"
```

A complete fresh run needs **5,250 successful API calls**, including the smoke
cases, before retries: three shared calls plus two calls per arm per patient.
This is not a small no-priors-only replay. No paid calls have been run locally.

```bash
python "$SCRIPT" --stage collect --run-root "$RUN"
```

Outputs: `table4.md`, `table4.tex`, `results.csv`, `subgroups.csv`, per-case
`agent/` decisions, shared reports, validated call caches and configuration
hashes. Keep raw patient reports and demographic input bundles private under
the dataset's data-use terms. Existing experiments are not overwritten.

## Resume After an Unavailable CDR

The SLO tool can return `"Not Available"` or `-1` when segmentation fails. The
ablation adapter now preserves this as `"Not Available"`, not zero, a diagnosis,
or a reason to remove the patient. Valid CDR measurements are unchanged.

For an existing run that crashed at `float(cdr)`, pull the correction and run:

```bash
python "$SCRIPT" --stage repair-cdr --run-root "$RUN"
```

This stage makes **no API calls** and does not launch inference. It accepts only
the recognized code correction, refuses an active run or unrelated source
changes, and records provenance under `cdr_missing_value_repair_v1/` without
changing the original experiment fingerprint or upstream cache keys.

Completed cases with valid CDR, validation predictions, reliability priors and
all shared paid reports are retained. If a cached case has an invalid numeric
CDR (including the AMD smoke case's `-1`), its original shared evidence and both
arms' downstream caches/decisions are archived. Its cached evidence is corrected;
only the two audits and two final decisions require fresh API calls on resume.
The case that crashed before saving shared evidence reuses its successful
Bio-Profiler/OCT/SLO API caches while reconstructing that evidence locally.

The repair is resumable and idempotent. Use the existing `--stage run` command
only when ready to resume paid inference; it still runs all three tasks.

## Local Verification

```bash
python -m unittest discover -s OphthalmicAgent/tests -p test_fairvision_ablation.py -v
```

The offline comparisons have been computed on the saved predictions. Tests
exercise synthetic clients, strict parsing, paired completeness, label-blind
evidence construction, no-priors isolation and cache resumption. GPU inference
and live Azure calls require the CECSL smoke test; they are not claimed tested
by the local suite. No improvement of the full arm is guaranteed.
