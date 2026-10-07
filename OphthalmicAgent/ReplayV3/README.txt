GLAUCOMA V3: ACQUISITION FRAMING AND TRUE SLICE LABELS
==================================================

This experiment changes OCT specialist presentation, not the classifier or
diagnostic decision rules. It is separate from V1 and the frozen V2 replay.
No original prompt, prediction, manuscript or V2 artifact is overwritten.

Changes
-------
1. Remove the assumed macular/foveal framing and blanket prohibition on visible
   optic-disc observations. Describe the expected optic-nerve-head context, with
   its provenance: offline visual dataset inspection, NOT verified acquisition
   metadata. Ask for visible anatomy, permit uncertainty, and do not infer
   glaucoma from an excavation alone. No invented CDR, RNFL/rim measurements,
   laterality, physical distances or anatomical sectors.
2. Derive every caption and request index from the actual volume. Indices are
   zero-based along array axis 0. Each context panel has its own slice label.
   For depth 200: central 100; context 28,56,113,142. The old captions 64 and
   16,32,80,96 were incorrect for these inputs. These are spread-out context
   slices, not adjacent slices or proof of continuity between displayed views.
3. Remove the unsupported unconditional statement that the classifier was
   uncertain. No classifier score or reference label enters the OCT request.

Held fixed
----------
- Central slice depth//2 and context slots 1,2,4,5 of the existing eight-slice
  linspace selection. No new scan-axis interpretation, rotation, crop or sampling.
- Legacy intensity conversion, CLAHE (1.5; 8x8), central 2x cubic enlargement,
  context image sizes and JPEG encoding. Only caption text/placement changes.
- Saved RETFound probability, narrative, SLO report, CDR and reliability value.
- Existing counterfactual prompt and temperature 0.
- Frozen V2 final reasoning instructions, output schema and generation settings.
  Only the final input's OCT provenance changes to reflect actual reassessment.
- OCT max_completion_tokens=500, with no temperature sent, as in the existing
  specialist. Truncation now blocks downstream use; do not silently increase the
  limit or splice incomplete reports into a final diagnosis.

Execution
---------
For each eligible historical error:
  1. New OCT specialist call with the correctly labelled image and V3 framing.
  2. New evidence-ablation/counterfactual call using the new OCT report.
  3. New final call using that report and the NEW counterfactual trace.

The old OCT report and old counterfactual diagnoses are never passed downstream.
This is not another final-prompt-only replay. It is also not a full pipeline
rerun: unchanged upstream sources are cached. SLO/CDR quality concerns and other
possible changes are deliberately outside this experiment.

The fixed cohort is all 57 original errors. Forty-nine have complete saved
non-OCT evidence and can run; eight remain explicitly unassessed despite having
raw images. The case order is inherited from V2, not selected by V2 outcomes.
No correct-case control requests are scheduled. Evaluation labels and old
predictions are used only for selecting the historical error cohort and offline
analysis, never as model inputs.

Budget and safety
-----------------
At most 49 x 3 = 147 request attempts, including failures/interrupted attempts.
The downloaded V2 receipts contained 100 attempts. Together these are 247,
within the previously stated 250-call allowance, assuming no other new jobs.
This code cannot account for unrelated jobs or manual reruns elsewhere.

No extra V3 control arm, specialist SLO call, CDR segmentation, Bio-Profiler,
classifier inference or training. API SDK retries are disabled. No automatic
application retries. A two-case pilot uses at most six attempts, included in
the same 147-attempt ledger, not added to it.

Run requires BOTH --allow-api and an explicit --max-cases. The latter is a
prefix of the frozen order, not an additional count: 2 then 49 resumes the
same run without repeating the first two. Do not copy/reprepare this experiment
into a fresh directory to retry requests; that would evade its persistent cap.

Reservations are committed before each request. Errors, refusals, malformed
responses, truncation or model drift stop the run. A stopped/interrupted ledger
blocks later runs until inspected. Missing outcomes are never negative labels.
New counterfactual/final requests are reconstructed from validated predecessor
receipts and checked on resume and collection. The returned model and endpoint
identity are pinned; keys are not written. Images, source NPZs, source bundle,
V2 results, rendered requests and runtime code are hashed. The actual sent JPEG
bytes are frozen during preparation, not re-rendered at run time.

Cluster commands (USER executes; from ~/RETFound)
------------------------------------------------
Requirements: existing replay environment plus numpy, opencv-python and
Pillow >= 10.1. Paid stages also require openai and the existing Azure environment
variables AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY. No .env auto-loading.
Preparation/preflight need no GPU, model weights, credentials or network.

  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py --stage prepare
  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py --stage preflight

Default raw source is Datasets/FairVision/Glaucoma/Test. There is no fallback to
AMD/DR files with identical names. --data-root may supply another Glaucoma/Test
directory. Alternatively --archive reads the transferred Glaucoma-only tar.gz
without extracting it. Only oct_bscans is loaded from NPZ; no reference fields.
The existing V2 bundle and failed_case_results.csv supply frozen saved evidence
and offline evaluation. Explicit path overrides are available in --help.

Inspect images/ and bundle.json before launching a pilot:

  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py \
    --stage run --allow-api --max-cases 2

Stop and inspect the pilot OCT reports and downstream receipts. There is no
automatic continuation. After review, the user may run all eligible errors:

  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py \
    --stage run --allow-api --max-cases 49

Offline analysis / re-export:

  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py --stage collect

Default experiment directory:
  OphthalmicAgent/outputs/glaucoma_v3_replay/
    bundle.json, images/*.jpg
    run/ledger.sqlite3, run_identity.json, api_receipts.jsonl,
        case_results.csv, summary.json, report.txt

Use the same directory for pilot and continuation. Preparation refuses to
overwrite it. Generated patient images/requests are local research artifacts,
not source files for publication or version control.

Interpretation
--------------
Reports retain all 57 rows and give repairs, persistent errors and unassessed
cases, plus additional/lost repairs relative to saved V2 on matched valid cases.
V2 is not a contemporaneous control; new stochastic outputs can also change
results. This cannot establish causality, generalization, significance or a
new 250-case F1. Regression on previously correct cases is not measured here.
No promised improvement and no substitution into historical paper results.

Offline tests
-------------
  python -m unittest discover -s OphthalmicAgent/tests \
    -p test_glaucoma_v3_replay.py -v

Synthetic clients only. Covers index derivation, pixel-transform parity, framing,
label blindness, upstream evidence replacement, stale-trace exclusion, frozen
request integrity, missing-row retention, cumulative budgets, resumption,
truncation/errors/model drift, task-qualified paths and offline CLI opt-in.
