GLAUCOMA V3: ACQUISITION FRAMING AND TRUE SLICE LABELS
==================================================

This experiment changes OCT specialist presentation, not the classifier or
diagnostic decision rules. It is separate from V1 and the frozen V2 replay.
No original prompt, prediction, manuscript or V2 artifact is overwritten.

Recovery for a first-request token-limit stop
--------------------------------------------
The original V3 OCT request retained a 500-token completion cap. A provider stop
other than "stop" was rejected before any dependent call. The old console error
did not print the actual finish reason, so it alone does not prove truncation.
Do NOT delete/reprepare the experiment, reset the ledger or accept partial text.

After syncing the updated code, inspect the saved receipt offline:

  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py --stage inspect

This prints finish reason, token usage, response character count, refusal and
error status without making API calls or editing the ledger. If the first and
only attempt has finish_reason="length", the following explicit amendment is
available (also offline):

  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py --stage amend-oct-limit
  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py --stage preflight

The amendment changes ONLY OCT max_completion_tokens from 500 to 2000 for every
eligible case. Images, prompts, deployment, omitted reasoning-effort parameter,
case order, non-OCT evidence, counterfactual settings and final settings stay
unchanged. This is an explicit generation-setting revision, not a claim that
the original and amended requests have identical sampling behavior or cost.
A larger cap allows greater token expenditure; it does not guarantee completion.
The completion cap includes visible output and reasoning tokens, not just the
visible report. Official parameter definition:
https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create

The command checks the exact original request, first-case identity and provider
receipt. It refuses any other failure reason, refusal, missing response, already
successful run or run with additional attempts. The failed row is NOT removed,
renamed or overwritten. Its request, raw response and token usage stay intact.
bundle.json stays unchanged. oct_completion_amendment.json freezes the revised
request hashes, original failed receipt, runtime hashes and one additional slot.
The same ledger gains a distinct oct_completion_repair attempt for the first
case. Budget becomes 148, INCLUDING the original failed request, rather than a
new 147-call allowance that forgets the failure. Recorded V2 100 + V3 148 = 248,
assuming no unrelated jobs. Returned-model and endpoint pins remain enforced.

After confirming the offline amendment, restart only the two-case pilot:

  python OphthalmicAgent/scripts/run_glaucoma_v3_replay.py \
    --stage run --allow-api --max-cases 2

This permits at most six NEW calls, hence seven V3 attempts including the failed
one. A repeat run reuses successful stages. A second failure stops and blocks
continuation; rerunning amend-oct-limit does not grant another retry or budget
increase. Never delete a failed row to bypass this safeguard. The exact original
V3 runner hash is supported for this amendment; arbitrary runtime drift is not.
If the offline amendment is interrupted, repeating that command can finish the
same recorded migration without resetting attempts or adding another allowance.

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
- The original OCT request has max_completion_tokens=500 and no temperature
  sent. Only the explicit recovery amendment above changes that cap to 2000;
  truncation still blocks downstream use. Never splice incomplete reports into
  a final diagnosis or silently change the frozen generation settings.

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
The explicit one-failure amendment above instead caps the same ledger at 148.
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
    oct_completion_amendment.json (only after explicit eligible amendment)
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
    -p 'test_glaucoma_v3*.py' -v

Synthetic clients only. Covers index derivation, pixel-transform parity, framing,
label blindness, upstream evidence replacement, stale-trace exclusion, frozen
request integrity, missing-row retention, cumulative budgets, resumption,
truncation/errors/model drift, task-qualified paths and offline CLI opt-in.
Amendment tests additionally cover preservation of the original receipt, cap-only
request changes, idempotent repair, failed-call accounting, refusal/filter guards,
interrupted migration, read-only inspection and rejection of arbitrary code drift.
