GLAUCOMA V2: SMALL EXPLORATORY FINAL-STAGE REPLAY
==============================================

This is separate from the manuscript run, the paired reliability ablation, and
the large frozen-confirmation experiment. Do not launch run_frozen_confirmation
for this task. No old prediction, checkpoint, prompt or manuscript is modified.

What is being tested
--------------------
The diagnostic orchestrator receives one frozen saved evidence packet per case.
The packet contains the existing narrative, OCT/SLO reports, RETFound probability,
CDR, reliability value and counterfactual audit. No upstream stages are rerun.

Two arms receive the same clinical evidence:
  legacy_prompt_control: current source-extracted legacy orchestrator prompt.
  v2: same reasoning instructions, replacing the old output section with the
      existing strict JSON/explicit-review contract and three fixed safeguards.

The safeguards distinguish non-assessable imaging from evidence against disease,
identify CDR as unverified without segmentation QC, and prohibit invented anatomy
when scan location is uncertain. They do not introduce a numerical disease
threshold, discard high CDR values, or force agreement with RETFound. The V2
request includes deterministic measurement-provenance flags; the value itself
is retained. This is one combined engineering revision, not an attribution study
of which individual safeguard helps.

Malformed or refused responses never become negative predictions in either arm.
The control uses the legacy prompt, but strict audit parsing: it does NOT emulate
the known silent-negative fallback. This is not a recreation of unknown historical
requests. The corrected age lookup is NOT recomputed for this saved-evidence
replay; historical reliability values are held fixed in both arms.

Selection and cost
------------------
All 57 historical errors are listed. Of these, 49 have evidence in the named
run-specific cache. The eight without evidence are untested, not filled from
other caches. All 23 historical RetinAgent corrections of RETFound errors are
included as regression checks. Both-correct cases are not included, so even a
successful replay cannot establish safety or accuracy across all 250 cases.

72 eligible cases x 2 final-stage requests = at most 144 API attempts.
The cap includes failed/invalid/interrupted attempts. No SDK retries, automatic
repair requests, vision calls, counterfactual calls, training or model inference.
Each request has a fixed 2,000 completion-token cap. Input token costs still apply;
the long saved reports and traces are preserved, not summarized with another API.
No dollar-cost estimate is asserted.

Evidence selection is the first record in source-file order, independent of its
diagnosis/correctness. This is NOT asserted to be the packet used historically.
Case order and arm order are deterministic hashes, not sorted by outcome.
Reference labels, historical predictions and error/control membership are used
only for selection/scoring and are never included in API requests.

Portable preparation
--------------------
bundle.json contains frozen inputs, requests, evaluation labels, source hashes
and runtime-code hashes. Include it with the new script and tests when YOU commit
and push. The cluster does not need the local audit directory to run the bundle.
Do not commit API keys or the paid-run output directory. The assistant performs
no commit/push/pull/SSH or paid requests.

Preparation (already performed locally; zero API calls):
  python OphthalmicAgent/scripts/run_glaucoma_v2_replay.py --stage prepare

An existing bundle must match exactly; it cannot be silently overwritten or
changed after seeing results. V1 files and the frozen-confirmation files are
read only. The script uses the existing repository analysis helpers and
Confirmation/contract.py, without altering that frozen experiment.

Commands for the user on the cluster, from repository root
--------------------------------------------------------
Use your existing analysis/API environment. No GPU, images or checkpoints needed.
The prepare stage uses the existing NumPy/SciPy analysis environment; the run
stage requires the OpenAI SDK only when explicitly enabled. No packages or
credentials are downloaded or installed automatically.

1. Free integrity/preflight check:
  python OphthalmicAgent/scripts/run_glaucoma_v2_replay.py --stage preflight

2. Optional small pilot, five frozen cases / at most ten calls:
  python OphthalmicAgent/scripts/run_glaucoma_v2_replay.py \
    --stage run --allow-api --max-cases 5

3. Continue the SAME frozen run through the remaining cases:
  python OphthalmicAgent/scripts/run_glaucoma_v2_replay.py --stage run --allow-api

4. Collect at any time, including after a failure (zero calls):
  python OphthalmicAgent/scripts/run_glaucoma_v2_replay.py --stage collect

Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in the environment yourself.
Deployment is frozen to gpt-5.1, API version 2024-12-01-preview, temperature 0.3.
If the deployment rejects a setting/schema, the first failed attempt stops the
run. Do not try alternate prompts or settings on successive failed test cases.
Pilot review should check execution/parsing, not select the best-performing prompt.

Resumption and safeguards
-------------------------
The default run root is OphthalmicAgent/outputs/glaucoma_v2_replay/.
Keep using that directory. Do not delete/copy its ledger to obtain more attempts;
the cap is per persistent run ledger, not an account-wide spending restriction.

SQLite reservations commit before any request is sent. Each case/arm can be
reserved once. A crash, timeout, refusal or invalid response consumes its slot;
resuming never automatically retries that pair. The first API/parse failure stops
execution for inspection; repeating run skips the failed pair and continues with
unattempted pairs. To inspect after failure, use collect first. A process lock
rejects concurrent workers on this root. Returned-model drift stops the run and
blocks further requests. Endpoint identity is also pinned for this run.

Replaying already valid stages is free. Frozen request/input/code checks prevent
resuming with different prompts or mixing outputs from different revisions.
If a request was reserved just before a crash, it stays incomplete even when it
may never have reached the provider. This intentionally favors underspending.

Outputs and interpretation
--------------------------
ledger.sqlite3: durable exact requests, response bodies, raw text, parsed labels,
API metadata including usage/model/response ID, timestamps and failure status.
api_receipts.jsonl: readable export of that ledger after collect.
case_results.csv: all 80 cases, including eight missing evidence, separate status
and prediction columns for both arms, review flag, historical labels and source
evidence hashes. Missing/invalid predictions remain blank.
summary.json / report.txt: error repairs and correction losses, both valid-arm
counts and matched-pair counts. Partial reports remain explicitly incomplete.

Compare V2 with the paired legacy-prompt replay, not just the old predictions:
fresh sampling and differences in the selected evidence version can change both.
Inspect errors repaired AND previous corrections lost. Raw model reasoning is
an explanation to audit, not ground-truth proof of the mechanism.

No full-250 F1, new significance claim, or replacement manuscript result is
produced. These cases were selected after inspecting errors. A promising V2 needs
an independent frozen evaluation to support a generalization claim.

Still unresolved by this experiment
----------------------------------
Original segmentation masks and source images were not checked or regenerated.
No new OCT/SLO interpretation or scan-location verification is obtained. Missing
historical final responses cannot be reconstructed. The eight missing packets
remain missing. Any image/tool fixes require a separately specified experiment,
not patching this run after its outcomes are inspected.

Offline verification:
  python -m unittest discover -s OphthalmicAgent/tests -p test_glaucoma_v2_replay.py -v

Local verification on 2026-10-06: 16 replay tests plus 18 existing glaucoma
trace/statistics tests passed. The actual 72-case bundle also completed a fully
mocked pilot, continuation and duplicate-free resume (144 synthetic requests).
Mock outputs were kept outside the repository under /private/tmp; no provider
was contacted and no new diagnostic result is claimed.
