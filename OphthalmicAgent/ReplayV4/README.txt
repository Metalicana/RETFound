GLAUCOMA V4: SOURCE-QUALIFICATION PILOT
=====================================

Purpose
-------
Test evidence handling, not tune diagnoses to the known errors. This separate
experiment preserves all V1/V2/V3 prompts, receipts and predictions. It is a
two-case downstream development pilot, NOT a new full-pipeline evaluation.
The cohort is the two completed cases from the existing V3 pilot, in its frozen
order. Their reference labels and prior decisions never enter new API requests.
These are inspected historical errors, not held-out cases or a causal ablation.

Concrete evidence for the change
-------------------------------
In data_07137 the V3 OCT source leaves disease presence OR degree uncertain.
The counterfactual/final responses recast the qualification mainly as uncertainty
about severity; the final says disease presence is not the main uncertainty.
In data_07014 the original SLO source explicitly says it cannot confidently be
called reassuring, but downstream reasoning calls it broadly reassuring. An
unverified CDR is acknowledged and then used for strong negative reassurance.
These are reasoning/attribution defects, not proof of the images' true diagnosis.

Preparation exports all 57 historical error packets, including eight missing
saved-evidence cases, and qualification_candidates.csv for OCT/SLO versus V2
reasoning. Broad lexical matches nominate passages for review; they do NOT
establish a semantic-error rate or adjudicate all 49 available reports.
The two V3 packets are included separately. No historical predictions change.

CDR source inspection
---------------------
vision_slo_glaucoma.py:89-125 computes vertical bounding-box cup height divided
by disc height, with disc=(label 1 OR label 2), cup=(label 2), rounded to three
decimals. Missing cup/disc returns -1. It does not validate connected components,
segmentation quality, image gradability or clinical measurement accuracy.
Mask rendering is commented out in the inspected call path. Console cutoffs at
0.4/0.7 in analyze() are printed descriptions, not a quality-validation step.

Synthetic tests execute only this pure method: an isolated cup-labelled pixel
can change the bounding-box ratio, and an all-cup region returns 1.0. This does
NOT prove either happened in a patient. The downloaded 57-NPZ image archive
contains no matched segmentation masks. No replacement CDR, class-map reversal,
new clinical cutoff or mask validity is inferred. No segmentation inference runs.

V4 design
---------
1. Review: use the existing V3 OCT report and unchanged SLO report. No RETFound
   score, demographics, reference label, historical diagnosis, V2/V3 decision or
   counterfactual diagnosis is given to this stage. Cite original full-line IDs
   for findings and limitations, keeping uncertainty in the interpretation.
2. Final: integrate the full original reports, that new review, unchanged saved
   RETFound probability, CDR, narrative and reliability. Preserve all retained
   limitation IDs and the exact review interpretations. Give a forced binary
   diagnosis and separate escalation flag.

The previous counterfactual diagnosis generator is REPLACED by the qualification
review, not simply rerun or fed forward. This is an explicit architecture/output
contract intervention, not the unchanged RetinAgent pipeline. No new OCT, SLO,
Bio-Profiler, classifier, CDR, counterfactual-ablation or training calls occur.
The saved V3 comparison is noncontemporaneous; it cannot isolate the causal
effect of each simultaneous change, and stochastic changes remain possible.

Original reports remain complete. A broad fixed lexical rule protects lines
containing limitations; the model may add further limitation IDs. Validators
reject unknown/cross-source citations, dropped protected limitations, rewritten
source-review interpretations, invalid labels, and a CDR represented in the
structured field as verified. Both positive
and negative diagnoses remain allowed. No outcome-directed thresholds exist.

These are mechanical checks, NOT a semantic entailment checker. A response can
cite a limitation and still misinterpret it in prose. Inspect the source-linked
response packets before any larger run; do not call schema compliance clinical
correctness or claim the problem has been solved. The qualifier matcher can
overselect or miss lines; it neither grades image quality nor decides diagnosis.

Budget and preservation
-----------------------
Hard cap: FOUR request attempts, including failed/interrupted calls. Two cases,
two stages each. No automatic retries, no repair calls, no 49-case continuation.
Completion caps are 2000 tokens per request; maximum requested completion-token
allowance is 8000, plus input tokens. This is not a dollar-price guarantee.
Deployment, API version and final temperature stay inherited from V3; review
temperature is 0. Endpoint and returned model are pinned to the V3 receipts.

Known source receipts: 100 V2 + 7 V3 = 107; complete V4 would total 111.
This is NOT a global account counter. Other jobs/manual requests count too.
Do NOT resume the old full V3 plan after V4: 100+148+4 would exceed 250.
The V4 runner verifies source snapshots before each new request and takes the
existing run locks during execution, blocking concurrent V2/V3 continuations.
It cannot stop a user launching an older runner later. Do not delete ledgers,
copy V4 to a new run location to retry, or reprepare after a failed request.

Prepare on the cluster, where paths resolve to the original saved runs. Do not
copy the locally prepared Mac bundle to the cluster: it records local source
paths. Preparation is offline and refuses to overwrite an existing V4 directory.
Preflight/inspect are read-only. Collect exports V4 results only. The runner
compares a live V2 ledger to its export when that ledger exists, so additional
attempts are not silently omitted. Changes to upstream artifacts stop V4.

Commands (USER runs from ~/RETFound after transferring the new source files)
-------------------------------------------------------------------------
  python OphthalmicAgent/scripts/run_glaucoma_v4_replay.py --stage prepare
  python OphthalmicAgent/scripts/run_glaucoma_v4_replay.py --stage preflight
  python OphthalmicAgent/scripts/run_glaucoma_v4_replay.py --stage run --allow-api --max-cases 2

Successful stages are reused; a repeated run cannot repeat successful requests.
Any failed stage blocks continuation, preserving the original response/usage.
Inspect without spending:
  python OphthalmicAgent/scripts/run_glaucoma_v4_replay.py --stage inspect
  python OphthalmicAgent/scripts/run_glaucoma_v4_replay.py --stage collect

Default output: OphthalmicAgent/outputs/glaucoma_v4_replay/
  bundle.json, offline_audit/qualification_candidates.csv, offline_audit/cases/
  run/ledger.sqlite3, run/api_receipts.jsonl, run/case_results.csv, run/summary.json
  run/data_07014_source_review.json, run/data_07137_source_review.json

The source-review packets include original lines and all valid structured outputs.
Keep unresolved/invalid cases explicit. No negative fallback, imputed outcome,
full-250 F1, significance claim, or assessment of correct-case regressions.

Offline tests:
  python -m unittest discover -s OphthalmicAgent/tests -p 'test_glaucoma_v4_replay.py' -v

API schema/refusal handling follows the existing local contract pattern and
official Structured Outputs documentation (read 2026-10-07):
https://developers.openai.com/api/docs/guides/structured-outputs
The documentation explicitly notes that structured responses can still contain
mistakes; the schema is not a guarantee of faithful or medically correct reasoning.
