V5: native-resolution SLO development pilot
==========================================

Scope: the two already-completed V3/V4 pilot cases, data_07014 and data_07137.
This is an error-selected, post-inspection development experiment. It changes
both SLO input presentation and the SLO reporting contract. It cannot isolate
resolution from prompt effects and is not a held-out evaluation.

Changes
-------
Send the inspected original 512x664 SLO JPEG bytes, with high image detail.
No resize, aspect-ratio change, crop, CLAHE or JPEG re-encoding is performed.
The two image hashes are pinned, so a renamed/wrong-case image is rejected.
Request one compact structured report distinguishing observed, tentative,
not observed with adequate visibility, and not assessable for each feature.
Remove the unsupported assumption that the OCT classifier was uncertain.
The SLO request has no reference labels, prior predictions, reports, CDR,
classifier probabilities, demographics or reliability estimates.

Reuse V4's exact review/final system prompts, schemas, settings and parser.
Only the new SLO report and its provenance change in those downstream inputs.
V3 OCT, narrative, RETFound probability, reliability and CDR remain cached.
The CDR is NOT recomputed from the native image and remains unverified. Its
200x200 NPZ input differs from the native SLO view of the same acquisition;
the updated provenance explicitly states that distinction.
No old counterfactual votes or V4 decisions are passed into the new requests.
The unchanged final system still groups SLO/CDR by shared acquisition; the
new provenance does not claim identical pixels or independent confirmation.

Budget and validation
---------------------
SLO -> qualification review -> final: three requests per case; six total.
SLO temperature 0.2, review 0, final inherits V4 (0.3). Each cap is 2000
completion tokens; no SDK or application retries. Returned model and endpoint
must match the previous receipts. Every attempt is reserved durably before
sending. Truncated, refused, invalid, failed or interrupted attempts stop the
run; they are not replaced, silently retried, or assigned a negative label.
Schema/citation checks do not establish clinical truth or semantic fidelity.

Known source receipts: V2=100, V3=7, V4=4. Thus 111 prior + 6 V5 = 117 maximum.
The runner checks live upstream ledgers/exports where available and pins their
snapshots. This is not a global account counter for unrelated jobs. Do not
resume V2/V3/V4 alongside V5 or prepare new directories to bypass the cap.
In particular, a full V3 continuation plus V4 would exceed the earlier 250
ceiling even before adding V5.

Cluster commands, after transferring the new source files yourself
----------------------------------------------------------------
  cd ~/RETFound
  python OphthalmicAgent/scripts/run_glaucoma_v5_replay.py --stage prepare
  python OphthalmicAgent/scripts/run_glaucoma_v5_replay.py --stage preflight
  python OphthalmicAgent/scripts/run_glaucoma_v5_replay.py --stage run --allow-api --max-cases 2

Prepare is offline and refuses an existing experiment directory. On resume,
skip prepare. --max-cases selects a frozen prefix, not additional cases.
The default original-image directory is Datasets/FairVision/Glaucoma/Test.
The default sources are the existing V2, V3 and V4 experiment directories.
Local audits can supply --v4-dir, --v3-dir, --v2-root, --v2-bundle, --image-dir
at prepare time. These paths are then frozen, not silently overridden on run.

  python OphthalmicAgent/scripts/run_glaucoma_v5_replay.py --stage inspect
  python OphthalmicAgent/scripts/run_glaucoma_v5_replay.py --stage collect

Inspect is read-only. Collect exports receipts, per-case evidence packets,
case_results.csv and summary.json under outputs/glaucoma_v5_replay/run/.
All 57 historical error rows remain, with the 55 untested cases unassessed.
Report repair counts only. Saved V4 is noncontemporaneous; no full-cohort F1,
significance, superiority or correct-case regression claim is supported.
No original manuscript, prediction, prompt or V1-V4 experiment is overwritten.

Offline tests (fake API client, no inference)
-------------------------------------------
  python -m unittest discover -s OphthalmicAgent/tests -p 'test_glaucoma_v5_replay.py' -v
