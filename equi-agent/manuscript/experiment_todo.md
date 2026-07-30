# RetinAgent Manuscript and Experiment TODO

The immediate goal is a structurally complete draft. Empty cells remain `--`
until a held-out test result is verified. Metrics collection must not modify
agent prompts or use test labels for threshold or policy selection. AMD and DR
are owned by a collaborator and are outside this execution queue.

## 0. Complete the Draft Structure

- [x] Insert `disease_tables_draft.tex` after the glaucoma tables.
- [x] Add the clinician-workflow framing paragraph before the disease results.
- [x] Standardize the system name as `RetinAgent` in the Chapter 4 prose.
- [x] Standardize missing values as `--` and use `N/A` only when a metric is
      mathematically or scientifically inapplicable.
- [x] Correct table column declarations in the Drishti-GS and REFUGE2 tables.
- [x] Replace the PAPILA placeholders with the completed 81-case results.
- [x] Replace the GAMMA placeholders and describe modalities accurately:
      foundation baselines use CFP except RETFound, while Ours combines the OCT
      prediction with CFP-derived evidence.

## 1. Lock the Metric Contract

- [ ] Confirm binary positive-class definitions for AMD, DR, glaucoma, and
      progression.
- [ ] Confirm validation-selected thresholds and locked held-out test cohorts.
- [ ] Define worst-group F1 attributes, intersection rules, and minimum support.
- [ ] Emit TP, FP, TN, FN and subgroup support alongside every reported metric.
- [ ] Keep forced prediction metrics separate from accepted-case selective
      metrics and report coverage whenever selective metrics are shown.

## 2. Finalize GDP Progression

- [ ] Verify the operating endpoint is
      `td_pointwise_no_p_cut` in every source file and caption.
- [x] Directly report the documented native-helper and RetinAgent test metrics
      in Chapter 4.
- [ ] Restore the exact 60-positive native-helper and live RetinAgent prediction
      artifacts; the similarly named local aggregate currently points to the
      older 18-positive endpoint.
- [x] Remove unsupported VisionFM, URFound, and LLM placeholder rows unless they
      are rerun on the same 60-positive endpoint.
- [ ] Export the final progression comparison and its provenance manifest.

## 3. Complete the FairVision Glaucoma Table

- [x] Rebuild every row from the existing 3,000-case prediction/metric artifacts;
      no foundation-model inference rerun is needed.
- [x] Resolve the mismatch between the manually entered RETFound/MIRAGE values
      and the current validation-threshold metric artifacts.
- [x] Define whether worst-group F1 includes supported intersections or only
      single demographic attributes.
- [x] Report RetinAgent forced performance separately from accepted-case
      performance and include coverage for the accepted row.
- [ ] Generate the LaTeX rows from the audited CSV rather than editing manually.

## 4. Complete the Harvard GDP Glaucoma Table

- [x] Populate RETFound, VisionFM, and URFound from the existing locked 400-case
      test artifacts.
- [ ] Locate or rerun the LLM baseline and RetinAgent glaucoma-detection outputs;
      their current manuscript values have no local source artifact.
- [ ] Use one 400-case protocol for every row and remove the older conflicting
      RETFound result.

## 5. Restore PAPILA and GAMMA Results

- [ ] Restore the completed benchmark summaries and agent prediction files from
      the cluster.
- [x] Populate PAPILA from the locked 81-case protocol.
- [x] Populate GAMMA from the locked 20-case protocol.
- [x] Correct modalities: GAMMA RETFound is OCT, the other foundation models are
      CFP, and RetinAgent combines the OCT prediction with CFP-derived evidence.

## 6. Run Genuinely Missing External Glaucoma Experiments

- [x] Build and validate a locked Drishti-GS manifest and split/evaluation
      protocol.
- [ ] Run the five CFP foundation models and the unchanged agent pipeline on
      Drishti-GS.
- [ ] Do not present the current Kaggle REFUGE2 mirror as official test
      classification: only train labels are available.
- [ ] Either obtain legitimate REFUGE2 classification labels or replace that
      table with the completed repeated-holdout structural CDR validation.

## 7. Add Statistical Support

- [ ] Add patient-level bootstrap 95% confidence intervals.
- [ ] Add paired bootstrap or McNemar comparisons for key model differences.
- [ ] Report subgroup coverage and escalation rates, not only accepted-case
      worst-group F1.
- [ ] Add repeated-run stability and invalid-output rate for LLM rows.

## Execution Order

1. Finish the draft structure.
2. Restore the exact progression artifacts.
3. Regenerate FairVision glaucoma from existing outputs.
4. Regenerate GDP glaucoma from existing outputs.
5. Restore and report PAPILA/GAMMA.
6. Run Drishti-GS.
7. Resolve REFUGE2 labels or report structural validation instead.
8. Add uncertainty intervals.
