# Clinical Decision Examples

**Figure caption.** Retrospective FairVision examples illustrate disagreement
between a foundation-model score and the agent's interpretation of additional
evidence. **a**, A glaucoma case receives a 19.4% RETFound score and a negative
decision; the final agent decision is glaucoma. **b**, A non-glaucoma case
receives a 72.7% score and a positive RETFound decision; the final agent decision
is non-glaucoma. *Image descriptions are taken from saved AI reports, not
clinician annotations; vertical cup-to-disc ratios are automated estimates.
**c**, For case b, the saved evidence-ablation response reports an inconclusive
diagnosis when OCT/SLO reports are withheld and non-glaucoma with full evidence.
These are model-reported scenarios within the same call, not independently
rerun interventions or proof of a causal mechanism. The final agent CSV also
records non-glaucoma. The selected cases illustrate why additional evidence
may matter; they do not establish overall superiority, correction frequency
or clinical benefit. Full paired outcomes, including adverse changes, are
retained below and in the source data rather than plotted as motivation.

## Provenance

- Original OCT/SLO images and the author's case-panel edits are retained.
  Images are the stored `oct_bscans[100, :, :]` and `slo_fundus` arrays, without
  enhancement or diagnostic markup.
- Final saved predictions determine outcomes. Last recorded evidence traces
  illustrate reasoning; their exact linkage to the final orchestrator call is
  not recorded. They do not prove why the decision changed.
- Counts are copied from the verified paired comparison, not a new run or a
  new threshold. Both systems use the same 124 glaucoma and 125 non-glaucoma
  cases. The saved baseline threshold is 0.5.
- `data_07199.npz` has nonbinary saved truth and is absent from the baseline
  file. It is excluded from both sides, leaving 249 of 250 agent-file rows.
- There are 23 corrected baseline errors and 18
  introduced errors: 17 corrected missed glaucoma cases
  versus 7 newly missed cases, and
  6 corrected false alarms versus
  11 new false alarms. None are dropped from the paired
  cohort or the results below.
- This run uses RETFound plus imaging reports and structural evidence. It is
  not an experiment demonstrating multi-foundation-model selection.
- The manuscript-reported RETFound worst-group F1 remains **0.6344**. Reported
  F1 and worst-group F1 values are recorded separately in `source_data.json`,
  attributed to the manuscript and not replaced by locally recalculated values.
  Neither F1 nor worst-group F1 is plotted in this motivation figure.

## Paired Outcomes For Results And Limitations

| System | Cases | Missed glaucoma | False alarms | Correct glaucoma | Correct non-glaucoma |
|---|---:|---:|---:|---:|---:|
| RETFound | 249 | 45 | 17 | 79 | 108 |
| RetinAgent | 249 | 35 | 22 | 89 | 103 |

The system has 10 fewer missed glaucoma cases and 5 more false alarms in this
paired cohort. This trade-off must remain in the experimental reporting; it
is not removed by choosing a case-based motivation figure.

## Files

`clinical_decisions.drawio` is the editable master. PDF, SVG and PNG are exports
of that same file. Only one current motivation figure is retained.

Manual draw.io edits are authoritative. To add the three panel borders without
rebuilding any edited labels:

```bash
python equi-agent/scripts/build_clinical_decisions_figure.py --borders-only
```

Without `--borders-only`, the script rebuilds the evidence panel and can restore
labels previously deleted from that strip. Export the edited master through
draw.io after updating. The script does not run models or change predictions.
