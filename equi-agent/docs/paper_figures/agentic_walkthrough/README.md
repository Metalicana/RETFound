# RetinAgent: Supporting a Glaucoma Assessment

**Figure caption.** A clinician-facing walkthrough using recorded outputs for
Drishti-GS case drishtiGS_053. **1**, A clinician asks whether a fundus photograph
suggests glaucoma. No history or visual-field evidence was supplied to this
image-only run. **2**, the RETFound raw score (24.040%) is below its
validation-selected threshold (53.015%), giving a non-glaucoma model vote. The
AI-generated image report instead describes vertical cupping and suspected rim
thinning. The segmented vertical cup-to-disc ratio is 0.690. The white square
identifies the enlarged region; cyan and magenta contours are predicted disc
and cup boundaries, not expert annotations. **3**, the saved evidence-removal
reasoning reports a different label when the image report is excluded. The
agent's final recorded label is glaucoma, with its reasoning emphasizing the
structural report. A short brief presents the AI assessment, supporting
findings and image-only limitation for clinician review.

The request and formatted clinical brief are illustrative paraphrases, not an
actual doctor conversation or tested interface. This figure has not been
validated in a clinician comprehension or reader study. It does not establish
clinical benefit or show that the agent's self-reported reasoning is a causal
explanation. The data and the full four-scenario trace are retained below.

## Plain-Language Roles

| Figure action | Implementation role |
|---|---|
| Read the image | Vision Agent and its image/model/segmentation tools |
| Check the reasoning | Counterfactual Agent: hypothetical evidence removal within a saved response |
| Explain the assessment | Orchestrator: reconcile the supplied evidence |
| Summarize patient history | BioProfiler |
| Compare other AI image models | Compatible foundation-model adapters / cached model outputs |
| Check reliability by patient group | Equity Agent and validation-derived reliability priors |
| Search diagnostic literature (PubMed) | Guidelines Agent; PubMed and web retrieval |
| Interpret visual-field tests | Functional Interpretation Agent |
| Flag missing or conflicting data | Safety Agent |

The bottom strip is deliberately separate from the worked case. Those
capabilities were not invoked in this saved run and are not shown as contributing
to its result. Their availability and exact behavior depend on the runner.

## What Was Actually Recorded

- The saved external-case run used **RETFound, an AI-generated fundus report and
  vertical CDR**. It did not use the other four foundation models listed below,
  BioProfiler, Equity, Guidelines, Functional or Safety agents.
- The five raw foundation-model scores retained in the source data and table
  below are separate benchmark predictions on the same image. Only RETFound
  appears in the case's main path; the others were removed from that path to
  avoid implying they contributed to the saved assessment.
  The RETFound raw score agrees exactly with the saved agent record. The agent
  consumed the threshold-aligned RETFound score (21.905%), not the raw probability
  (24.040%). Raw model scores are neither reliability scores nor directly
  interchangeable model thresholds.
- The four evidence-ablation labels are scenarios in one saved LLM response,
  not independently rerun interventions or causal evidence. Label 0 is shown
  as non-glaucoma, although its saved reasoning describes an equivocal case.
- The saved trace incorrectly calls the visual report "human" in places.
  It was produced by an LLM; this illustration explicitly labels it AI.
- The new SegFormer export uses the existing external-CDR preprocessing and
  reproduces the saved vertical CDR to its six-decimal precision. The raw mask,
  image hashes and pixel-derived measurements are checked before rendering.
  Horizontal CDR (0.626609) and area CDR (0.422035) are retained in the source
  data; only vertical CDR (0.690265) was passed to this saved agent and is shown
  on the figure. This export is not asserted to be the
  original historical mask, which was not retained.
- All CFP-derived representations remain correlated. No calibrated confidence,
  subgroup reliability score, retrieved paper, patient demographics, safety
  outcome or clinician finding is fabricated for this example.
- This is a framework overview across runners, not a claim that a single
  evaluated pipeline invokes every module. No high-confidence bypass is
  illustrated: this is the full reasoning path, and the selected case did not
  use a bypass. Existing experiment metrics and prompts are unchanged.

### Saved Evidence Scenarios

| Evidence scenario | Saved label |
|---|---|
| All evidence | Glaucoma |
| Without RETFound probability | Glaucoma |
| Without CDR | Glaucoma |
| Without visual interpretation | Non-glaucoma |

## Source Code Map

| Component | Implementation / scope |
|---|---|
| BioProfiler | `equi-agent/BioProfilerAgent/bio_profiler.py`: metadata to narrative |
| Vision | `equi-agent/VisionAgent/vision.py`: visual reports and supported model calls |
| Expanded FM bank | `equi-agent/scripts/run_equi_agent_fairvision_live.py`: precomputed model scores / validation priors; not all model calls occur online |
| CFP CDR | `OphthalmicAgent/scripts/precompute_external_cdr.py`; one-case export replicates preprocessing |
| Equity / reliability | Runner-specific; `equi-agent/main.py`, `OphthalmicAgent/main_new.py` and FairVision live runner are not identical formulas |
| Guidelines | `equi-agent/GuidelinesAgent/guidelines_agent.py` and `orchestrator.py`: PubMed/web diagnostic literature, not patient-case matching |
| Functional | `equi-agent/FunctionalInterpretationAgent/function_interpreter.py`: visual field / TD / MD when available |
| Recorded ablations | `OphthalmicAgent/CounterfactualAgent/external_glaucoma.py`: evidence removal, not demographic counterfactual patients |
| Recorded orchestration | `OphthalmicAgent/Orchestrator/external_glaucoma.py` |
| Recorded case runner | `OphthalmicAgent/scripts/run_external_glaucoma_agent.py` |
| Safety | `equi-agent/SafetyAgent/safety_agent.py`; invoked by `equi-agent/main.py`, not the external-case runner |

## Files

- `agentic_walkthrough.pdf`: vector text/shapes with genuine raster fundus images.
- `agentic_walkthrough.drawio`: native editable agents, arrows, labels and scores.
- `agentic_walkthrough.png` / `.svg`: preview and alternative vector export.
- `case_data.json`: recorded outputs, trace, segmentation metadata and source hashes.
- `assets/`: original fundus, prediction mask and derived overlay; no fake anatomy.

The PDF is 183 mm wide. Text and vector elements are editable in draw.io.
Real images are embedded. Do not overwrite manual draw.io edits with the builder.

Rebuild from the committed assets and source snapshot:

```bash
python equi-agent/scripts/build_agentic_walkthrough_figure.py
```

## Standalone Model Outputs

These are separate benchmark outputs, not a multi-model agent call.

| Model | Raw score | Validation threshold | Model vote |
|---|---:|---:|---|
| RETFound | 0.240405 | 0.530151 | Non-glaucoma |
| MIRAGE | 0.431597 | 0.354362 | Glaucoma |
| RET-CLIP | 0.927545 | 0.731676 | Glaucoma |
| RetiZero | 0.432606 | 0.296001 | Glaucoma |
| URFound | 0.678208 | 0.313967 | Glaucoma |
