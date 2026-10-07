RETINAGENT ARCHITECTURE - CONCISE EDITABLE WORKFLOWS

Master: retinagent_architecture.drawio
Page 1: Diagnosis. Preview/print: retinagent_architecture.png / .pdf
Page 2: GDP progression. Preview/print: retinagent_progression.png / .pdf
Captions: captions.tex
Earlier detailed figure: previous_detailed/ (preserved before replacement).

Both pages have five workflow connectors and fewer than 40 label words.
Inputs -> evidence assembly -> counterfactual audit -> orchestrator -> output.
A separate direct arrow supplies the original case evidence to the orchestrator.
The grouping represents evidence assembly, not a claim that all components
receive identical inputs or execute simultaneously. Burgundy circles are LLM
roles; teal glyphs within evidence assembly represent models and lookup/tools.

All labels, glyphs, shapes and connectors are native draw.io objects. Agent
glyphs and titles are grouped; arrows attach to their circles and move with
them. The three clinical thumbnails are embedded raster assets.

SCOPE (kept outside the figure to avoid subtitle clutter)
The diagnostic page summarizes the diagnostic architecture, with RETFound as
the probability-producing anchor. Modalities and reliability availability vary
by cohort: FairVision uses OCT/SLO, external diagnostic cohorts use CFP. The
thumbnails are modality examples, not three images from a single patient.
Demographics select reliability context; the lookup is not a separate Equity
LLM. Counterfactual scenarios are dependence audits, not independent votes.

The GDP page separately depicts baseline imaging, baseline visual-field TD
values, structural/functional reports, helper predictions and reliability.
Baseline imaging is RNFLT plus optional OCT; the diagram does not imply access
to follow-up visits. It ends in six forecasts; the staged output contract also
retains rationale and review flags, described in the caption.

Neither schematic certifies historical run configuration, complete predictions,
or clinical benefit. In particular, historical AMD/DR counterfactual caches
use a glaucoma-labelled prompt; this known implementation limitation remains
documented in the saved evidence-ablation audit. Drawing a schematic does not
resolve it. Inactive guideline, web, safety and equity reviewers are omitted,
as are proposed clinical handoff and online-learning loops. The historical
diagnostic output is not depicted as a calibrated uncertainty percentage.

ASSETS AND CODE
CFP: agentic_walkthrough/assets/fundus.png (existing Drishti example).
OCT/SLO: cells c5/c7 of clinical_decisions/clinical_decisions.drawio.
Original pixels are unchanged; source hashes are in provenance.json.

OphthalmicAgent/evaluate_fairvision_glaucoma_agentic.py
OphthalmicAgent/CounterfactualAgent/counterfactual_agent.py
OphthalmicAgent/Progression/workflow.py
OphthalmicAgent/Progression/prompts.py

REBUILD / CHECK
python3 equi-agent/scripts/build_retinagent_architecture_reference.py
python3 -m unittest discover -s equi-agent/tests -p test_retinagent_architecture_reference.py -v

The builder refuses to overwrite manual changes to a generated master. Use a
different --output-dir to regenerate alongside a manually edited version.
PNG/PDF files are exported locally using the installed draw.io application.
No API calls, inference, SSH, commits, pushes or pulls are involved.
