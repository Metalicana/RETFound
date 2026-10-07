RETINAGENT ARCHITECTURE - EDITABLE DRAW.IO FIGURE

Master: retinagent_architecture.drawio
Preview: retinagent_architecture.png
Print export: retinagent_architecture.pdf

The master opens directly in draw.io / diagrams.net. Text, agents, glyphs,
panels, arrows and the legend are native editable cells. Agent glyphs and their
labels are grouped; workflow connectors attach to groups and follow movement.
The three clinical thumbnails are embedded raster assets, not vector traces.
No external image downloads are needed. Existing figures were not overwritten.

The reference image guided the four-part composition, resource shelf, burgundy
agent circles, teal arrows and stacked clinical images. Content is aligned to
the local implementations: an evidence-ablation counterfactual agent replaces
an unverified active safety/equity review loop. Logging and offline assessment
are shown instead of claiming continuous online model learning.

SUGGESTED CAPTION
RetinAgent's task-dependent ophthalmic reasoning architecture. (A) Available
patient metadata and retinal imaging enter task-specific interpretation paths.
The GDP progression branch additionally uses baseline RNFLT and visual-field
total-deviation values. (B) Model predictions, reliability reference tables,
demographic lookup and cup-to-disc measurements supply complementary inputs
where configured. (C) Patient summaries and specialist reports are assembled
with model and reliability evidence. An evidence-ablation agent reports source
dependence, and the ophthalmologist orchestrator integrates the evidence and
audit trace. (D) The workflow produces a task-specific forced binary assessment
with rationale and preserves available outputs and traces for offline analysis.
The explicit escalation flag is an updated-protocol extension; clinician
handoff is proposed, not a measured reader-study outcome. The separate lower
strip lists optional Equity LLM, Guideline/Web and Safety modules not active in
the inspected task-specific FairVision evaluators. This is an architecture
overview, not proof of the exact execution behind every historical result.

IMAGE SCOPE
The fundus thumbnail is the existing Drishti example in
agentic_walkthrough/assets/fundus.png. The OCT and SLO thumbnails are embedded
cells c5 and c7 from clinical_decisions/clinical_decisions.drawio. Original image
bytes are reused without pixel editing. These thumbnails illustrate available
modalities and do NOT represent one patient with all modalities. Source hashes
are recorded in provenance.json. No synthetic clinical measurements are shown.

IMPLEMENTATION REFERENCES
OphthalmicAgent/evaluate_fairvision_glaucoma_agentic.py
OphthalmicAgent/CounterfactualAgent/counterfactual_agent.py
OphthalmicAgent/Progression/workflow.py
OphthalmicAgent/Confirmation/contract.py

REBUILD (overwrites this generated master, not other figures)
python3 equi-agent/scripts/build_retinagent_architecture_reference.py

CHECKS
python3 -m unittest discover -s equi-agent/tests -p test_retinagent_architecture_reference.py -v

The installed draw.io desktop renderer generates the PNG and PDF from the native
master. Generation and checks perform no model inference or API calls. Git
commit/push/pull remain with the user.
