RETINAGENT SUPPLEMENTARY PROMPT EXPORT

supplementary_prompts.tex is a self-contained LaTeX document. It has no external
prompt-file dependencies and does not require shell escape. Compile twice with
pdflatex for the contents page; required packages are geometry, fontenc, inputenc,
fvextra, xurl and hyperref. No LaTeX engine was available on the export machine,
so the rendered PDF has not been verified there.

The export includes current source templates for the three FairVision tasks,
external diagnostic variants, GDP detection and progression, independent LLM
baselines, legacy/optional modules, and prospective ablation/escalation changes.
The latter two groups are separate appendices, NOT evidence of historical
execution. Do not describe this as the verified prompt snapshot that generated
the original manuscript results. No prompt content is revised by this export.

Static prompts are shown as decoded text. Dynamic prompts retain their readable
Python expressions and placeholders. Branches are labelled; source comments,
logging strings and archived non-executable drafts are excluded. Construction
functions and response schemas are explicitly distinguished from prompt text.
No patient-specific requests, image payloads, API keys or execution are needed.

prompt_index.csv maps each block to its source location and hashes.
provenance.json records source-file, exporter and final LaTeX hashes.

Regenerate from the repository root:
  python3 equi-agent/scripts/export_supplementary_prompts_latex.py

Offline checks:
  python3 -m unittest discover -s equi-agent/tests -p test_supplementary_prompts_latex.py -v

Git commit/push/pull and any eventual experiment execution remain with the user.
