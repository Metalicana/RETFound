"""Export exact original and adapted system prompts without importing model code."""
from __future__ import annotations

import ast
import hashlib
from html import escape
from pathlib import Path

from .prompts import DEFAULT_RUN_NAME, SOURCE_PROMPTS, SYSTEM_PROMPTS, VERSION

ROOT = Path(__file__).resolve().parents[1]
NOTES = {
    "bio_profiler": "Same medical-scribe role. Demographics are explicitly descriptive only; no outcomes, MD field or arbitrary manifest columns enter the request.",
    "rnflt_specialist": "Based on OphthalmicAgent's GDP RNFLT specialist, replacing the unavailable SLO/CFP view. Receives an actual RNFLT image and descriptive statistics, never helper scores. Colors are case-scaled, not normative.",
    "oct_specialist": "Optional --include-oct branch. Receives actual baseline B-scans, not longitudinal visits. This adds an input beyond the RNFLT/TDS native helper and must be labelled as an expanded-input comparison.",
    "functional_specialist": "Adapts the existing FunctionalSpecialist component. It is not active in the current FairVision main_new.py path; GDP has baseline TDS evidence, so this adaptation explicitly activates it. The mean of TD values is not relabelled as MD.",
    "counterfactual": "Same evidence-ablation role, five scenarios in one separate call. CDR ablation becomes functional-evidence ablation; probabilities become target-specific native-helper predictions. Scenarios are not votes or independent experiments.",
    "orchestrator": "Based on the active Orchestrator/new.py glaucoma prompt. Six progression labels replace one diagnostic label. Uses original full evidence plus the audit, not a numerical clamp. Final labels are stored directly, including review flags.",
}


def literal(node, constants):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(literal(part, constants) for part in node.values)
    if isinstance(node, ast.FormattedValue):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "join" and len(call.args) == 1
                and isinstance(call.args[0], ast.Name)):
            return literal(call.func.value, constants).join(constants[call.args[0].id])
    raise ValueError("Unsupported dynamic prompt expression; inspect the source instead of approximating it")


def original_prompt(role):
    relative, class_name, method_name = SOURCE_PROMPTS[role]
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    matches = []
    for node in ast.walk(method):
        if isinstance(node, ast.Dict):
            entries = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
            if isinstance(entries.get("role"), ast.Constant) and entries["role"].value == "system":
                matches.append((literal(entries["content"], constants).strip(), node.lineno))
    if len(matches) != 1:
        raise ValueError(f"Expected one active system prompt for {role}; got {len(matches)}")
    return matches[0][0], f"{relative}:{matches[0][1]}", hashlib.sha256(path.read_bytes()).hexdigest()


def export(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for role, adapted in SYSTEM_PROMPTS.items():
        original, source, sha = original_prompt(role)
        sections.append(f'''<section id="{role}"><h2>{escape(role.replace('_', ' ').title())}</h2>
<p>{escape(NOTES[role])}</p><div class="pair">
<article><h3>Existing OphthalmicAgent</h3><small>{escape(source)}</small><pre>{escape(original)}</pre></article>
<article><h3>New Progression Adaptation</h3><small>Progression/prompts.py: SYSTEM_PROMPTS["{role}"]</small><pre>{escape(adapted)}</pre></article>
</div><details><summary>Original source SHA-256</summary><code>{sha}</code></details></section>''')
    commands = '''cd ~/RETFound
conda activate retfound
RUN="$HOME/RETFound/OphthalmicAgent/outputs/__RUN_NAME__"
SCRIPT=OphthalmicAgent/scripts/run_gdp_progression_ophthalmic_agent.py
mkdir -p "$RUN"

# No API calls: checks all six native runs and all 200 baseline inputs.
python "$SCRIPT" --stage prepare --out-dir "$RUN"

# Five real staged calls for one patient; cached for the subsequent full run.
python "$SCRIPT" --stage smoke --out-dir "$RUN"

# Only after successful preflight and smoke. No retraining or GPU is needed.
nohup env PYTHONUNBUFFERED=1 python "$SCRIPT" --stage run --out-dir "$RUN" \\
  > "$RUN/run.log" 2>&1 < /dev/null &
echo $! > "$RUN/run.pid"
tail -n 30 "$RUN/run.log"
'''.replace("__RUN_NAME__", DEFAULT_RUN_NAME)
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OphthalmicAgent: GDP progression prompt review</title>
<style>
body{{font:16px/1.5 Arial,sans-serif;color:#20252b;background:white;max-width:1440px;margin:24px auto;padding:0 24px;letter-spacing:0}}
h1{{font-size:28px}}h2{{font-size:22px}}h3{{font-size:17px;margin:0 0 8px}}small{{display:block;overflow-wrap:anywhere;color:#51565a}}
section{{border-top:2px solid #33474b;margin-top:32px;padding-top:12px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
article{{border:1px solid #a3afb4;padding:16px;min-width:0}}article:nth-child(2){{border-top:4px solid #087f73}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 ui-monospace,monospace}}code{{overflow-wrap:anywhere}}
nav a{{display:inline-block;margin-right:18px;color:#07695f}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbc3c7;text-align:left;padding:10px}}
@media(max-width:800px){{.pair{{grid-template-columns:1fr}}body{{padding:0 12px}}}}
@media print{{body{{font-size:10pt;padding:0}}pre{{font-size:9pt}}.pair{{grid-template-columns:1fr 1fr}}}}
</style></head><body><h1>OphthalmicAgent: GDP Progression Prompt Review</h1>
<p>Version: <code>{VERSION}</code>. Left: exact active system prompts extracted with Python AST, not imported or rewritten. Right: exact new system prompts sent by the staged progression runner. Existing diagnostic prompts are unchanged.</p>
<p>The v2 progression prompts distinguish forecasting a future outcome from confirming observed change. Follow-up examinations are not required inputs; label 0 means a negative forecast, not an unconfirmed outcome. Review flags remain separate from binary forecasts, and helper dependence is not treated as automatic evidence of error.</p>
<p>This revision followed inspection of v1 test-case reasoning. Preserve v1 outputs and disclose the prompt revision when reporting a rerun on the same test cohort; that cohort is no longer an untouched evaluation for prompt development. Use the new <code>{DEFAULT_RUN_NAME}</code> directory. Old prompt caches cannot be migrated with <code>--upgrade-output-contract</code>. Performance improvement has not been established.</p>
<nav>{''.join(f'<a href="#{r}">{escape(r.replace("_", " ").title())}</a>' for r in SYSTEM_PROMPTS)}</nav>
<h2>Execution Contract</h2>
<p>Separate calls: Bio-Profiler &rarr; RNFLT specialist &rarr; baseline visual-field specialist &rarr; evidence-counterfactual audit &rarr; final orchestrator. Optional OCT adds a separate image specialist. All six endpoints share the reports; the audit and final stage return all six endpoints together. Default: 5 calls/patient, 1,000 calls for 200 cases before retries. With OCT: 6/patient, 1,200 total. Successful stages are cached.</p>
<table><tr><th>Preserved</th><th>Necessary, disclosed adaptations</th></tr>
<tr><td>Medical scribe, independent specialist observations, reliability context, evidence ablation, final full-evidence integration.</td><td>Future-progression forecasting replaces current-glaucoma diagnosis. RNFLT and baseline TDS replace unavailable fundus/CDR inputs. Optional OCT is explicitly additional input.</td></tr>
<tr><td>Original reliability-tool risk_score and weights, support-weighted local risks, intersection shrinkage k=50.</td><td>Computed from 300 development-OOF cases per endpoint, not test metrics or FairVision priors. Sparse subgroups (&lt;20 positives or &lt;20 negatives) use global risk. Reliability age bins match main_new.py: &lt;40, 40-59, &ge;60. No claim that a trust score means freedom from bias.</td></tr>
<tr><td>Final orchestrator's actual binary label is retained.</td><td>No transferred glaucoma 10%/90% bypass, no &plusmn;0.15 clamp, no artificial thresholds, no helper-label lock. Review flags never exclude a case.</td></tr>
<tr><td>Baseline-only evidence and explicit uncertainty.</td><td>No future examinations, slopes, p-values, target labels, or held-out performance enter prompts. The single-time-point evidence supports risk forecasting, not a claim of observed worsening.</td></tr></table>
<p>There is no new standalone Equity LLM: reliability is a deterministic tool, as in the active glaucoma path. There is no CDR tool without a fundus image. PubMed and a separate Safety Agent are not invoked by the inspected active glaucoma path, so neither is invented here. Review/uncertainty remains in the final response.</p>
{''.join(sections)}
<section><h2>Run on CECSL After Your Push and Pull</h2><pre>{escape(commands)}</pre>
<p>The smoke run checks execution only, not whether the first test patient was predicted correctly. Do not adjust prompts based on held-out labels. Completion writes six paired tables under <code>$RUN/results/</code> with positive-class F1, macro-F1, sensitivity, specificity and balanced accuracy. Worst-group F1 is explicitly labelled. Agent AUROC is not fabricated from binary labels. <code>paired_changes.csv</code> records actual label changes, corrections and regressions against the native helper.</p>
<p>For OCT, choose a different output directory and pass <code>--include-oct</code> at every stage. This must be labelled as an expanded-input comparison. This implementation is a new staged adaptation, not a retroactive repair of historical results. Better performance is not guaranteed.</p></section></body></html>'''
    destination.write_text(document, encoding="utf-8")
    return destination
