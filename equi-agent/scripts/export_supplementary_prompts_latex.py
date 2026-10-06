"""Export local prompt templates without importing or executing model code."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import textwrap
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "equi-agent/manuscript/supplementary_prompts_20261006"
AGENT = "OphthalmicAgent/"
SCRIPTS = "equi-agent/scripts/"

SECTIONS = {
    "fairvision": (
        "FairVision Diagnostic Agents",
        "Current task-specific diagnostic templates. Their presence in this checkout does not "
        "establish the exact templates used for the historical manuscript predictions. The shared "
        "counterfactual template is glaucoma-specific; task adaptations are listed separately in "
        "the prospective-protocol appendix.",
    ),
    "external": (
        "External Glaucoma Diagnostic Variants",
        "CFP and modality-aware external variants. These are source variants, not proof of which "
        "variant generated a particular historical external-cohort result.",
    ),
    "gdp": (
        "GDP Detection and Specialist Templates",
        "Detection and baseline-measurement interpretation templates. These must not be confused "
        "with the staged progression-forecasting prompts in the next section.",
    ),
    "progression": (
        "Staged GDP Progression",
        "Current staged progression system templates, endpoint definitions, evidence-packet "
        "construction and response schemas. The SOURCE_PROMPTS mapping records source lineage; "
        "it does not mean the legacy prompts are additional messages in this workflow.",
    ),
    "baselines": (
        "Independent Language-Model Baselines",
        "Independent diagnostic and progression baseline templates, including provider-specific "
        "routes. Shared text is retained per implementation to expose differences. File names "
        "identify source implementations, not verified historical model snapshots. The GAMMA and "
        "PAPILA input descriptions below use the shared glaucoma baseline routes, with no additional "
        "cohort-specific system prompt.",
    ),
    "legacy": (
        "Legacy and Optional Agent Templates",
        "General-purpose and optional source variants. The standalone Equity, Guideline and Safety "
        "modules are not invoked by the inspected task-specific FairVision evaluators. Listing "
        "them here is not evidence that they contributed to the reported results.",
    ),
    "prospective": (
        "Prospective Ablation and Escalation Extensions",
        "Separate, not evidence for historical manuscript results. These newer source extensions "
        "have not been established as executed study runs. The ablation adapter first modifies "
        "the task-specific prompts and response schema. In the frozen confirmation route, endpoint "
        "instructions are then appended except for the Bio-Profiler; final-message replacements "
        "and the explicit escalation schema are subsequently applied to the orchestrator. "
        "This section documents templates only and does not authorize or launch an experiment.",
    ),
}


def source_specs():
    specs = []

    def add(section, path, title, functions=(), dependencies=()):
        specs.append(dict(section=section, path=path, title=title,
                          functions=functions, dependencies=dependencies))

    for task, label in (("glaucoma", "Glaucoma"), ("amd", "AMD"), ("dr", "DR")):
        for folder, stem, role in (
            ("BioProfilerAgent", "bio_profiler", "Bio-Profiler"),
            ("VisionAgent", "vision_oct", "OCT Specialist"),
            ("VisionAgent", "vision_slo", "SLO Specialist"),
            ("Orchestrator", "fairvision", "Final Orchestrator"),
        ):
            add("fairvision", AGENT + f"{folder}/{stem}_{task}.py", f"{label}: {role}")
    add("fairvision", AGENT + "CounterfactualAgent/counterfactual_agent.py",
        "Shared Glaucoma Evidence-Ablation Agent", dependencies=("SCENARIOS", "evidence"))

    for path, title, deps in (
        ("VisionAgent/vision_cfp.py", "CFP Specialist", ()),
        ("Orchestrator/drishti.py", "Drishti Orchestrator", ()),
        ("Orchestrator/external_glaucoma.py", "Modality-Aware External Orchestrator", ("relationship",)),
        ("CounterfactualAgent/counterfactual_cfp.py", "CFP Evidence-Ablation Agent",
         ("SCENARIO_NAMES", "CounterfactualCFPAgent._messages:scenarios", "evidence")),
        ("CounterfactualAgent/external_glaucoma.py", "Modality-Aware External Evidence Audit",
         ("scenarios", "relationship", "evidence")),
    ):
        add("external", AGENT + path, title, dependencies=deps)
    for path, title in (
        ("VisionAgent/vision_rnflt.py", "RNFLT Specialist"),
        ("FunctionalInterpretationAgent/function_interpreter.py", "Visual-Field Specialist"),
        ("Orchestrator/gdp.py", "GDP Detection Orchestrator"),
    ):
        add("gdp", AGENT + path, title)
    add("progression", AGENT + "Progression/prompts.py", "Progression System Prompts",
        dependencies=("VERSION", "ENDPOINTS", "SCENARIOS", "SOURCE_PROMPTS"))
    add("progression", AGENT + "Progression/workflow.py", "Progression Input and Output Contracts",
        functions=("response_format", "messages", "ablation_packets", "run_case"))

    add("baselines", AGENT + "llm_baseline_utils.py", "Shared Glaucoma Baseline Routes")
    for task in ("glaucoma", "amd", "dr"):
        add("baselines", AGENT + f"evaluate_fairvision_{task}_baseline.py",
            f"FairVision {task.upper()} Baseline", functions=() if task == "glaucoma" else ("user_text",))
    for cohort in ("drishti", "refuge", "gdp"):
        add("baselines", AGENT + f"evaluate_{cohort}_llm_baseline.py", f"{cohort.upper()} Baseline")
    for cohort in ("gamma", "papila"):
        add("baselines", AGENT + f"evaluate_{cohort}_llm_baseline.py",
            f"{cohort.upper()}: Shared Baseline Input Description", dependencies=("IMAGE_DESCRIPTION",))
    add("baselines", SCRIPTS + "run_gdp_progression_llm_baseline.py",
        "Single-Endpoint Progression Baseline", functions=("build_user_prompt", "round_numbers"),
        dependencies=("PROGRESSION_TARGET_DESCRIPTIONS",))
    add("baselines", SCRIPTS + "run_gdp_progression_llm_multitarget_baseline.py",
        "Joint Six-Endpoint Progression Baseline", functions=("build_user_prompt",),
        dependencies=("TARGETS",))

    for path, title in (
        ("BioProfilerAgent/bio_profiler.py", "General Bio-Profiler"),
        ("VisionAgent/vision.py", "General Vision Specialist"),
        ("VisionAgent/vision_oct.py", "General OCT Specialist"),
        ("VisionAgent/vision_slo.py", "General SLO Specialist"),
        ("Orchestrator/new.py", "Legacy Multimodal Orchestrator"),
        ("Orchestrator/ophthalmic_agent.py", "General Ophthalmic Orchestrator"),
        ("EquityAgent/equity_agent.py", "Standalone Equity Auditor"),
        ("GuidelinesAgent/guidelines_agent.py", "Guideline Query and Evidence Summarizer"),
        ("GuidelinesAgent/evidence_tool.py", "Retrieved Evidence Summarizer"),
        ("SafetyAgent/safety_agent.py", "Safety Auditor"),
    ):
        add("legacy", AGENT + path, title)
    add("prospective", AGENT + "Ablation/fairvision_live.py", "Paired Reliability Ablation Adapter",
        functions=("response_format", "counterfactual_messages", "orchestrator_messages"),
        dependencies=("DISEASES",))
    add("prospective", AGENT + "Confirmation/contract.py", "Explicit Escalation Contract",
        functions=("endpoint_messages", "final_schema", "final_messages"), dependencies=("ENDPOINTS",))
    add("prospective", AGENT + "Confirmation/workflow.py", "Prospective External CFP Instructions",
        dependencies=("external:evidence",))
    return specs


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ancestors(node, parents):
    while node in parents:
        node = parents[node]
        yield node


def qualified_name(node, parents):
    names = [parent.name for parent in ancestors(node, parents)
             if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return ".".join(reversed(names)) or "module"


def condition_context(node, parents):
    parts = []
    child = node
    for parent in ancestors(node, parents):
        if isinstance(parent, ast.If):
            branch = "if" if child in parent.body else "else of"
            parts.append(f"{branch} {ast.unparse(parent.test)}")
        child = parent
    return "; ".join(reversed(parts))


def is_text_expression(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and bool(node.value.strip())
    return isinstance(node, (ast.JoinedStr, ast.BinOp, ast.IfExp)) and any(
        isinstance(part, ast.Constant) and isinstance(part.value, str) and part.value.strip()
        for part in ast.walk(node)
    )


def without_comments(source):
    lines = source.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            row, start = token.start
            _, end = token.end
            lines[row - 1] = lines[row - 1][:start] + " " * (end - start) + lines[row - 1][end:]
    return "".join(lines)


def extract(source, spec):
    """Select executable templates, not commented-out drafts, docstrings or logs."""
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    clean_source = without_comments(source)
    chosen = {}
    covered = set()

    def add(node, label, code=False):
        if node in covered or node in chosen:
            return
        segment = ast.get_source_segment(source, node)
        if segment is None:
            raise ValueError(f"No source segment for {label}")
        if not code and isinstance(node, ast.Constant) and isinstance(node.value, str):
            body = textwrap.dedent(node.value).strip("\n")
            kind = "Literal text (layout indentation normalized)"
        else:
            # Keep multiline templates readable, while excluding commented-out drafts.
            body = textwrap.dedent(ast.get_source_segment(clean_source, node, padded=True)).strip()
            if isinstance(node, ast.expr):
                try:
                    ast.parse(body, mode="eval")
                except SyntaxError:
                    body = "(\n" + body + "\n)"
            kind = "Python template or construction code (not evaluated)"
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        chosen[node] = dict(label=label, context=qualified_name(node, parents),
                            condition=condition_context(node, parents), kind=kind,
                            line_start=node.lineno, line_end=node.end_lineno, body=body,
                            source_expression_sha256=sha(segment.encode()),
                            display_sha256=sha(body.encode()))

    wanted_functions = set(spec["functions"])
    found_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            add(node, f"Construction function: {node.name}", code=True)
            covered.update(ast.walk(node))
            found_functions.add(node.name)
    if found_functions != wanted_functions:
        raise ValueError(f"Missing functions in {spec['path']}: {wanted_functions - found_functions}")

    prompt_names = {"prompt", "system_prompt", "user_prompt", "system_content", "base_content",
                    "safety_agent_system_prompt", "escalation_instruction"}
    dependencies = set(spec["dependencies"])
    for node in ast.walk(tree):
        if node in covered:
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and node.value is not None:
                    name = target.id
                    if name == "SYSTEM_PROMPTS" and isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
                            add(value, f"System prompt: {ast.literal_eval(key)}")
                    elif name in dependencies or qualified_name(node, parents) + ":" + name in dependencies:
                        add(node, f"Template dependency: {name}", code=True)
                    elif name.lower() in prompt_names and is_text_expression(node.value):
                        prefix = "Append to" if isinstance(node, ast.AugAssign) else "Template"
                        add(node.value, f"{prefix}: {name}")
                elif (isinstance(node, ast.AugAssign) and isinstance(target, ast.Subscript)
                      and isinstance(target.slice, ast.Constant) and target.slice.value == "content"
                      and is_text_expression(node.value)):
                    add(node.value, "Append to message content")
        if isinstance(node, ast.Dict):
            role = next((value.value for key, value in zip(node.keys, node.values)
                         if isinstance(key, ast.Constant) and key.value == "role"
                         and isinstance(value, ast.Constant)), "message")
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {"content", "text", "system"}:
                    if is_text_expression(value):
                        add(value, f"{role.capitalize()} {key.value}")
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in {"system", "content", "text"} and is_text_expression(keyword.value):
                    add(keyword.value, f"Message field: {keyword.arg}")
                elif keyword.arg == "response_format" and isinstance(keyword.value, ast.Dict):
                    add(keyword.value, "API response format", code=True)

    # A selected container already includes its constituent message expressions.
    entries = [entry for node, entry in chosen.items()
               if not any(parent in chosen for parent in ancestors(node, parents))]
    return sorted(entries, key=lambda entry: (entry["line_start"], entry["line_end"]))


def latex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def verbatim(body):
    if r"\end{Verbatim}" in body:
        raise ValueError("Prompt collides with LaTeX verbatim delimiter")
    return "\\begin{Verbatim}\n" + body + "\n\\end{Verbatim}\n"


def render(modules, export_date):
    preamble = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\DeclareUnicodeCharacter{2011}{-}
\usepackage{fvextra}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\fvset{fontsize=\footnotesize,breaklines=true,breakanywhere=true,tabsize=4}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setcounter{tocdepth}{2}
\title{RetinAgent: Supplementary Prompt Templates}
\author{}
"""
    chunks = [preamble, "\\date{Source export: " + latex_escape(export_date) + "}\n",
              "\\begin{document}\n\\maketitle\n", r"\section*{Scope and Provenance}" + "\n",
              latex_escape(
                  "This supplement reproduces current executable prompt templates in the local source "
                  "checkout, with dynamic input construction and structured-output contracts where relevant. "
                  "It is not a verified frozen snapshot of the prompts that generated the historical "
                  "manuscript results. Historical canonical FairVision request snapshots were not established "
                  "by the saved-output audit. Legacy/optional modules and prospective, unverified extensions "
                  "are explicitly separated; their inclusion does not establish study execution."
              ) + "\n\n",
              latex_escape(
                  "Literal prompts are decoded from Python string literals with layout indentation "
                  "normalized. Dynamic f-strings, conditional fragments and construction functions are "
                  "shown as Python templates, not fabricated patient-specific requests. Expressions in "
                  "braces denote runtime interpolation. Function blocks document construction and schemas, "
                  "not additional text sent to the model. The applicable inputs, images, upstream reports "
                  "and retrieved evidence are supplied at runtime; no patient data or image payloads are "
                  "embedded. Conditional additions are labelled and must not be concatenated across "
                  "mutually exclusive branches. Source comments, logging strings, credentials, test "
                  "fixtures and archived non-executable prompt drafts are excluded."
              ) + "\n\n",
              latex_escape(
                  "Paths are repository-relative. Every module lists its source-file SHA256; blocks "
                  "list source line spans and stable export identifiers. The accompanying CSV index and "
                  "JSON manifest provide expression and display hashes. The source's non-breaking "
                  "hyphen (U+2011) is displayed as a hyphen for pdfLaTeX compatibility. No source prompt "
                  "is modified and no model import, inference or API request is performed by this export."
              ) + "\n\n\\tableofcontents\n\\clearpage\n"]
    for section, (title, note) in SECTIONS.items():
        if section == "legacy":
            chunks.append("\\appendix\n")
        chunks.extend(["\\section{" + latex_escape(title) + "}\n", latex_escape(note) + "\n"])
        for module in modules:
            if module["section"] != section:
                continue
            chunks.extend(["\\subsection{" + latex_escape(module["title"]) + "}\n",
                           "Source: \\path{" + module["path"] + "}\\\\\n",
                           "SHA256: {\\scriptsize\\path{" + module["sha256"] + "}}\n"])
            for entry in module["entries"]:
                chunks.append("\\paragraph{" + latex_escape(entry["id"] + ": " + entry["label"]) + "}\n")
                chunks.append(latex_escape(
                    f"{entry['kind']}. Context: {entry['context']}. "
                    f"Source lines {entry['line_start']}-{entry['line_end']}."
                ) + "\n")
                if entry["condition"]:
                    chunks.append("\\textbf{Conditional fragment:} " + latex_escape(entry["condition"]) + "\n")
                chunks.append(verbatim(entry["body"]))
        chunks.append("\\clearpage\n")
    chunks.append("\\end{document}\n")
    return "".join(chunks)


def export(root, out, export_date="2026-10-06"):
    modules = []
    index = []
    for spec in source_specs():
        raw = (root / spec["path"]).read_bytes()
        entries = extract(raw.decode("utf-8"), spec)
        if not entries:
            raise ValueError(f"No templates extracted from {spec['path']}")
        module = {key: spec[key] for key in ("section", "path", "title")}
        module.update(sha256=sha(raw), entries=entries)
        for entry in entries:
            entry["id"] = f"P{len(index) + 1:03d}"
            index.append(dict(id=entry["id"], section=spec["section"], source=spec["path"],
                              source_sha256=module["sha256"], **{
                                  key: value for key, value in entry.items() if key not in {"id", "body"}
                              }))
        modules.append(module)
    document = render(modules, export_date)
    non_ascii = set(document) - set(chr(code) for code in range(128))
    if non_ascii - {"\u2011"}:
        raise ValueError(f"Add explicit LaTeX support for: {sorted(non_ascii)!r}")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "supplementary_prompts.tex"
    target.write_text(document, encoding="utf-8")
    with (out / "prompt_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    manifest = dict(export_date=export_date, artifact=target.name,
                    artifact_sha256=sha(target.read_bytes()),
                    exporter="equi-agent/scripts/export_supplementary_prompts_latex.py",
                    exporter_sha256=sha(Path(__file__).read_bytes()),
                    historical_execution_verified=False, model_calls=0,
                    scope="Current executable templates; legacy and prospective extensions separated",
                    source_count=len(modules), block_count=len(index), sources=[{
                        key: value for key, value in module.items() if key != "entries"
                    } for module in modules])
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--export-date", default="2026-10-06")
    args = parser.parse_args()
    result = export(ROOT, args.output_dir, args.export_date)
    print(f"Exported {result['block_count']} blocks from {result['source_count']} sources to "
          f"{args.output_dir / result['artifact']}; no model/API calls.")


if __name__ == "__main__":
    main()
