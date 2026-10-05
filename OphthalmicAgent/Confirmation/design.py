"""Offline cohort selection and immutable source receipts. No inference imports."""
from __future__ import annotations
import ast
import json
import random
import subprocess
from pathlib import Path

import run_fairvision_ablation as base
from build_manifests import normalize_binary_label, require_data_libs
from EquityAgent.demographics import fairvision_age_group

ROOT = base.ROOT
DESIGN = ROOT / "OphthalmicAgent/Confirmation"


def read_protocol():
    return json.loads((DESIGN / "protocol.json").read_text())


def locked_json(path, data):
    if path.exists():
        base.require(json.loads(path.read_text()) == data, f"Frozen artifact differs: {path}; do not overwrite")
    else:
        base.write_json(path, data)


def demographics_profiles(case):
    result = []
    original = case["metadata"]
    for field, values in read_protocol()["counterfactual_replacements"].items():
        for value in values:
            same = (fairvision_age_group(value) == fairvision_age_group(original[field]) if field == "Age"
                    else str(value).lower() == str(original[field]).lower())
            if same:
                continue
            metadata = {**original, field: value}
            name = f"{field.lower()}_{str(value).replace('-', '_')}"
            result.append(dict(name=name, field=field, original=original[field], replacement=value,
                               metadata=metadata, **base.demographic(metadata)))
    base.require(len(result) == 6, f"Expected six demographic changes for {case['case_id']}; inspect unknown metadata")
    return result


def select_cohorts():
    require_data_libs()
    protocol = read_protocol()
    excluded, sources = {t: set() for t in base.TASKS}, {}
    # Include attempted cases, even if their old responses could not be parsed.
    for path in sorted((ROOT / "OphthalmicAgent/results/fairvision").rglob("*agentic*.csv")):
        sources[str(path.relative_to(ROOT))] = base.sha(path)
        for r in base.read_csv(path):
            task = (r.get("Task_Folder") or r.get("Disease") or
                    next((t for t in base.TASKS if f"_{t}_" in path.name), "")).lower()
            base.require(task in excluded, f"Cannot identify prior task: {path}")
            excluded[task].add(Path(r.get("Filename") or r.get("filename", "")).stem)
    locked = ROOT / "OphthalmicAgent/data/fairvision_250each.csv"
    sources[str(locked.relative_to(ROOT))] = base.sha(locked)
    for r in base.read_csv(locked):
        excluded[r["Task_Folder"].lower()].add(Path(r["filename"]).stem)
    extra = DESIGN / "additional_exclusions.csv"
    if extra.exists():
        sources[str(extra.relative_to(ROOT))] = base.sha(extra)
        for r in base.read_csv(extra):
            excluded[r["task"]].add(Path(r["case_id"]).stem)
    result, summary = [], []
    for task in base.TASKS:
        folder = base.DISEASE_FOLDERS[task]
        path = ROOT / f"OphthalmicAgent/data/{folder}/data_summary_{task}.csv"
        sources[str(path.relative_to(ROOT))] = base.sha(path)
        population = [r for r in base.read_csv(path) if r["use"] == "test"]
        base.require(len(population) == 3000, f"Expected 3000 {task} test records")
        eligible = [r for r in population if Path(r["filename"]).stem not in excluded[task]]
        chosen = []
        for label in (0, 1):
            pool = sorted((r for r in eligible if base.binary(normalize_binary_label(r[task], task)) == label),
                          key=lambda r: r["filename"])
            base.require(len(pool) >= 125, f"Only {len(pool)} unseen {task} class {label} cases; cannot draw 125")
            random.Random(f"{protocol['selection_seed']}:{task}:{label}").shuffle(pool)
            for r in pool[:125]:
                record = dict(filename=f"data/{folder}/Test/{r['filename']}", Task_Folder=folder,
                    Ground_Truth=label, Age=r["age"], Gender=r["gender"], Race=r["race"], Ethnicity=r["ethnicity"])
                demographics_profiles(dict(case_id=Path(r["filename"]).stem,
                    metadata={k: record[k] for k in ("Age", "Gender", "Race", "Ethnicity")}))
                chosen.append(record)
        result.extend(sorted(chosen, key=lambda r: r["filename"]))
        summary.append(dict(task=task, population=3000, excluded=len(population)-len(eligible),
                            eligible=len(eligible), selected=250, positive=125, negative=125))
    target = DESIGN / "cohorts.csv"
    if target.exists():
        previous = base.read_csv(target)
        base.require(previous == [{k: str(v) for k, v in r.items()} for r in result], "Cohort already frozen")
    else:
        base.audit.write_csv(target, result)
    locked_json(DESIGN / "selection_receipt.json", dict(seed=protocol["selection_seed"],
        sampling="class-stratified without replacement; Python Random string seeds task:label; no model outputs used",
        summary=summary, sources=sources, cohort_sha256=base.sha(target),
        exclusions={t: sorted(ids) for t, ids in excluded.items()},
        scope="All locally identified prior live-agent case IDs; excludes failed attempts; not proof of no remote exposure"))
    print(json.dumps(summary, indent=2))


def freeze_paths():
    paths = set(base.code_paths())
    paths.add(ROOT / "equi-agent/scripts/build_manifests.py")
    for folder in ("Confirmation", "Orchestrator", "VisionAgent", "BioProfilerAgent", "CounterfactualAgent", "EquityAgent"):
        paths.update((ROOT / "OphthalmicAgent" / folder).glob("*.py"))
    paths.update(DESIGN / name for name in ("protocol.json", "cohorts.csv", "selection_receipt.json"))
    paths.add(DESIGN / "README.txt")
    paths.add(ROOT / "OphthalmicAgent/tests/test_frozen_confirmation.py")
    paths.add(ROOT / "OphthalmicAgent/tests/test_fairvision_ablation.py")
    paths.update(ROOT / "OphthalmicAgent" / name for name in (
        "main_new.py", "evaluate_fairvision_glaucoma_agentic.py", "evaluate_fairvision_amd_agentic.py",
        "evaluate_fairvision_dr_agentic.py", "scripts/run_frozen_confirmation.py",
        "scripts/precompute_external_cdr.py", "scripts/run_external_glaucoma_agent.py",
        "scripts/run_frozen_confirmation.sh"))
    return sorted(paths)


def freeze():
    sources = {str(p.relative_to(ROOT)): base.sha(p) for p in freeze_paths()}
    text = ["FROZEN RETINAGENT CONFIRMATION PROMPTS\n",
            "Source string inventory, including interpolations and legacy/inactive templates.\n"
            "Inclusion does not mean execution. Actual requests are saved under api/ during the run.\n"
            "See protocol.json and README.txt for executed stages and exact generation settings.\n"]
    for path in freeze_paths():
        if path.suffix != ".py":
            continue
        source = path.read_text()
        tree = ast.parse(source)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Constant, ast.JoinedStr)) and not isinstance(parents.get(node), (ast.JoinedStr, ast.FormattedValue)):
                value = node.value if isinstance(node, ast.Constant) else ast.get_source_segment(source, node)
                if isinstance(value, str) and len(value) >= 100:
                    text.append(f"\nSOURCE {path.relative_to(ROOT)}:{node.lineno}\n{value}\n")
    snapshot = DESIGN / "prompts_snapshot.txt"
    content = "\n".join(text)
    if snapshot.exists():
        base.require(snapshot.read_text() == content, "Frozen prompt snapshot differs; do not overwrite")
    else:
        snapshot.write_text(content)
    locked_json(DESIGN / "source_lock.json", dict(version=read_protocol()["version"], source_sha256=sources,
                                                prompts_sha256=base.sha(snapshot)))
    print("Source and prompts frozen. Commit this tree before --stage prepare or --stage run.")


def verify_freeze(require_commit=True):
    path = DESIGN / "source_lock.json"
    base.require(path.exists(), "Run --stage freeze and commit before preparing inference")
    lock = json.loads(path.read_text())
    base.require(lock["source_sha256"] == {str(p.relative_to(ROOT)): base.sha(p) for p in freeze_paths()},
                 "Frozen source changed; do not revise the protocol after test inspection")
    snapshot = DESIGN / "prompts_snapshot.txt"
    base.require(base.sha(snapshot) == lock["prompts_sha256"], "Frozen prompt export changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if require_commit:
        for rel in [*lock["source_sha256"], str(path.relative_to(ROOT)), str(snapshot.relative_to(ROOT))]:
            saved = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
            import hashlib
            base.require(hashlib.sha256(saved).hexdigest() == base.sha(ROOT / rel), f"Uncommitted frozen file: {rel}")
    return commit, base.digest(lock)
