"""Preserve edited case panels and illustrate a recorded evidence ablation.

Uses saved, verified counts and embedded real images. No model calls, fitting,
subgroup recalculation, image alteration or manuscript metric replacement.
"""

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from build_paper_drawio import Page


OUT = Path(__file__).resolve().parents[1] / "docs/paper_figures/clinical_decisions"
INK, GRAY, TEAL = "#202326", "#606A70", "#007F78"


def compact_source(source):
    cohort = source["cohort"]
    counts = {key: cohort[key] for key in (
        "n", "raw_agent_rows", "excluded", "corrected", "introduced",
        "both_correct", "both_wrong", "corrected_positive", "corrected_negative",
        "introduced_positive", "introduced_negative",
    )}
    for system in ("baseline", "agent"):
        counts[system] = {key: cohort[system][key] for key in ("n", "tn", "fp", "fn", "tp")}
    validate_counts(counts)
    return dict(
        cohort=counts, examples=source["examples"], sources=source["sources"],
        images_verified=source["images_verified"],
        manuscript_reported_metrics=dict(
            provenance="Author-supplied manuscript values, not recalculated for this figure; not plotted.",
            baseline=dict(weighted_f1=.7477, worst_group_f1=.6344),
            agent=dict(weighted_f1=.7704, worst_group_f1=.6380),
        ),
    )


def validate_counts(c):
    b, a = c["baseline"], c["agent"]
    for system in (b, a):
        if any(type(system[k]) is not int or system[k] < 0 for k in ("n", "tn", "fp", "fn", "tp")):
            raise ValueError("Invalid confusion counts")
        if sum(system[k] for k in ("tn", "fp", "fn", "tp")) != system["n"] or system["n"] != c["n"]:
            raise ValueError("Confusion counts do not match cohort")
    if b["tp"] + b["fn"] != a["tp"] + a["fn"] or b["tn"] + b["fp"] != a["tn"] + a["fp"]:
        raise ValueError("Reference class counts differ")
    if b["fn"] - c["corrected_positive"] + c["introduced_positive"] != a["fn"]:
        raise ValueError("Positive transitions do not match errors")
    if b["fp"] - c["corrected_negative"] + c["introduced_negative"] != a["fp"]:
        raise ValueError("Negative transitions do not match errors")
    if c["corrected"] != c["corrected_positive"] + c["corrected_negative"]:
        raise ValueError("Correction total differs")
    if c["introduced"] != c["introduced_positive"] + c["introduced_negative"]:
        raise ValueError("Introduced-error total differs")
    if sum(c[k] for k in ("corrected", "introduced", "both_correct", "both_wrong")) != c["n"]:
        raise ValueError("Paired transition total differs")


def add_panel_borders(tree):
    """Frame the two case panels and evidence panel without rebuilding content."""
    tree = copy.deepcopy(tree)
    graph = tree.find("diagram/mxGraphModel")
    if graph is None:
        raise ValueError("Expected one uncompressed draw.io diagram")
    root = graph.find("root")
    frames = {"panel_a_border", "panel_b_border", "panel_c_border"}
    for cell in list(root):
        if cell.get("id") in frames:
            root.remove(cell)
        elif cell.get("id") == "c32" and cell.get("edge") == "1":
            # The panel frames replace the previous single horizontal divider.
            root.remove(cell)
    for index, (ident, x, y, w, h) in enumerate((
        ("panel_a_border", 4, 0, 904, 642),
        ("panel_b_border", 922, 0, 904, 642),
        ("panel_c_border", 4, 656, 1822, 172),
    )):
        cell = ET.Element("mxCell", id=ident, value="", vertex="1", parent="1",
                          style="rounded=0;fillColor=none;strokeColor=#737D84;strokeWidth=2;")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h),
                      **{"as": "geometry"})
        root.insert(2 + index, cell)
    graph.set("pageHeight", "842")
    return tree


def evidence_example(source):
    candidates = [e for e in source["examples"] if e["case_id"] == "data_07062.npz"]
    if len(candidates) != 1:
        raise ValueError("Missing or duplicate evidence example for case b")
    example = candidates[0]
    if example["truth"] != 0 or example["baseline"] != 1 or example["agent"] != 0:
        raise ValueError("Case b final outcomes changed")
    if example["scenarios"].get("full_evidence") != 0 or example["scenarios"].get("without_visual_interpretation") != -1:
        raise ValueError("Recorded evidence-ablation diagnoses changed")
    if not example.get("trace_fingerprint"):
        raise ValueError("Missing evidence-trace provenance")
    return example


def update_diagram(tree, source):
    validate_counts(source["cohort"])
    evidence_example(source)
    tree = copy.deepcopy(tree)
    diagrams = list(tree.iter("diagram"))
    if len(diagrams) != 1:
        raise ValueError("Expected one editable figure page")
    diagram = diagrams[0]
    graph = diagram.find("mxGraphModel")
    if graph is None:
        raise ValueError("Save an uncompressed draw.io XML diagram first")
    root = graph.find("root")
    # Keep every case-panel cell, including author edits and image bytes.
    for cell in list(root):
        geometry = cell.find("mxGeometry")
        old_footer = cell.get("vertex") == "1" and geometry is not None and float(geometry.get("y", 0)) >= 650
        if cell.get("id", "").startswith("summary_") or old_footer:
            root.remove(cell)
    images = [c for c in root if "shape=image;" in c.get("style", "")]
    if len(images) != 4:
        raise ValueError("Expected the four original OCT/SLO images")
    diagram.set("name", "Clinical decision examples")
    graph.set("pageHeight", "855")

    p = Page("Evidence ablation")
    p.text("c  Case b: recorded evidence ablation", 20, 663, 1200, 35, 23, INK, True)
    p.text("Without OCT/SLO reports", 20, 716, 700, 32, 24, GRAY)
    p.text("Inconclusive", 20, 758, 700, 50, 34, GRAY, True)
    p.text("With OCT/SLO reports", 930, 716, 860, 32, 24, INK)
    p.text("Non-glaucoma", 930, 758, 860, 50, 34, TEAL, True)
    for cell in list(p.root)[2:]:
        cell.set("id", "summary_" + cell.get("id"))
        root.append(cell)
    return add_panel_borders(tree)


def caption(source):
    c = source["cohort"]
    b, a = c["baseline"], c["agent"]
    return f"""# Clinical Decision Examples

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
- There are {c['corrected']} corrected baseline errors and {c['introduced']}
  introduced errors: {c['corrected_positive']} corrected missed glaucoma cases
  versus {c['introduced_positive']} newly missed cases, and
  {c['corrected_negative']} corrected false alarms versus
  {c['introduced_negative']} new false alarms. None are dropped from the paired
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
| RETFound | {b['n']} | {b['fn']} | {b['fp']} | {b['tp']} | {b['tn']} |
| RetinAgent | {a['n']} | {a['fn']} | {a['fp']} | {a['tp']} | {a['tn']} |

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
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-diagram", type=Path, default=OUT / "clinical_decisions.drawio")
    parser.add_argument("--source-data", type=Path, default=OUT / "source_data.json")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--borders-only", action="store_true",
                        help="Preserve all author-edited content; only add the three panel frames.")
    args = parser.parse_args()
    if args.borders_only:
        tree = add_panel_borders(ET.parse(args.case_diagram))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output = args.out_dir / "clinical_decisions.drawio"
        tree.write(output, encoding="utf-8", xml_declaration=True)
        print(f"Added panel borders: {output}; all content preserved.")
        return
    source = compact_source(json.loads(args.source_data.read_text()))
    tree = update_diagram(ET.parse(args.case_diagram), source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "clinical_decisions.drawio"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    (args.out_dir / "source_data.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "caption.md").write_text(caption(source))
    print(f"Updated {output}; export PDF/PNG/SVG with draw.io.")


if __name__ == "__main__":
    main()
