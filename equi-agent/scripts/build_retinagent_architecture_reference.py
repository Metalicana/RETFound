"""Build a native, editable draw.io architecture using existing figure assets.

No model inference, training, API access or changes to historical figures.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from build_paper_drawio import Page, write_document


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "equi-agent/docs/paper_figures/architecture_reference"
INK = "#222629"
MUTED = "#647076"
TEAL = "#397F87"
WINE = "#80344D"
BLUE = "#527FA3"
GREEN = "#719A79"
GOLD = "#C99946"
PALE = "#F3F4F4"
LINE = "#CCD1D3"
WHITE = "#FFFFFF"


class Diagram(Page):
    def __init__(self):
        super().__init__("RetinAgent architecture", width=2400, height=1460)
        self.element.set("id", "retinagent-architecture-reference")
        self.parent = "1"
        self.graph = self.element.find("mxGraphModel")
        self.graph.set("dx", "2400")
        self.graph.set("dy", "1460")

    def cell(self, text, x, y, w, h, style):
        ident = super().cell(text, x, y, w, h, style)
        self.node(ident).set("parent", self.parent)
        return ident

    def node(self, ident):
        return self.root.find(f"mxCell[@id='{ident}']")

    def group(self, x, y, w, h, name):
        ident = self.cell("", x, y, w, h, "group;")
        self.node(ident).set("data-name", name)
        return ident

    def rect(self, x, y, w, h, fill=WHITE, stroke="none", width=1.5, dashed=False, rounded=True):
        return self.cell("", x, y, w, h,
                         f"rounded={int(rounded)};absoluteArcSize=1;arcSize=8;fillColor={fill};"
                         f"strokeColor={stroke};strokeWidth={width};dashed={int(dashed)};dashPattern=5 4;")

    def oval(self, x, y, w, h, fill="none", stroke=INK, width=3):
        return self.cell("", x, y, w, h,
                         f"ellipse;fillColor={fill};strokeColor={stroke};strokeWidth={width};")

    def line(self, points, color=INK, dashed=False, arrow=False, width=2.5):
        super().line(points, color, dashed, arrow, width)
        self.node(f"c{self.counter}").set("parent", self.parent)

    def link(self, source, target, start=(1, .5), end=(0, .5), points=(), color=TEAL,
             dashed=False, width=4, arrow=True):
        self.counter += 1
        edge = ET.SubElement(self.root, "mxCell", id=f"c{self.counter}", parent="1", edge="1",
                             source=source, target=target,
                             style=f"edgeStyle=none;rounded=1;arcSize=20;html=0;strokeColor={color};"
                             f"strokeWidth={width};endArrow={'block' if arrow else 'none'};endSize=10;endFill=1;"
                             f"dashed={int(dashed)};dashPattern=4 4;"
                             f"exitX={start[0]};exitY={start[1]};exitDx=0;exitDy=0;exitPerimeter=0;"
                             f"entryX={end[0]};entryY={end[1]};entryDx=0;entryDy=0;entryPerimeter=0;")
        geo = ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            array = ET.SubElement(geo, "Array", **{"as": "points"})
            for x, y in points:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))
        return edge.get("id")

    def label(self, text, x, y, w, h, size=24, color=INK, bold=False, align="left"):
        return self.text(text, x, y, w, h, size, color, bold, align)

    def icon(self, kind, x, y, size=70, color=INK):
        """Each glyph consists of editable native geometry, not an SVG image."""
        previous = self.parent
        group = self.group(x, y, size, size, f"{kind} icon")
        self.parent = group
        s = size / 80

        def ln(points, width=3):
            self.line([(a * s, b * s) for a, b in points], color, width=width * s)

        def ov(a, b, w, h, fill="none", width=3):
            self.oval(a * s, b * s, w * s, h * s, fill, color, width * s)

        def box(a, b, w, h, fill="none", width=3):
            self.rect(a * s, b * s, w * s, h * s, fill, color, width * s)

        if kind in {"person", "doctor"}:
            ov(25, 4, 30, 34)
            ln([(29, 36), (27, 44), (10, 52), (6, 71), (74, 71), (70, 52), (53, 44), (51, 36)])
            if kind == "doctor":
                ln([(27, 44), (39, 60), (53, 44)])
                ln([(26, 49), (25, 60), (20, 60), (20, 53)])
                ov(50, 57, 10, 10, width=2.5)
                ln([(51, 47), (55, 50), (55, 57)])
            else:
                ln([(27, 44), (40, 49), (53, 44)])
        elif kind == "robot":
            ln([(40, 8), (40, 18)])
            ov(36, 1, 8, 8)
            box(13, 19, 54, 37)
            box(5, 29, 8, 17)
            box(67, 29, 8, 17)
            ov(25, 29, 6, 7, color, width=1)
            ov(49, 29, 6, 7, color, width=1)
            ln([(28, 46), (52, 46)])
            ln([(40, 56), (40, 63), (20, 63), (13, 75)])
            ln([(40, 63), (60, 63), (67, 75)])
        elif kind == "network":
            nodes = [(10, 14), (10, 40), (10, 66), (39, 10), (39, 36), (39, 63), (70, 25), (70, 54)]
            for a, b in ((0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 6), (4, 6), (4, 7), (5, 7)):
                ln([nodes[a], nodes[b]], 1.8)
            for a, b in nodes:
                ov(a - 5, b - 5, 10, 10, PALE, 2.5)
        elif kind == "heads":
            box(5, 27, 24, 26)
            ln([(29, 40), (44, 40), (44, 13), (57, 13)])
            ln([(44, 40), (57, 40)])
            ln([(44, 40), (44, 67), (57, 67)])
            for b in (4, 31, 58):
                box(57, b, 19, 18)
        elif kind == "table":
            box(9, 7, 62, 64)
            ln([(9, 23), (71, 23)])
            ln([(27, 23), (27, 71)])
            for b, length in ((35, 26), (47, 18), (59, 31)):
                ln([(36, b), (36 + length, b)], 4)
        elif kind == "lookup":
            ov(9, 6, 43, 43)
            ov(17, 14, 27, 27, width=1.8)
            ln([(47, 44), (71, 68)], 6)
            ln([(16, 62), (37, 62)], 2)
            ln([(16, 72), (46, 72)], 2)
        elif kind == "cdr":
            ov(11, 5, 51, 66)
            ov(26, 22, 25, 37)
            ln([(71, 5), (71, 71)], 2)
            ln([(67, 5), (76, 5)], 2)
            ln([(67, 71), (76, 71)], 2)
        elif kind == "document":
            ln([(15, 5), (54, 5), (69, 20), (69, 75), (15, 75), (15, 5)])
            ln([(54, 5), (54, 20), (69, 20)], 2)
            for b, right in ((31, 58), (43, 58), (55, 48)):
                ln([(26, b), (right, b)], 3)
        elif kind == "grid":
            box(5, 5, 69, 69)
            for b in (22, 39, 56):
                ln([(5, b), (74, b)], 1.5)
                ln([(b, 5), (b, 74)], 1.5)
        elif kind == "book":
            ln([(40, 16), (13, 8), (7, 8), (7, 64), (15, 64), (40, 73), (65, 64), (73, 64), (73, 8), (67, 8), (40, 16), (40, 73)])
        elif kind == "shield":
            ln([(40, 4), (68, 16), (65, 49), (53, 65), (40, 76), (27, 65), (15, 49), (12, 16), (40, 4)])
            ln([(25, 39), (36, 50), (56, 27)], 4)
        self.parent = previous
        return group

    def agent(self, x, y, w, h, title, subtitle, diameter=112, kind="robot", outline=False, title_top=False):
        group = self.group(x, y, w, h, title.replace("\n", " "))
        previous, self.parent = self.parent, group
        cy = 80 if title_top else 8
        cx = (w - diameter) / 2
        self.oval(cx, cy, diameter, diameter, WHITE if outline else WINE, WINE, 3)
        self.icon(kind, cx + diameter * .22, cy + diameter * .20, diameter * .56, WINE if outline else WHITE)
        label_y = 0 if title_top else cy + diameter + 8
        title_height = 62 if "\n" in title else 34
        self.label(title, 0, label_y, w, title_height,
                   29 if title_top else 25, INK, True, "center")
        if subtitle:
            subtitle_y = h - 40 if title_top else label_y + title_height + 7
            self.label(subtitle, -10, subtitle_y, w + 20, 28, 18, MUTED, align="center")
        self.parent = previous
        return group, ((cx + diameter) / w, (cy + diameter / 2) / h), (cx / w, (cy + diameter / 2) / h)


def embedded_images():
    path = ROOT / "equi-agent/docs/paper_figures/clinical_decisions/clinical_decisions.drawio"
    tree = ET.parse(path)
    images = {}
    for name, ident in (("oct", "c5"), ("slo", "c7")):
        cell = next(cell for cell in tree.iter("mxCell") if cell.get("id") == ident)
        images[name] = next(part.split("=", 1)[1] for part in cell.get("style").split(";") if part.startswith("image="))
    fundus_path = ROOT / "equi-agent/docs/paper_figures/agentic_walkthrough/assets/fundus.png"
    images["cfp"] = "data:image/png," + base64.b64encode(fundus_path.read_bytes()).decode()
    return images, [path, fundus_path]


def build():
    p = Diagram()
    images, asset_paths = embedded_images()
    p.label("RETINAGENT", 45, 25, 500, 45, 35, INK, True)
    p.label("Reliability-aware ophthalmic reasoning", 575, 27, 1250, 40, 28, MUTED)
    p.line([(45, 87), (2355, 87)], LINE, width=1.5)
    for title, x, w in (("A  Patient inputs", 45, 520), ("B  Shared models and tools", 650, 1170),
                        ("D  Outputs and review", 1895, 470)):
        p.label(title, x, 112, w, 50, 33, INK, True)

    # White-space and the resource/core backplates establish the reference's four regions.
    shelf = p.rect(650, 190, 1180, 250, PALE)
    p.rect(650, 190, 1180, 55, "#E5E8E8")
    p.label("Task-matched resources shared by the reasoning workflow", 675, 200, 1130, 35, 23, INK, True)
    for x, kind, title in ((688, "network", "Retinal foundation\nmodels"),
                           (913, "heads", "Task-specific\nprediction heads"),
                           (1138, "table", "Reliability\nreference tables"),
                           (1363, "lookup", "Demographic\nlookup"),
                           (1588, "cdr", "Cup-to-disc\ntool")):
        p.icon(kind, x + 66, 268, 64)
        p.label(title, x, 341, 205, 62, 25, INK, align="center")
    p.label("Resources and modalities vary by diagnostic or progression task.", 678, 410, 1120, 25, 18, MUTED)
    p.label("C  Multi-agent reasoning core", 650, 476, 730, 50, 33, INK, True)
    p.rect(650, 548, 1180, 683, "#F4F5F5")

    # Patient context is a schema, not invented clinical information for a case.
    p.icon("person", 236, 193, 86)
    context = p.group(65, 310, 455, 140, "Available patient context")
    p.parent = context
    p.rect(0, 0, 455, 140, PALE, LINE)
    p.label("Available patient context", 18, 8, 420, 35, 24, INK, True)
    p.line([(0, 51), (455, 51)], LINE, width=1)
    p.label("Age / sex / race / ethnicity\nObserved values or explicit missingness", 18, 58, 420, 72, 21, MUTED)
    p.parent = "1"
    p.line([(279, 280), (279, 310)], INK, width=2)
    input_nodes = []
    for key, y, color, title, subtitle in (
        ("cfp", 502, GOLD, "Fundus photograph", "External cohorts"),
        ("oct", 746, BLUE, "OCT B-scans", "Structural imaging"),
        ("slo", 990, GREEN, "SLO image", "Paired retinal imaging"),
    ):
        g = p.group(70, y, 455, 210, f"{key.upper()} input")
        p.parent = g
        p.rect(17, -10, 208, 199, color, INK, 1, rounded=False)
        p.rect(7, -2, 208, 199, WHITE, INK, 1, rounded=False)
        p.cell("", 0, 5, 210, 192, f"shape=image;imageAspect=1;image={images[key]};fillColor=#101515;strokeColor=none;")
        p.label(title, 245, 49, 207, 64, 24, INK, True)
        p.label(subtitle, 245, 119, 207, 58, 20, MUTED)
        p.parent = "1"
        input_nodes.append(g)

    gdp = p.group(65, 1248, 455, 124, "GDP progression inputs")
    p.parent = gdp
    p.rect(0, 0, 455, 124, "#F0F5F8", "#BDCDD9")
    p.icon("grid", 20, 32, 57, BLUE)
    p.label("GDP progression branch", 95, 12, 340, 33, 23, BLUE, True)
    p.label("RNFLT + baseline visual field\n52 total-deviation values", 95, 50, 340, 62, 21, INK)
    p.parent = "1"

    bio, bio_e, bio_w = p.agent(680, 598, 235, 230, "Bio-Profiler\nAgent", "Patient narrative", kind="doctor")
    vision, vis_e, vis_w = p.agent(680, 827, 235, 230, "Vision\nSpecialists", "OCT / SLO / CFP / RNFLT")
    functional, fn_e, fn_w = p.agent(870, 1040, 220, 184, "Functional agent", "GDP progression only", diameter=85, outline=True)
    cf, cf_e, cf_w = p.agent(1152, 1030, 246, 194, "Counterfactual", "Evidence-ablation scenarios", diameter=105)
    orch, orch_e, orch_w = p.agent(1388, 699, 385, 322, "Ophthalmologist\nOrchestrator",
                                 "Integrates evidence + reliability", diameter=190, kind="doctor", title_top=True)

    packet = p.rect(1100, 667, 9, 412, TEAL, TEAL, 0, rounded=False)
    p.node(packet).set("data-name", "Evidence packet junction")
    p.label("Evidence packet", 977, 624, 255, 30, 21, TEAL, True, "center")
    scores = p.rect(1325, 581, 430, 68, WHITE, LINE)
    p.label("Model scores, reliability\nand tool measurements", 1340, 586, 400, 58, 22, INK, align="center")
    p.link(shelf, scores, start=(.78, 1), end=(.5, 0), points=((1570, 500), (1540, 500)),
           color=MUTED, dashed=True, width=2.8)
    p.link(scores, packet, start=(0, .5), end=(.5, 0), points=((1255, 615), (1255, 667)),
           color=MUTED, dashed=True, width=2.8)
    p.link(context, bio, start=(1, .5), end=bio_w, points=((615, 380), (615, 662)), width=4)
    bus = p.rect(577, 602, 5, 493, TEAL, TEAL, 0, rounded=False)
    for index, node in enumerate(input_nodes):
        p.link(node, bus, start=(1, .5), end=(.5, (index * 244 + 5) / 493), arrow=False, width=2.8)
    p.link(bus, vision, start=(1, .59), end=vis_w, points=((628, 892),), width=4)
    p.link(gdp, functional, start=(1, .5), end=fn_w,
           points=((611, 1310), (611, 1190), (843, 1190), (843, 1090)), color=BLUE, width=3)
    p.link(gdp, vision, start=(1, .5), end=(.27, .44),
           points=((611, 1310), (611, 935), (716, 935)), color=BLUE, width=3)
    p.label("RNFLT", 627, 902, 93, 25, 17, BLUE, align="center")
    p.label("Visual field", 675, 1153, 150, 28, 18, BLUE, align="center")
    p.link(bio, packet, start=bio_e, end=(0, 0), width=4)
    p.link(vision, packet, start=vis_e, end=(0, .55), width=4)
    p.link(functional, packet, start=fn_e, end=(0, 1), width=3, color=BLUE)
    p.link(packet, orch, start=(1, .5), end=orch_w, width=5)
    p.label("Reports + scores + trust", 1160, 835, 280, 30, 20, TEAL, align="center")
    p.link(packet, cf, start=(1, .88), end=cf_w, points=((1134, 1030), (1134, 1090)), width=3.5)
    p.link(cf, orch, start=cf_e, end=(.29, .68), points=((1425, 1090), (1425, 938)), width=3.5)
    p.label("Audit trace", 1400, 1031, 167, 29, 19, TEAL, align="center")

    # A native document, not fabricated example predictions or calibrated uncertainty.
    report = p.group(1930, 678, 400, 386, "Diagnostic or prognostic output")
    p.parent = report
    p.rect(12, 9, 381, 364, "#E6E9E9", "none")
    p.rect(0, 0, 381, 364, WHITE, "#657379", 2)
    p.icon("document", 22, 20, 55, TEAL)
    p.label("Task-specific output", 93, 26, 265, 45, 27, INK, True)
    p.line([(23, 93), (358, 93)], LINE, width=1)
    for y, text in ((109, "Forced binary prediction"), (156, "Evidence-based rationale"), (203, "Source-dependence trace")):
        p.oval(25, y + 10, 9, 9, TEAL, TEAL, 0)
        p.label(text, 48, y, 314, 35, 23, INK)
    p.rect(22, 263, 337, 78, "#FAF5F7", WINE, 1.5, dashed=True)
    p.label("Explicit escalation flag*", 36, 269, 309, 31, 22, WINE, True)
    p.label("Updated protocol only", 36, 304, 309, 26, 19, MUTED)
    p.parent = "1"
    p.link(orch, report, start=orch_e, end=(0, .51), width=5)
    p.label("Final assessment", 1758, 825, 180, 31, 20, TEAL, align="center")

    audit = p.group(2005, 264, 290, 275, "Saved audit trail")
    p.parent = audit
    p.icon("document", 91, 3, 102, WINE)
    p.label("Saved audit trail", 0, 123, 290, 44, 27, INK, True, "center")
    p.label("Outputs, traces and settings\nfor offline evaluation", -35, 175, 360, 65, 22, MUTED, align="center")
    p.parent = "1"
    p.link(report, audit, start=(.985, .40), end=(.98, .54),
           points=((2365, 832), (2365, 596), (2348, 430)), color=MUTED, dashed=True, width=3)
    p.label("No online model updates", 1960, 554, 385, 32, 20, WINE, align="center")
    clinician = p.group(2010, 1133, 280, 86, "Clinician review")
    p.parent = clinician
    p.icon("doctor", 5, 2, 72, WINE)
    p.label("Clinician review", 100, 8, 222, 38, 26, INK, True)
    p.label("Proposed handoff", 100, 51, 222, 30, 20, MUTED)
    p.parent = "1"
    p.link(report, clinician, start=(.50, .97), end=(.45, 0), width=3.5)

    p.rect(650, 1290, 1180, 98, WHITE, "#ACB5B9", 1.5, dashed=True)
    p.label("OPTIONAL MODULES  /  not active in task-specific FairVision runs", 671, 1251, 1150, 32, 20, MUTED, True)
    for x, icon, name in ((692, "lookup", "Equity LLM"), (1067, "book", "Guidelines / web"), (1474, "shield", "Safety reviewer")):
        p.icon(icon, x, 1310, 53, WINE)
        p.label(name, x + 77, 1311, 267, 50, 25, INK)

    p.label("KEY", 1930, 1260, 420, 26, 18, MUTED, True)
    for y, color, dashed, text in ((1302, TEAL, False, "Evidence / inference"),
                                  (1340, MUTED, True, "Retrieval / logging"),
                                  (1378, WINE, True, "Optional / prospective")):
        p.line([(1930, y), (2005, y)], color, dashed=dashed, width=3.5)
        p.label(text, 2024, y - 15, 321, 30, 21, INK)
    p.line([(45, 1413), (2355, 1413)], LINE, width=1)
    p.label("Task-dependent architecture, not a historical execution receipt.  *The explicit escalation flag belongs to the updated protocol.",
            45, 1424, 2310, 25, 18, MUTED)
    return p, asset_paths


def validate(page):
    cells = list(page.root)
    ids = {cell.get("id") for cell in cells}
    if len(ids) != len(cells):
        raise ValueError("Duplicate diagram IDs")
    for cell in cells:
        if cell.get("id") == "0":
            continue
        if cell.get("parent") not in ids:
            raise ValueError("Missing cell parent")
        if cell.get("edge") == "1":
            for endpoint in ("source", "target"):
                if cell.get(endpoint) and cell.get(endpoint) not in ids:
                    raise ValueError("Dangling connector")
        elif cell.get("vertex") == "1":
            geo = cell.find("mxGeometry")
            if float(geo.get("width")) <= 0 or float(geo.get("height")) <= 0:
                raise ValueError("Non-positive shape size")
    return dict(cells=len(cells), groups=sum(cell.get("style") == "group;" for cell in cells),
                attached_connectors=sum(bool(cell.get("source")) for cell in cells),
                embedded_images=sum("image=data:" in cell.get("style", "") for cell in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    page, assets = build()
    counts = validate(page)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "retinagent_architecture.drawio"
    write_document(target, [page])
    manifest = dict(figure=target.name, scope="Task-dependent current-source architecture, not historical execution proof",
                    counts=counts, model_calls=0, raster_policy="Existing image bytes embedded without pixel modification",
                    image_sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in assets},
                    figure_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (args.output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(target)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
