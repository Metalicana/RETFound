"""Build two concise, native draw.io workflows; no model or network calls.

Diagnostic and GDP progression workflows are separate pages. Implementation
qualifications belong in the accompanying caption, not inside the diagram.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from build_paper_drawio import Page, write_document

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "equi-agent/docs/paper_figures/architecture_reference"
INK = "#222629"
TEAL = "#397F87"
WINE = "#80344D"
PALE = "#F3F5F5"
WHITE = "#FFFFFF"
LINE = "#CCD1D3"
WIDTH, HEIGHT = 2400, 850


class Diagram(Page):
    def __init__(self, name="Diagnosis"):
        super().__init__(name, width=WIDTH, height=HEIGHT)
        self.element.set("id", "retinagent-" + name.lower().replace(" ", "-"))
        self.parent = "1"
        self.graph = self.element.find("mxGraphModel")
        self.graph.set("dx", str(WIDTH))
        self.graph.set("dy", str(HEIGHT))

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

    def rect(self, x, y, w, h, fill=WHITE, stroke="none", width=1.5):
        return self.cell("", x, y, w, h, f"rounded=1;absoluteArcSize=1;arcSize=8;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={width};")

    def oval(self, x, y, w, h, fill="none", stroke=INK, width=3):
        return self.cell("", x, y, w, h,
                         f"ellipse;fillColor={fill};strokeColor={stroke};strokeWidth={width};")

    def line(self, points, color=INK, width=2.5):
        super().line(points, color, width=width)
        self.node(f"c{self.counter}").set("parent", self.parent)

    def link(self, source, target, start=(1, .5), end=(0, .5), points=(), color=TEAL, width=4):
        self.counter += 1
        edge = ET.SubElement(self.root, "mxCell", id=f"c{self.counter}", parent="1", edge="1",
            source=source, target=target,
            style=f"edgeStyle=none;rounded=1;arcSize=16;html=0;strokeColor={color};"
            f"strokeWidth={width};endArrow=block;endSize=10;endFill=1;"
            f"exitX={start[0]};exitY={start[1]};exitDx=0;exitDy=0;exitPerimeter=0;"
            f"entryX={end[0]};entryY={end[1]};entryDx=0;entryDy=0;entryPerimeter=0;")
        geo = ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            array = ET.SubElement(geo, "Array", **{"as": "points"})
            for x, y in points:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))
        return edge.get("id")

    def label(self, text, x, y, w, h, size=26, color=INK, bold=False, align="center"):
        return self.text(text, x, y, w, h, size, color, bold, align)

    def icon(self, kind, x, y, size=70, color=INK):
        """Native editable glyphs; clinical images remain embedded bitmaps."""
        previous = self.parent
        group = self.group(x, y, size, size, kind + " icon")
        self.parent = group
        s = size / 80

        def ln(points, width=3):
            self.line([(a*s, b*s) for a, b in points], color, width*s)

        def ov(a, b, w, h, fill="none", width=3):
            self.oval(a*s, b*s, w*s, h*s, fill, color, width*s)

        def box(a, b, w, h):
            self.rect(a*s, b*s, w*s, h*s, "none", color, 3*s)

        if kind in {"person", "doctor"}:
            ov(25, 4, 30, 34)
            ln([(29, 36), (27, 44), (10, 52), (6, 71), (74, 71), (70, 52), (53, 44), (51, 36)])
            if kind == "doctor":
                ln([(27, 44), (39, 60), (53, 44)])
                ln([(26, 49), (25, 60), (20, 60), (20, 53)])
                ov(50, 57, 10, 10, width=2.5)
                ln([(51, 47), (55, 50), (55, 57)])
        elif kind == "robot":
            ln([(40, 8), (40, 18)])
            ov(36, 1, 8, 8)
            box(13, 19, 54, 37)
            box(5, 29, 8, 17)
            box(67, 29, 8, 17)
            ov(25, 29, 6, 7, color, 1)
            ov(49, 29, 6, 7, color, 1)
            ln([(28, 46), (52, 46)])
            ln([(40, 56), (40, 63), (20, 63), (13, 75)])
            ln([(40, 63), (60, 63), (67, 75)])
        elif kind == "network":
            nodes = [(10, 14), (10, 40), (10, 66), (39, 10), (39, 36), (39, 63), (70, 25), (70, 54)]
            for a, b in ((0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 6), (4, 6), (4, 7), (5, 7)):
                ln([nodes[a], nodes[b]], 1.8)
            for a, b in nodes:
                ov(a-5, b-5, 10, 10, PALE, 2.5)
        elif kind in {"table", "grid"}:
            box(9, 7, 62, 64)
            ln([(9, 23), (71, 23)])
            ln([(27, 23), (27, 71)])
            for b, length in ((35, 26), (47, 18), (59, 31)):
                ln([(36, b), (36+length, b)], 4)
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
        else:
            raise ValueError(f"Unknown icon {kind}")
        self.parent = previous
        return group

    def agent(self, x, y, w, title, diameter=110, kind="robot", font=27):
        group = self.group(x, y, w, diameter+94, title.replace("\n", " "))
        previous, self.parent = self.parent, group
        cx = (w-diameter)/2
        circle = self.oval(cx, 0, diameter, diameter, WINE, WINE, 0)
        self.node(circle).set("data-role", "agent-port")
        self.icon(kind, cx+diameter*.22, diameter*.20, diameter*.56, WHITE)
        self.label(title, 0, diameter+14, w, 72, font, INK, True)
        self.parent = previous
        return circle


def embedded_images():
    path = ROOT / "equi-agent/docs/paper_figures/clinical_decisions/clinical_decisions.drawio"
    tree = ET.parse(path)
    images = {}
    for name, ident in (("oct", "c5"), ("slo", "c7")):
        cell = next(cell for cell in tree.iter("mxCell") if cell.get("id") == ident)
        images[name] = next(part.split("=", 1)[1] for part in cell.get("style").split(";") if part.startswith("image="))
    fundus = ROOT / "equi-agent/docs/paper_figures/agentic_walkthrough/assets/fundus.png"
    images["cfp"] = "data:image/png," + base64.b64encode(fundus.read_bytes()).decode()
    return images, [path, fundus]


def build(progression=False):
    p = Diagram("GDP progression" if progression else "Diagnosis")
    images, asset_paths = embedded_images()
    p.label("RetinAgent" + (" / GDP progression" if progression else " / Diagnosis"),
            50, 28, 1000, 56, 39, INK, True, "left")
    for title, x, w in (("1  Inputs", 50, 380), ("2  Evidence", 530, 610),
                        ("3  Reasoning", 1230, 650), ("4  Output", 2040, 310)):
        p.label(title, x, y=123, w=w, h=46, size=30, color=INK, bold=True, align="left")
        p.line([(x, 187), (x+w, 187)], LINE, 1.5)

    # A single patient-input interface feeds evidence assembly, not every tool.
    inputs = p.group(50, 250, 380, 450, "Patient inputs")
    p.parent = inputs
    p.icon("person", 140, 0, 94)
    p.label("Demographics", 0, 103, 380, 40, 28, INK, True)
    if not progression:
        p.label("Retinal imaging", 0, 190, 380, 38, 27, INK, True)
        for key, x in (("oct", 0), ("slo", 132), ("cfp", 264)):
            p.cell("", x, 246, 116, 128, f"shape=image;imageAspect=1;image={images[key]};"
                   "fillColor=#101515;strokeColor=none;")
            p.label(key.upper(), x, 386, 116, 35, 23)
    else:
        for y, kind, name in ((206, "grid", "Baseline imaging"), (328, "table", "Baseline visual field")):
            p.icon(kind, 12, y, 62, TEAL)
            p.label(name, 92, y, 280, 70, 26, INK, True, "left")
    p.parent = "1"

    packet = p.group(530, 246, 610, 480, "Case evidence")
    p.parent = packet
    p.rect(0, 0, 610, 480, PALE)
    if progression:
        for x, title, kind in ((15, "Bio-Profiler", "doctor"), (215, "Structural\nspecialists", "robot"),
                               (415, "Functional\nspecialist", "robot")):
            p.agent(x, 51, 180, title, 94, kind, 24)
        tools = ((105, "network", "Helper models"), (335, "table", "Reliability lookup"))
    else:
        p.agent(58, 45, 230, "Bio-Profiler", kind="doctor")
        p.agent(323, 45, 230, "Vision\nspecialists")
        tools = ((15, "network", "RETFound"), (215, "cdr", "CDR tool"), (415, "table", "Reliability\nlookup"))
    p.line([(32, 277), (578, 277)], LINE, 1.5)
    for x, kind, name in tools:
        p.icon(kind, x+58, 310, 64, TEAL)
        p.label(name, x, 388, 180, 64, 25)
    p.parent = "1"

    counterfactual = p.agent(1220, 409, 265, "Counterfactual\nAgent", 132, font=28)
    orchestrator = p.agent(1630, 389, 295, "Ophthalmologist\nOrchestrator", 172, "doctor", 30)
    output = p.group(2080, 412, 250, 238, "Progression forecast" if progression else "Diagnosis output")
    p.parent = output
    p.icon("document", 69, 0, 112, TEAL)
    p.label("Six endpoint\nforecasts" if progression else "Diagnosis\n+ rationale", 0, 141, 250, 83, 30, INK, True)
    p.parent = "1"

    p.link(inputs, packet, end=(0, 229/480))
    p.link(packet, counterfactual, start=(1, 229/480))
    p.link(counterfactual, orchestrator)
    p.label("Audit", 1485, 428, 125, 32, 23, TEAL)
    p.link(orchestrator, output, end=(.28, 63/238))
    # The orchestrator receives original evidence as well as the audit, not only
    # the preceding agent's answer. This is the sole bypass in both workflows.
    p.link(packet, orchestrator, start=(1, 65/480), end=(.5, 0),
           points=((1777.5, 311),), width=3)
    p.label("Case evidence", 1280, 269, 320, 32, 23, TEAL)
    return p, asset_paths if not progression else []


def bounds(page, cell):
    geometry = cell.find("mxGeometry")
    x, y = float(geometry.get("x", 0)), float(geometry.get("y", 0))
    parent = page.node(cell.get("parent"))
    while parent is not None and parent.get("id") not in {"0", "1"}:
        g = parent.find("mxGeometry")
        x, y = x+float(g.get("x", 0)), y+float(g.get("y", 0))
        parent = page.node(parent.get("parent"))
    return x, y, float(geometry.get("width")), float(geometry.get("height"))


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
            x, y, w, h = bounds(page, cell)
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > WIDTH or y+h > HEIGHT:
                raise ValueError(f"Invalid shape bounds: {cell.get('value') or cell.get('id')}")
    texts = [c for c in cells if c.get("value")]
    for i, cell in enumerate(texts):
        x, y, w, h = bounds(page, cell)
        for other in texts[i+1:]:
            a, b, c, d = bounds(page, other)
            if min(x+w, a+c)-max(x, a) > 1 and min(y+h, b+d)-max(y, b) > 1:
                raise ValueError(f"Overlapping labels: {cell.get('value')} / {other.get('value')}")
    return dict(cells=len(cells), groups=sum(cell.get("style") == "group;" for cell in cells),
        attached_connectors=sum(bool(cell.get("source")) for cell in cells),
        embedded_images=sum("image=data:" in cell.get("style", "") for cell in cells),
        label_words=sum(len(c.get("value", "").split()) for c in cells))


def archive_existing(out):
    target, manifest = out / "retinagent_architecture.drawio", out / "provenance.json"
    if not target.exists():
        return
    if not manifest.exists():
        raise ValueError("Existing master has no provenance; refusing to overwrite")
    previous = json.loads(manifest.read_text())
    if previous["figure_sha256"] != hashlib.sha256(target.read_bytes()).hexdigest():
        raise ValueError("Master has manual edits; use a new --output-dir")
    if previous.get("layout_version") == "concise_v2":
        return
    archive = out / "previous_detailed"
    archive.mkdir(exist_ok=True)
    for name in ("retinagent_architecture.drawio", "retinagent_architecture.png", "retinagent_architecture.pdf",
                 "README.txt", "provenance.json"):
        source, destination = out / name, archive / name
        if source.exists():
            if destination.exists() and source.read_bytes() != destination.read_bytes():
                raise ValueError(f"Different archive already exists: {destination}")
            shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    diagnosis, assets = build()
    progression, _ = build(progression=True)
    pages = [diagnosis, progression]
    counts = {page.element.get("name"): validate(page) for page in pages}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_existing(args.output_dir)
    target = args.output_dir / "retinagent_architecture.drawio"
    write_document(target, pages)
    manifest = dict(figure=target.name, layout_version="concise_v2", pages=counts,
        scope="Separate diagnostic and staged GDP workflows; qualifications in caption, not historical execution proof",
        model_calls=0, raster_policy="Existing image bytes embedded without pixel modification",
        image_sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in assets},
        figure_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (args.output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(target)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
