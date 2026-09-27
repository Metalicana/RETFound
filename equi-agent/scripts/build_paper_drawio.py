"""Build editable paper figure layouts without inventing empirical results."""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "equi-agent/docs/paper_figures"
INK = "#253238"
BLUE = "#356887"
TEAL = "#087F78"
AMBER = "#AD692D"
GRAY = "#6A757C"


class Page:
    def __init__(self, name, width=1480, height=900):
        self.element = ET.Element("diagram", name=name, id=name.split()[0])
        graph = ET.SubElement(self.element, "mxGraphModel", dx="1480", dy="900",
                              grid="1", gridSize="10", guides="1", tooltips="1",
                              connect="1", arrows="1", fold="1", page="1",
                              pageScale="1", pageWidth=str(width), pageHeight=str(height),
                              math="0", shadow="0", background="#ffffff")
        self.root = ET.SubElement(graph, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.counter = 1

    def cell(self, text, x, y, w, h, style):
        self.counter += 1
        ident = f"c{self.counter}"
        node = ET.SubElement(self.root, "mxCell", id=ident, value=text,
                             style=style, vertex="1", parent="1")
        ET.SubElement(node, "mxGeometry", x=str(x), y=str(y), width=str(w),
                      height=str(h), **{"as": "geometry"})
        return ident

    def text(self, text, x, y, w, h, size=20, color=INK, bold=False, align="left"):
        return self.cell(text, x, y, w, h,
                         f"text;html=0;whiteSpace=wrap;overflow=hidden;align={align};"
                         f"verticalAlign=middle;fontFamily=Arial;fontSize={size};fontColor={color};"
                         f"fontStyle={1 if bold else 0};spacing=0;strokeColor=none;fillColor=none;")

    def box(self, heading, lines, x, y, w, h, color=BLUE, fill="#F3F7FA"):
        ident = self.cell("", x, y, w, h,
                          f"rounded=0;strokeColor={color};strokeWidth=1.5;fillColor={fill};")
        self.text(heading, x + 18, y + 15, w - 36, 50, 23, color, True)
        self.text(lines, x + 18, y + 75, w - 36, h - 88, 19)
        return ident

    def line(self, points, color=GRAY, dashed=False, arrow=False, width=1.6):
        self.counter += 1
        edge = ET.SubElement(self.root, "mxCell", id=f"c{self.counter}", parent="1", edge="1",
                             style=f"edgeStyle=none;rounded=0;html=0;strokeColor={color};"
                             f"strokeWidth={width};dashed={int(dashed)};"
                             f"endArrow={'block' if arrow else 'none'};endFill=1;")
        geo = ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
        for key, point in (("sourcePoint", points[0]), ("targetPoint", points[-1])):
            ET.SubElement(geo, "mxPoint", x=str(point[0]), y=str(point[1]), **{"as": key})
        if len(points) > 2:
            arr = ET.SubElement(geo, "Array", **{"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))

    def header(self, number, question, subtitle, pending=False):
        self.text(f"FIGURE {number}  /  RETINAGENT", 60, 35, 1200, 28, 17, TEAL, True)
        self.text(question, 60, 82, 1360, 52, 32, INK, True)
        self.text(subtitle, 60, 142, 1340, 50, 19, GRAY)
        if pending:
            self.text("LAYOUT ONLY / RESULTS NOT YET INSERTED", 60, 211, 1320, 30, 17, AMBER, True)

    def pending_panel(self, letter, heading, detail, x, y, w, h):
        self.text(f"{letter}  {heading}", x, y, w, 64, 24, INK, True)
        self.cell("", x, y + 82, w, h - 82,
                  "rounded=0;fillColor=#FAFBFC;strokeColor=#CBD1D4;dashed=1;strokeWidth=1;")
        self.text(detail, x + 24, y + 106, w - 48, h - 128, 21, GRAY)


def architecture():
    p = Page("02 Architecture")
    p.header(2, "A reliability layer between retinal models and the clinician",
             "Conceptual workflow: evidence informs the diagnosis; reliability qualifies how much each source is trusted.")
    for label, x in (("a  Evidence", 60), ("b  Reliability", 405),
                     ("c  Arbitration", 755), ("d  Clinical handoff", 1110)):
        p.text(label, x, 235, 290, 40, 23, INK, True)
    p.box("Case evidence", "Available CFP / SLO / OCT\nFM scores + modality\nClinical or structural findings\nMissingness + provenance",
          60, 350, 270, 230)
    p.box("Validation reference", "Task / model / subgroup\nError + calibration", 405, 645, 270, 145,
          TEAL, "#F0F8F6")
    p.box("Case-specific trust", "Match task + subgroup\nShrink sparse estimates\nFallback to global reliability\nFlag unstable evidence",
          405, 350, 270, 245, TEAL, "#F0F8F6")
    p.box("Patient context", "Age / sex / race, if available\nReliability context only", 60, 645, 270, 145,
          GRAY, "#F5F6F7")
    p.box("Evidence arbitration", "Combine findings + trust\nCheck conflicting evidence\nEvidence ablations, if enabled\nAssess need for escalation",
          755, 350, 270, 245, TEAL, "#F0F8F6")
    p.box("Prediction + evidence", "Provisional assessment\nSources and uncertainty", 1110, 340, 270, 145,
          TEAL, "#F0F8F6")
    p.box("Escalation + reasons", "Unreliable or conflicting\nevidence requires review", 1110, 530, 270, 145,
          AMBER, "#FCF6EF")
    p.cell("", 1110, 715, 270, 78, "rounded=0;fillColor=#F4F5F6;strokeColor=#6A757C;strokeWidth=1.5;")
    p.text("Ophthalmologist\nFinal decision", 1125, 725, 240, 58, 21, INK, True, "center")
    p.line([(330, 445), (405, 445)], BLUE, arrow=True)
    p.line([(540, 645), (540, 595)], TEAL, dashed=True, arrow=True)
    p.line([(330, 717), (365, 717), (365, 560), (405, 560)], GRAY, dashed=True, arrow=True)
    p.line([(675, 465), (755, 465)], TEAL, arrow=True)
    p.line([(195, 350), (195, 312), (890, 312), (890, 350)], BLUE, arrow=True)
    p.text("Disease evidence", 460, 282, 245, 24, 17, BLUE, align="center")
    p.line([(1025, 425), (1064, 425), (1064, 390), (1110, 390)], TEAL, arrow=True)
    p.line([(1025, 520), (1064, 520), (1064, 598), (1110, 598)], AMBER, arrow=True)
    p.line([(1380, 390), (1412, 390), (1412, 754), (1380, 754)], TEAL, arrow=True)
    p.line([(1245, 675), (1245, 715)], AMBER, arrow=True)
    p.text("Trust changes the interpretation of evidence; it is not a disease label.", 755, 645, 290, 115, 19, GRAY)
    p.line([(60, 835), (110, 835)], GRAY, dashed=True, arrow=True)
    p.text("Reliability conditioning", 120, 820, 255, 30, 17, GRAY)
    p.line([(460, 835), (510, 835)], BLUE, arrow=True)
    p.text("Evidence / decision flow", 520, 820, 270, 30, 17, GRAY)
    p.text("Sources and modules vary by experiment; no clinical benefit is assumed.", 840, 815, 550, 42, 17, GRAY)
    return p


def problem():
    p = Page("01 Heterogeneous reliability")
    p.header(1, "When do retinal models fail differently?",
             "Separate variation in average performance from case-level complementary errors.", True)
    panels = [
        ("a", "Across tasks", "Planned: aligned model-performance dot plots.\n\nAMD / DR / glaucoma\nSame held-out cohort per task\nFixed F1 convention\nPatient-bootstrap intervals"),
        ("b", "Across subgroups", "Planned: subgroup performance profiles.\n\nRace / sex / age groups\nDisplay subgroup support\nFlag undefined or sparse cells\nDo not infer winners from noise"),
        ("c", "Paired errors", "Planned: pairwise error-overlap map.\n\nJoin predictions on case IDs\nCommon cohort denominator\nDistinguish disagreement from\ncomplementary correct decisions"),
    ]
    for x, args in zip((60, 525, 990), panels):
        p.pending_panel(*args, x, 280, 425, 425)
    p.text("Required: matched test predictions, labels, subgroup metadata, model identities, thresholds and run provenance.",
           60, 750, 1320, 50, 22, INK)
    p.text("Evidence status: local metric files exist, but the final paper cohort and live-run identity need confirmation.",
           60, 822, 1340, 35, 18, AMBER)
    return p


def coverage():
    p = Page("03 Risk and coverage")
    p.header(3, "What performance is retained as cases are deferred?",
             "Compare every method on the same cases, using deployable ranking scores and a validation-fixed policy.", True)
    for x, letter, heading, ylabel in ((100, "a", "Risk versus coverage", "Accepted-case error rate"),
                                     (810, "b", "F1 versus coverage", "Accepted-case F1")):
        p.text(f"{letter}  {heading}", x, 275, 555, 45, 25, INK, True)
        p.line([(x, 365), (x, 650), (x + 520, 650)], INK, width=1.5)
        p.text(ylabel, x, 330, 550, 25, 18, GRAY)
        p.text("Fraction of cases accepted", x + 40, 665, 440, 35, 20, INK, align="center")
        p.text("Awaiting matched predictions\nand ranking scores.\nNo results plotted.",
               x + 55, 440, 410, 100, 23, GRAY, align="center")
    p.text("Planned comparators", 60, 727, 250, 32, 20, INK, True)
    p.text("Validation-selected single FM  /  Static ensemble  /  Confidence selection  /  Reliability routing  /  RetinAgent",
           60, 767, 1340, 45, 19, INK)
    p.text("Report coverage and class support alongside accepted-case F1. Keep deterministic arbitration separate from live LLM runs.",
           60, 832, 1340, 34, 17, AMBER)
    return p


def external():
    p = Page("04 External validation")
    p.header(4, "Where does evidence arbitration help or hurt?",
             "Paired changes against a prespecified baseline; retain negative findings and disclose adaptation and overlap.", True)
    p.text("a  Paired performance difference", 60, 280, 690, 48, 25, INK, True)
    p.line([(720, 348), (720, 732)], GRAY, width=1.5)
    p.line([(400, 732), (1035, 732)], GRAY, width=1.5)
    for i, name in enumerate(("FairVision30K", "Harvard GDP", "Drishti-GS", "REFUGE2", "PAPILA", "GAMMA")):
        y = 365 + i * 61
        p.text(name, 75, y, 245, 35, 23)
        p.line([(395, y + 18), (1040, y + 18)], "#E5E9EC", dashed=True, width=1)
    p.text("0", 700, 744, 40, 28, 20, align="center")
    p.text("Favors baseline", 375, 791, 280, 35, 21, GRAY)
    p.text("Favors RetinAgent", 800, 791, 290, 35, 21, TEAL)
    p.text("No estimates plotted", 1100, 355, 295, 38, 22, AMBER, True)
    p.text("Required per cohort:\n\nMatched case IDs\nIdentical metric definition\nForced-output metrics\nInvalid-output accounting\nPaired bootstrap intervals\nCheckpoint provenance",
           1100, 415, 295, 310, 21, INK)
    p.text("Main comparison uses all eligible cases. Accepted-only performance belongs in Figure 3 with coverage explicitly reported.",
           60, 849, 1320, 35, 17, GRAY)
    return p


def write_document(path, pages):
    document = ET.Element("mxfile", host="Electron", agent="RETFound paper figure builder",
                          version="31.5.2", type="device")
    for page in pages:
        document.append(page.element)
    ET.indent(document, space="  ")
    ET.ElementTree(document).write(path, encoding="utf-8", xml_declaration=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Open directly on the complete architecture, with quantitative layouts in other tabs.
    write_document(OUT / "retinagent_paper_figures.drawio", [architecture(), problem(), coverage(), external()])
    write_document(OUT / "figure_02_architecture.drawio", [architecture()])
    print(OUT / "retinagent_paper_figures.drawio")


if __name__ == "__main__":
    main()
