"""Five editable motivation figures from archived predictions and real images.

No inference, fitting, image enhancement or manuscript-metric replacement.
PDF/SVG/PNG and native draw.io use the same geometry, at 183 mm print width.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/retinagent-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_clinical_decisions_figure import validate_counts
from build_paper_drawio import Page, write_document


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "equi-agent/docs/paper_figures/motivation_options"
CASE_SOURCE = ROOT / "equi-agent/docs/paper_figures/clinical_decisions/source_data.json"
INK, GRAY, LIGHT = "#20282D", "#59666E", "#CCD3D7"
TEAL, RED, BLUE = "#087F78", "#B84632", "#376B9C"
PAPER = "#F3F5F6"
W, H, PT = 1830, 1070, 72 / 254
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.unicode_minus": False})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(rows, id_col, truth_col, pred_col):
    result = {}
    for row in rows:
        key = Path(row[id_col]).name
        require(key not in result, f"Duplicate case {key}")
        require(row[truth_col] in {"0", "1"}, f"Invalid truth for {key}")
        require(row[pred_col] in {"0", "1"}, f"Invalid prediction for {key}")
        result[key] = row
    return result


def paired_foundations(retfound, visionfm):
    r = keyed(retfound, "image_id", "y_true", "y_pred")
    v = keyed(visionfm, "image_id", "y_true", "y_pred")
    require(set(r) == set(v), "Foundation cohorts differ; intersection is not sufficient")
    counts = Counter()
    for key in r:
        require(r[key]["y_true"] == v[key]["y_true"], f"Truth mismatch: {key}")
        a, b = r[key]["y_pred"] == r[key]["y_true"], v[key]["y_pred"] == v[key]["y_true"]
        counts[("both_correct" if b else "retfound_only") if a else
               ("visionfm_only" if b else "both_wrong")] += 1
    return {"n": len(r), **counts}, r, v


def calibration_rows(matrix):
    result = []
    require(all(str(r["y_true"]) in {"0", "1"} and 0 <= float(r["y_prob"]) <= 1
                for r in matrix), "Invalid calibration label or probability")
    for model in ("retfound_oct", "urfound_oct", "visionfm_oct"):
        rows = [r for r in matrix if r["model_name"] == model and
                .9 <= float(r["y_prob"]) <= 1]
        require(rows and all(r["split"] == "val" for r in rows), "Expected validation bins")
        require(len({r["image_id"] for r in rows}) == len(rows), "Duplicate bin cases")
        positive = sum(int(r["y_true"]) for r in rows)
        result.append(dict(model=model, n=len(rows), positives=positive,
                           mean_probability=sum(float(r["y_prob"]) for r in rows) / len(rows),
                           observed_positive_fraction=positive / len(rows)))
    return result


def load_evidence(inputs, image_root):
    source = json.loads(CASE_SOURCE.read_text())
    validate_counts(source["cohort"])
    hashes = {str(CASE_SOURCE.relative_to(ROOT)): digest(CASE_SOURCE)}

    def path(name, expected=None):
        candidate = inputs / name
        if not candidate.is_file():
            candidate = ROOT / name
        require(candidate.is_file(), f"Missing source: {name}")
        sha = digest(candidate)
        if expected:
            require(sha == expected, f"Changed source: {name}")
        hashes[name] = sha
        return candidate

    for name, metadata in source["sources"].items():
        path(name, metadata["sha256"])
    agent_path = "OphthalmicAgent/outputs/glaucoma_counterfactual_250/predictions.csv"
    baseline_path = "OphthalmicAgent/_extras/CSVs/raw_model_predictions/retfound_glaucoma_predictions.csv"
    trace_path = "OphthalmicAgent/outputs/glaucoma_counterfactual_250/counterfactual_traces.jsonl"
    agent = {Path(r["Filename"]).name: r for r in read_rows(path(agent_path))}
    baseline = keyed(read_rows(path(baseline_path)), "Filename", "Ground_Truth", "Prediction")
    with path(trace_path).open() as stream:
        traces = [json.loads(line) for line in stream]
    for e in source["examples"]:
        a, b = agent[e["case_id"]], baseline[e["case_id"]]
        require(int(a["Ground_Truth"]) == int(b["Ground_Truth"]) == e["truth"], "Case truth mismatch")
        require(int(a["Pred_GL"]) == e["agent"] and int(b["Prediction"]) == e["baseline"], "Case decisions changed")
        require(abs(float(b["Probability_Positive"]) - e["probability"]) < 1e-10, "Score changed")
        require(abs(float(b["Decision_Threshold"]) - e["threshold"]) < 1e-10, "Threshold changed")
        matched = [t for t in traces if Path(t["case_id"]).name == e["case_id"]]
        require(matched and matched[-1]["fingerprint"] == e["trace_fingerprint"], "Trace selection changed")
        scenarios = {s["name"]: s["diagnosis"] for s in matched[-1]["scenarios"]}
        require(scenarios == e["scenarios"], "Ablation scenarios changed")
        require(abs(float(matched[-1]["evidence"]["vertical_cup_to_disc_ratio"]) - e["cdr"]) < .001,
                "CDR estimate changed")

    names = ["equi-agent/outputs/predictions/fairvision_oct_retfound_test_thresholded.csv",
             "equi-agent/outputs/predictions/fairvision_glaucoma_visionfm_oct_test_thresholded.csv"]
    cohorts = [[r for r in read_rows(path(name)) if r["task"] == "glaucoma" and r["split"] == "test"]
               for name in names]
    paired, r, v = paired_foundations(*cohorts)
    require(paired["n"] == 3000, "Expected the saved 3,000-case foundation benchmark")
    examples = [dict(case_id=key, truth=int(r[key]["y_true"]),
                     retfound_probability=float(r[key]["y_prob"]), retfound_prediction=int(r[key]["y_pred"]),
                     visionfm_probability=float(v[key]["y_prob"]), visionfm_prediction=int(v[key]["y_pred"]))
                for key in ("data_08423.npz", "data_09154.npz")]
    require([e["truth"] for e in examples] == [1, 0] and
            all(e["retfound_prediction"] == 0 and e["visionfm_prediction"] == 1 for e in examples),
            "The opposite-model illustrative decisions have changed")
    matrix_path = "equi-agent/outputs/metrics/foundation_reliability_boundaries_glaucoma/model_case_matrix.csv"
    matrix = read_rows(path(matrix_path))
    require(all(r["task"] == "glaucoma" for r in matrix), "Calibration task is not glaucoma")
    bins = calibration_rows(matrix)
    images = {}
    for e in source["examples"] + examples:
        p = image_root / e["case_id"]
        require(p.is_file(), f"Missing downloaded image: {p}")
        sha = digest(p)
        if "npz_sha256" in e:
            require(e["npz_sha256"] == sha, f"Changed image: {p}")
        hashes[e["case_id"]] = sha
        with np.load(p, allow_pickle=False) as npz:
            require(int(np.asarray(npz["glaucoma"]).item()) == e["truth"], "Image label mismatch")
            oct_scan, slo = npz["oct_bscans"], npz["slo_fundus"]
            require(oct_scan.shape == (200, 200, 200) and slo.shape == (200, 200), "Unexpected image shape")
            require(oct_scan.dtype == np.uint8 and slo.dtype == np.uint8, "Expected native uint8 images")
            images[e["case_id"]] = {"OCT": oct_scan[100].copy(), "SLO": slo.copy()}
    evidence = dict(agent=source, foundations=paired, foundation_examples=examples,
                    calibration=bins, sha256=hashes,
                    image_display="OCT oct_bscans[100, :, :]; SLO slo_fundus; native 0-255 grayscale; no enhancement",
                    scope="Selected cases are illustrative; the three data cohorts are not interchangeable.")
    return evidence, images


class Canvas:
    """One geometry for publication exports and native editable draw.io cells."""

    def __init__(self, name, title, subtitle):
        self.name = name
        self.page = Page(name, W, H)
        self.fig = plt.figure(figsize=(W / 254, H / 254), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, W), ylim=(H, 0))
        self.ax.axis("off")
        self.text_bounds = []
        self.footprint = []
        self.text(title, 25, 17, 1780, 55, 36, bold=True)
        self.text(subtitle, 25, 79, 1780, 43, 25, GRAY)
        self.line([(25, 133), (1805, 133)], LIGHT, width=1.5)

    def text(self, value, x, y, w, h=42, size=27, color=INK, bold=False, align="left"):
        # Text is never auto-shrunk: a layout overflow fails the build.
        a = self.ax.text(x + (w / 2 if align == "center" else w if align == "right" else 0), y,
                         value, fontsize=size * PT, color=color, weight="bold" if bold else "normal",
                         ha=align, va="top", linespacing=1.3)
        self.text_bounds.append((value, a, (x, y, w, h)))
        self.page.cell(value, x, y, w, h,
                       f"text;html=0;whiteSpace=wrap;overflow=hidden;align={align};verticalAlign=top;"
                       f"fontFamily=Helvetica;fontSize={size};fontColor={color};fontStyle={int(bold)};"
                       "spacing=0;strokeColor=none;fillColor=none;")

    def rect(self, x, y, w, h, fill="none", edge=LIGHT, width=1.5):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=width * PT))
        self.page.cell("", x, y, w, h,
                       f"rounded=0;fillColor={fill};strokeColor={edge};strokeWidth={width};")

    def line(self, points, color=GRAY, width=2, arrow=False, dashed=False):
        self.page.line(points, color, dashed=dashed, arrow=arrow, width=width)
        xs, ys = zip(*points)
        self.ax.plot(xs, ys, color=color, lw=width * PT, ls=(0, (4, 3)) if dashed else "-", zorder=2)
        if arrow:
            self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                              mutation_scale=24 * PT, color=color, lw=width * PT,
                                              shrinkA=0, shrinkB=0, zorder=3))

    def dot(self, x, y, radius, color, fill=True):
        self.ax.add_patch(Circle((x, y), radius, facecolor=color if fill else "white",
                                 edgecolor=color, linewidth=2.5 * PT, zorder=5))
        self.page.cell("", x-radius, y-radius, radius*2, radius*2,
                       f"ellipse;fillColor={color if fill else '#FFFFFF'};strokeColor={color};strokeWidth=2.5;")

    def image(self, array, x, y, size, label):
        self.ax.imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="none",
                       extent=(x, x+size, y+size, y), aspect="equal", zorder=1)
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="PNG")
        data = base64.b64encode(stream.getvalue()).decode("ascii")
        self.page.cell("", x, y, size, size,
                       f"shape=image;imageAspect=0;aspect=fixed;image=data:image/png,{data};")
        self.text(label, x, y+size+10, size, 36, 24, GRAY)
        self.footprint.append((x, y, size, size))

    def mark(self, x, y, correct):
        if correct:
            self.line([(x, y+15), (x+10, y+25), (x+30, y)], TEAL, 4)
        else:
            self.line([(x, y), (x+25, y+25)], RED, 4)
            self.line([(x, y+25), (x+25, y)], RED, 4)

    def footer(self, text):
        self.text(text, 25, 1018, 1780, 38, 23, GRAY)

    def validate(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        errors = []
        rendered = []
        for value, artist, (x, y, w, h) in self.text_bounds:
            box = artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            left, right = sorted((box.x0, box.x1))
            top, bottom = sorted((box.y0, box.y1))
            if left < x-2 or right > x+w+2 or top < y-2 or bottom > y+h+2:
                errors.append(f"Text outside its box: {value!r}: {(left,top,right,bottom)} vs {(x,y,w,h)}")
            if left < 0 or right > W or top < 0 or bottom > H:
                errors.append(f"Text outside page: {value!r}")
            rendered.append((value, left, top, right, bottom))
        for i, a in enumerate(rendered):
            for b in rendered[i+1:]:
                if min(a[3], b[3]) - max(a[1], b[1]) > 2 and min(a[4], b[4]) - max(a[2], b[2]) > 2:
                    errors.append(f"Overlapping text: {a[0]!r} and {b[0]!r}")
            for x, y, w, h in self.footprint:
                if min(a[3], x+w) - max(a[1], x) > 2 and min(a[4], y+h) - max(a[2], y) > 2:
                    errors.append(f"Text obscures image: {a[0]!r}")
        require(not errors, "\n".join(errors))


def option_cases(evidence, images):
    p = Canvas("01_case_corrections", "A model score can miss the clinical picture",
               "Two selected FairVision cases: incorrect RETFound calls, correct final agent calls")
    for x, label, width in ((40, "Patient images", 570), (665, "Recorded AI evidence", 570),
                             (1320, "Decision", 450)):
        p.text(label, x, 156, width, 38, 25, GRAY, True)
    for index, (e, y) in enumerate(zip(evidence["agent"]["examples"], (215, 602))):
        p.rect(25, y-5, 1780, 363)
        p.image(images[e["case_id"]]["OCT"], 40, y+42, 267, "OCT")
        p.image(images[e["case_id"]]["SLO"], 329, y+42, 267, "SLO")
        truth = "Glaucoma" if e["truth"] else "Non-glaucoma"
        p.text(f"{'a' if index == 0 else 'b'}  Reference: {truth}", 40, y+2, 660, 35, 26, bold=True)
        p.text("Thin inferior rim;\nvertically enlarged cup" if index == 0 else
               "Focal OCT distortion;\nno marked SLO notch", 665, y+81, 540, 105, 30)
        p.text(f"Automated vCDR  {e['cdr']:.3f}", 665, y+225, 535, 45, 27, GRAY)
        p.line([(1207, y+177), (1280, y+177)], arrow=True)
        p.mark(1320, y+53, False)
        p.text("RETFound", 1370, y+47, 400, 38, 25, GRAY)
        p.text(f"{'Negative' if index == 0 else 'Positive'}  ({e['probability']:.1%})",
               1320, y+90, 440, 50, 33, RED, True)
        p.line([(1510, y+152), (1510, y+190)], LIGHT, arrow=True)
        p.mark(1320, y+217, True)
        p.text("RetinAgent", 1370, y+211, 400, 38, 25, GRAY)
        p.text(truth, 1320, y+258, 450, 50, 33, TEAL, True)
    p.footer("Illustrative cases, not aggregate benefit. Image findings are AI-reported, not clinician-adjudicated.")
    return p


def option_complementarity(evidence, images):
    p = Canvas("02_model_disagreement", "The more confident model is not always the right one",
               "Nearly identical model disagreements; opposite reference diagnoses")
    for e, x, letter in zip(evidence["foundation_examples"], (25, 932), ("a", "b")):
        p.rect(x, 158, 873, 660)
        p.text(f"{letter}  Reference: {'Glaucoma' if e['truth'] else 'Non-glaucoma'}", x+18, 176, 830, 42, 29, bold=True)
        p.image(images[e["case_id"]]["OCT"], x+18, 240, 380, "OCT: model input")
        p.image(images[e["case_id"]]["SLO"], x+435, 240, 380, "SLO: context only")
        for model, label, y in (("retfound", "RETFound", 690), ("visionfm", "VisionFM", 756)):
            score, vote = e[f"{model}_probability"], e[f"{model}_prediction"]
            correct = vote == e["truth"]
            color = TEAL if correct else RED
            p.text(label, x+18, y-8, 180, 38, 27)
            p.rect(x+217, y, 423, 26, PAPER, "none")
            p.rect(x+217, y, score*423, 26, color, "none")
            p.text(f"{score:.1%}", x+665, y-9, 122, 40, 30, color, True)
            p.mark(x+815, y, correct)
    p.text("Bars: predicted glaucoma probability", 25, 838, 900, 36, 23, GRAY)
    p.mark(1300, 840, True)
    p.text("Correct", 1345, 836, 180, 35, 23, GRAY)
    p.mark(1545, 840, False)
    p.text("Incorrect", 1590, 836, 200, 35, 23, GRAY)
    c = evidence["foundations"]
    p.text("Across 3,000 paired test cases", 25, 918, 665, 43, 29, bold=True)
    p.text(f"{c['retfound_only']} RETFound only correct", 720, 918, 520, 43, 29, BLUE, True)
    p.text(f"{c['visionfm_only']} VisionFM only correct", 1280, 918, 520, 43, 29, TEAL, True)
    p.footer("Foundation benchmark, not the agent cohort. Selected disagreements motivate case-specific trust; no oracle selection is evaluated.")
    return p


def option_calibration(evidence, images):
    p = Canvas("03_confidence_reliability", "A 95% model score is not a 95% guarantee",
               "FairVision validation cases with model scores between 90% and 100%")
    p.rect(305, 169, 36, 27, "white", BLUE, 3)
    p.text("Mean model score", 362, 163, 540, 42, 26, BLUE)
    p.rect(1050, 169, 36, 27, TEAL, TEAL)
    p.text("Observed glaucoma fraction", 1108, 163, 690, 42, 26, TEAL)
    bottom, height = 799, 506
    for tick in (0, .5, 1):
        y = bottom-height*tick
        p.line([(190, y), (1775, y)], LIGHT, 1.3)
        p.text(f"{tick:.0%}", 45, y-18, 110, 38, 24, GRAY, align="right")
    for row, x, name in zip(evidence["calibration"], (452, 980, 1508), ("RETFound", "URFound", "VisionFM")):
        for value, offset, color, fill in (
            (row["mean_probability"], -150, BLUE, "white"),
            (row["observed_positive_fraction"], 30, TEAL, TEAL),
        ):
            y = bottom-height*value
            p.rect(x+offset, y, 120, height*value, fill, color, 3)
            p.text(f"{value:.1%}", x+offset-25, y-58, 170, 44, 30, color, True, "center")
        p.text(name, x-230, 829, 460, 49, 34, bold=True, align="center")
        p.text(f"{row['positives']} / {row['n']} cases positive", x-230, 889, 460, 40, 25, GRAY, align="center")
    p.text("Trust needs an empirical reference, not just a raw score.", 25, 955, 1780, 51, 32, bold=True)
    p.footer("Different model-specific case sets; descriptive bin estimates, not a paired ranking or a validated routing rule.")
    return p


def option_ablation(evidence, images):
    e = evidence["agent"]["examples"][1]
    p = Canvas("04_evidence_ablation", "The report can change how a model score is interpreted",
               "One non-glaucoma case: recorded full-evidence and leave-one-evidence-out responses")
    p.image(images[e["case_id"]]["OCT"], 25, 170, 355, "OCT")
    p.image(images[e["case_id"]]["SLO"], 402, 170, 355, "SLO")
    p.text("RETFound", 835, 180, 400, 40, 26, GRAY)
    p.text(f"{e['probability']:.1%}", 835, 228, 425, 68, 53, RED, True)
    p.text("glaucoma probability", 835, 303, 440, 40, 26, GRAY)
    p.text("Automated vCDR", 1330, 180, 440, 40, 26, GRAY)
    p.text(f"{e['cdr']:.3f}", 1330, 228, 440, 68, 53, bold=True)
    p.text("Saved visual report", 835, 383, 915, 40, 26, GRAY)
    p.text("Focal OCT distortion;\nno marked SLO rim notch", 835, 435, 930, 94, 34)
    p.line([(915, 575), (915, 627)], LIGHT, 2)
    p.line([(305, 627), (1525, 627)], LIGHT, 2)
    for x in (305, 915, 1525):
        p.line([(x, 627), (x, 680)], GRAY, 2, arrow=True)
    for x, heading, detail, outcome, color in (
        (25, "Full evidence", "FM score + CDR + reports", "Non-glaucoma", TEAL),
        (635, "Without visual reports", "FM score + CDR", "Inconclusive", GRAY),
        (1245, "Without FM score", "CDR + reports", "Non-glaucoma", TEAL),
    ):
        p.rect(x, 696, 560, 247, "white", LIGHT, 1.5)
        p.text(heading, x+16, 718, 528, 46, 28, bold=True, align="center")
        p.text(detail, x+16, 783, 528, 40, 24, GRAY, align="center")
        p.text(outcome, x+16, 856, 528, 53, 36, color, True, "center")
    p.footer("Scenarios from one saved LLM response, not independent interventions. Full evidence also matches the final agent decision.")
    return p


def option_handoff(evidence, images):
    e = evidence["agent"]["examples"][0]
    p = Canvas("05_clinical_handoff", "From a model output to a reviewable patient assessment",
               "A recorded glaucoma case anchors the proposed clinician-in-the-loop workflow")
    for x, width, title in ((25, 580, "a  See the evidence"), (680, 480, "b  Reconcile the conflict"),
                             (1260, 545, "c  Review the assessment")):
        p.text(title, x, 172, width, 44, 29, bold=True)
    p.image(images[e["case_id"]]["OCT"], 25, 248, 270, "OCT")
    p.image(images[e["case_id"]]["SLO"], 310, 248, 270, "SLO")
    p.text("RETFound glaucoma score", 25, 609, 550, 43, 26, GRAY)
    p.text(f"{e['probability']:.1%}", 25, 663, 430, 77, 58, RED, True)
    p.text("Negative model call", 25, 766, 550, 44, 29, RED)
    p.line([(592, 445), (655, 445)], arrow=True)
    p.text("Structural evidence", 680, 270, 510, 44, 28, GRAY)
    p.text(f"vCDR  {e['cdr']:.3f}", 680, 327, 510, 59, 40, bold=True)
    p.text("Saved AI report", 680, 445, 510, 44, 28, GRAY)
    p.text("Thin inferior rim\nVertical cupping", 680, 502, 535, 118, 35)
    p.text("Low score does not\nsettle the disagreement", 680, 716, 535, 107, 30, BLUE, True)
    p.line([(1192, 445), (1247, 445)], arrow=True)
    p.rect(1260, 247, 545, 570, PAPER, LIGHT)
    p.text("RetinAgent", 1282, 272, 500, 43, 27, GRAY)
    p.text("Glaucoma", 1282, 330, 500, 71, 48, TEAL, True)
    p.text("Matches reference diagnosis", 1282, 414, 500, 73, 25, TEAL)
    p.line([(1282, 496), (1780, 496)], LIGHT, 1.5)
    p.text("Traceable evidence", 1282, 527, 500, 47, 30, bold=True)
    p.text("Original images\nRecorded model score\nAutomated structural estimate", 1282, 592, 500, 132, 25)
    p.line([(1532, 817), (1532, 865)], arrow=True, dashed=True)
    p.rect(1260, 880, 545, 90, "white", BLUE, 2)
    p.text("Ophthalmologist review", 1278, 905, 510, 42, 29, BLUE, True, "center")
    p.text("Recorded example", 25, 913, 1140, 44, 29, GRAY)
    p.footer("Clinician handoff is proposed, not evaluated here. AI-reported findings and automated CDR require clinical verification.")
    return p


BUILDERS = (option_cases, option_complementarity, option_calibration, option_ablation, option_handoff)


def captions(e):
    paired, bins = e["foundations"], e["calibration"]
    return f"""# Five Motivation Figure Options

Start with **motivation_review.pdf**, a five-page review pack. All pages are
183 mm wide. **motivation_options.drawio** contains the same five editable tabs.
Individual PDF, PNG, SVG and draw.io files use numbered matching names.
The existing author-edited clinical_decisions figure is not overwritten.

## 1. Case Corrections

**Recommended for the main image-led motivation.** Two retrospective FairVision
examples show incorrect RETFound decisions followed by correct final RetinAgent
decisions. Case data_07057.npz is glaucoma: RETFound probability 0.1938, saved
threshold 0.5, agent label 1. Case data_07062.npz is non-glaucoma: probability
0.7269, saved threshold 0.5, agent label 0. Image summaries are AI-reported;
vCDR values are automated estimates, not expert annotations. The two cases were
selected to illustrate opposite error directions, not to estimate benefit.

## 2. Model Disagreement

**Recommended for motivating multiple-model reliability.** Two held-out cases
have similar opposing RETFound/VisionFM scores but different correct sources.
In data_08423.npz (glaucoma), RETFound predicts negative and VisionFM positive.
In data_09154.npz (non-glaucoma), RETFound predicts negative and VisionFM positive.
OCT is the model input; SLO is shown only as anatomical context. In the full
3,000-case matched glaucoma test benchmark, only RETFound is correct on {paired['retfound_only']}
cases and only VisionFM on {paired['visionfm_only']}; both are correct on {paired['both_correct']:,} and both wrong on {paired['both_wrong']}.
These are saved thresholded decisions, not predictions rethresholded for the
figure. This benchmark is separate from the 249-case agent cohort. It motivates
case-specific trust but does not demonstrate that the agent chooses the right
FM, achieves oracle performance, or was tested on these illustrated cases.

## 3. Confidence and Reliability

For each model, validation cases scoring in [0.9, 1.0] are grouped separately.
Outlined bars show mean predicted glaucoma probability; filled bars show
the observed glaucoma fraction in that same model's bin. RETFound: {bins[0]['positives']}/{bins[0]['n']},
URFound: {bins[1]['positives']}/{bins[1]['n']}, VisionFM: {bins[2]['positives']}/{bins[2]['n']}. Bin frequencies are descriptive, have
different sample sizes and case composition, and are not a paired model ranking.
There is no fitted curve or inferential interval. This is validation evidence,
not independent evidence of improvement by a learned reliability selector.

## 4. Evidence Ablation

In case data_07062.npz, the stored response gives non-glaucoma for full evidence,
inconclusive without the OCT/SLO reports, and non-glaucoma without the FM score.
Other available inputs, including demographics, are held conceptually unchanged
and omitted from the diagram for readability. These scenarios were generated
within one response; they are not independently rerun ablations, ground-truth
causal effects, or proof of the final orchestrator's decision mechanism. The
full-evidence label agrees with the final agent CSV. The saved AI visual report
also notes image artifacts and uncertain subtle findings. Its anatomical
interpretation has not been clinically adjudicated.

## 5. Clinical Handoff

The observed inputs and agent decision for data_07057.npz anchor a proposed
clinical workflow: review the original images, reconcile an FM score with
reported structural evidence, then present a source-linked assessment for
ophthalmologist review. The arrangement is a conceptual case brief, not an
existing tested interface. The clinician handoff (dashed arrow) is proposed;
no clinician accuracy, adoption, time-saving, or patient benefit is established.

## Shared Image and Evidence Notes

Images are genuine downloaded FairVision arrays, displayed as stored: central
OCT slice oct_bscans[100, :, :] and slo_fundus, grayscale 0-255. No denoising,
contrast enhancement, lesion annotations, generated medical imagery, or
anatomically targeted slice selection is used. The report can refer to imaging
evidence beyond what the single displayed slice supports. All source hashes,
case labels, predictions and recorded ablation outcomes are checked on build.
The inference prompts, existing experiment files, and manuscript metrics are
not changed. No numbers are inferred from a rendered table.

## Full Paired Agent Results (Not a Motivation Claim)

The illustrative examples must not replace the full results. Among 249 matched
evaluable cases, baseline TN/FP/FN/TP are 108/17/45/79; agent values are
103/22/35/89. There are 23 corrected errors and 18 introduced errors: 17 missed
positives and 6 false positives corrected, but 7 missed positives and 11 false
positives introduced. Thus false alarms increase overall even though case b
illustrates an individual false-positive correction. One raw row (data_07199)
has nonbinary truth and is excluded from both. The source data retains all
these outcomes, not only the illustrated favorable transitions. The
author-reported RETFound worst-group F1 remains 0.6344 and is not recomputed.

## Rebuild

```bash
python equi-agent/scripts/build_motivation_options.py \\
  --inputs-root /path/to/extracted/retinagent_figure_inputs \\
  --image-root /path/to/downloaded/npz_files
python -m unittest discover -s equi-agent/tests -p test_motivation_options.py
```

The source_data.json file preserves the independent validation, foundation-test,
and paired-agent cohorts. Keep those distinctions when reusing any panel.
Rebuilding replaces generated options. After manual draw.io edits, use a new
--out-dir for regeneration so your edited variants are retained.
"""


def contact_sheet(out, canvases):
    cards = []
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 23) if Path(font_path).exists() else ImageFont.load_default()
    for c in canvases:
        img = Image.open(out / f"{c.name}.png").convert("RGB")
        img.thumbnail((990, 590), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (1030, 653), "white")
        draw = ImageDraw.Draw(card)
        draw.text((22, 12), c.name.replace("_", " "), fill=INK, font=font)
        card.paste(img, (20, 58))
        draw.rectangle((0, 0, 1029, 652), outline=LIGHT, width=2)
        cards.append(card)
    sheet = Image.new("RGB", (2105, 2030), "#E8ECEE")
    for i, card in enumerate(cards):
        sheet.paste(card, (15+(i % 2)*1060, 15+(i // 2)*675))
    sheet.save(out / "contact_sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    evidence, images = load_evidence(args.inputs_root, args.image_root)
    canvases = [builder(evidence, images) for builder in BUILDERS]
    for c in canvases:
        c.validate()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out_dir / "motivation_review.pdf", metadata={"Title": "Five motivation figure options"}) as pdf:
        for c in canvases:
            pdf.savefig(c.fig)
            for extension in ("pdf", "svg", "png"):
                c.fig.savefig(args.out_dir / f"{c.name}.{extension}", dpi=300)
            write_document(args.out_dir / f"{c.name}.drawio", [c.page])
    write_document(args.out_dir / "motivation_options.drawio", [c.page for c in canvases])
    (args.out_dir / "source_data.json").write_text(json.dumps(evidence, indent=2)+"\n")
    (args.out_dir / "README.md").write_text(captions(evidence))
    contact_sheet(args.out_dir, canvases)
    for c in canvases:
        plt.close(c.fig)
    print(f"Five variants; all text/image bounds checked. Review: {args.out_dir / 'motivation_review.pdf'}")


if __name__ == "__main__":
    main()
