"""Editable framework illustration with a verified Drishti-GS CFP example.

Reads transferred outputs only. Never calls an LLM or runs model inference.
The framework includes optional modules; the case trace used RETFound only.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from build_motivation_options import Canvas, INK, GRAY, LIGHT, TEAL, BLUE, PT, plt
from matplotlib.patches import FancyBboxPatch, Polygon
from build_paper_drawio import write_document
from export_figure_cdr_case import mask_measurements, overlay_mask


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "equi-agent/docs/paper_figures/agentic_walkthrough"
CASE = "drishtiGS_053"
NAMES = {"retfound": "RETFound", "mirage": "MIRAGE", "ret_clip": "RET-CLIP",
         "retizero": "RetiZero", "urfound": "URFound"}
DISC, CUP, GOLD = "#00AFC5", "#D94DAB", "#987127"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify_segmentation(folder, source_image):
    metadata = json.loads((folder / "measurements.json").read_text())
    require(metadata["case_id"] == CASE, "Wrong exported case")
    for name in ("fundus.png", "mask.png", "overlay.png", "model_input_equalized.png"):
        require(digest(folder / name) == metadata["files"][name], f"Export checksum mismatch: {name}")
    require(digest(source_image) == metadata["source_image_sha256"], "Original fundus file differs")
    original = Image.open(source_image).convert("RGB")
    fundus = Image.open(folder / "fundus.png").convert("RGB")
    require(original.size == fundus.size and np.array_equal(original, fundus), "Fundus pixels differ")
    require(hashlib.sha256(np.asarray(fundus).tobytes()).hexdigest() == metadata["rgb_pixel_sha256"],
            "Fundus pixel hash differs")
    mask = np.asarray(Image.open(folder / "mask.png"))
    require(mask.shape == (fundus.height, fundus.width), "Mask dimensions differ")
    derived = mask_measurements(mask)
    for key, value in derived.items():
        require(np.allclose(value, metadata[key], rtol=0, atol=1e-10), f"Incorrect mask measurement: {key}")
    overlay = Image.open(folder / "overlay.png").convert("RGB")
    require(np.array_equal(overlay, overlay_mask(fundus, mask)), "Overlay does not match the real mask")
    return metadata


def collect_case(inputs):
    paths = {}
    foundations = {}
    for model in NAMES:
        name = f"equi-agent/outputs/benchmarks/drishti_glaucoma_foundations_v1/{model}/predictions_test.csv"
        path = inputs / name
        rows = [r for r in csv_rows(path) if r["case_id"] == CASE and r["split"] == "test"]
        require(len(rows) == 1, f"Missing or duplicated foundation case: {model}")
        row = rows[0]
        require(row["y_true"] == "1" and row["dataset"] == "drishti", "Case label/dataset mismatch")
        foundations[model] = {"probability": float(row["y_prob"]), "threshold": float(row["threshold"]),
                              "prediction": int(row["y_pred"])}
        paths[name] = digest(path)
    name = "OphthalmicAgent/outputs/drishti/agentic_retfound_cfp_v1/predictions.csv"
    path = inputs / name
    rows = [r for r in csv_rows(path) if r["case_id"] == CASE and r["split"] == "test" and r["Pred_GL"] in {"0", "1"}]
    require(len(rows) == 1, "Expected one valid saved agent attempt; review duplicates explicitly")
    row = rows[0]
    paths[name] = digest(path)
    trace = json.loads(row["Counterfactual_Trace"])
    return dict(case_id=CASE, dataset="Drishti-GS", truth=int(row["Ground_Truth"]),
                foundations=foundations, agent=dict(raw_probability=float(row["Raw_RETFound_Probability"]),
                    input_probability_pct=float(row["Agent_RETFound_Probability_Pct"]),
                    vertical_cdr=float(row["Vertical_CDR"]), prediction=int(row["Pred_GL"]),
                    visual_report=row["CFP_Report"], decision=row["Agentic_Decision"],
                    scenarios={s["name"]: s["diagnosis"] for s in trace["scenarios"]}),
                source_csv_sha256=paths,
                scope="Standalone scores for five FMs; recorded agent uses RETFound, AI visual report and vertical CDR only.")


def validate_case(case, measurements):
    require(case["case_id"] == measurements["case_id"] == CASE, "Case IDs differ")
    require(set(case["foundations"]) == set(NAMES), "Incomplete model bank")
    for model, row in case["foundations"].items():
        require(0 <= row["probability"] <= 1 and 0 <= row["threshold"] <= 1, f"Invalid score: {model}")
        require(int(row["probability"] >= row["threshold"]) == row["prediction"], "Threshold mismatch")
    agent = case["agent"]
    require(abs(agent["raw_probability"] - case["foundations"]["retfound"]["probability"]) < 1e-10,
            "RETFound standalone score and agent input use different predictions")
    require(abs(agent["vertical_cdr"] - measurements["vertical_cdr"]) < 1e-6,
            "New segmentation does not reproduce historical CDR; review before illustrating this trace")
    require(agent["prediction"] == 1 and case["truth"] == 1, "Recorded example decision changed")
    require(agent["scenarios"] == {"full_evidence": 1, "without_retfound_probability": 1,
                                  "without_visual_interpretation": 0, "without_cdr_tool": 1},
            "Recorded evidence scenarios changed")
    require("inferior" in agent["visual_report"].lower() and "vertically" in agent["visual_report"].lower(),
            "Illustrated visual-report paraphrase not supported")


def photo(c, picture, x, y, width, border=True):
    height = width * picture.height / picture.width
    c.ax.imshow(np.asarray(picture), extent=(x, x+width, y+height, y), interpolation="none", aspect="equal")
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    c.page.cell("", x, y, width, height, f"shape=image;imageAspect=0;image=data:image/png,{data};")
    c.footprint.append((x, y, width, height))
    if border:
        c.rect(x, y, width, height, edge=INK, width=1)
    return height


def robot(c, x, y, size=70, color=BLUE):
    """Small native vector agent glyph, editable in draw.io."""
    s = size / 70
    c.line([(x+35*s, y+14*s), (x+35*s, y+5*s)], color, 2.5)
    c.dot(x+35*s, y+3*s, 3*s, color)
    c.rect(x+8*s, y+16*s, 54*s, 37*s, "#FFFFFF", color, 2.5)
    c.rect(x+1*s, y+26*s, 7*s, 16*s, color, color, 1)
    c.rect(x+62*s, y+26*s, 7*s, 16*s, color, color, 1)
    c.dot(x+24*s, y+31*s, 3.3*s, color)
    c.dot(x+46*s, y+31*s, 3.3*s, color)
    c.line([(x+26*s, y+43*s), (x+44*s, y+43*s)], color, 2)
    c.line([(x+20*s, y+59*s), (x+50*s, y+59*s)], color, 2.5)
    c.line([(x+35*s, y+53*s), (x+35*s, y+59*s)], color, 2.5)
    c.line([(x+20*s, y+59*s), (x+12*s, y+69*s)], color, 2.5)
    c.line([(x+50*s, y+59*s), (x+58*s, y+69*s)], color, 2.5)


def human(c, x, y, color=INK):
    c.dot(x+31, y+16, 14, color, fill=False)
    c.line([(x+1, y+72), (x+5, y+46), (x+19, y+36), (x+31, y+59),
            (x+43, y+36), (x+57, y+46), (x+61, y+72)], color, 2.5)
    c.line([(x+31, y+59), (x+31, y+73)], color, 2)
    c.line([(x+45, y+51), (x+45, y+64)], TEAL, 2.5)
    c.line([(x+39, y+57), (x+51, y+57)], TEAL, 2.5)


def document(c, x, y, color, width=37, height=48):
    c.rect(x, y, width, height, "#FFFFFF", color, 1.8)
    for offset in (12, 22, 32):
        c.line([(x+7, y+offset), (x+width-7, y+offset)], color, 1.8)


def bubble(c, text, x, y, w, h, color=BLUE, fill="#F2F6FA", size=26, bold=False,
           tail="left", name=None):
    """Editable rounded speech bubble; prose stays inside a measured text box."""
    center = y+min(h*.5, 39)
    if tail == "left":
        points = [(x-17, center+10), (x+4, center-13), (x+4, center+13)]
        c.ax.add_patch(Polygon(points, facecolor=fill, edgecolor=color, linewidth=1.5*PT))
        c.page.cell("", x-17, center-13, 21, 26,
                    f"triangle;direction=west;fillColor={fill};strokeColor={color};strokeWidth=1.5;")
    else:
        points = [(x+25, y-15), (x+22, y+4), (x+48, y+4)]
        c.ax.add_patch(Polygon(points, facecolor=fill, edgecolor=color, linewidth=1.5*PT))
        c.page.cell("", x+22, y-15, 26, 19,
                    f"triangle;direction=north;fillColor={fill};strokeColor={color};strokeWidth=1.5;")
    c.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                                 facecolor=fill, edgecolor=color, linewidth=1.5*PT))
    ident = c.page.cell("", x, y, w, h,
                        f"rounded=1;arcSize=8;absoluteArcSize=1;fillColor={fill};"
                        f"strokeColor={color};strokeWidth=1.5;")
    if name:
        c.page.root.find(f"mxCell[@id='{ident}']").set("data-bubble", name)
    if tail == "left":
        c.rect(x-1, center-10, 3, 20, fill, "none", 0)
    else:
        c.rect(x+24, y-1, 22, 3, fill, "none", 0)
    lines = len(text.splitlines())
    text_height = size*(1.3*lines+.1)
    c.text(text, x+14, y+(h-text_height)/2+2, w-28, text_height+5, size, color, bold, align="center")


def make_figure(case, measurements, assets):
    c = Canvas("RetinAgent walkthrough", "RetinAgent: supporting a glaucoma assessment", "", height=1110)
    for x, w, title in ((25, 390, "1  Clinical question"), (470, 745, "2  Image analysis"),
                        (1280, 525, "3  Assessment for review")):
        c.text(title, x, 105, w, 39, 29, bold=True)
        c.line([(x, 153), (x+w, 153)], INK, 1.6)

    human(c, 42, 195)
    bubble(c, "Could this be\nglaucoma?", 136, 177, 272, 110,
           color=INK, fill="#F5F6F7", size=26, name="clinician-request")
    original = Image.open(assets / "fundus.png").convert("RGB")
    overlay = Image.open(assets / "overlay.png").convert("RGB")
    c.text("Fundus photograph", 42, 322, 360, 36, 25, bold=True)
    photo(c, original, 42, 367, 360)
    c.text("No patient history or\nvisual field was supplied.", 42, 714, 360, 77, 25, GRAY)
    c.line([(403, 519), (444, 519), (444, 223), (464, 223)], BLUE, 2.5, arrow=True)

    # Plain-language action labels; implementation names belong in the caption.
    robot(c, 474, 182, color=BLUE)
    c.text("Read the image", 566, 195, 530, 44, 33, BLUE, True)
    c.page.root[-1].set("data-module", "Vision Agent")
    c.line([(841, 250), (841, 281), (647, 281), (647, 308)], BLUE, 1.8, arrow=True)
    c.line([(841, 281), (1042, 281), (1042, 308)], BLUE, 1.8, arrow=True)
    retfound = case["foundations"]["retfound"]
    c.text("RETFound model score", 474, 322, 358, 36, 25, bold=True)
    c.text(f"{retfound['probability']*100:.0f}%", 474, 373, 335, 55, 43, BLUE, True)
    c.rect(474, 447, 330, 16, "#E9EEF1", "none", 0)
    c.rect(474, 447, 330*retfound["probability"], 16, BLUE, "none", 0)
    threshold_x = 474+330*retfound["threshold"]
    c.line([(threshold_x, 438), (threshold_x, 473)], GRAY, 2)
    c.text(f"Model cutoff: {retfound['threshold']*100:.0f}%", 474, 488, 344, 35, 24, GRAY)
    c.text("Model vote: non-glaucoma", 474, 537, 356, 34, 24, GRAY)
    c.text("AI's image report", 474, 596, 350, 36, 25, BLUE, True)
    bubble(c, "Vertical cupping;\nsuspected rim thinning.", 477, 648, 346, 98,
           tail="top", size=25, name="image-report")

    c.text("Disc and cup", 875, 322, 337, 36, 25, bold=True)
    x0, y0, x1, y1 = measurements["disc_bbox_xyxy"]
    side = int(max(x1-x0, y1-y0)*1.45)
    cx, cy = (x0+x1)//2, (y0+y1)//2
    crop_box = (cx-side//2, cy-side//2, cx-side//2+side, cy-side//2+side)
    require(0 <= crop_box[0] < crop_box[2] <= overlay.width and
            0 <= crop_box[1] < crop_box[3] <= overlay.height, "Disc crop outside image")
    photo(c, overlay.crop(crop_box), 875, 367, 337)
    scale = 360/original.width
    c.rect(42+crop_box[0]*scale, 367+crop_box[1]*scale, side*scale, side*scale,
           edge="#F2F3F4", width=1.8)
    c.line([(880, 730), (913, 730)], DISC, 4)
    c.text("Disc", 925, 718, 90, 31, 23)
    c.line([(1041, 730), (1074, 730)], CUP, 4)
    c.text("Cup", 1086, 718, 90, 31, 23)
    c.text("Vertical cup-to-disc ratio", 875, 762, 337, 32, 22, TEAL, True, "center")
    c.text(f"{measurements['vertical_cdr']:.3f}", 875, 798, 337, 40, 33, TEAL, True, "center")
    c.text("The model vote and image findings disagree.", 474, 858, 744, 35, 26, GOLD, True)

    # The evidence check is a saved hypothetical-removal scenario, not a new patient diagnosis.
    robot(c, 1291, 182, color=TEAL)
    c.text("Check the reasoning", 1380, 194, 421, 43, 29, TEAL, True)
    c.page.root[-1].set("data-module", "Counterfactual Agent")
    bubble(c, "The answer changes if\nthe image report is removed.", 1301, 289, 480, 116,
           TEAL, "#F0F8F6", 27, tail="top", name="evidence-ablation")
    c.text("The AI says the image findings are key.", 1291, 432, 501, 36, 24, GRAY)
    c.line([(1216, 223), (1268, 223)], BLUE, 2.5, arrow=True)
    c.line([(1540, 484), (1540, 509)], TEAL, 2.5, arrow=True)
    robot(c, 1291, 519, color=TEAL)
    c.text("Explain the assessment", 1380, 532, 421, 42, 29, TEAL, True)
    c.page.root[-1].set("data-module", "Orchestrator")
    bubble(c, "AI assessment: glaucoma\nCupping + rim thinning\nPhotograph-only evidence", 1301, 617, 480, 150,
           TEAL, "#EDF7F4", 28, tail="top", name="recorded-assessment")
    c.line([(1540, 768), (1540, 794)], TEAL, 2.5, arrow=True)
    human(c, 1297, 806)
    c.text("Doctor reviews evidence\nand makes the final decision.", 1380, 804, 421, 78, 24, bold=True)

    # Separate capabilities from the case actually executed. No optional path is drawn into its result.
    c.line([(25, 908), (1805, 908)], LIGHT, 1.5)
    c.text("Additional framework tools (not used in this case)", 25, 927, 1780, 38, 26, GRAY, True)
    extensions = [
        (25, "Summarize\npatient history", "BioProfiler", GOLD),
        (325, "Compare other AI\nimage models", "Model tools", BLUE),
        (625, "Check reliability\nby patient group", "Equity Agent", TEAL),
        (925, "Search diagnostic\nliterature (PubMed)", "Guidelines Agent", GOLD),
        (1225, "Interpret\nvisual-field tests", "Functional Agent", BLUE),
        (1525, "Flag missing or\nconflicting data", "Safety Agent", GOLD),
    ]
    for x, action, name, color in extensions:
        if name == "Model tools":
            for offset in (0, 13, 26):
                document(c, x+offset, 986+offset*.3, color, 31, 40)
        else:
            robot(c, x, 982, 48, color)
        c.text(action, x+60, 982, 223, 75, 23, color)
        # Machine-readable provenance preserves technical names without forcing clinicians to decode them.
        c.page.root[-1].set("data-module", name)
    c.text("Recorded case: RETFound + AI image report + cup-to-disc ratio. Tool names and full trace are in the caption.",
           25, 1080, 1780, 27, 22, GRAY)
    c.validate()
    return c, crop_box


CAPTION = """# RetinAgent: Supporting a Glaucoma Assessment

**Figure caption.** A clinician-facing walkthrough using recorded outputs for
Drishti-GS case drishtiGS_053. **1**, A clinician asks whether a fundus photograph
suggests glaucoma. No history or visual-field evidence was supplied to this
image-only run. **2**, the RETFound raw score (24.040%) is below its
validation-selected threshold (53.015%), giving a non-glaucoma model vote. The
AI-generated image report instead describes vertical cupping and suspected rim
thinning. The segmented vertical cup-to-disc ratio is 0.690. The white square
identifies the enlarged region; cyan and magenta contours are predicted disc
and cup boundaries, not expert annotations. **3**, the saved evidence-removal
reasoning reports a different label when the image report is excluded. The
agent's final recorded label is glaucoma, with its reasoning emphasizing the
structural report. A short brief presents the AI assessment, supporting
findings and image-only limitation for clinician review.

The request and formatted clinical brief are illustrative paraphrases, not an
actual doctor conversation or tested interface. This figure has not been
validated in a clinician comprehension or reader study. It does not establish
clinical benefit or show that the agent's self-reported reasoning is a causal
explanation. The data and the full four-scenario trace are retained below.

## Plain-Language Roles

| Figure action | Implementation role |
|---|---|
| Read the image | Vision Agent and its image/model/segmentation tools |
| Check the reasoning | Counterfactual Agent: hypothetical evidence removal within a saved response |
| Explain the assessment | Orchestrator: reconcile the supplied evidence |
| Summarize patient history | BioProfiler |
| Compare other AI image models | Compatible foundation-model adapters / cached model outputs |
| Check reliability by patient group | Equity Agent and validation-derived reliability priors |
| Search diagnostic literature (PubMed) | Guidelines Agent; PubMed and web retrieval |
| Interpret visual-field tests | Functional Interpretation Agent |
| Flag missing or conflicting data | Safety Agent |

The bottom strip is deliberately separate from the worked case. Those
capabilities were not invoked in this saved run and are not shown as contributing
to its result. Their availability and exact behavior depend on the runner.

## What Was Actually Recorded

- The saved external-case run used **RETFound, an AI-generated fundus report and
  vertical CDR**. It did not use the other four foundation models listed below,
  BioProfiler, Equity, Guidelines, Functional or Safety agents.
- The five raw foundation-model scores retained in the source data and table
  below are separate benchmark predictions on the same image. Only RETFound
  appears in the case's main path; the others were removed from that path to
  avoid implying they contributed to the saved assessment.
  The RETFound raw score agrees exactly with the saved agent record. The agent
  consumed the threshold-aligned RETFound score (21.905%), not the raw probability
  (24.040%). Raw model scores are neither reliability scores nor directly
  interchangeable model thresholds.
- The four evidence-ablation labels are scenarios in one saved LLM response,
  not independently rerun interventions or causal evidence. Label 0 is shown
  as non-glaucoma, although its saved reasoning describes an equivocal case.
- The saved trace incorrectly calls the visual report "human" in places.
  It was produced by an LLM; this illustration explicitly labels it AI.
- The new SegFormer export uses the existing external-CDR preprocessing and
  reproduces the saved vertical CDR to its six-decimal precision. The raw mask,
  image hashes and pixel-derived measurements are checked before rendering.
  Horizontal CDR (0.626609) and area CDR (0.422035) are retained in the source
  data; only vertical CDR (0.690265) was passed to this saved agent and is shown
  on the figure. This export is not asserted to be the
  original historical mask, which was not retained.
- All CFP-derived representations remain correlated. No calibrated confidence,
  subgroup reliability score, retrieved paper, patient demographics, safety
  outcome or clinician finding is fabricated for this example.
- This is a framework overview across runners, not a claim that a single
  evaluated pipeline invokes every module. No high-confidence bypass is
  illustrated: this is the full reasoning path, and the selected case did not
  use a bypass. Existing experiment metrics and prompts are unchanged.

### Saved Evidence Scenarios

| Evidence scenario | Saved label |
|---|---|
| All evidence | Glaucoma |
| Without RETFound probability | Glaucoma |
| Without CDR | Glaucoma |
| Without visual interpretation | Non-glaucoma |

## Source Code Map

| Component | Implementation / scope |
|---|---|
| BioProfiler | `equi-agent/BioProfilerAgent/bio_profiler.py`: metadata to narrative |
| Vision | `equi-agent/VisionAgent/vision.py`: visual reports and supported model calls |
| Expanded FM bank | `equi-agent/scripts/run_equi_agent_fairvision_live.py`: precomputed model scores / validation priors; not all model calls occur online |
| CFP CDR | `OphthalmicAgent/scripts/precompute_external_cdr.py`; one-case export replicates preprocessing |
| Equity / reliability | Runner-specific; `equi-agent/main.py`, `OphthalmicAgent/main_new.py` and FairVision live runner are not identical formulas |
| Guidelines | `equi-agent/GuidelinesAgent/guidelines_agent.py` and `orchestrator.py`: PubMed/web diagnostic literature, not patient-case matching |
| Functional | `equi-agent/FunctionalInterpretationAgent/function_interpreter.py`: visual field / TD / MD when available |
| Recorded ablations | `OphthalmicAgent/CounterfactualAgent/external_glaucoma.py`: evidence removal, not demographic counterfactual patients |
| Recorded orchestration | `OphthalmicAgent/Orchestrator/external_glaucoma.py` |
| Recorded case runner | `OphthalmicAgent/scripts/run_external_glaucoma_agent.py` |
| Safety | `equi-agent/SafetyAgent/safety_agent.py`; invoked by `equi-agent/main.py`, not the external-case runner |

## Files

- `agentic_walkthrough.pdf`: vector text/shapes with genuine raster fundus images.
- `agentic_walkthrough.drawio`: native editable agents, arrows, labels and scores.
- `agentic_walkthrough.png` / `.svg`: preview and alternative vector export.
- `case_data.json`: recorded outputs, trace, segmentation metadata and source hashes.
- `assets/`: original fundus, prediction mask and derived overlay; no fake anatomy.

The PDF is 183 mm wide. Text and vector elements are editable in draw.io.
Real images are embedded. Do not overwrite manual draw.io edits with the builder.

Rebuild from the committed assets and source snapshot:

```bash
python equi-agent/scripts/build_agentic_walkthrough_figure.py
```
"""


def caption_with_scores(case):
    rows = ["\n## Standalone Model Outputs", "",
            "These are separate benchmark outputs, not a multi-model agent call.", "",
            "| Model | Raw score | Validation threshold | Model vote |", "|---|---:|---:|---|"]
    for model, name in NAMES.items():
        row = case["foundations"][model]
        vote = "Glaucoma" if row["prediction"] else "Non-glaucoma"
        rows.append(f"| {name} | {row['probability']:.6f} | {row['threshold']:.6f} | {vote} |")
    return CAPTION + "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, help="Extracted CECSL result archive for the first build")
    parser.add_argument("--segmentation-dir", type=Path, help="One-case CECSL export directory for the first build")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    require(bool(args.inputs_root) == bool(args.segmentation_dir), "Provide both input directories, or neither")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assets = args.out_dir / "assets"
    if args.inputs_root:
        original = ROOT / f"OphthalmicAgent/data_drishti/Glaucoma/{CASE}.png"
        measurements = verify_segmentation(args.segmentation_dir, original)
        case = collect_case(args.inputs_root)
        validate_case(case, measurements)
        assets.mkdir(exist_ok=True)
        for name in ("fundus.png", "mask.png", "overlay.png"):
            shutil.copy2(args.segmentation_dir / name, assets / name)
        case["segmentation"] = measurements
    else:
        case = json.loads((args.out_dir / "case_data.json").read_text())
        measurements = case["segmentation"]
        for name in ("fundus.png", "mask.png", "overlay.png"):
            require(digest(assets / name) == measurements["files"][name], f"Changed figure asset: {name}")
        validate_case(case, measurements)
    c, crop_box = make_figure(case, measurements, assets)
    case["display"] = dict(crop_box_xyxy=crop_box, image_enhancement="none; original RGB and exported mask contours",
                           width_mm=183, height_mm=111, smallest_font_pt=22*PT)
    base = args.out_dir / "agentic_walkthrough"
    write_document(base.with_suffix(".drawio"), [c.page])
    for ext in ("pdf", "svg", "png"):
        c.fig.savefig(base.with_suffix("."+ext), dpi=240, facecolor="white")
    plt.close(c.fig)
    (args.out_dir / "case_data.json").write_text(json.dumps(case, indent=2)+"\n")
    (args.out_dir / "README.md").write_text(caption_with_scores(case))
    print(base.with_suffix(".pdf"))
    print("Verified: fundus, mask, overlay, CDR, model inputs and saved evidence-ablation labels.")


if __name__ == "__main__":
    main()
