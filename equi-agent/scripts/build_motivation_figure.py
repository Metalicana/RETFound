"""Show model error complementarity on the archived full FairVision glaucoma test set."""

import argparse
import json
from pathlib import Path

from build_paper_drawio import Page, write_document, INK, GRAY
from build_recovered_experiment_figures import Inputs, SELECTIVE, binary, metrics, require, unique_rows


REFERENCE = "retfound_oct"
LABELS = {
    "retfound_oct": "RETFound (OCT)",
    "visionfm_oct": "VisionFM (OCT)",
    "urfound_oct": "URFound (OCT)",
    "mirage_slo": "MIRAGE (SLO)",
    "retizero_slo": "RetiZero (SLO)",
    "ret_clip_slo": "RET-CLIP (SLO)",
    "urfound_slo": "URFound (SLO)",
    "visionfm_slo": "VisionFM (SLO)",
    "flair_slo": "FLAIR (SLO)",
}
CORRECT = "#087F8C"
WRONG = "#BD6955"
BAR = "#A5B5C0"


def compare(reference, alternative):
    require(reference.keys() == alternative.keys(), "Models have different case sets")
    counts = dict(both_correct=0, both_wrong=0, reference_only_correct=0, alternative_only_correct=0)
    for key, row in reference.items():
        other = alternative[key]
        truth = binary(row["y_true"])
        require(truth == binary(other["y_true"]), "Models have different ground truth")
        a = binary(row["y_pred"]) == truth
        b = binary(other["y_pred"]) == truth
        field = "both_correct" if a and b else "both_wrong" if not a and not b else (
            "reference_only_correct" if a else "alternative_only_correct")
        counts[field] += 1
    return counts


def load_data(inputs):
    summary = inputs.json(f"{SELECTIVE}/selective_arbitration_summary.json")
    sources = [r for r in summary["test_loaded_files"] if r["task"] == "glaucoma"]
    require(len(sources) == 9 and {s["model"] for s in sources} == set(LABELS),
            "Expected the archived nine-model glaucoma panel")
    streams = {}
    for source in sources:
        require("test_thresholded" in source["path"], "Expected saved thresholded test predictions")
        rows = [r for r in inputs.csv(source["path"]) if r["task"] == "glaucoma"]
        require(len(rows) == int(source["rows"]) == 3000, "Expected 3000 test cases per model")
        require(all(r["split"] == "test" and r["model_name"] == source["model"] for r in rows),
                "Wrong model or split in prediction file")
        streams[source["model"]] = unique_rows(rows, "image_id")
    base = streams[REFERENCE]
    models = []
    for name, rows in streams.items():
        counts = compare(base, rows)
        models.append(dict(model=name, label=LABELS[name], paired=counts,
                           metrics=metrics((r["y_true"], r["y_pred"]) for r in rows.values())))
    models.sort(key=lambda m: (-m["metrics"]["balanced_accuracy"], m["model"]))
    errors = [key for key, row in base.items() if binary(row["y_true"]) != binary(row["y_pred"])]
    alternatives_correct = sum(any(binary(rows[key]["y_pred"]) == binary(base[key]["y_true"])
                                   for name, rows in streams.items() if name != REFERENCE) for key in errors)
    require(models[0]["model"] == REFERENCE, "The proposed highest-score motivation needs revisiting")
    return dict(dataset="Harvard FairVision30K", task="glaucoma", split="test", n=len(base),
                reference=REFERENCE, reference_errors=len(errors),
                reference_errors_with_correct_alternative=alternatives_correct,
                models=models, sources=inputs.sources)


def rectangle(page, x, y, w, h, color):
    page.cell("", x, y, w, h, f"rounded=0;fillColor={color};strokeColor=none;")


def figure(data):
    page = Page("01 Why case-specific trust", width=1600, height=1060)
    page.text("Overall performance hides complementary errors", 60, 40, 1480, 60, 34, INK, True)
    page.text(f"FairVision glaucoma  |  Same {data['n']:,} test cases for all nine models", 60, 112, 1460, 40, 22, GRAY)
    page.text("a  Overall performance", 60, 190, 680, 45, 27, INK, True)
    page.text("b  Different models are right on different cases", 850, 190, 695, 65, 26, INK, True)
    page.text("Balanced accuracy", 310, 273, 425, 32, 21, GRAY)
    page.text("RETFound correct\nOther model wrong", 850, 265, 320, 62, 21, WRONG, align="center")
    page.text("RETFound wrong\nOther model correct", 1210, 265, 320, 62, 21, CORRECT, align="center")
    start, width, center, half, limit = 310, 400, 1190, 340, 700
    bottom = 844
    for percent in (0, 20, 40, 60, 80, 100):
        x = start + percent / 100 * width
        page.line([(x, 340), (x, bottom)], "#E8EBEE", width=1)
        page.text(str(percent), x - 30, bottom + 13, 60, 30, 20, GRAY, align="center")
    for count in (-600, -300, 0, 300, 600):
        x = center + count / limit * half
        page.line([(x, 340), (x, bottom)], GRAY if count == 0 else "#E8EBEE", width=1)
        page.text(str(abs(count)), x - 30, bottom + 13, 60, 30, 20, GRAY, align="center")
    for i, model in enumerate(data["models"]):
        y = 365 + 55 * i
        reference = model["model"] == REFERENCE
        page.text(model["label"], 60, y - 19, 235, 38, 22, INK, reference)
        score = model["metrics"]["balanced_accuracy"]
        rectangle(page, start, y - 14, score * width, 28, INK if reference else BAR)
        page.text(f"{100 * score:.1f}", start + score * width + 10, y - 18, 65, 36, 21, INK, reference)
        if reference:
            page.text("Reference model", 960, y - 20, 460, 40, 21, GRAY, align="center")
            continue
        lost = model["paired"]["reference_only_correct"]
        gained = model["paired"]["alternative_only_correct"]
        require(max(lost, gained) <= limit, "Paired bar exceeds its axis")
        a, b = lost / limit * half, gained / limit * half
        rectangle(page, center - a, y - 14, a, 28, WRONG)
        rectangle(page, center, y - 14, b, 28, CORRECT)
        page.text(str(lost), center - a - 65, y - 18, 55, 36, 20, WRONG, align="right")
        page.text(str(gained), center + b + 10, y - 18, 55, 36, 20, CORRECT)
    page.line([(start, bottom), (start + width, bottom)], INK, width=1.2)
    page.line([(center - half, bottom), (center + half, bottom)], INK, width=1.2)
    page.text("Balanced accuracy (%)", start, 899, width, 36, 22, align="center")
    page.text("Number of cases", center - half, 899, 2 * half, 36, 22, align="center")
    n, errors = data["reference_errors_with_correct_alternative"], data["reference_errors"]
    page.text(f"At least one alternative is correct on {n:,} of RETFound's {errors:,} errors.",
              60, 969, 1480, 34, 24, INK, True)
    page.text("But every alternative also makes errors that RETFound avoids. The problem is deciding when to trust it.",
              60, 1009, 1480, 34, 22, GRAY)
    return page


def caption(data):
    return (
        "# Motivation Figure\n\n"
        "**Overall performance hides complementary errors.** "
        "(a) Observed balanced accuracy of nine foundation-model/modality combinations on the "
        "same 3,000 held-out FairVision glaucoma cases. RETFound has the highest observed score. "
        "(b) Paired correctness relative to RETFound, using the same row order as panel a. "
        "Teal counts cases where the alternative is correct and RETFound is wrong; coral counts "
        "the reverse. Counts overlap across models and must not be added across rows. "
        f"Of RETFound's {data['reference_errors']} errors, "
        f"{data['reference_errors_with_correct_alternative']} have at least one correct alternative "
        "prediction. This retrospective observation motivates studying case-specific source trust; "
        "it does not establish a deployable selector, achievable improvement or clinical benefit.\n\n"
        "## Use In The Paper\n\n"
        "Use this as the **problem/motivation figure**, before the architecture. The message is "
        "not that every model is equally good: lower-ranked models can contribute correct evidence, "
        "but also introduce errors. The proposed method must separately demonstrate that it can "
        "distinguish those situations without access to test labels.\n\n"
        "## Provenance\n\n"
        "This is a descriptive reanalysis of the saved full 3,000-case test predictions used by "
        "the historical deterministic-arbitration experiment. It is **not** the manuscript's "
        "250-case agent evaluation. Model names, source paths and SHA-256 hashes are recorded "
        "in `motivation_source_data.json`. Saved thresholded decisions are used unchanged; no "
        "fitting, rethresholding, LLM calls, case selection or prompt changes were performed. "
        "All nine available models are shown, ordered by observed balanced accuracy for readability. "
        "These are point estimates, not tests of statistical superiority. Case IDs and binary "
        "labels are checked for exact agreement across files. Balanced accuracy is used to avoid "
        "mixing the binary and support-weighted F1 conventions of earlier tables.\n\n"
        "## Rebuild\n\n```bash\n"
        "python equi-agent/scripts/build_motivation_figure.py \\\n"
        "  --inputs-root /path/to/separate/extracted/archive\n```\n\n"
        "The draw.io file is editable. Rebuilding overwrites generated files; keep manual edits "
        "in a separately named copy.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/paper_figures/motivation")
    args = parser.parse_args()
    data = load_data(Inputs(args.inputs_root))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_document(args.out_dir / "motivation.drawio", [figure(data)])
    (args.out_dir / "motivation_source_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "caption.md").write_text(caption(data))
    print(f"Matched {data['n']} cases across {len(data['models'])} models; wrote {args.out_dir}")


if __name__ == "__main__":
    main()
