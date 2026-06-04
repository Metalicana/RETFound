from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def fnum(value: object) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    q1 = ordered[int(0.25 * (len(ordered) - 1))]
    q3 = ordered[int(0.75 * (len(ordered) - 1))]
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "max": round(max(values), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize REFUGE2 manifest labels and mask-derived CDR features.")
    parser.add_argument("--manifest", type=Path, default=Path("equi-agent/outputs/manifests/refuge2_manifest.csv"))
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    print(f"manifest: {args.manifest}")
    print(f"rows: {len(rows)}")

    print("\nLabel counts by split:")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row.get("split", "")][row.get("label_glaucoma", "")] += 1
    for split in sorted(counts):
        print(split, dict(sorted(counts[split].items())))

    print("\nMask feature rules:")
    print(dict(Counter(row.get("mask_feature_rule", "") for row in rows).most_common()))

    for column in ["cup_to_disc_area_ratio", "vertical_cup_to_disc_ratio"]:
        print(f"\n{column} by split and label:")
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            value = fnum(row.get(column))
            if value is None:
                continue
            grouped[(row.get("split", ""), row.get("label_glaucoma", ""))].append(value)
        for key in sorted(grouped):
            split, label = key
            print(f"{split:5s} label={label}: {stats(grouped[key])}")

    print("\nSample rows:")
    for row in rows[:10]:
        print(
            {
                "split": row.get("split"),
                "image_id": row.get("image_id"),
                "label": row.get("label_glaucoma"),
                "vCDR": row.get("vertical_cup_to_disc_ratio"),
                "areaCDR": row.get("cup_to_disc_area_ratio"),
                "mask_rule": row.get("mask_feature_rule"),
            }
        )


if __name__ == "__main__":
    main()
