from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TABLE_EXTS = {".csv", ".tsv", ".txt"}
MASK_HINTS = ("mask", "seg", "cup", "disc", "gt", "label", "annotation")
LABEL_HINTS = ("glaucoma", "label", "class", "diagnosis", "ground", "truth")


def read_table_preview(path: Path, max_rows: int) -> tuple[list[str], list[dict[str, str]]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = []
            for _, row in zip(range(max_rows), reader):
                rows.append(dict(row))
            return list(reader.fieldnames or []), rows
    except UnicodeDecodeError:
        with path.open(newline="", encoding="latin-1") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = []
            for _, row in zip(range(max_rows), reader):
                rows.append(dict(row))
            return list(reader.fieldnames or []), rows
    except Exception:
        return [], []


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an external glaucoma dataset layout.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--max-files", type=int, default=200, help="Maximum sample image files to inspect for dimensions.")
    parser.add_argument("--max-table-rows", type=int, default=5, help="Rows to preview per metadata table.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    all_files = [path for path in root.rglob("*") if path.is_file()]
    image_files = [path for path in all_files if path.suffix.lower() in IMAGE_EXTS]
    table_files = [path for path in all_files if path.suffix.lower() in TABLE_EXTS]
    other_metadata = [
        path
        for path in all_files
        if path.suffix.lower() in {".json", ".xlsx", ".xls", ".mat", ".xml"}
    ]

    print(f"root: {root}")
    print(f"total_files: {len(all_files)}")
    print(f"image_like_files: {len(image_files)}")
    print(f"table_files: {len(table_files)}")
    print(f"other_metadata_files: {len(other_metadata)}")

    print("\nTop-level entries:")
    for path in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:80]:
        kind = "dir " if path.is_dir() else "file"
        print(f"  {kind} {path.relative_to(root)}")

    print("\nImage extensions:")
    print(dict(sorted(Counter(path.suffix.lower() for path in image_files).items())))

    print("\nImage parent folders:")
    for folder, count in Counter(path.parent.relative_to(root).as_posix() for path in image_files).most_common(40):
        print(f"  {count:6d}  {folder}")

    print("\nMask/annotation-looking image folders:")
    mask_like = Counter()
    for path in image_files:
        text = path.parent.relative_to(root).as_posix().lower()
        if any(hint in text for hint in MASK_HINTS):
            mask_like[path.parent.relative_to(root).as_posix()] += 1
    for folder, count in mask_like.most_common(40):
        print(f"  {count:6d}  {folder}")

    print("\nSample image dimensions:")
    dims = Counter()
    for path in image_files[: args.max_files]:
        size = image_size(path)
        if size:
            dims[size] += 1
    for size, count in dims.most_common(20):
        print(f"  {count:6d}  {size}")

    print("\nSample image files:")
    for path in image_files[:40]:
        print(f"  {path.relative_to(root)}")

    print("\nTable/metadata files:")
    for path in table_files[:60]:
        rel = path.relative_to(root)
        fields, rows = read_table_preview(path, args.max_table_rows)
        label_fields = [field for field in fields if any(hint in field.lower() for hint in LABEL_HINTS)]
        print(f"  {rel}")
        print(f"    columns: {fields[:30]}")
        if label_fields:
            print(f"    label_like_columns: {label_fields}")
        for row in rows[:2]:
            preview = {key: row.get(key, "") for key in fields[:8]}
            print(f"    row: {preview}")

    print("\nOther metadata files:")
    for path in other_metadata[:80]:
        print(f"  {path.relative_to(root)}")


if __name__ == "__main__":
    main()
