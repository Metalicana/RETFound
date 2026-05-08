from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_table(
    df: pd.DataFrame,
    csv_path: str | Path,
    tex_path: str | Path | None = None,
    json_path: str | Path | None = None,
    float_format: str = "%.3f",
) -> None:
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)

    if tex_path is not None:
        tex_file = Path(tex_path)
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        latex = df.to_latex(index=False, escape=True, float_format=lambda x: float_format % x)
        tex_file.write_text(latex, encoding="utf-8")

    if json_path is not None:
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(json_file, orient="records", indent=2)

