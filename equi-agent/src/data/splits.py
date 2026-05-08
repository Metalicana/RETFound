from __future__ import annotations

import pandas as pd


def validate_patient_level_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    split_col: str = "split",
    eye_col: str | None = "eye_id",
) -> pd.DataFrame:
    """Return split leakage rows; raise if any patient or eye crosses splits."""
    for col in [patient_col, split_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required split column: {col}")

    patient_counts = df.groupby(patient_col)[split_col].nunique(dropna=False)
    leaked_patients = patient_counts[patient_counts > 1].index
    leaks = df[df[patient_col].isin(leaked_patients)].copy()

    if eye_col and eye_col in df.columns:
        eye_key = [patient_col, eye_col]
        eye_counts = df.groupby(eye_key)[split_col].nunique(dropna=False)
        leaked_eyes = eye_counts[eye_counts > 1].reset_index()[eye_key]
        if not leaked_eyes.empty:
            eye_leaks = df.merge(leaked_eyes, on=eye_key, how="inner")
            leaks = pd.concat([leaks, eye_leaks], ignore_index=True).drop_duplicates()

    if not leaks.empty:
        raise ValueError(
            "Patient/eye split leakage detected. "
            f"Leaked rows: {len(leaks)}; leaked patients: {leaks[patient_col].nunique()}"
        )
    return leaks

