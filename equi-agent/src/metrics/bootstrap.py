from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    cluster_col: str = "patient_id",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Cluster bootstrap a scalar metric, patient-level by default."""
    if cluster_col not in df.columns:
        raise ValueError(f"Missing bootstrap cluster column: {cluster_col}")
    rng = np.random.default_rng(random_state)
    clusters = df[cluster_col].dropna().unique()
    estimates = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([df[df[cluster_col] == cluster] for cluster in sampled], ignore_index=True)
        estimates.append(metric_fn(sample))
    alpha = 1.0 - confidence_level
    values = np.asarray(estimates, dtype=float)
    return {
        "estimate": float(metric_fn(df)),
        "ci_lower": float(np.nanquantile(values, alpha / 2)),
        "ci_upper": float(np.nanquantile(values, 1 - alpha / 2)),
        "n_bootstrap": int(n_bootstrap),
        "cluster_col": cluster_col,
    }


def paired_bootstrap_ci(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    key_cols: list[str],
    cluster_col: str = "patient_id",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Cluster bootstrap paired metric difference, computed as A minus B."""
    merged = df_a[key_cols + [cluster_col]].drop_duplicates().merge(
        df_b[key_cols + [cluster_col]].drop_duplicates(),
        on=key_cols,
        suffixes=("_a", "_b"),
    )
    if len(merged) != len(df_a.drop_duplicates(key_cols)):
        raise ValueError("Paired bootstrap inputs do not share the same sample keys.")

    rng = np.random.default_rng(random_state)
    clusters = df_a[cluster_col].dropna().unique()
    diffs = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample_a = pd.concat([df_a[df_a[cluster_col] == cluster] for cluster in sampled], ignore_index=True)
        sample_b = pd.concat([df_b[df_b[cluster_col] == cluster] for cluster in sampled], ignore_index=True)
        diffs.append(metric_fn(sample_a) - metric_fn(sample_b))
    alpha = 1.0 - confidence_level
    values = np.asarray(diffs, dtype=float)
    return {
        "difference": float(metric_fn(df_a) - metric_fn(df_b)),
        "ci_lower": float(np.nanquantile(values, alpha / 2)),
        "ci_upper": float(np.nanquantile(values, 1 - alpha / 2)),
        "n_bootstrap": int(n_bootstrap),
        "cluster_col": cluster_col,
    }

