"""Bootstrap helpers for clustered uncertainty intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def clustered_delta_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    delta_col: str = "delta_composite",
    n_boot: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap delta summaries by resampling clusters with replacement."""
    if frame.empty or cluster_col not in frame.columns or delta_col not in frame.columns:
        return {}

    grouped: list[np.ndarray] = []
    for _, group in frame.groupby(cluster_col, sort=False):
        values = group[delta_col].dropna().to_numpy(dtype=float)
        if len(values):
            grouped.append(values)

    n_clusters = len(grouped)
    if n_clusters < 2:
        return {}

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level

    mean_values = np.empty(n_boot, dtype=float)
    median_values = np.empty(n_boot, dtype=float)
    positive_share_values = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sampled_ids = rng.integers(0, n_clusters, size=n_clusters)
        sample = np.concatenate([grouped[idx] for idx in sampled_ids])
        mean_values[i] = float(np.mean(sample))
        median_values[i] = float(np.median(sample))
        positive_share_values[i] = float(np.mean(sample > 0))

    def interval(values: np.ndarray) -> list[float]:
        lo = float(np.quantile(values, alpha / 2.0))
        hi = float(np.quantile(values, 1.0 - alpha / 2.0))
        return [lo, hi]

    return {
        "cluster_col": cluster_col,
        "n_clusters": n_clusters,
        "n_boot": n_boot,
        "confidence_level": confidence_level,
        "mean_delta_composite_ci": interval(mean_values),
        "median_delta_composite_ci": interval(median_values),
        "positive_share_ci": interval(positive_share_values),
    }


def flatten_bootstrap_interval(
    bootstrap_summary: dict[str, object] | None,
    *,
    prefix: str,
) -> dict[str, object]:
    """Flatten a bootstrap summary into CSV-friendly scalar columns."""
    if not bootstrap_summary:
        return {}

    result: dict[str, object] = {
        f"{prefix}_cluster_col": bootstrap_summary.get("cluster_col"),
        f"{prefix}_n_clusters": bootstrap_summary.get("n_clusters"),
        f"{prefix}_n_boot": bootstrap_summary.get("n_boot"),
        f"{prefix}_confidence_level": bootstrap_summary.get("confidence_level"),
    }

    for key in ("mean_delta_composite_ci", "median_delta_composite_ci", "positive_share_ci"):
        value = bootstrap_summary.get(key)
        if not isinstance(value, list) or len(value) != 2:
            result[f"{prefix}_{key}_lo"] = math.nan
            result[f"{prefix}_{key}_hi"] = math.nan
            continue
        result[f"{prefix}_{key}_lo"] = float(value[0])
        result[f"{prefix}_{key}_hi"] = float(value[1])

    return result
