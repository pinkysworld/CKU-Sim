"""Composite structural opacity score with PCA-based weight derivation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def derive_pca_weights(df: pd.DataFrame) -> dict[str, float]:
    """Derive composite weights from first principal component of opacity metrics.

    Args:
        df: DataFrame with columns ci_gzip, ci_lzma, ci_zstd,
            shannon_entropy, cyclomatic_density, halstead_volume.

    Returns:
        Dict mapping metric group to weight (sums to 1.0).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Average CI across algorithms
    df = df.copy()
    df["ci_mean"] = df[["ci_gzip", "ci_lzma", "ci_zstd"]].mean(axis=1)

    features = ["ci_mean", "shannon_entropy", "cyclomatic_density", "halstead_volume"]
    X = df[features].dropna()

    if len(X) < 3:
        logger.warning("Too few samples for PCA, using default weights")
        return {
            "compressibility": 0.35,
            "entropy": 0.25,
            "cyclomatic_density": 0.25,
            "halstead_volume": 0.15,
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    pca.fit(X_scaled)

    # First PC loadings as weights (absolute values, normalised)
    loadings = np.abs(pca.components_[0])
    loadings = loadings / loadings.sum()

    weights = {
        "compressibility": float(loadings[0]),
        "entropy": float(loadings[1]),
        "cyclomatic_density": float(loadings[2]),
        "halstead_volume": float(loadings[3]),
    }

    logger.info(f"PCA-derived weights: {weights}")
    logger.info(f"Explained variance: {pca.explained_variance_ratio_[0]:.3f}")

    return weights


def recompute_composite(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Recompute composite scores for an entire DataFrame.

    Args:
        df: DataFrame with opacity metric columns.
        weights: Weight dict.

    Returns:
        Series of composite scores.
    """
    ci_mean = df[["ci_gzip", "ci_lzma", "ci_zstd"]].mean(axis=1)

    total_w = sum(weights.values())
    score = (
        weights.get("compressibility", 0) * ci_mean
        + weights.get("entropy", 0) * df["shannon_entropy"]
        + weights.get("cyclomatic_density", 0) * df["cyclomatic_density"]
        + weights.get("halstead_volume", 0) * df["halstead_volume"]
    ) / total_w

    return score
