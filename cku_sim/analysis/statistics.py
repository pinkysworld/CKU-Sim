"""Statistical analysis: correlation, regression, classification, robustness."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def full_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Spearman correlations between all opacity metrics.

    Useful for checking metric independence and algorithm robustness.
    """
    cols = [c for c in df.columns if c.startswith("ci_") or c in
            ("shannon_entropy", "cyclomatic_density", "halstead_volume", "composite_score")]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr(method="spearman")


def algorithm_robustness(df: pd.DataFrame) -> dict:
    """Test whether CI is consistent across compression algorithms.

    Returns pairwise Spearman correlations between gzip, lzma, zstd.
    """
    pairs = [("ci_gzip", "ci_lzma"), ("ci_gzip", "ci_zstd"), ("ci_lzma", "ci_zstd")]
    results = {}
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            valid = df[[a, b]].dropna()
            rho, p = stats.spearmanr(valid[a], valid[b])
            results[f"{a}_vs_{b}"] = {"spearman_rho": float(rho), "p_value": float(p)}
    return results


def permutation_test(
    x: np.ndarray, y: np.ndarray, n_perms: int = 10_000, seed: int = 42
) -> dict:
    """Permutation test for Spearman correlation significance.

    More robust than parametric p-values for small samples.
    """
    rng = np.random.RandomState(seed)
    observed_rho, _ = stats.spearmanr(x, y)

    null_rhos = np.empty(n_perms)
    for i in range(n_perms):
        perm_y = rng.permutation(y)
        null_rhos[i], _ = stats.spearmanr(x, perm_y)

    p_value = np.mean(np.abs(null_rhos) >= np.abs(observed_rho))

    return {
        "observed_rho": float(observed_rho),
        "p_value_permutation": float(p_value),
        "null_mean": float(null_rhos.mean()),
        "null_std": float(null_rhos.std()),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def weight_sensitivity_analysis(
    df: pd.DataFrame,
    outcome: str = "cve_density",
    n_samples: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test robustness of composite score by perturbing weights.

    Randomly samples weight vectors from a Dirichlet distribution and
    recomputes composite → outcome correlation for each.

    Returns:
        DataFrame with columns: w_compress, w_entropy, w_cc, w_halstead,
        spearman_rho, p_value.
    """
    from cku_sim.metrics.composite import recompute_composite

    rng = np.random.RandomState(seed)
    valid = df.dropna(subset=[outcome])
    if len(valid) < 5:
        return pd.DataFrame()

    rows = []
    for _ in range(n_samples):
        w = rng.dirichlet([1, 1, 1, 1])
        weights = {
            "compressibility": float(w[0]),
            "entropy": float(w[1]),
            "cyclomatic_density": float(w[2]),
            "halstead_volume": float(w[3]),
        }

        scores = recompute_composite(valid, weights)
        rho, p = stats.spearmanr(scores, valid[outcome])

        rows.append({
            "w_compress": w[0],
            "w_entropy": w[1],
            "w_cc": w[2],
            "w_halstead": w[3],
            "spearman_rho": float(rho),
            "p_value": float(p),
        })

    result = pd.DataFrame(rows)
    result = result.dropna(subset=["spearman_rho", "p_value"]).reset_index(drop=True)
    if result.empty:
        logger.warning(
            "Weight sensitivity analysis produced no finite correlations; "
            "the outcome may be constant or insufficiently variable."
        )
        return result

    logger.info(
        f"Weight sensitivity: mean ρ={result['spearman_rho'].mean():.4f}, "
        f"std={result['spearman_rho'].std():.4f}, "
        f"range=[{result['spearman_rho'].min():.4f}, {result['spearman_rho'].max():.4f}]"
    )
    return result
