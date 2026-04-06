"""Simulation 1: Opacity class separation.

Replicates the original CKU paper result showing that compressibility index
cleanly separates regular from irregular codebases, then validates against
the real corpus.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from cku_sim.metrics.compressibility import compressibility_index
from cku_sim.simulation.scenario_generator import generate_spectrum

logger = logging.getLogger(__name__)


def run_synthetic_separation(
    n_samples: int = 50,
    size_bytes: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Simulation 1 on synthetic data.

    Generates a spectrum of synthetic codebases and computes CI for each.

    Returns:
        DataFrame with columns: label, regularity, ci_gzip, ci_lzma, ci_zstd.
    """
    logger.info(f"Generating {n_samples} synthetic codebases...")
    samples = generate_spectrum(n_samples, size_bytes, seed)

    results = []
    for label, source, regularity in samples:
        ci_gzip = compressibility_index(source, "gzip")
        ci_lzma = compressibility_index(source, "lzma")
        ci_zstd = compressibility_index(source, "zstd")

        results.append({
            "label": label,
            "regularity": regularity,
            "ci_gzip": ci_gzip,
            "ci_lzma": ci_lzma,
            "ci_zstd": ci_zstd,
            "ci_mean": np.mean([ci_gzip, ci_lzma, ci_zstd]),
        })

    df = pd.DataFrame(results)

    # Correlation between regularity and CI (should be strongly negative:
    # more regular code -> lower CI in our opacity-oriented metric)
    rho, p = stats.spearmanr(df["regularity"], df["ci_mean"])
    logger.info(f"Synthetic separation: Spearman ρ={rho:.4f}, p={p:.2e}")

    return df


def validate_against_corpus(
    corpus_df: pd.DataFrame,
) -> dict:
    """Validate synthetic separation result against real corpus.

    Tests whether projects labelled "high_opacity" have significantly higher
    CI than those labelled "low_opacity".

    Args:
        corpus_df: DataFrame from compute_corpus_opacity with 'category' column.

    Returns:
        Dict with test statistics.
    """
    high = corpus_df[corpus_df["category"] == "high_opacity"]["ci_gzip"]
    low = corpus_df[corpus_df["category"] == "low_opacity"]["ci_gzip"]

    if len(high) < 2 or len(low) < 2:
        logger.warning("Insufficient data for group comparison")
        return {"error": "insufficient data"}

    # Mann-Whitney U test (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(high, low, alternative="greater")

    # Effect size: rank-biserial correlation
    n1, n2 = len(high), len(low)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    # Descriptive stats
    result = {
        "high_opacity_mean_ci": float(high.mean()),
        "high_opacity_std_ci": float(high.std()),
        "low_opacity_mean_ci": float(low.mean()),
        "low_opacity_std_ci": float(low.std()),
        "mann_whitney_U": float(u_stat),
        "mann_whitney_p": float(u_p),
        "rank_biserial_r": float(r_rb),
        "n_high": n1,
        "n_low": n2,
    }

    logger.info(
        f"Corpus validation: high CI={result['high_opacity_mean_ci']:.4f}, "
        f"low CI={result['low_opacity_mean_ci']:.4f}, "
        f"U={u_stat:.1f}, p={u_p:.4f}, r_rb={r_rb:.4f}"
    )

    return result
