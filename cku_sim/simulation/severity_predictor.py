"""Simulation 2: Opacity → vulnerability severity mapping.

Tests whether structural opacity predicts CVE severity (CVSS scores)
across the corpus, controlling for confounds.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def merge_opacity_and_cves(
    opacity_df: pd.DataFrame,
    cve_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge opacity metrics with CVE data per project.

    Args:
        opacity_df: DataFrame from compute_corpus_opacity.
        cve_data: Dict mapping project name to DataFrame of CVERecords.

    Returns:
        DataFrame with one row per project: opacity metrics + CVE summary stats.
    """
    rows = []
    for _, row in opacity_df.iterrows():
        name = row["name"]
        cves = cve_data.get(name)

        if cves is None or len(cves) == 0:
            cve_count = 0
            mean_cvss = np.nan
            median_cvss = np.nan
            max_cvss = np.nan
            high_sev_count = 0
            critical_count = 0
        else:
            scores = cves["cvss_v3_score"].dropna()
            cve_count = len(cves)
            mean_cvss = float(scores.mean()) if len(scores) > 0 else np.nan
            median_cvss = float(scores.median()) if len(scores) > 0 else np.nan
            max_cvss = float(scores.max()) if len(scores) > 0 else np.nan
            high_sev_count = int((scores >= 7.0).sum())
            critical_count = int((scores >= 9.0).sum())

        loc = row.get("total_loc", 1)
        merged = row.to_dict()
        merged.update({
            "cve_count": cve_count,
            "cve_density": cve_count / max(loc / 1000, 1),  # CVEs per KLOC
            "mean_cvss": mean_cvss,
            "median_cvss": median_cvss,
            "max_cvss": max_cvss,
            "high_severity_count": high_sev_count,
            "critical_count": critical_count,
        })
        rows.append(merged)

    return pd.DataFrame(rows)


def correlation_analysis(merged_df: pd.DataFrame) -> dict:
    """Run correlation analysis between opacity metrics and CVE outcomes.

    Returns:
        Dict with Spearman correlations, p-values, and confidence intervals.
    """
    results = {}

    opacity_metrics = ["ci_gzip", "ci_lzma", "ci_zstd", "shannon_entropy",
                       "cyclomatic_density", "halstead_volume", "composite_score"]
    cve_outcomes = ["cve_density", "mean_cvss", "median_cvss"]

    for metric in opacity_metrics:
        for outcome in cve_outcomes:
            valid = merged_df[[metric, outcome]].dropna()
            if len(valid) < 5:
                continue

            rho, p = stats.spearmanr(valid[metric], valid[outcome])

            # Bootstrap CI
            ci_low, ci_high = _bootstrap_spearman_ci(
                valid[metric].values, valid[outcome].values
            )

            key = f"{metric}_vs_{outcome}"
            results[key] = {
                "spearman_rho": float(rho),
                "p_value": float(p),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "n": len(valid),
            }

            logger.info(
                f"  {key}: ρ={rho:.4f} (p={p:.4f}) "
                f"95% CI [{ci_low:.4f}, {ci_high:.4f}]"
            )

    return results


def regression_analysis(merged_df: pd.DataFrame) -> dict:
    """Multivariate regression: opacity → CVE density, controlling for confounds.

    Controls: total_loc, num_files.

    Returns:
        Dict with regression summary.
    """
    import statsmodels.api as sm

    df = merged_df.dropna(subset=["composite_score", "cve_density"]).copy()
    if len(df) < 8:
        logger.warning("Too few complete cases for regression")
        return {"error": "insufficient data"}

    # Log-transform skewed variables
    df["log_loc"] = np.log1p(df["total_loc"])
    df["log_cve_density"] = np.log1p(df["cve_density"])

    X = df[["composite_score", "log_loc", "num_files"]]
    X = sm.add_constant(X)
    y = df["log_cve_density"]

    model = sm.OLS(y, X).fit()

    result = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": {
            name: {
                "coef": float(model.params[name]),
                "std_err": float(model.bse[name]),
                "t_stat": float(model.tvalues[name]),
                "p_value": float(model.pvalues[name]),
            }
            for name in model.params.index
        },
        "n_obs": int(model.nobs),
    }

    logger.info(f"Regression R²={result['r_squared']:.4f}, "
                f"composite_score p={result['coefficients']['composite_score']['p_value']:.4f}")

    return result


def _bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Bootstrap confidence interval for Spearman correlation."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rhos[i], _ = stats.spearmanr(x[idx], y[idx])

    return float(np.percentile(rhos, 100 * alpha / 2)), float(
        np.percentile(rhos, 100 * (1 - alpha / 2))
    )
