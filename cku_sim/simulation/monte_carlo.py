"""Simulation 3: Monte Carlo cyber insurance model.

Demonstrates that ignoring structural opacity (CKU) leads to systematic
underestimation of tail risk in cyber insurance portfolios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InsuranceConfig:
    """Configuration for the Monte Carlo insurance simulation."""

    n_firms: int = 100
    n_simulations: int = 10_000
    base_incident_rate: float = 0.15  # base annual Poisson rate
    pareto_alpha_low_opacity: float = 3.0  # lighter tail
    pareto_alpha_high_opacity: float = 1.5  # heavier tail
    pareto_scale: float = 10_000.0  # minimum loss in USD
    seed: int = 42


def simulate_portfolio(
    opacity_scores: np.ndarray,
    config: InsuranceConfig | None = None,
) -> dict:
    """Run Monte Carlo simulation for a portfolio of software firms.

    Each firm has a structural opacity score. Incident arrival rate and
    severity distribution are modulated by opacity.

    Args:
        opacity_scores: Array of opacity scores for each firm, in [0, 1].
        config: Simulation configuration.

    Returns:
        Dict with simulation results including loss distributions and risk metrics.
    """
    config = config or InsuranceConfig()
    rng = np.random.RandomState(config.seed)

    n_firms = len(opacity_scores)
    n_sims = config.n_simulations

    # --- Opacity-aware model ---
    # Incident rate scales with opacity: λ_i = λ_0 * (1 + 2 * ω_i)
    aware_rates = config.base_incident_rate * (1.0 + 2.0 * opacity_scores)

    # Pareto shape parameter: α_i = α_high + (α_low - α_high) * (1 - ω_i)
    # High opacity → low α → heavier tail
    aware_alphas = (
        config.pareto_alpha_high_opacity
        + (config.pareto_alpha_low_opacity - config.pareto_alpha_high_opacity)
        * (1.0 - opacity_scores)
    )

    # --- Opacity-blind model ---
    # All firms get the mean rate and alpha
    blind_rate = np.mean(aware_rates)
    blind_alpha = np.mean(aware_alphas)

    # Simulate
    aware_portfolio_losses = np.zeros(n_sims)
    blind_portfolio_losses = np.zeros(n_sims)

    for sim in range(n_sims):
        # Opacity-aware
        for i in range(n_firms):
            n_incidents = rng.poisson(aware_rates[i])
            if n_incidents > 0:
                losses = config.pareto_scale * (
                    rng.pareto(aware_alphas[i], size=n_incidents) + 1
                )
                aware_portfolio_losses[sim] += losses.sum()

        # Opacity-blind
        for i in range(n_firms):
            n_incidents = rng.poisson(blind_rate)
            if n_incidents > 0:
                losses = config.pareto_scale * (
                    rng.pareto(blind_alpha, size=n_incidents) + 1
                )
                blind_portfolio_losses[sim] += losses.sum()

    # Compute risk metrics
    percentiles = [90, 95, 99, 99.5]

    result = {
        "n_firms": n_firms,
        "n_simulations": n_sims,
        "opacity_distribution": {
            "mean": float(opacity_scores.mean()),
            "std": float(opacity_scores.std()),
            "min": float(opacity_scores.min()),
            "max": float(opacity_scores.max()),
        },
        "aware_model": _compute_risk_metrics(aware_portfolio_losses, percentiles),
        "blind_model": _compute_risk_metrics(blind_portfolio_losses, percentiles),
    }

    # Compute gaps
    gaps = {}
    for p in percentiles:
        a = result["aware_model"][f"VaR_{p}"]
        b = result["blind_model"][f"VaR_{p}"]
        gaps[f"VaR_{p}_gap_pct"] = ((a - b) / b * 100) if b > 0 else 0.0
    for p in percentiles:
        a = result["aware_model"][f"CVaR_{p}"]
        b = result["blind_model"][f"CVaR_{p}"]
        gaps[f"CVaR_{p}_gap_pct"] = ((a - b) / b * 100) if b > 0 else 0.0

    result["gaps"] = gaps

    logger.info(
        f"Monte Carlo complete: "
        f"VaR99 aware={result['aware_model']['VaR_99']:,.0f}, "
        f"blind={result['blind_model']['VaR_99']:,.0f}, "
        f"gap={gaps['VaR_99_gap_pct']:.1f}%"
    )

    return result


def simulate_concentrated_portfolios(
    opacity_scores: np.ndarray,
    config: InsuranceConfig | None = None,
) -> pd.DataFrame:
    """Compare risk metrics for portfolios concentrated in different opacity quartiles.

    Returns:
        DataFrame with one row per quartile, columns for risk metrics.
    """
    config = config or InsuranceConfig()
    quartiles = np.percentile(opacity_scores, [25, 50, 75])

    rows = []
    for label, mask in [
        ("Q1 (lowest)", opacity_scores <= quartiles[0]),
        ("Q2", (opacity_scores > quartiles[0]) & (opacity_scores <= quartiles[1])),
        ("Q3", (opacity_scores > quartiles[1]) & (opacity_scores <= quartiles[2])),
        ("Q4 (highest)", opacity_scores > quartiles[2]),
        ("Full portfolio", np.ones(len(opacity_scores), dtype=bool)),
    ]:
        subset = opacity_scores[mask]
        if len(subset) == 0:
            continue

        result = simulate_portfolio(subset, config)
        row = {
            "quartile": label,
            "n_firms": len(subset),
            "mean_opacity": float(subset.mean()),
        }
        row.update({
            f"aware_{k}": v
            for k, v in result["aware_model"].items()
            if isinstance(v, (int, float))
        })
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_risk_metrics(losses: np.ndarray, percentiles: list[int]) -> dict:
    """Compute standard actuarial risk metrics from a loss distribution."""
    result = {
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses)),
        "median_loss": float(np.median(losses)),
        "max_loss": float(np.max(losses)),
    }

    for p in percentiles:
        var = float(np.percentile(losses, p))
        cvar = float(np.mean(losses[losses >= var])) if np.any(losses >= var) else var
        result[f"VaR_{p}"] = var
        result[f"CVaR_{p}"] = cvar

    return result
