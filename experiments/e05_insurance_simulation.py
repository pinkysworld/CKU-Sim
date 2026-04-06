"""Experiment 5: Monte Carlo cyber insurance simulation.

Demonstrates that opacity-blind actuarial models systematically
underestimate tail risk compared to opacity-aware models.

Usage:
    python -m experiments.e05_insurance_simulation --config experiments/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cku_sim.core.config import Config
from cku_sim.simulation.monte_carlo import (
    InsuranceConfig,
    simulate_portfolio,
    simulate_concentrated_portfolios,
)
from cku_sim.viz.plots import plot_insurance_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Insurance simulation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-sims", type=int, default=None, help="Override Monte Carlo runs")
    parser.add_argument("--portfolio-size", type=int, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    results_dir = config.results_dir / "e05_insurance"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 5: Monte Carlo Insurance Simulation")
    logger.info("=" * 60)

    # Try to use real opacity distribution from Experiment 2
    opacity_path = config.results_dir / "e02_corpus" / "corpus_opacity.parquet"
    if opacity_path.exists():
        corpus_df = pd.read_parquet(opacity_path)
        real_scores = corpus_df["composite_score"].dropna().values
        logger.info(f"Using real opacity distribution from {len(real_scores)} repos")
        logger.info(
            f"  Mean={real_scores.mean():.4f}, Std={real_scores.std():.4f}, "
            f"Range=[{real_scores.min():.4f}, {real_scores.max():.4f}]"
        )
    else:
        real_scores = None
        logger.info("No corpus data found — using synthetic opacity distribution")

    # Insurance config
    n_firms = args.portfolio_size or config.portfolio_size
    n_sims = args.n_sims or config.monte_carlo_runs

    ins_config = InsuranceConfig(
        n_firms=n_firms,
        n_simulations=n_sims,
        seed=config.random_seed,
    )

    # Generate portfolio opacity scores
    rng = np.random.RandomState(config.random_seed)
    if real_scores is not None and len(real_scores) >= 5:
        # Bootstrap from real distribution
        opacity_scores = rng.choice(real_scores, size=n_firms, replace=True)
        # Add small noise to avoid exact duplicates
        opacity_scores += rng.normal(0, 0.01, size=n_firms)
        opacity_scores = np.clip(opacity_scores, 0.01, 0.99)
        source = "empirical (bootstrapped from corpus)"
    else:
        # Synthetic: bimodal distribution mimicking real corpus
        n_high = n_firms // 3
        n_low = n_firms // 3
        n_mid = n_firms - n_high - n_low
        opacity_scores = np.concatenate([
            rng.beta(2, 5, size=n_low) * 0.4 + 0.1,    # low opacity cluster
            rng.beta(5, 5, size=n_mid) * 0.3 + 0.35,    # mid
            rng.beta(5, 2, size=n_high) * 0.4 + 0.5,    # high opacity cluster
        ])
        rng.shuffle(opacity_scores)
        source = "synthetic (bimodal beta mixture)"

    logger.info(f"Portfolio: {n_firms} firms, opacity source: {source}")
    logger.info(f"Simulations: {n_sims:,}")

    # Run main simulation
    logger.info("Running opacity-aware vs opacity-blind simulation...")
    result = simulate_portfolio(opacity_scores, ins_config)

    # Save results
    with open(results_dir / "simulation_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Concentrated portfolio analysis
    logger.info("Running concentrated portfolio analysis...")
    quartile_df = simulate_concentrated_portfolios(opacity_scores, ins_config)
    quartile_df.to_csv(results_dir / "quartile_analysis.csv", index=False)

    # Convergence check at higher sample count
    logger.info("Running convergence check (100k sims)...")
    conv_config = InsuranceConfig(
        n_firms=n_firms,
        n_simulations=100_000,
        seed=config.random_seed + 1,
    )
    conv_result = simulate_portfolio(opacity_scores, conv_config)
    with open(results_dir / "convergence_check.json", "w") as f:
        json.dump(conv_result, f, indent=2)

    # Plots
    logger.info("Generating figures...")
    plot_insurance_comparison(result, results_dir / "insurance_comparison.pdf")
    plot_insurance_comparison(result, results_dir / "insurance_comparison.png")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("MONTE CARLO RESULTS")
    logger.info("=" * 50)
    logger.info(f"Portfolio: {n_firms} firms, {n_sims:,} simulations")
    logger.info(f"Opacity source: {source}")
    logger.info("")

    for model_name in ("aware_model", "blind_model"):
        m = result[model_name]
        logger.info(f"  {model_name.upper()}:")
        logger.info(f"    Mean loss:  ${m['mean_loss']:>15,.0f}")
        logger.info(f"    VaR 95:     ${m['VaR_95']:>15,.0f}")
        logger.info(f"    VaR 99:     ${m['VaR_99']:>15,.0f}")
        logger.info(f"    CVaR 99:    ${m['CVaR_99']:>15,.0f}")

    logger.info("")
    logger.info("  GAPS (aware - blind, % of blind):")
    for key, val in result["gaps"].items():
        logger.info(f"    {key:30s}: {val:>+8.1f}%")

    # Convergence comparison
    var99_10k = result["aware_model"]["VaR_99"]
    var99_100k = conv_result["aware_model"]["VaR_99"]
    conv_diff = abs(var99_10k - var99_100k) / var99_100k * 100
    logger.info(f"\n  Convergence: VaR99 differs by {conv_diff:.2f}% between 10k and 100k sims")

    logger.info("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
