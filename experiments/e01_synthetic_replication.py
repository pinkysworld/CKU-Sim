"""Experiment 1: Replicate synthetic opacity separation from Nguyen (2026).

Generates a spectrum of synthetic codebases and shows that compressibility
index cleanly separates regular from irregular code.

Usage:
    python -m experiments.e01_synthetic_replication
    python -m experiments.e01_synthetic_replication --config experiments/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cku_sim.core.config import Config
from cku_sim.simulation.opacity_separator import run_synthetic_separation
from cku_sim.viz.plots import plot_synthetic_separation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Synthetic separation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of synthetic samples")
    parser.add_argument("--size", type=int, default=50_000, help="Bytes per sample")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    results_dir = config.results_dir / "e01_synthetic"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 1: Synthetic Opacity Separation")
    logger.info("=" * 60)

    # Run simulation
    df = run_synthetic_separation(
        n_samples=args.n_samples,
        size_bytes=args.size,
        seed=config.random_seed,
    )

    # Save data
    df.to_parquet(results_dir / "synthetic_separation.parquet")
    df.to_csv(results_dir / "synthetic_separation.csv", index=False)
    logger.info(f"Saved {len(df)} samples to {results_dir}")

    # Summary statistics
    summary = {
        "n_samples": len(df),
        "ci_mean_range": [float(df["ci_mean"].min()), float(df["ci_mean"].max())],
        "correlation_regularity_ci": float(df[["regularity", "ci_mean"]].corr().iloc[0, 1]),
    }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_synthetic_separation(df, results_dir / "synthetic_separation.pdf")
    plot_synthetic_separation(df, results_dir / "synthetic_separation.png")
    logger.info("Figures saved.")

    logger.info(f"\nSummary: {json.dumps(summary, indent=2)}")
    logger.info("Experiment 1 complete.")


if __name__ == "__main__":
    main()
