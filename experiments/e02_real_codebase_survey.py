"""Experiment 2: Compute opacity metrics across the real corpus.

Clones all repositories and computes structural opacity metrics.
This is the data collection step that feeds all subsequent experiments.

Usage:
    python -m experiments.e02_real_codebase_survey --config experiments/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cku_sim.core.config import Config, DEFAULT_CORPUS
from cku_sim.collectors.git_collector import clone_corpus
from cku_sim.collectors.repo_metrics import compute_corpus_opacity
from cku_sim.simulation.opacity_separator import validate_against_corpus
from cku_sim.analysis.statistics import full_correlation_matrix, algorithm_robustness
from cku_sim.viz.plots import plot_corpus_opacity, plot_algorithm_robustness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Real codebase survey")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--no-clone", action="store_true", help="Skip cloning (use existing)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    results_dir = config.results_dir / "e02_corpus"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 2: Real Codebase Survey")
    logger.info(f"Corpus: {len(corpus)} repositories")
    logger.info("=" * 60)

    # Step 1: Clone repositories
    if not args.no_clone:
        logger.info("Step 1: Cloning repositories...")
        repo_paths = clone_corpus(corpus, config.raw_dir, shallow=True)
    else:
        logger.info("Step 1: Skipping clone, using existing repos...")
        repo_paths = {
            entry.name: config.raw_dir / entry.name
            for entry in corpus
            if (config.raw_dir / entry.name).exists()
        }

    logger.info(f"Available repos: {list(repo_paths.keys())}")

    # Step 2: Compute opacity metrics
    logger.info("Step 2: Computing opacity metrics...")
    opacity_df = compute_corpus_opacity(repo_paths, corpus, config)

    if opacity_df.empty:
        logger.error("No opacity data computed. Check repository clones.")
        return

    # Save
    opacity_df.to_parquet(results_dir / "corpus_opacity.parquet")
    opacity_df.to_csv(results_dir / "corpus_opacity.csv", index=False)
    logger.info(f"Saved opacity metrics for {len(opacity_df)} repos")

    # Step 3: Validation — do categories separate?
    logger.info("Step 3: Validating category separation...")
    validation = validate_against_corpus(opacity_df)
    with open(results_dir / "validation.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    # Step 4: Algorithm robustness
    logger.info("Step 4: Checking algorithm robustness...")
    robustness = algorithm_robustness(opacity_df)
    with open(results_dir / "algorithm_robustness.json", "w") as f:
        json.dump(robustness, f, indent=2)

    # Step 5: Correlation matrix
    corr_matrix = full_correlation_matrix(opacity_df)
    corr_matrix.to_csv(results_dir / "correlation_matrix.csv")

    # Step 6: Plots
    logger.info("Step 6: Generating figures...")
    plot_corpus_opacity(opacity_df, results_dir / "corpus_opacity.pdf")
    plot_corpus_opacity(opacity_df, results_dir / "corpus_opacity.png")
    plot_algorithm_robustness(opacity_df, results_dir / "algorithm_robustness.pdf")
    plot_algorithm_robustness(opacity_df, results_dir / "algorithm_robustness.png")

    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 40)
    for _, row in opacity_df.sort_values("composite_score", ascending=False).iterrows():
        logger.info(
            f"  {row['name']:20s}  category={row['category']:14s}  "
            f"CI(gzip)={row['ci_gzip']:.4f}  composite={row['composite_score']:.4f}"
        )

    logger.info(f"\nValidation: {json.dumps(validation, indent=2, default=str)}")
    logger.info("Experiment 2 complete.")


if __name__ == "__main__":
    main()
