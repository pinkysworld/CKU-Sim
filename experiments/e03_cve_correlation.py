"""Experiment 3: Correlate opacity metrics with CVE data.

Fetches CVE records from NVD for each corpus project and tests whether
structural opacity predicts vulnerability density and severity.

Usage:
    python -m experiments.e03_cve_correlation --config experiments/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cku_sim.core.config import Config, DEFAULT_CORPUS
from cku_sim.collectors.nvd_collector import fetch_cves_for_cpe
from cku_sim.simulation.severity_predictor import (
    merge_opacity_and_cves,
    correlation_analysis,
    regression_analysis,
)
from cku_sim.analysis.statistics import permutation_test, weight_sensitivity_analysis
from cku_sim.viz.plots import plot_opacity_vs_cve, plot_weight_sensitivity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: CVE correlation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    results_dir = config.results_dir / "e03_cve"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.processed_dir / "nvd_cache"

    # Load opacity data from Experiment 2
    opacity_path = config.results_dir / "e02_corpus" / "corpus_opacity.parquet"
    if not opacity_path.exists():
        logger.error(
            "Opacity data not found. Run experiment 2 first:\n"
            "  python -m experiments.e02_real_codebase_survey --config experiments/config.yaml"
        )
        return

    opacity_df = pd.read_parquet(opacity_path)
    logger.info(f"Loaded opacity data for {len(opacity_df)} repos")

    # Step 1: Fetch CVE data
    logger.info("=" * 60)
    logger.info("Experiment 3: CVE Correlation Analysis")
    logger.info("=" * 60)

    cve_data = {}
    for entry in corpus:
        if entry.name not in opacity_df["name"].values:
            continue
        if not entry.cpe_id:
            logger.info("Skipping %s: no CPE identifier configured", entry.name)
            continue

        logger.info(f"Fetching CVEs for {entry.name} ({entry.cpe_id})...")
        records = fetch_cves_for_cpe(
            entry.cpe_id,
            api_key=config.nvd_api_key,
            rate_limit=config.nvd_rate_limit,
            cache_dir=cache_dir,
        )
        if records:
            cve_data[entry.name] = pd.DataFrame([r.to_dict() for r in records])
            logger.info(f"  {entry.name}: {len(records)} CVEs")
        else:
            cve_data[entry.name] = pd.DataFrame()
            logger.info(f"  {entry.name}: 0 CVEs")

    # Step 2: Merge opacity + CVE data
    logger.info("Merging opacity and CVE data...")
    merged = merge_opacity_and_cves(opacity_df, cve_data)
    merged.to_parquet(results_dir / "merged_opacity_cve.parquet")
    merged.to_csv(results_dir / "merged_opacity_cve.csv", index=False)

    # Step 3: Correlation analysis
    logger.info("Running correlation analysis...")
    correlations = correlation_analysis(merged)
    with open(results_dir / "correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)

    # Step 4: Permutation test on key result
    valid = merged.dropna(subset=["composite_score", "cve_density"])
    if len(valid) >= 5:
        logger.info("Running permutation test (composite vs CVE density)...")
        perm = permutation_test(
            valid["composite_score"].values,
            valid["cve_density"].values,
        )
        with open(results_dir / "permutation_test.json", "w") as f:
            json.dump(perm, f, indent=2)

    # Step 5: Multivariate regression
    logger.info("Running regression analysis...")
    reg = regression_analysis(merged)
    with open(results_dir / "regression.json", "w") as f:
        json.dump(reg, f, indent=2, default=str)

    # Step 6: Weight sensitivity
    logger.info("Running weight sensitivity analysis...")
    sensitivity = weight_sensitivity_analysis(merged, outcome="cve_density")
    if not sensitivity.empty:
        sensitivity.to_parquet(results_dir / "weight_sensitivity.parquet")

    # Step 7: Plots
    logger.info("Generating figures...")
    plot_opacity_vs_cve(merged, results_dir / "opacity_vs_cve.pdf")
    plot_opacity_vs_cve(merged, results_dir / "opacity_vs_cve.png")
    plot_weight_sensitivity(sensitivity, results_dir / "weight_sensitivity.pdf")
    plot_weight_sensitivity(sensitivity, results_dir / "weight_sensitivity.png")

    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("KEY RESULTS")
    logger.info("=" * 40)
    for key, val in correlations.items():
        if "composite_score" in key:
            logger.info(f"  {key}: ρ={val['spearman_rho']:.4f} (p={val['p_value']:.4f})")

    logger.info("Experiment 3 complete.")


if __name__ == "__main__":
    main()
