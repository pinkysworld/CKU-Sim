"""Experiment 16: frozen external holdout for the prospective file-level panel."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from cku_sim.analysis.predictive_validation import plot_pooled_roc_curves
from cku_sim.analysis.prospective_file_panel import (
    build_prospective_prediction_dataset,
    evaluate_external_holdout,
    plot_model_comparison,
)
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 16: frozen external holdout validation for the prospective panel"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--train-e12-subdir",
        type=str,
        default="e12_prospective_file_panel__curated15_h730_l10_t5",
        help="Experiment-12 results subdirectory used for frozen training",
    )
    parser.add_argument(
        "--holdout-e12-subdir",
        type=str,
        default="e12_prospective_file_panel__external_holdout_h730_l10_t5",
        help="Experiment-12 results subdirectory used for external holdout evaluation",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e16_external_holdout__curated15_to_external",
        help="Results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    train_pairs_path = config.results_dir / args.train_e12_subdir / "pair_level.parquet"
    holdout_pairs_path = config.results_dir / args.holdout_e12_subdir / "pair_level.parquet"
    if not train_pairs_path.exists():
        logger.error("Training pair data not found: %s", train_pairs_path)
        return
    if not holdout_pairs_path.exists():
        logger.error("Holdout pair data not found: %s", holdout_pairs_path)
        return

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = pd.read_parquet(train_pairs_path)
    holdout_pairs = pd.read_parquet(holdout_pairs_path)
    train_dataset = build_prospective_prediction_dataset(train_pairs)
    holdout_dataset = build_prospective_prediction_dataset(holdout_pairs)
    predictions_df, repo_metrics_df, summary = evaluate_external_holdout(
        train_dataset,
        holdout_dataset,
    )
    summary["train_e12_subdir"] = args.train_e12_subdir
    summary["holdout_e12_subdir"] = args.holdout_e12_subdir

    predictions_df.to_parquet(results_dir / "heldout_predictions.parquet")
    predictions_df.to_csv(results_dir / "heldout_predictions.csv", index=False)
    repo_metrics_df.to_csv(results_dir / "repo_metrics.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    if summary.get("models"):
        plot_model_comparison(summary, results_dir / "model_comparison.pdf")
        plot_model_comparison(summary, results_dir / "model_comparison.png")
    if not predictions_df.empty:
        plot_pooled_roc_curves(predictions_df, results_dir / "pooled_roc.pdf")
        plot_pooled_roc_curves(predictions_df, results_dir / "pooled_roc.png")

    logger.info(
        "External holdout: train repos=%d, holdout repos=%d, holdout pairs=%d",
        summary["n_train_repos"],
        summary["n_holdout_repos"],
        summary["n_holdout_pairs"],
    )
    if summary.get("models"):
        for model_name, metrics in summary["models"].items():
            logger.info(
                "  %s: AUC=%.3f AP=%.3f Brier=%.3f PairAcc=%.3f",
                model_name,
                metrics["roc_auc"],
                metrics["average_precision"],
                metrics["brier_score"],
                metrics["pairwise_accuracy"],
            )


if __name__ == "__main__":
    main()
