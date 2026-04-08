"""Experiment 7: leave-one-repository-out predictive validation."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from cku_sim.analysis.predictive_validation import (
    build_prediction_dataset,
    evaluate_leave_one_repo_out,
    plot_model_comparison,
    plot_pooled_roc_curves,
)
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 7: predictive validation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--e06-subdir",
        type=str,
        default="e06_file_case_control",
        help="Results subdirectory containing experiment 6 pair_level.parquet",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=None,
        help="Optional results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    results_subdir = args.results_subdir or (
        "e07_predictive_validation"
        if args.e06_subdir == "e06_file_case_control"
        else f"e07_predictive_validation__{args.e06_subdir}"
    )
    results_dir = config.results_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = config.results_dir / args.e06_subdir / "pair_level.parquet"
    if not pairs_path.exists():
        logger.error(
            "File-level pair data not found. Run experiment 6 first:\n"
            "  python -m experiments.e06_file_level_case_control --config experiments/config.yaml"
        )
        return

    pairs_df = pd.read_parquet(pairs_path)
    dataset = build_prediction_dataset(pairs_df)
    predictions, fold_metrics, summary = evaluate_leave_one_repo_out(dataset)

    dataset.to_parquet(results_dir / "file_level_dataset.parquet")
    dataset.to_csv(results_dir / "file_level_dataset.csv", index=False)
    predictions.to_parquet(results_dir / "heldout_predictions.parquet")
    predictions.to_csv(results_dir / "heldout_predictions.csv", index=False)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_model_comparison(summary, results_dir / "model_comparison.pdf")
    plot_model_comparison(summary, results_dir / "model_comparison.png")
    plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.pdf")
    plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.png")

    logger.info("=" * 60)
    logger.info("Experiment 7: Predictive validation")
    logger.info("=" * 60)
    logger.info(
        "Dataset: %d files from %d matched pairs across %d commits and %d repos",
        summary["n_files"],
        summary["n_pairs"],
        summary["n_commits"],
        summary["n_repos"],
    )
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
