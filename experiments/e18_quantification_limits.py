"""Experiment 18: calibration and model-disagreement diagnostics for the prospective panel."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from cku_sim.analysis.quantification_limits import (
    build_calibration_curve_table,
    build_model_disagreement_frame,
    merge_prediction_diagnostics,
    plot_calibration_curves,
    plot_quantification_limits,
    summarise_calibration_by_stratum,
    summarise_disagreement_by_stratum,
    summarise_opacity_strata,
    summarise_quantification_limits,
)
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 18: calibration and model-disagreement diagnostics"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--e12-subdir",
        type=str,
        default="e12_prospective_file_panel__curated15_h730_l10_t5__supported",
        help="Existing e12 results subdirectory containing dataset and held-out predictions",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=None,
        help="Optional results subdirectory name under data/results/",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=6,
        help="Number of rank-based probability bins for calibration summaries",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of cluster bootstrap resamples for high-vs-low-opacity gaps",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    source_dir = config.results_dir / args.e12_subdir
    results_subdir = args.results_subdir or f"e18_quantification_limits__{args.e12_subdir}"
    results_dir = config.results_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = source_dir / "file_level_dataset.parquet"
    predictions_path = source_dir / "heldout_predictions.parquet"
    if not dataset_path.exists() or not predictions_path.exists():
        logger.error(
            "Missing required e12 artifacts under %s (dataset=%s, predictions=%s)",
            source_dir,
            dataset_path.exists(),
            predictions_path.exists(),
        )
        return

    dataset = pd.read_parquet(dataset_path)
    predictions = pd.read_parquet(predictions_path)

    scored_predictions = merge_prediction_diagnostics(dataset, predictions)
    disagreement = build_model_disagreement_frame(scored_predictions)
    opacity_summary = summarise_opacity_strata(disagreement)
    calibration_summary = summarise_calibration_by_stratum(
        scored_predictions,
        n_bins=args.n_bins,
    )
    disagreement_summary = summarise_disagreement_by_stratum(disagreement)
    calibration_curve = build_calibration_curve_table(
        scored_predictions,
        n_bins=args.n_bins,
    )
    summary = summarise_quantification_limits(
        scored_predictions,
        disagreement,
        calibration_summary,
        n_bins=args.n_bins,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )
    summary["source_e12_subdir"] = args.e12_subdir
    summary["n_bins"] = int(args.n_bins)
    summary["n_bootstrap"] = int(args.n_bootstrap)

    scored_predictions.to_parquet(results_dir / "scored_predictions.parquet")
    scored_predictions.to_csv(results_dir / "scored_predictions.csv", index=False)
    disagreement.to_parquet(results_dir / "file_level_disagreement.parquet")
    disagreement.to_csv(results_dir / "file_level_disagreement.csv", index=False)
    opacity_summary.to_csv(results_dir / "opacity_strata_summary.csv", index=False)
    calibration_summary.to_csv(results_dir / "calibration_summary.csv", index=False)
    disagreement_summary.to_csv(results_dir / "disagreement_summary.csv", index=False)
    calibration_curve.to_csv(results_dir / "calibration_curve.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_quantification_limits(
        opacity_summary,
        calibration_summary,
        disagreement_summary,
        output_path=results_dir / "quantification_limits.png",
    )
    plot_quantification_limits(
        opacity_summary,
        calibration_summary,
        disagreement_summary,
        output_path=results_dir / "quantification_limits.pdf",
    )
    plot_calibration_curves(
        calibration_curve,
        output_path=results_dir / "calibration_curves.png",
    )
    plot_calibration_curves(
        calibration_curve,
        output_path=results_dir / "calibration_curves.pdf",
    )

    logger.info("=" * 60)
    logger.info("Experiment 18: quantification-limit diagnostics")
    logger.info("=" * 60)
    logger.info("Files: %d | Pairs: %d | Events: %d | Repos: %d", summary["n_files"], summary["n_pairs"], summary["n_events"], summary["n_repos"])
    logger.info(
        "Primary model high-vs-low ECE gap: %.4f",
        (
            summary.get("primary_model_calibration", {})
            .get("high_minus_low_ece", {})
            .get("observed_gap", float("nan"))
        ),
    )
    logger.info(
        "High-vs-low model score-range gap: %.4f",
        (
            summary.get("model_disagreement", {})
            .get("high_minus_low_score_range", {})
            .get("observed_gap", float("nan"))
        ),
    )


if __name__ == "__main__":
    main()
