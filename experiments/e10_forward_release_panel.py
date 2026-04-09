"""Experiment 10: forward-looking release-level panel validation."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from cku_sim.analysis.forward_panel import (
    build_forward_panel,
    evaluate_forward_prediction,
    plot_forward_event_rate_by_quartile,
    plot_forward_model_comparison,
    summarise_forward_panel,
)
from cku_sim.core.config import Config, DEFAULT_CORPUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 10: forward-looking release-level panel validation"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated repo names to include",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        default=20,
        help="Maximum release tags per repository",
    )
    parser.add_argument(
        "--min-tag-gap-days",
        type=int,
        default=30,
        help="Minimum spacing between sampled release tags",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=365,
        help="Future outcome window in days",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=None,
        help="Restrict sampled release tags to the trailing fully observed lookback window",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e10_forward_release_panel",
        help="Results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    if args.repos:
        selected = {name.strip() for name in args.repos.split(",") if name.strip()}
        corpus = [entry for entry in corpus if entry.name in selected]

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    repo_paths = {
        entry.name: config.raw_dir / entry.name
        for entry in corpus
        if (config.raw_dir / entry.name).exists()
    }

    logger.info("=" * 60)
    logger.info("Experiment 10: Forward-looking release panel")
    logger.info("=" * 60)
    logger.info(
        "Repos=%d, horizon=%d days, max_tags=%d, min_tag_gap_days=%d",
        len(repo_paths),
        args.horizon_days,
        args.max_tags,
        args.min_tag_gap_days,
    )

    panel_df, events_df = build_forward_panel(
        repo_paths,
        corpus,
        config,
        max_tags=args.max_tags,
        min_tag_gap_days=args.min_tag_gap_days,
        horizon_days=args.horizon_days,
        lookback_years=args.lookback_years,
    )
    if panel_df.empty:
        logger.error("No usable panel rows were generated.")
        return

    prediction_df, fold_metrics_df, prediction_summary = evaluate_forward_prediction(panel_df)
    summary = {
        "panel": summarise_forward_panel(panel_df),
        "prediction": prediction_summary,
        "horizon_days": args.horizon_days,
        "max_tags": args.max_tags,
        "min_tag_gap_days": args.min_tag_gap_days,
        "lookback_years": args.lookback_years,
    }

    panel_df.to_parquet(results_dir / "release_panel.parquet")
    panel_df.to_csv(results_dir / "release_panel.csv", index=False)
    events_df.to_csv(results_dir / "forward_events.csv", index=False)
    prediction_df.to_parquet(results_dir / "heldout_predictions.parquet")
    prediction_df.to_csv(results_dir / "heldout_predictions.csv", index=False)
    fold_metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_forward_event_rate_by_quartile(panel_df, results_dir / "future_event_rate_by_quartile.pdf")
    plot_forward_event_rate_by_quartile(panel_df, results_dir / "future_event_rate_by_quartile.png")
    plot_forward_model_comparison(prediction_summary, results_dir / "model_comparison.pdf")
    plot_forward_model_comparison(prediction_summary, results_dir / "model_comparison.png")

    panel_summary = summary["panel"]
    logger.info(
        "Panel: %d snapshots across %d repos, future-event rate %.3f, mean future-event count %.3f",
        panel_summary["n_snapshots"],
        panel_summary["n_repos"],
        panel_summary["future_event_rate"],
        panel_summary["mean_future_event_count"],
    )
    if panel_summary.get("association_models"):
        models = panel_summary["association_models"]
        if "logit_future_any_event" in models:
            model = models["logit_future_any_event"]
            logger.info(
                "Logit composite coefficient: %.4f (p=%.4g, 95%% CI [%.4f, %.4f])",
                model["coef"],
                model["pvalue"],
                model["ci_lo"],
                model["ci_hi"],
            )
        if "poisson_future_event_count" in models:
            model = models["poisson_future_event_count"]
            logger.info(
                "Poisson composite coefficient: %.4f (p=%.4g, 95%% CI [%.4f, %.4f])",
                model["coef"],
                model["pvalue"],
                model["ci_lo"],
                model["ci_hi"],
            )

    if prediction_summary.get("models"):
        base = prediction_summary["models"]["baseline_size"]
        plus = prediction_summary["models"]["baseline_plus_composite"]
        logger.info(
            "Held-out AUC: baseline=%.3f baseline+composite=%.3f",
            base["roc_auc"],
            plus["roc_auc"],
        )
        logger.info(
            "Held-out AP: baseline=%.3f baseline+composite=%.3f",
            base["average_precision"],
            plus["average_precision"],
        )


if __name__ == "__main__":
    main()
