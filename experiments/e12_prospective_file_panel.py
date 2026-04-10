"""Experiment 12: prospective file-level panel for later security-fix involvement."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.analysis.predictive_validation import plot_pooled_roc_curves
from cku_sim.analysis.prospective_file_panel import (
    PROSPECTIVE_GROUND_TRUTH_METADATA,
    build_prospective_file_panel,
    build_prospective_prediction_dataset,
    evaluate_leave_one_repo_out,
    fit_repo_fixed_effect_models,
    plot_model_comparison,
    plot_repo_deltas,
    prospective_policy_metadata,
    sample_audit_rows,
    summarise_prospective_pairs,
    summarise_repo_pairs,
)
from cku_sim.core.config import Config, DEFAULT_CORPUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _plot_delta_histogram(pairs: pd.DataFrame, output_path: Path) -> None:
    if pairs.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairs["delta_composite"], bins=30, color="#4C78A8", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Case - control composite opacity")
    ax.set_ylabel("Matched file pairs")
    ax.set_title("Prospective file-level opacity delta")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 12: prospective file-level panel"
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
        default=2,
        help="Maximum release tags per repository",
    )
    parser.add_argument(
        "--min-tag-gap-days",
        type=int,
        default=365,
        help="Minimum spacing between sampled release tags",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=730,
        help="Forward outcome window in days",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=10,
        help="Restrict sampled release tags to a recent fully observed window",
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=20,
        help="Minimum LOC for case/control files",
    )
    parser.add_argument(
        "--ground-truth-policy",
        type=str,
        default="expanded_advisory_plus_explicit",
        choices=sorted(PROSPECTIVE_GROUND_TRUTH_METADATA),
        help="Future-event definition policy",
    )
    parser.add_argument(
        "--severity-band",
        type=str,
        default="all",
        choices=["all", "high_critical"],
        help="Severity filter for future events",
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=40,
        help="Number of distinct event observations to sample for manual audit",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e12_prospective_file_panel",
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
    policy_meta = prospective_policy_metadata(args.ground_truth_policy)

    logger.info("=" * 60)
    logger.info("Experiment 12: Prospective file-level panel")
    logger.info("=" * 60)
    logger.info("Ground-truth policy: %s", policy_meta["label"])
    logger.info(
        "Repos=%d, horizon=%d days, max_tags=%d, min_tag_gap_days=%d, lookback_years=%d, severity_band=%s",
        len(repo_paths),
        args.horizon_days,
        args.max_tags,
        args.min_tag_gap_days,
        args.lookback_years,
        args.severity_band,
    )

    pairs_df, event_catalog_df, audit_df = build_prospective_file_panel(
        repo_paths,
        corpus,
        config,
        max_tags=args.max_tags,
        min_tag_gap_days=args.min_tag_gap_days,
        horizon_days=args.horizon_days,
        lookback_years=args.lookback_years,
        min_loc=args.min_loc,
        ground_truth_policy=args.ground_truth_policy,
        severity_band=args.severity_band,
    )
    if pairs_df.empty:
        logger.error("No usable prospective matched pairs were generated.")
        return

    dataset = build_prospective_prediction_dataset(pairs_df)
    if dataset["repo"].nunique() >= 2:
        predictions_df, fold_metrics_df, prediction_summary = evaluate_leave_one_repo_out(dataset)
    else:
        predictions_df = pd.DataFrame()
        fold_metrics_df = pd.DataFrame()
        prediction_summary = {
            "note": "Leave-one-repository-out evaluation requires at least two repositories."
        }
    repo_summary_df = summarise_repo_pairs(pairs_df)
    pair_summary = summarise_prospective_pairs(pairs_df)
    fixed_effects = (
        fit_repo_fixed_effect_models(dataset)
        if dataset["repo"].nunique() >= 2
        else {
            "note": "Repository fixed-effects estimation requires at least two repositories."
        }
    )
    audit_sample_df = sample_audit_rows(
        audit_df,
        sample_size=args.audit_sample_size,
        random_seed=config.random_seed,
    )

    summary = {
        "pair_level": pair_summary,
        "prediction": prediction_summary,
        "repo_fixed_effects": fixed_effects,
        "event_catalog": {
            "n_rows": int(len(event_catalog_df)),
            "n_event_observations": int(event_catalog_df["event_id"].nunique())
            if not event_catalog_df.empty
            else 0,
            "n_snapshot_keys": int(event_catalog_df["snapshot_key"].nunique())
            if not event_catalog_df.empty
            else 0,
            "source_breakdown": (
                event_catalog_df["source"].value_counts().to_dict()
                if not event_catalog_df.empty and "source" in event_catalog_df
                else {}
            ),
            "source_family_breakdown": (
                event_catalog_df["source_family"].value_counts().to_dict()
                if not event_catalog_df.empty and "source_family" in event_catalog_df
                else {}
            ),
            "severity_label_breakdown": (
                event_catalog_df["severity_label"].fillna("UNKNOWN").value_counts().to_dict()
                if not event_catalog_df.empty and "severity_label" in event_catalog_df
                else {}
            ),
            "known_severity_rate": (
                float(
                    (
                        event_catalog_df["severity_label"].fillna("UNKNOWN").ne("UNKNOWN")
                        | event_catalog_df["severity_score"].notna()
                    ).mean()
                )
                if not event_catalog_df.empty and "severity_label" in event_catalog_df
                else 0.0
            ),
        },
        "audit": {
            "n_rows_full": int(len(audit_df)),
            "n_rows_sampled": int(len(audit_sample_df)),
        },
        "parameters": {
            "max_tags": args.max_tags,
            "min_tag_gap_days": args.min_tag_gap_days,
            "horizon_days": args.horizon_days,
            "lookback_years": args.lookback_years,
            "min_loc": args.min_loc,
            "ground_truth_policy": args.ground_truth_policy,
            "ground_truth_label": policy_meta["label"],
            "ground_truth_source_filter": policy_meta.get("source_filter_label", ""),
            "severity_band": args.severity_band,
            "results_subdir": args.results_subdir,
        },
    }

    pairs_df.to_parquet(results_dir / "pair_level.parquet")
    pairs_df.to_csv(results_dir / "pair_level.csv", index=False)
    event_catalog_df.to_csv(results_dir / "event_catalog.csv", index=False)
    audit_df.to_csv(results_dir / "audit_full.csv", index=False)
    audit_sample_df.to_csv(results_dir / "audit_sample.csv", index=False)
    repo_summary_df.to_csv(results_dir / "repo_summary.csv", index=False)
    dataset.to_parquet(results_dir / "file_level_dataset.parquet")
    dataset.to_csv(results_dir / "file_level_dataset.csv", index=False)
    predictions_df.to_parquet(results_dir / "heldout_predictions.parquet")
    predictions_df.to_csv(results_dir / "heldout_predictions.csv", index=False)
    fold_metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    _plot_delta_histogram(pairs_df, results_dir / "delta_histogram.pdf")
    _plot_delta_histogram(pairs_df, results_dir / "delta_histogram.png")
    plot_repo_deltas(repo_summary_df, results_dir / "delta_by_repo.pdf")
    plot_repo_deltas(repo_summary_df, results_dir / "delta_by_repo.png")
    if prediction_summary.get("models"):
        plot_model_comparison(prediction_summary, results_dir / "model_comparison.pdf")
        plot_model_comparison(prediction_summary, results_dir / "model_comparison.png")
    if not predictions_df.empty:
        plot_pooled_roc_curves(predictions_df, results_dir / "pooled_roc.pdf")
        plot_pooled_roc_curves(predictions_df, results_dir / "pooled_roc.png")

    logger.info("\n" + "=" * 40)
    logger.info("KEY RESULTS")
    logger.info("=" * 40)
    logger.info(
        "Pairs: %d across %d future event observations in %d repos and %d snapshots",
        pair_summary["n_pairs"],
        pair_summary["n_events"],
        pair_summary["n_repos"],
        pair_summary["n_snapshots"],
    )
    logger.info(
        "Pair-level composite delta: mean=%.4f median=%.4f positive_share=%.3f",
        pair_summary["mean_delta_composite"],
        pair_summary["median_delta_composite"],
        pair_summary["positive_share"],
    )
    pair_bootstrap = pair_summary.get("bootstrap_primary_cluster", {})
    if pair_bootstrap:
        median_ci = pair_bootstrap.get("median_delta_composite_ci")
        if isinstance(median_ci, list) and len(median_ci) == 2:
            logger.info(
                "Pair-level clustered 95%% CI for median delta (%s): [%.4f, %.4f]",
                pair_bootstrap.get("cluster_col"),
                median_ci[0],
                median_ci[1],
            )
    if "wilcoxon_pvalue_greater" in pair_summary:
        logger.info(
            "Pair-level Wilcoxon one-sided p=%.4g",
            pair_summary["wilcoxon_pvalue_greater"],
        )
    if prediction_summary.get("models"):
        for model_name, metrics in prediction_summary["models"].items():
            logger.info(
                "  %s: AUC=%.3f AP=%.3f Brier=%.3f PairAcc=%.3f",
                model_name,
                metrics["roc_auc"],
                metrics["average_precision"],
                metrics["brier_score"],
                metrics["pairwise_accuracy"],
            )
    if fixed_effects.get("baseline_plus_composite_repo_fixed_effects"):
        model = fixed_effects["baseline_plus_composite_repo_fixed_effects"]
        logger.info(
            "Repo fixed effects composite coefficient: %.4f (p=%.4g, 95%% CI [%.4f, %.4f])",
            model["composite_coef"],
            model["composite_pvalue"],
            model["composite_ci_lo"],
            model["composite_ci_hi"],
        )


if __name__ == "__main__":
    main()
