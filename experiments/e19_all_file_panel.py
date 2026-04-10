"""Experiment 19: audited all-file prospective panel."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.analysis.audited_panel import (
    PRIMARY_BASELINE_MODEL,
    PRIMARY_EXTERNAL_CANDIDATES,
    PRIMARY_PLUS_MODEL,
    build_audited_all_file_panel,
    evaluate_all_file_leave_one_repo_out,
    fit_repo_fixed_effect_models_all_file,
    load_audited_security_table,
    load_repo_paths,
    seed_security_audit_table,
    summarise_all_file_panel,
    summarise_audit_table,
    summarise_repo_panel,
)
from cku_sim.analysis.predictive_validation import plot_pooled_roc_curves
from cku_sim.core.config import Config, DEFAULT_CORPUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_AUDIT_SOURCES = [
    "e12_prospective_file_panel__curated15_h730_l10_t5__supported",
    "e12_prospective_file_panel__external_python5_h730_l10_t5__supported",
    "e12_prospective_file_panel__external_mix9_h730_l10_t3__supported",
    "e12_prospective_file_panel__external_holdout_flask_requests_h730_l10_t5",
]


def _plot_model_comparison(summary: dict[str, object], output_path: Path) -> None:
    models = summary.get("models", {})
    if not models:
        return
    model_names = list(models.keys())
    aucs = [models[name]["roc_auc"] for name in model_names]
    aps = [models[name]["average_precision"] for name in model_names]
    briers = [models[name]["brier_score"] for name in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].bar(model_names, aucs, color="#4C78A8")
    axes[0].axhline(0.5, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title("Held-out discrimination")

    axes[1].bar(model_names, aps, color="#72B7B2")
    axes[1].set_ylabel("Average precision")
    axes[1].set_title("Held-out precision-recall")

    axes[2].bar(model_names, briers, color="#F58518")
    axes[2].set_ylabel("Brier score")
    axes[2].set_title("Held-out calibration loss")

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 19: audited all-file prospective panel"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--audit-path",
        type=str,
        default="data/processed/security_event_file_audit_curated.csv",
        help="Path to the curated security audit CSV",
    )
    parser.add_argument(
        "--seed-audit",
        action="store_true",
        help="Refresh the curated security audit table from event catalogs before building the panel",
    )
    parser.add_argument(
        "--seed-sources",
        type=str,
        default=",".join(DEFAULT_AUDIT_SOURCES),
        help="Comma-separated results subdirectories containing event_catalog.csv files",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Optional comma-separated repo names to include",
    )
    parser.add_argument(
        "--include-external-candidates",
        action="store_true",
        help="Include the deterministic external-holdout candidate repositories in the primary audited training panel",
    )
    parser.add_argument("--max-tags", type=int, default=5, help="Maximum release tags per repo")
    parser.add_argument(
        "--min-tag-gap-days",
        type=int,
        default=365,
        help="Minimum spacing between sampled release tags",
    )
    parser.add_argument("--horizon-days", type=int, default=730, help="Forward horizon in days")
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=10,
        help="Trailing fully observed lookback window for snapshot sampling",
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=5,
        help="Minimum LOC for eligible files in the audited all-file panel",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e19_all_file_panel__curated15_h730_l10_t5__audited",
        help="Results subdirectory under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    audit_path = Path(args.audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    if args.seed_audit or not audit_path.exists():
        seed_paths = [
            config.results_dir / subdir.strip() / "event_catalog.csv"
            for subdir in args.seed_sources.split(",")
            if subdir.strip()
        ]
        seeded = seed_security_audit_table(seed_paths)
        seeded.to_csv(audit_path, index=False)
        logger.info("Seeded security audit table with %d rows at %s", len(seeded), audit_path)

    security_audit = load_audited_security_table(audit_path)
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    if args.repos:
        selected = {value.strip() for value in args.repos.split(",") if value.strip()}
        corpus = [entry for entry in corpus if entry.name in selected]
    elif not args.include_external_candidates:
        corpus = [
            entry
            for entry in corpus
            if entry.name not in set(PRIMARY_EXTERNAL_CANDIDATES)
        ]
    repo_paths = load_repo_paths(config, corpus)

    dataset = build_audited_all_file_panel(
        repo_paths,
        corpus,
        config,
        security_audit,
        repos=[entry.name for entry in corpus],
        max_tags=args.max_tags,
        min_tag_gap_days=args.min_tag_gap_days,
        horizon_days=args.horizon_days,
        lookback_years=args.lookback_years,
        min_loc=args.min_loc,
    )
    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    if dataset.empty:
        logger.error("No audited all-file rows were generated.")
        return

    predictions, fold_metrics, prediction_summary = evaluate_all_file_leave_one_repo_out(dataset)
    repo_summary = summarise_repo_panel(dataset)
    panel_summary = summarise_all_file_panel(dataset)
    audit_summary = summarise_audit_table(security_audit)
    fixed_effects = fit_repo_fixed_effect_models_all_file(dataset)

    summary = {
        "audit": audit_summary,
        "panel": panel_summary,
        "prediction": prediction_summary,
        "repo_fixed_effects": fixed_effects,
        "parameters": {
            "audit_path": str(audit_path),
            "max_tags": args.max_tags,
            "min_tag_gap_days": args.min_tag_gap_days,
            "horizon_days": args.horizon_days,
            "lookback_years": args.lookback_years,
            "min_loc": args.min_loc,
            "primary_baseline_model": PRIMARY_BASELINE_MODEL,
            "primary_plus_model": PRIMARY_PLUS_MODEL,
            "results_subdir": args.results_subdir,
            "include_external_candidates": bool(args.include_external_candidates),
        },
        "gating": {
            "has_no_range_only_labels": bool(
                not security_audit["source_family"].isin({"range_only"}).any()
            ),
            "train_repos_disjoint_from_primary_external_candidates": bool(
                not set(dataset["repo"].unique()).intersection(PRIMARY_EXTERNAL_CANDIDATES)
            ),
        },
    }

    dataset.to_parquet(results_dir / "file_level_dataset.parquet")
    dataset.to_csv(results_dir / "file_level_dataset.csv", index=False)
    predictions.to_parquet(results_dir / "heldout_predictions.parquet")
    predictions.to_csv(results_dir / "heldout_predictions.csv", index=False)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    repo_summary.to_csv(results_dir / "repo_summary.csv", index=False)
    security_audit.to_csv(results_dir / "accepted_security_audit.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    if prediction_summary.get("models"):
        _plot_model_comparison(prediction_summary, results_dir / "model_comparison.png")
        _plot_model_comparison(prediction_summary, results_dir / "model_comparison.pdf")
    if not predictions.empty:
        plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.png")
        plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.pdf")

    logger.info("Audited all-file panel: %d files, %d positive files, %d repos", panel_summary["n_files"], panel_summary["n_positive_files"], panel_summary["n_repos"])
    if prediction_summary.get("models"):
        baseline = prediction_summary["models"].get(PRIMARY_BASELINE_MODEL, {})
        plus = prediction_summary["models"].get(PRIMARY_PLUS_MODEL, {})
        if baseline and plus:
            logger.info(
                "Primary comparison (%s -> %s): AUC %.3f -> %.3f, AP %.3f -> %.3f, Brier %.3f -> %.3f",
                PRIMARY_BASELINE_MODEL,
                PRIMARY_PLUS_MODEL,
                baseline.get("roc_auc", float("nan")),
                plus.get("roc_auc", float("nan")),
                baseline.get("average_precision", float("nan")),
                plus.get("average_precision", float("nan")),
                baseline.get("brier_score", float("nan")),
                plus.get("brier_score", float("nan")),
            )


if __name__ == "__main__":
    main()
