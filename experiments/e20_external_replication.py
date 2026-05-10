"""Experiment 20: audited frozen external replication for the all-file panel."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cku_sim.analysis.audited_panel import (
    PRIMARY_BASELINE_MODEL,
    PRIMARY_EXTERNAL_CANDIDATES,
    PRIMARY_PLUS_MODEL,
    build_audited_all_file_panel,
    build_holdout_screen,
    evaluate_all_file_external_holdout,
    load_audited_security_table,
    load_repo_paths,
    seed_security_audit_table,
    split_corpora_for_external_replication,
    summarise_all_file_panel,
    summarise_audit_table,
    summarise_external_replication,
)
from cku_sim.analysis.predictive_validation import plot_model_comparison, plot_pooled_roc_curves
from cku_sim.analysis.prospective_file_panel import PROSPECTIVE_MODEL_SPECS
from cku_sim.analysis.quantification_limits import expected_calibration_error
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
    "e12_prospective_file_panel__external_django_traefik_h730_l10_t12_g90__supported",
    "e12_prospective_file_panel__external_fastapi_requests_scrapy_h730_l10_t12_g90__supported",
    "e12_prospective_file_panel__external_remaining5_nocpython_h730_l10_t5__supported",
]

E20_MODEL_SPECS = {
    **PROSPECTIVE_MODEL_SPECS,
    "gradient_boosted_trees_history_plus_structure": {
        **PROSPECTIVE_MODEL_SPECS[PRIMARY_BASELINE_MODEL],
        "estimator": "hist_gradient_boosting",
    },
}

DEFAULT_COMPOSITE_WEIGHTS = {
    "compressibility": 0.35,
    "entropy": 0.25,
    "cyclomatic_density": 0.25,
    "halstead_volume": 0.15,
}


def _config_from_path(path: str | None) -> Config:
    if path:
        return Config.from_yaml(path)
    return Config()


def _with_ece(
    predictions: pd.DataFrame,
    repo_metrics: pd.DataFrame,
    summary: dict[str, object],
    *,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if predictions.empty:
        return repo_metrics, summary

    repo_metrics = repo_metrics.copy()
    ece_rows: list[dict[str, object]] = []
    for (model_name, repo), group in predictions.groupby(["model", "repo"], sort=False):
        ece_rows.append(
            {
                "model": str(model_name),
                "repo": str(repo),
                "ece": expected_calibration_error(group["label"], group["score"], n_bins=n_bins),
            }
        )
    if ece_rows:
        repo_metrics = repo_metrics.merge(
            pd.DataFrame(ece_rows),
            on=["model", "repo"],
            how="left",
        )
    for model_name, group in predictions.groupby("model", sort=False):
        model_summary = summary.setdefault("models", {}).setdefault(str(model_name), {})
        model_summary["ece"] = expected_calibration_error(
            group["label"], group["score"], n_bins=n_bins
        )
        model_summary["pairwise_accuracy"] = float("nan")
    return repo_metrics, summary


def _load_panel(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    panel_path = Path(path)
    if panel_path.suffix.lower() == ".csv":
        return pd.read_csv(panel_path)
    return pd.read_parquet(panel_path)


def _implied_compressibility_component(df: pd.DataFrame) -> pd.Series:
    """Recover the released mean-compressibility component from default composite."""
    weights = DEFAULT_COMPOSITE_WEIGHTS
    return (
        df["composite_score"]
        - weights["entropy"] * df["shannon_entropy"]
        - weights["cyclomatic_density"] * df["cyclomatic_density"]
        - weights["halstead_volume"] * df["halstead_volume"]
    ) / weights["compressibility"]


def _weighted_composite(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(weights.values())
    return (
        weights["compressibility"] * _implied_compressibility_component(df)
        + weights["entropy"] * df["shannon_entropy"]
        + weights["cyclomatic_density"] * df["cyclomatic_density"]
        + weights["halstead_volume"] * df["halstead_volume"]
    ) / total


def run_holdout_weight_sensitivity(
    train_dataset: pd.DataFrame,
    holdout_dataset: pd.DataFrame,
    baseline_metrics: dict[str, float],
    *,
    n_samples: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Perturb composite weights and refit the file-level holdout endpoint."""
    rng = np.random.default_rng(seed)
    numeric = [
        "weighted_composite_score" if value == "composite_score" else value
        for value in PROSPECTIVE_MODEL_SPECS[PRIMARY_PLUS_MODEL]["numeric"]
    ]
    model_specs = {
        "baseline_plus_weighted_composite": {
            "numeric": numeric,
            "categorical": PROSPECTIVE_MODEL_SPECS[PRIMARY_PLUS_MODEL]["categorical"],
        }
    }

    rows: list[dict[str, object]] = []
    for draw in range(n_samples):
        sampled = rng.dirichlet(np.ones(4))
        weights = {
            "compressibility": float(sampled[0]),
            "entropy": float(sampled[1]),
            "cyclomatic_density": float(sampled[2]),
            "halstead_volume": float(sampled[3]),
        }
        train = train_dataset.copy()
        holdout = holdout_dataset.copy()
        train["weighted_composite_score"] = _weighted_composite(train, weights)
        holdout["weighted_composite_score"] = _weighted_composite(holdout, weights)

        _, _, summary = evaluate_all_file_external_holdout(train, holdout, model_specs)
        metrics = summary["models"]["baseline_plus_weighted_composite"]
        rows.append(
            {
                "draw": draw,
                "w_compressibility": weights["compressibility"],
                "w_entropy": weights["entropy"],
                "w_cyclomatic_density": weights["cyclomatic_density"],
                "w_halstead_volume": weights["halstead_volume"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "roc_auc_lift_vs_baseline": metrics["roc_auc"] - baseline_metrics["roc_auc"],
                "average_precision_lift_vs_baseline": (
                    metrics["average_precision"] - baseline_metrics["average_precision"]
                ),
                "brier_lift_vs_baseline": metrics["brier_score"] - baseline_metrics["brier_score"],
            }
        )
    return pd.DataFrame(rows)


def summarise_holdout_weight_sensitivity(sensitivity: pd.DataFrame) -> dict[str, object]:
    if sensitivity.empty:
        return {}

    def quantiles(column: str) -> dict[str, float]:
        values = sensitivity[column]
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
        }

    return {
        "n_draws": int(len(sensitivity)),
        "roc_auc_lift_vs_baseline": quantiles("roc_auc_lift_vs_baseline"),
        "average_precision_lift_vs_baseline": quantiles(
            "average_precision_lift_vs_baseline"
        ),
        "share_positive_auc_lift": float(
            (sensitivity["roc_auc_lift_vs_baseline"] > 0).mean()
        ),
        "share_positive_average_precision_lift": float(
            (sensitivity["average_precision_lift_vs_baseline"] > 0).mean()
        ),
    }


def plot_holdout_weight_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    if sensitivity.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(
        sensitivity["roc_auc_lift_vs_baseline"],
        bins=30,
        color="#4C78A8",
        edgecolor="white",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("ROC AUC lift versus baseline history + structure")
    ax.set_ylabel("Weight draws")
    ax.set_title("File-level holdout opacity-weight sensitivity")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 20: audited frozen external replication for the all-file panel"
    )
    parser.add_argument("--train-config", type=str, default="experiments/config.forward_panel_curated.yaml")
    parser.add_argument("--holdout-config", type=str, default="experiments/config.external_holdout.yaml")
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        default=None,
        help="Optional existing train dataset (.parquet or .csv) to reuse instead of rebuilding the train panel",
    )
    parser.add_argument(
        "--holdout-dataset-path",
        type=str,
        default=None,
        help="Optional existing holdout dataset (.parquet or .csv) to reuse instead of rebuilding the holdout panel",
    )
    parser.add_argument(
        "--holdout-repos",
        type=str,
        default=None,
        help="Optional comma-separated explicit holdout repos to use instead of the default candidate pool",
    )
    parser.add_argument(
        "--audit-path",
        type=str,
        default="data/processed/security_event_file_audit_curated.csv",
        help="Path to the curated security audit CSV",
    )
    parser.add_argument(
        "--seed-audit",
        action="store_true",
        help="Refresh the curated security audit table from event catalogs before external replication",
    )
    parser.add_argument(
        "--seed-sources",
        type=str,
        default=",".join(DEFAULT_AUDIT_SOURCES),
        help="Comma-separated results subdirectories containing event_catalog.csv files",
    )
    parser.add_argument("--max-tags", type=int, default=5)
    parser.add_argument("--min-tag-gap-days", type=int, default=365)
    parser.add_argument("--horizon-days", type=int, default=730)
    parser.add_argument("--lookback-years", type=int, default=10)
    parser.add_argument("--min-loc", type=int, default=5)
    parser.add_argument(
        "--train-repos",
        type=str,
        default=None,
        help="Optional comma-separated training repos to retain after the frozen train/holdout split",
    )
    parser.add_argument("--min-holdout-snapshots", type=int, default=3)
    parser.add_argument("--min-holdout-events", type=int, default=5)
    parser.add_argument("--weight-sensitivity-samples", type=int, default=1_000)
    parser.add_argument("--weight-sensitivity-seed", type=int, default=42)
    parser.add_argument(
        "--skip-weight-sensitivity",
        action="store_true",
        help="Skip the file-level opacity-weight sensitivity sweep",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e20_external_replication__audited_curated_to_external",
    )
    args = parser.parse_args()

    train_config = _config_from_path(args.train_config)
    holdout_config = _config_from_path(args.holdout_config)
    audit_path = Path(args.audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    if args.seed_audit or not audit_path.exists():
        seed_paths = [
            train_config.results_dir / subdir.strip() / "event_catalog.csv"
            for subdir in args.seed_sources.split(",")
            if subdir.strip()
        ]
        seeded = seed_security_audit_table(seed_paths)
        seeded.to_csv(audit_path, index=False)
        logger.info("Seeded security audit table with %d rows at %s", len(seeded), audit_path)

    security_audit = load_audited_security_table(audit_path)
    explicit_holdout_repos = (
        [value.strip() for value in args.holdout_repos.split(",") if value.strip()]
        if args.holdout_repos
        else None
    )
    screening = build_holdout_screen(
        security_audit,
        candidate_repos=explicit_holdout_repos or PRIMARY_EXTERNAL_CANDIDATES,
        min_snapshots=args.min_holdout_snapshots,
        min_events=args.min_holdout_events,
    )

    holdout_corpus = holdout_config.corpus if holdout_config.corpus else DEFAULT_CORPUS
    holdout_repo_paths = load_repo_paths(holdout_config, holdout_corpus)
    screening["in_holdout_config"] = screening["repo"].isin({entry.name for entry in holdout_corpus})
    screening["has_local_repo"] = screening["repo"].isin(set(holdout_repo_paths))
    screening["eligible_primary_holdout"] = (
        screening["eligible_holdout"]
        & screening["in_holdout_config"]
        & screening["has_local_repo"]
    )

    eligible_holdout_repos = screening.loc[
        screening["eligible_primary_holdout"], "repo"
    ].tolist()
    fallback_repos = screening.loc[
        (~screening["eligible_primary_holdout"])
        & screening["in_holdout_config"]
        & screening["has_local_repo"]
        & (screening["n_events"] > 0)
        & (screening["n_snapshots"] > 0),
        "repo",
    ].tolist()
    if explicit_holdout_repos:
        selected_holdout_repos = screening.loc[
            screening["in_holdout_config"]
            & screening["has_local_repo"]
            & (screening["n_events"] > 0)
            & (screening["n_snapshots"] > 0),
            "repo",
        ].tolist()
    else:
        selected_holdout_repos = eligible_holdout_repos or fallback_repos

    train_corpus = train_config.corpus if train_config.corpus else DEFAULT_CORPUS
    if explicit_holdout_repos:
        explicit_holdout_set = set(explicit_holdout_repos)
        train_corpus = [entry for entry in train_corpus if entry.name not in explicit_holdout_set]
        holdout_corpus = [entry for entry in holdout_corpus if entry.name in explicit_holdout_set]
    else:
        train_corpus, holdout_corpus = split_corpora_for_external_replication(
            train_corpus,
            holdout_corpus,
            candidate_repos=PRIMARY_EXTERNAL_CANDIDATES,
        )
    if args.train_repos:
        selected_train = {
            value.strip() for value in args.train_repos.split(",") if value.strip()
        }
        train_corpus = [entry for entry in train_corpus if entry.name in selected_train]
    holdout_corpus = [entry for entry in holdout_corpus if entry.name in set(selected_holdout_repos)]
    holdout_repo_paths = load_repo_paths(holdout_config, holdout_corpus)

    train_dataset = _load_panel(args.train_dataset_path)
    if train_dataset is None:
        train_repo_paths = load_repo_paths(train_config, train_corpus)
        train_dataset = build_audited_all_file_panel(
            train_repo_paths,
            train_corpus,
            train_config,
            security_audit,
            repos=[entry.name for entry in train_corpus],
            max_tags=args.max_tags,
            min_tag_gap_days=args.min_tag_gap_days,
            horizon_days=args.horizon_days,
            lookback_years=args.lookback_years,
            min_loc=args.min_loc,
        )
    holdout_dataset = _load_panel(args.holdout_dataset_path)
    if holdout_dataset is None:
        holdout_dataset = build_audited_all_file_panel(
            holdout_repo_paths,
            holdout_corpus,
            holdout_config,
            security_audit,
            repos=[entry.name for entry in holdout_corpus],
            max_tags=args.max_tags,
            min_tag_gap_days=args.min_tag_gap_days,
            horizon_days=args.horizon_days,
            lookback_years=args.lookback_years,
            min_loc=args.min_loc,
        )

    predictions, repo_metrics, summary = evaluate_all_file_external_holdout(
        train_dataset,
        holdout_dataset,
        E20_MODEL_SPECS,
    )
    repo_metrics, summary = _with_ece(predictions, repo_metrics, summary)
    results_dir = train_config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    summary["audit"] = summarise_audit_table(security_audit)
    summary["screening"] = {
        "candidate_repos": explicit_holdout_repos or PRIMARY_EXTERNAL_CANDIDATES,
        "eligible_holdout_repos": eligible_holdout_repos,
        "fallback_repos": fallback_repos,
        "selected_holdout_repos": selected_holdout_repos,
        "primary_holdout_rule_met": bool(eligible_holdout_repos),
    }
    summary["train_panel"] = summarise_all_file_panel(train_dataset)
    summary["holdout_panel"] = summarise_all_file_panel(holdout_dataset)
    summary["external_replication"] = summarise_external_replication(summary)
    summary["parameters"] = {
        "train_config": args.train_config,
        "holdout_config": args.holdout_config,
        "audit_path": str(audit_path),
        "max_tags": args.max_tags,
        "min_tag_gap_days": args.min_tag_gap_days,
        "horizon_days": args.horizon_days,
        "lookback_years": args.lookback_years,
        "min_loc": args.min_loc,
        "train_repos": sorted({str(value) for value in train_dataset["repo"].dropna().tolist()})
        if not train_dataset.empty and "repo" in train_dataset
        else [entry.name for entry in train_corpus],
        "train_dataset_path": args.train_dataset_path,
        "holdout_dataset_path": args.holdout_dataset_path,
        "min_holdout_snapshots": args.min_holdout_snapshots,
        "min_holdout_events": args.min_holdout_events,
        "weight_sensitivity_samples": args.weight_sensitivity_samples,
        "weight_sensitivity_seed": args.weight_sensitivity_seed,
    }
    summary["gating"] = {
        "has_no_range_only_labels": bool(
            not security_audit["source_family"].isin({"range_only"}).any()
        ),
        "train_holdout_disjoint": bool(
            set(train_dataset["repo"].unique()).isdisjoint(set(holdout_dataset["repo"].unique()))
            if not train_dataset.empty and not holdout_dataset.empty
            else True
        ),
        "holdout_min_positive_observations_met": bool(
            summary["holdout_panel"].get("n_positive_files", 0) >= 40
        ),
        "holdout_min_repo_count_met": bool(summary["holdout_panel"].get("n_repos", 0) >= 6),
    }

    screening.to_csv(results_dir / "screening.csv", index=False)
    train_dataset.to_parquet(results_dir / "train_file_level_dataset.parquet")
    train_dataset.to_csv(results_dir / "train_file_level_dataset.csv", index=False)
    holdout_dataset.to_parquet(results_dir / "holdout_file_level_dataset.parquet")
    holdout_dataset.to_csv(results_dir / "holdout_file_level_dataset.csv", index=False)
    predictions.to_parquet(results_dir / "heldout_predictions.parquet")
    predictions.to_csv(results_dir / "heldout_predictions.csv", index=False)
    repo_metrics.to_csv(results_dir / "repo_metrics.csv", index=False)
    security_audit.to_csv(results_dir / "accepted_security_audit.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    baseline = summary.get("models", {}).get(PRIMARY_BASELINE_MODEL, {})
    plus = summary.get("models", {}).get(PRIMARY_PLUS_MODEL, {})
    if not args.skip_weight_sensitivity and baseline:
        sensitivity = run_holdout_weight_sensitivity(
            train_dataset,
            holdout_dataset,
            baseline,
            n_samples=args.weight_sensitivity_samples,
            seed=args.weight_sensitivity_seed,
        )
        sensitivity.to_csv(results_dir / "holdout_weight_sensitivity.csv", index=False)
        sensitivity.to_parquet(results_dir / "holdout_weight_sensitivity.parquet")
        sensitivity_summary = summarise_holdout_weight_sensitivity(sensitivity)
        summary["holdout_weight_sensitivity"] = sensitivity_summary
        with open(results_dir / "holdout_weight_sensitivity_summary.json", "w") as handle:
            json.dump(sensitivity_summary, handle, indent=2)
        with open(results_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        plot_holdout_weight_sensitivity(
            sensitivity,
            results_dir / "holdout_weight_sensitivity.png",
        )
        plot_holdout_weight_sensitivity(
            sensitivity,
            results_dir / "holdout_weight_sensitivity.pdf",
        )

    if summary.get("models"):
        plot_model_comparison(summary, results_dir / "model_comparison.png")
        plot_model_comparison(summary, results_dir / "model_comparison.pdf")
    if not predictions.empty:
        plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.png")
        plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.pdf")

    logger.info(
        "External replication: train repos=%d, holdout repos=%d, holdout positives=%d",
        summary["train_panel"].get("n_repos", 0),
        summary["holdout_panel"].get("n_repos", 0),
        summary["holdout_panel"].get("n_positive_files", 0),
    )
    if baseline and plus:
        logger.info(
            "Primary comparison (%s -> %s): AUC %.3f -> %.3f, AP %.3f -> %.3f, Brier %.3f -> %.3f, ECE %.3f -> %.3f",
            PRIMARY_BASELINE_MODEL,
            PRIMARY_PLUS_MODEL,
            baseline.get("roc_auc", float("nan")),
            plus.get("roc_auc", float("nan")),
            baseline.get("average_precision", float("nan")),
            plus.get("average_precision", float("nan")),
            baseline.get("brier_score", float("nan")),
            plus.get("brier_score", float("nan")),
            baseline.get("ece", float("nan")),
            plus.get("ece", float("nan")),
        )


if __name__ == "__main__":
    main()
