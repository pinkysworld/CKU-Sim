"""Experiment 22: audited quantification-limit diagnostics on frozen external holdout."""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.analysis.quantification_limits import (
    brier_reliability,
    bootstrap_stratum_gap,
    build_all_file_calibration_curve_table,
    build_all_file_disagreement_frame,
    expected_calibration_error,
    merge_all_file_prediction_diagnostics,
    summarise_all_file_calibration_by_stratum,
    summarise_all_file_disagreement_by_stratum,
    summarise_all_file_opacity_strata,
)
from cku_sim.analysis.audited_panel import PRIMARY_PLUS_MODEL
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _plot_gap_bars(calibration_summary: pd.DataFrame, disagreement_summary: pd.DataFrame, output_path) -> None:
    if calibration_summary.empty or disagreement_summary.empty:
        return
    strata = [value for value in calibration_summary["opacity_stratum"].tolist() if value != "overall"]
    primary = calibration_summary.loc[
        calibration_summary["model"] == PRIMARY_PLUS_MODEL
    ].set_index("opacity_stratum")
    disagreement = disagreement_summary.set_index("opacity_stratum")
    if not set(strata).issubset(primary.index) or not set(strata).issubset(disagreement.index):
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(strata, primary.loc[strata, "ece"].astype(float), color="#4C78A8")
    axes[0].set_title("ECE by opacity decile")
    axes[0].set_ylabel("ECE")

    axes[1].bar(strata, primary.loc[strata, "brier_reliability"].astype(float), color="#F58518")
    axes[1].set_title("Brier reliability")
    axes[1].set_ylabel("Reliability")

    axes[2].bar(strata, disagreement.loc[strata, "mean_score_range"].astype(float), color="#54A24B")
    axes[2].set_title("Cross-model score range")
    axes[2].set_ylabel("Mean range")

    for ax in axes:
        ax.set_xlabel("Opacity decile")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_calibration_curves(
    calibration_curve: pd.DataFrame,
    *,
    low_label: str,
    high_label: str,
    output_path,
) -> None:
    if calibration_curve.empty:
        return
    model_names = calibration_curve["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(model_names), figsize=(6.2 * len(model_names), 4.5))
    if len(model_names) == 1:
        axes = [axes]
    palette = {
        "overall": "#4C78A8",
        low_label: "#72B7B2",
        high_label: "#E45756",
    }
    for ax, model_name in zip(axes, model_names):
        subset = calibration_curve.loc[calibration_curve["model"] == model_name].copy()
        for label in ["overall", low_label, high_label]:
            curve = subset.loc[subset["opacity_stratum"] == label].copy()
            if curve.empty:
                continue
            ax.plot(
                curve["mean_score"],
                curve["event_rate"],
                marker="o",
                label=label,
                color=palette[label],
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 22: audited quantification-limit diagnostics on frozen external holdout"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--e20-subdir",
        type=str,
        default="e20_external_replication__audited_curated_to_external",
    )
    parser.add_argument("--n-opacity-strata", type=int, default=10)
    parser.add_argument("--n-calibration-bins", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e22_quantification_calibration__audited_external",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    e20_dir = config.results_dir / args.e20_subdir
    dataset_path = e20_dir / "holdout_file_level_dataset.parquet"
    predictions_path = e20_dir / "heldout_predictions.parquet"
    if not dataset_path.exists() or not predictions_path.exists():
        logger.error("External replication outputs not found under %s", e20_dir)
        return

    dataset = pd.read_parquet(dataset_path)
    predictions = pd.read_parquet(predictions_path)
    if dataset.empty or predictions.empty:
        logger.error("External replication inputs are empty; cannot run calibration diagnostics.")
        return

    scored_predictions = merge_all_file_prediction_diagnostics(
        dataset,
        predictions,
        n_strata=args.n_opacity_strata,
    )
    disagreement_frame = build_all_file_disagreement_frame(scored_predictions)
    decile_labels = [f"Q{i}" for i in range(1, args.n_opacity_strata + 1)]
    low_label = decile_labels[0]
    high_label = decile_labels[-1]
    calibration_summary = summarise_all_file_calibration_by_stratum(
        scored_predictions,
        group_labels=decile_labels,
        n_bins=args.n_calibration_bins,
    )
    disagreement_summary = summarise_all_file_disagreement_by_stratum(
        disagreement_frame,
        group_labels=decile_labels,
    )
    opacity_summary = summarise_all_file_opacity_strata(
        disagreement_frame,
        group_labels=decile_labels,
    )
    calibration_curve = build_all_file_calibration_curve_table(
        scored_predictions,
        model_names=["baseline_history_plus_structure", PRIMARY_PLUS_MODEL],
        group_labels=["overall", low_label, high_label],
        n_bins=args.n_calibration_bins,
    )

    primary_predictions = scored_predictions.loc[
        scored_predictions["model"] == PRIMARY_PLUS_MODEL
    ].copy()
    primary_predictions["absolute_error"] = (
        primary_predictions["label"].astype(float)
        - pd.to_numeric(primary_predictions["score"], errors="coerce")
    ).abs()

    def _metric_gap(frame: pd.DataFrame, statistic, *, cluster_col: str) -> dict[str, object]:
        return bootstrap_stratum_gap(
            frame,
            statistic=statistic,
            low_label=low_label,
            high_label=high_label,
            cluster_col=cluster_col,
            n_bootstrap=args.n_bootstrap,
        )

    ece_event_gap = _metric_gap(
        primary_predictions,
        lambda frame: expected_calibration_error(
            frame["label"], frame["score"], n_bins=args.n_calibration_bins
        ),
        cluster_col="event_observation_id",
    )
    ece_repo_gap = _metric_gap(
        primary_predictions,
        lambda frame: expected_calibration_error(
            frame["label"], frame["score"], n_bins=args.n_calibration_bins
        ),
        cluster_col="repo",
    )
    reliability_event_gap = _metric_gap(
        primary_predictions,
        lambda frame: brier_reliability(
            frame["label"], frame["score"], n_bins=args.n_calibration_bins
        ),
        cluster_col="event_observation_id",
    )
    reliability_repo_gap = _metric_gap(
        primary_predictions,
        lambda frame: brier_reliability(
            frame["label"], frame["score"], n_bins=args.n_calibration_bins
        ),
        cluster_col="repo",
    )
    abs_error_event_gap = _metric_gap(
        primary_predictions,
        lambda frame: float(pd.to_numeric(frame["absolute_error"], errors="coerce").mean()),
        cluster_col="event_observation_id",
    )
    abs_error_repo_gap = _metric_gap(
        primary_predictions,
        lambda frame: float(pd.to_numeric(frame["absolute_error"], errors="coerce").mean()),
        cluster_col="repo",
    )
    score_range_event_gap = bootstrap_stratum_gap(
        disagreement_frame,
        statistic=lambda frame: float(pd.to_numeric(frame["score_range"], errors="coerce").mean()),
        low_label=low_label,
        high_label=high_label,
        cluster_col="event_observation_id",
        n_bootstrap=args.n_bootstrap,
    )
    score_range_repo_gap = bootstrap_stratum_gap(
        disagreement_frame,
        statistic=lambda frame: float(pd.to_numeric(frame["score_range"], errors="coerce").mean()),
        low_label=low_label,
        high_label=high_label,
        cluster_col="repo",
        n_bootstrap=args.n_bootstrap,
    )

    summary = {
        "n_files": int(len(disagreement_frame)),
        "n_events": int(disagreement_frame["event_observation_id"].nunique()),
        "n_repos": int(disagreement_frame["repo"].nunique()),
        "primary_model": PRIMARY_PLUS_MODEL,
        "opacity_low_label": low_label,
        "opacity_high_label": high_label,
        "opacity_summary": opacity_summary.to_dict(orient="records"),
        "primary_model_calibration": calibration_summary.loc[
            calibration_summary["model"] == PRIMARY_PLUS_MODEL
        ].to_dict(orient="records"),
        "gaps": {
            "ece_event_cluster": ece_event_gap,
            "ece_repo_cluster": ece_repo_gap,
            "brier_reliability_event_cluster": reliability_event_gap,
            "brier_reliability_repo_cluster": reliability_repo_gap,
            "absolute_error_event_cluster": abs_error_event_gap,
            "absolute_error_repo_cluster": abs_error_repo_gap,
            "score_range_event_cluster": score_range_event_gap,
            "score_range_repo_cluster": score_range_repo_gap,
        },
        "gating": {
            "cku_consistent_primary_and_secondary_signal": False,
        },
    }
    event_ok = False
    secondary_ok = False
    for key, value in summary["gaps"].items():
        if not value:
            continue
        ci = value.get("ci_95", [0.0, 0.0])
        if key == "ece_event_cluster" and ci[0] > 0:
            event_ok = True
        if key in {
            "brier_reliability_event_cluster",
            "absolute_error_event_cluster",
            "score_range_event_cluster",
        } and ci[0] > 0:
            secondary_ok = True
    summary["gating"]["cku_consistent_primary_and_secondary_signal"] = bool(event_ok and secondary_ok)

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    scored_predictions.to_parquet(results_dir / "scored_predictions.parquet")
    scored_predictions.to_csv(results_dir / "scored_predictions.csv", index=False)
    disagreement_frame.to_parquet(results_dir / "disagreement_frame.parquet")
    disagreement_frame.to_csv(results_dir / "disagreement_frame.csv", index=False)
    opacity_summary.to_csv(results_dir / "opacity_summary.csv", index=False)
    calibration_summary.to_csv(results_dir / "calibration_summary.csv", index=False)
    disagreement_summary.to_csv(results_dir / "disagreement_summary.csv", index=False)
    calibration_curve.to_csv(results_dir / "calibration_curve.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    _plot_gap_bars(calibration_summary, disagreement_summary, results_dir / "gap_bars.png")
    _plot_gap_bars(calibration_summary, disagreement_summary, results_dir / "gap_bars.pdf")
    _plot_calibration_curves(
        calibration_curve,
        low_label=low_label,
        high_label=high_label,
        output_path=results_dir / "calibration_curves.png",
    )
    _plot_calibration_curves(
        calibration_curve,
        low_label=low_label,
        high_label=high_label,
        output_path=results_dir / "calibration_curves.pdf",
    )

    logger.info(
        "Quantification calibration: repos=%d, events=%d, CKU-consistent=%s",
        summary["n_repos"],
        summary["n_events"],
        summary["gating"]["cku_consistent_primary_and_secondary_signal"],
    )


if __name__ == "__main__":
    main()
