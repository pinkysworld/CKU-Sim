"""Experiment 14: horizon and severity robustness summary for the prospective panel."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_run_label(subdir: str) -> dict[str, object]:
    lower = subdir.lower()
    horizon_days = 365 if "h365" in lower else 730 if "h730" in lower else None
    severity_band = "high_critical" if "high_critical" in lower else "all"
    return {
        "run_subdir": subdir,
        "horizon_days": horizon_days,
        "severity_band": severity_band,
    }


def _plot_robustness(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    work = frame.copy()
    work["label"] = work.apply(
        lambda row: f"{row['severity_band']} | {int(row['horizon_days'])}d",
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(work["label"], work["pair_median_delta"], color="#4C78A8")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Median case-control delta")
    axes[0].set_title("Prospective association")

    axes[1].bar(work["label"], work["pred_auc_baseline_plus_composite"], color="#F58518")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Held-out pooled ROC AUC")
    axes[1].set_title("Prospective prediction")

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 14: horizon and severity robustness summary"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Comma-separated list of experiment 12 result subdirectories",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e14_prospective_robustness",
        help="Results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for subdir in [item.strip() for item in args.runs.split(",") if item.strip()]:
        summary_path = config.results_dir / subdir / "summary.json"
        if not summary_path.exists():
            logger.warning("Skipping missing summary: %s", summary_path)
            continue
        summary = json.loads(summary_path.read_text())
        pair = summary.get("pair_level", {})
        pred = summary.get("prediction", {}).get("models", {})
        row = {
            **_parse_run_label(subdir),
            "pair_n_pairs": pair.get("n_pairs"),
            "pair_n_events": pair.get("n_events"),
            "pair_n_repos": pair.get("n_repos"),
            "pair_n_snapshots": pair.get("n_snapshots"),
            "pair_mean_delta": pair.get("mean_delta_composite"),
            "pair_median_delta": pair.get("median_delta_composite"),
            "pair_positive_share": pair.get("positive_share"),
            "pair_wilcoxon_pvalue": pair.get("wilcoxon_pvalue_greater"),
            "pair_mean_ci_lo": (pair.get("bootstrap_primary_cluster") or {}).get("mean_delta_composite_ci", [None, None])[0],
            "pair_mean_ci_hi": (pair.get("bootstrap_primary_cluster") or {}).get("mean_delta_composite_ci", [None, None])[1],
            "pred_auc_baseline_size": (pred.get("baseline_size") or {}).get("roc_auc"),
            "pred_auc_baseline_history": (pred.get("baseline_history") or {}).get("roc_auc"),
            "pred_auc_baseline_plus_composite": (pred.get("baseline_plus_composite") or {}).get("roc_auc"),
            "pred_ap_baseline_history": (pred.get("baseline_history") or {}).get("average_precision"),
            "pred_ap_baseline_plus_composite": (pred.get("baseline_plus_composite") or {}).get("average_precision"),
            "pred_pairacc_baseline_history": (pred.get("baseline_history") or {}).get("pairwise_accuracy"),
            "pred_pairacc_baseline_plus_composite": (pred.get("baseline_plus_composite") or {}).get("pairwise_accuracy"),
            "repo_fe_composite_pvalue": (
                summary.get("repo_fixed_effects", {})
                .get("baseline_plus_composite_repo_fixed_effects", {})
                .get("composite_pvalue")
            ),
            "repo_fe_lr_pvalue": (
                summary.get("repo_fixed_effects", {})
                .get("baseline_plus_composite_repo_fixed_effects", {})
                .get("lr_pvalue_vs_baseline")
            ),
        }
        if row["pred_auc_baseline_history"] is not None and row["pred_auc_baseline_plus_composite"] is not None:
            row["pred_auc_lift_vs_history"] = (
                row["pred_auc_baseline_plus_composite"] - row["pred_auc_baseline_history"]
            )
        if row["pred_pairacc_baseline_history"] is not None and row["pred_pairacc_baseline_plus_composite"] is not None:
            row["pred_pairacc_lift_vs_history"] = (
                row["pred_pairacc_baseline_plus_composite"] - row["pred_pairacc_baseline_history"]
            )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["horizon_days", "severity_band"])
    frame.to_csv(results_dir / "robustness_summary.csv", index=False)
    _plot_robustness(frame, results_dir / "robustness_summary.png")
    _plot_robustness(frame, results_dir / "robustness_summary.pdf")

    logger.info("=" * 60)
    logger.info("Experiment 14: Prospective robustness summary")
    logger.info("=" * 60)
    logger.info("Compiled %d experiment-12 runs", len(frame))


if __name__ == "__main__":
    main()
