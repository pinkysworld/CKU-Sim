"""Experiment 24: direct external quantification-failure diagnostics on positive files."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from cku_sim.analysis.audited_panel import PRIMARY_PLUS_MODEL
from cku_sim.analysis.quantification_limits import (
    assign_grouped_opacity_band,
    bootstrap_stratum_gap,
    build_all_file_disagreement_frame,
    build_all_file_positive_failure_frame,
    merge_all_file_prediction_diagnostics,
    summarise_positive_failure_by_band,
)
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_labels(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 24: direct external quantification-failure diagnostics on positive files"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--e20-subdir",
        type=str,
        default="e20_external_replication__expanded7_no_gitea__audited_v1",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e24_external_quantification_failure__expanded7_no_gitea__audited_v1",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="baseline_history_plus_structure,baseline_plus_composite",
        help="Comma-separated models to evaluate",
    )
    parser.add_argument(
        "--low-opacity-labels",
        type=str,
        default="Q1,Q2,Q3,Q4",
        help="Comma-separated low-opacity strata labels",
    )
    parser.add_argument(
        "--high-opacity-labels",
        type=str,
        default="Q7,Q8,Q9,Q10",
        help="Comma-separated high-opacity strata labels",
    )
    parser.add_argument("--n-opacity-strata", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    e20_dir = config.results_dir / args.e20_subdir
    dataset_path = e20_dir / "holdout_file_level_dataset.parquet"
    predictions_path = e20_dir / "heldout_predictions.parquet"
    if not dataset_path.exists() or not predictions_path.exists():
        logger.error("External replication outputs not found under %s", e20_dir)
        return

    model_names = _parse_labels(args.models)
    low_labels = _parse_labels(args.low_opacity_labels)
    high_labels = _parse_labels(args.high_opacity_labels)
    dataset = pd.read_parquet(dataset_path)
    predictions = pd.read_parquet(predictions_path)
    scored_predictions = merge_all_file_prediction_diagnostics(
        dataset,
        predictions,
        n_strata=args.n_opacity_strata,
    )
    disagreement_frame = build_all_file_disagreement_frame(scored_predictions)
    positive_failure = build_all_file_positive_failure_frame(
        disagreement_frame,
        model_names=model_names,
    )
    positive_failure = assign_grouped_opacity_band(
        positive_failure,
        low_labels=low_labels,
        high_labels=high_labels,
        source_col="opacity_stratum",
        output_col="opacity_band",
    )
    positive_failure = positive_failure.loc[
        positive_failure["opacity_band"].isin(["low", "high"])
    ].copy()
    band_summary = summarise_positive_failure_by_band(
        positive_failure,
        band_col="opacity_band",
    )

    gap_rows: list[dict[str, object]] = []
    gap_lookup: dict[str, dict[str, object]] = {}
    for model_name in model_names:
        model_frame = positive_failure.loc[positive_failure["model"] == model_name].copy()
        if model_frame.empty:
            continue
        for metric_name in ("underprediction_loss", "rank_pct", "top10_miss", "top25_miss"):
            for cluster_col in ("event_observation_id", "repo"):
                gap = bootstrap_stratum_gap(
                    model_frame,
                    statistic=lambda frame, metric=metric_name: float(
                        pd.to_numeric(frame[metric], errors="coerce").mean()
                    ),
                    strata_col="opacity_band",
                    low_label="low",
                    high_label="high",
                    cluster_col=cluster_col,
                    n_bootstrap=args.n_bootstrap,
                )
                gap_key = f"{model_name}:{metric_name}:{cluster_col}"
                gap_lookup[gap_key] = gap
                if gap:
                    gap_rows.append(
                        {
                            "model": model_name,
                            "metric": metric_name,
                            "cluster_col": cluster_col,
                            "observed_gap": float(gap["observed_gap"]),
                            "ci_lo": float(gap["ci_95"][0]),
                            "ci_hi": float(gap["ci_95"][1]),
                            "n_bootstrap": int(gap["n_bootstrap"]),
                        }
                    )

    def _ci_above_zero(payload: dict[str, object]) -> bool:
        if not payload:
            return False
        ci = payload.get("ci_95", [0.0, 0.0])
        return float(ci[0]) > 0.0

    primary_underprediction = (
        _ci_above_zero(gap_lookup.get(f"{PRIMARY_PLUS_MODEL}:underprediction_loss:event_observation_id", {}))
        and _ci_above_zero(gap_lookup.get(f"{PRIMARY_PLUS_MODEL}:underprediction_loss:repo", {}))
    )
    primary_secondary = (
        _ci_above_zero(gap_lookup.get(f"{PRIMARY_PLUS_MODEL}:rank_pct:event_observation_id", {}))
        and _ci_above_zero(gap_lookup.get(f"{PRIMARY_PLUS_MODEL}:rank_pct:repo", {}))
    ) or (
        _ci_above_zero(gap_lookup.get(f"{PRIMARY_PLUS_MODEL}:top10_miss:event_observation_id", {}))
        and _ci_above_zero(gap_lookup.get(f"{PRIMARY_PLUS_MODEL}:top10_miss:repo", {}))
    )

    summary = {
        "n_positive_files": int(len(positive_failure.drop_duplicates(subset=["event_observation_id", "file_path"]))),
        "n_positive_rows": int(len(positive_failure)),
        "n_repos": int(positive_failure["repo"].nunique()) if not positive_failure.empty else 0,
        "models": model_names,
        "low_opacity_labels": low_labels,
        "high_opacity_labels": high_labels,
        "band_counts": positive_failure["opacity_band"].value_counts().to_dict(),
        "band_summary": band_summary.to_dict(orient="records"),
        "gaps": gap_lookup,
        "gating": {
            "primary_underprediction_signal": bool(primary_underprediction),
            "primary_secondary_failure_signal": bool(primary_secondary),
            "direct_quantification_failure_gate": bool(primary_underprediction and primary_secondary),
        },
    }

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    positive_failure.to_parquet(results_dir / "positive_failure_frame.parquet")
    positive_failure.to_csv(results_dir / "positive_failure_frame.csv", index=False)
    band_summary.to_csv(results_dir / "band_summary.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(results_dir / "gap_summary.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "External quantification failure: positives=%d, repos=%d, direct-gate=%s",
        summary["n_positive_files"],
        summary["n_repos"],
        summary["gating"]["direct_quantification_failure_gate"],
    )


if __name__ == "__main__":
    main()
