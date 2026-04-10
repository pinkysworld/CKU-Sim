"""Calibration and model-disagreement diagnostics for the prospective file panel."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

LOW_OPACITY_STRATUM = "Q1"
HIGH_OPACITY_STRATUM = "Q4"
OPACITY_STRATA = [LOW_OPACITY_STRATUM, "Q2", "Q3", HIGH_OPACITY_STRATUM]
DIAGNOSTIC_MODEL_ORDER = [
    "baseline_size",
    "baseline_history",
    "baseline_history_plus_structure",
    "baseline_plus_composite",
]
PRIMARY_MODEL = "baseline_plus_composite"


def assign_opacity_strata(
    frame: pd.DataFrame,
    *,
    score_col: str = "composite_score",
    output_col: str = "opacity_stratum",
    n_strata: int = 4,
) -> pd.DataFrame:
    """Assign quantile-based opacity strata using rank-based bins."""
    work = frame.copy()
    scores = pd.to_numeric(work[score_col], errors="coerce")
    fill_value = float(scores.dropna().median()) if scores.notna().any() else 0.0
    scores = scores.fillna(fill_value)
    labels = [f"Q{i}" for i in range(1, n_strata + 1)]
    ranks = scores.rank(method="first")
    work[output_col] = pd.qcut(ranks, q=n_strata, labels=labels).astype(str)
    return work


def build_quantile_bins(
    labels: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    *,
    n_bins: int = 6,
) -> pd.DataFrame:
    """Build rank-based probability bins for calibration diagnostics."""
    frame = pd.DataFrame(
        {
            "label": pd.to_numeric(pd.Series(labels), errors="coerce"),
            "score": pd.to_numeric(pd.Series(scores), errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_score", "event_rate", "abs_gap"])

    n_bins = max(1, min(int(n_bins), len(frame)))
    ranks = frame["score"].rank(method="first")
    frame["bin"] = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    grouped = (
        frame.groupby("bin", observed=False)
        .agg(
            n=("label", "size"),
            mean_score=("score", "mean"),
            event_rate=("label", "mean"),
        )
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(int) + 1
    grouped["abs_gap"] = (grouped["event_rate"] - grouped["mean_score"]).abs()
    return grouped


def expected_calibration_error(
    labels: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    *,
    n_bins: int = 6,
) -> float:
    """Compute expected calibration error from quantile bins."""
    bins = build_quantile_bins(labels, scores, n_bins=n_bins)
    if bins.empty:
        return float("nan")
    weights = bins["n"] / bins["n"].sum()
    return float((weights * bins["abs_gap"]).sum())


def brier_reliability(
    labels: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    *,
    n_bins: int = 6,
) -> float:
    """Approximate the reliability component of the Brier score via quantile bins."""
    bins = build_quantile_bins(labels, scores, n_bins=n_bins)
    if bins.empty:
        return float("nan")
    weights = bins["n"] / bins["n"].sum()
    return float((weights * (bins["mean_score"] - bins["event_rate"]) ** 2).sum())


def merge_prediction_diagnostics(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach opacity strata and file-level metadata to held-out predictions."""
    if dataset.empty or predictions.empty:
        return pd.DataFrame()

    enriched_dataset = assign_opacity_strata(dataset)
    key_cols = ["pair_id", "repo", "snapshot_tag", "event_id", "label", "kind", "file_path"]
    merge_cols = key_cols + [
        "event_observation_id",
        "ground_truth_source",
        "ground_truth_source_family",
        "opacity_stratum",
        "composite_score",
        "loc",
        "log_loc",
        "directory_depth",
        "suffix",
    ]
    merged = predictions.merge(
        enriched_dataset[merge_cols],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    if merged["opacity_stratum"].isna().any():
        raise ValueError("Prediction merge left opacity strata unresolved for some rows")
    return merged


def merge_all_file_prediction_diagnostics(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    n_strata: int = 10,
) -> pd.DataFrame:
    """Attach opacity strata and file metadata to all-file held-out predictions."""
    if dataset.empty or predictions.empty:
        return pd.DataFrame()

    enriched_dataset = assign_opacity_strata(dataset, n_strata=n_strata)
    merge_cols = [
        "repo",
        "snapshot_tag",
        "snapshot_key",
        "event_observation_id",
        "label",
        "file_path",
        "opacity_stratum",
        "composite_score",
        "loc",
        "log_loc",
        "directory_depth",
        "suffix",
        "advisory_ids",
        "source_families",
        "mapping_bases",
    ]
    merge_cols = [col for col in merge_cols if col in enriched_dataset.columns]
    merged = predictions.merge(
        enriched_dataset[merge_cols],
        on=["repo", "snapshot_tag", "snapshot_key", "event_observation_id", "label", "file_path"],
        how="left",
        validate="many_to_one",
    )
    if merged["opacity_stratum"].isna().any():
        raise ValueError("All-file prediction merge left opacity strata unresolved for some rows")
    return merged


def _safe_roc_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    if pd.Series(labels).nunique() < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_average_precision(labels: pd.Series, scores: pd.Series) -> float | None:
    if pd.Series(labels).nunique() < 2:
        return None
    return float(average_precision_score(labels, scores))


def summarise_calibration_by_stratum(
    scored_predictions: pd.DataFrame,
    *,
    n_bins: int = 6,
) -> pd.DataFrame:
    """Summarise calibration and forecasting error by model and opacity stratum."""
    rows: list[dict[str, object]] = []
    for model_name, model_df in scored_predictions.groupby("model", sort=False):
        strata = [("overall", model_df)] + [
            (label, model_df.loc[model_df["opacity_stratum"] == label].copy())
            for label in OPACITY_STRATA
        ]
        for stratum_label, subset in strata:
            subset = subset.dropna(subset=["score", "label"])
            if subset.empty:
                continue
            y_true = subset["label"].astype(float)
            y_score = pd.to_numeric(subset["score"], errors="coerce")
            rows.append(
                {
                    "model": str(model_name),
                    "opacity_stratum": str(stratum_label),
                    "n_files": int(len(subset)),
                    "n_pairs": int(subset["pair_id"].nunique()),
                    "n_events": int(subset["event_observation_id"].nunique()),
                    "n_repos": int(subset["repo"].nunique()),
                    "event_rate": float(y_true.mean()),
                    "mean_score": float(y_score.mean()),
                    "calibration_gap": float(y_score.mean() - y_true.mean()),
                    "absolute_error_mean": float(np.abs(y_true - y_score).mean()),
                    "brier_score": float(brier_score_loss(y_true, y_score)),
                    "ece": expected_calibration_error(y_true, y_score, n_bins=n_bins),
                    "roc_auc": _safe_roc_auc(y_true, y_score),
                    "average_precision": _safe_average_precision(y_true, y_score),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {"overall": 0, LOW_OPACITY_STRATUM: 1, "Q2": 2, "Q3": 3, HIGH_OPACITY_STRATUM: 4}
    frame["stratum_order"] = frame["opacity_stratum"].map(order).fillna(99)
    frame = frame.sort_values(["model", "stratum_order"]).drop(columns=["stratum_order"])
    return frame


def summarise_all_file_calibration_by_stratum(
    scored_predictions: pd.DataFrame,
    *,
    group_labels: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarise calibration diagnostics for all-file predictions by opacity group."""
    rows: list[dict[str, object]] = []
    for model_name, model_df in scored_predictions.groupby("model", sort=False):
        strata = [("overall", model_df)] + [
            (label, model_df.loc[model_df["opacity_stratum"] == label].copy())
            for label in group_labels
        ]
        for stratum_label, subset in strata:
            subset = subset.dropna(subset=["score", "label"])
            if subset.empty:
                continue
            y_true = subset["label"].astype(float)
            y_score = pd.to_numeric(subset["score"], errors="coerce")
            rows.append(
                {
                    "model": str(model_name),
                    "opacity_stratum": str(stratum_label),
                    "n_files": int(len(subset)),
                    "n_events": int(subset["event_observation_id"].nunique()),
                    "n_repos": int(subset["repo"].nunique()),
                    "event_rate": float(y_true.mean()),
                    "mean_score": float(y_score.mean()),
                    "calibration_gap": float(y_score.mean() - y_true.mean()),
                    "absolute_error_mean": float(np.abs(y_true - y_score).mean()),
                    "brier_score": float(brier_score_loss(y_true, y_score)),
                    "brier_reliability": brier_reliability(y_true, y_score, n_bins=n_bins),
                    "ece": expected_calibration_error(y_true, y_score, n_bins=n_bins),
                    "roc_auc": _safe_roc_auc(y_true, y_score),
                    "average_precision": _safe_average_precision(y_true, y_score),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {"overall": 0, **{label: index for index, label in enumerate(group_labels, start=1)}}
    frame["stratum_order"] = frame["opacity_stratum"].map(order).fillna(99)
    return frame.sort_values(["model", "stratum_order"]).drop(columns=["stratum_order"])


def build_model_disagreement_frame(scored_predictions: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-model held-out scores into one row per file observation."""
    if scored_predictions.empty:
        return pd.DataFrame()

    id_cols = [
        "pair_id",
        "repo",
        "snapshot_tag",
        "event_id",
        "event_observation_id",
        "label",
        "kind",
        "file_path",
        "ground_truth_source",
        "ground_truth_source_family",
        "opacity_stratum",
        "composite_score",
        "loc",
        "log_loc",
        "directory_depth",
        "suffix",
    ]
    wide = (
        scored_predictions.pivot_table(
            index=id_cols,
            columns="model",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    model_cols = [name for name in DIAGNOSTIC_MODEL_ORDER if name in wide.columns]
    if not model_cols:
        return wide
    wide["score_mean"] = wide[model_cols].mean(axis=1)
    wide["score_std"] = wide[model_cols].std(axis=1, ddof=0)
    wide["score_range"] = wide[model_cols].max(axis=1) - wide[model_cols].min(axis=1)
    if {"baseline_history", "baseline_plus_composite"}.issubset(wide.columns):
        wide["history_vs_composite_gap"] = (
            wide["baseline_plus_composite"] - wide["baseline_history"]
        ).abs()
    else:
        wide["history_vs_composite_gap"] = np.nan
    if {"baseline_history_plus_structure", "baseline_plus_composite"}.issubset(wide.columns):
        wide["structure_vs_composite_gap"] = (
            wide["baseline_plus_composite"] - wide["baseline_history_plus_structure"]
        ).abs()
    else:
        wide["structure_vs_composite_gap"] = np.nan
    return wide


def build_all_file_disagreement_frame(scored_predictions: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-model all-file held-out scores into one row per file observation."""
    if scored_predictions.empty:
        return pd.DataFrame()

    id_cols = [
        "repo",
        "snapshot_tag",
        "snapshot_key",
        "event_observation_id",
        "label",
        "file_path",
        "opacity_stratum",
        "composite_score",
        "loc",
        "log_loc",
        "directory_depth",
        "suffix",
    ]
    optional_cols = [
        "advisory_ids",
        "source_families",
        "mapping_bases",
    ]
    id_cols = id_cols + [col for col in optional_cols if col in scored_predictions.columns]
    wide = (
        scored_predictions.pivot_table(
            index=id_cols,
            columns="model",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    model_cols = [name for name in DIAGNOSTIC_MODEL_ORDER if name in wide.columns]
    if not model_cols:
        return wide
    wide["score_mean"] = wide[model_cols].mean(axis=1)
    wide["score_std"] = wide[model_cols].std(axis=1, ddof=0)
    wide["score_range"] = wide[model_cols].max(axis=1) - wide[model_cols].min(axis=1)
    if {"baseline_history", "baseline_plus_composite"}.issubset(wide.columns):
        wide["history_vs_composite_gap"] = (
            wide["baseline_plus_composite"] - wide["baseline_history"]
        ).abs()
    else:
        wide["history_vs_composite_gap"] = np.nan
    if {"baseline_history_plus_structure", "baseline_plus_composite"}.issubset(wide.columns):
        wide["structure_vs_composite_gap"] = (
            wide["baseline_plus_composite"] - wide["baseline_history_plus_structure"]
        ).abs()
    else:
        wide["structure_vs_composite_gap"] = np.nan
    return wide


def summarise_disagreement_by_stratum(disagreement_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise cross-model disagreement by opacity stratum."""
    if disagreement_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    strata = [("overall", disagreement_frame)] + [
        (label, disagreement_frame.loc[disagreement_frame["opacity_stratum"] == label].copy())
        for label in OPACITY_STRATA
    ]
    for stratum_label, subset in strata:
        if subset.empty:
            continue
        rows.append(
            {
                "opacity_stratum": str(stratum_label),
                "n_files": int(len(subset)),
                "n_pairs": int(subset["pair_id"].nunique()),
                "n_events": int(subset["event_observation_id"].nunique()),
                "n_repos": int(subset["repo"].nunique()),
                "event_rate": float(pd.to_numeric(subset["label"], errors="coerce").mean()),
                "mean_score_std": float(pd.to_numeric(subset["score_std"], errors="coerce").mean()),
                "median_score_std": float(pd.to_numeric(subset["score_std"], errors="coerce").median()),
                "mean_score_range": float(pd.to_numeric(subset["score_range"], errors="coerce").mean()),
                "median_score_range": float(pd.to_numeric(subset["score_range"], errors="coerce").median()),
                "mean_history_vs_composite_gap": float(
                    pd.to_numeric(subset["history_vs_composite_gap"], errors="coerce").mean()
                ),
                "mean_structure_vs_composite_gap": float(
                    pd.to_numeric(subset["structure_vs_composite_gap"], errors="coerce").mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    order = {"overall": 0, LOW_OPACITY_STRATUM: 1, "Q2": 2, "Q3": 3, HIGH_OPACITY_STRATUM: 4}
    frame["stratum_order"] = frame["opacity_stratum"].map(order).fillna(99)
    return frame.sort_values("stratum_order").drop(columns=["stratum_order"])


def summarise_all_file_disagreement_by_stratum(
    disagreement_frame: pd.DataFrame,
    *,
    group_labels: list[str],
) -> pd.DataFrame:
    """Summarise cross-model disagreement for all-file predictions by opacity group."""
    if disagreement_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    strata = [("overall", disagreement_frame)] + [
        (label, disagreement_frame.loc[disagreement_frame["opacity_stratum"] == label].copy())
        for label in group_labels
    ]
    for stratum_label, subset in strata:
        if subset.empty:
            continue
        rows.append(
            {
                "opacity_stratum": str(stratum_label),
                "n_files": int(len(subset)),
                "n_events": int(subset["event_observation_id"].nunique()),
                "n_repos": int(subset["repo"].nunique()),
                "event_rate": float(pd.to_numeric(subset["label"], errors="coerce").mean()),
                "mean_score_std": float(pd.to_numeric(subset["score_std"], errors="coerce").mean()),
                "median_score_std": float(pd.to_numeric(subset["score_std"], errors="coerce").median()),
                "mean_score_range": float(pd.to_numeric(subset["score_range"], errors="coerce").mean()),
                "median_score_range": float(pd.to_numeric(subset["score_range"], errors="coerce").median()),
                "mean_history_vs_composite_gap": float(
                    pd.to_numeric(subset["history_vs_composite_gap"], errors="coerce").mean()
                ),
                "mean_structure_vs_composite_gap": float(
                    pd.to_numeric(subset["structure_vs_composite_gap"], errors="coerce").mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    order = {"overall": 0, **{label: index for index, label in enumerate(group_labels, start=1)}}
    frame["stratum_order"] = frame["opacity_stratum"].map(order).fillna(99)
    return frame.sort_values("stratum_order").drop(columns=["stratum_order"])


def summarise_opacity_strata(disagreement_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise file counts and event rates by opacity stratum."""
    if disagreement_frame.empty:
        return pd.DataFrame()
    frame = (
        disagreement_frame.groupby("opacity_stratum", observed=False)
        .agg(
            n_files=("label", "size"),
            n_pairs=("pair_id", "nunique"),
            n_events=("event_observation_id", "nunique"),
            n_repos=("repo", "nunique"),
            event_rate=("label", "mean"),
            mean_composite=("composite_score", "mean"),
            median_composite=("composite_score", "median"),
        )
        .reset_index()
    )
    frame["stratum_order"] = frame["opacity_stratum"].map(
        {label: index for index, label in enumerate(OPACITY_STRATA, start=1)}
    )
    return frame.sort_values("stratum_order").drop(columns=["stratum_order"])


def summarise_all_file_opacity_strata(
    disagreement_frame: pd.DataFrame,
    *,
    group_labels: list[str],
) -> pd.DataFrame:
    """Summarise all-file counts and event rates by opacity group."""
    if disagreement_frame.empty:
        return pd.DataFrame()
    frame = (
        disagreement_frame.groupby("opacity_stratum", observed=False)
        .agg(
            n_files=("label", "size"),
            n_events=("event_observation_id", "nunique"),
            n_repos=("repo", "nunique"),
            event_rate=("label", "mean"),
            mean_composite=("composite_score", "mean"),
            median_composite=("composite_score", "median"),
        )
        .reset_index()
    )
    frame["stratum_order"] = frame["opacity_stratum"].map(
        {label: index for index, label in enumerate(group_labels, start=1)}
    )
    return frame.sort_values("stratum_order").drop(columns=["stratum_order"])


def bootstrap_stratum_gap(
    frame: pd.DataFrame,
    *,
    statistic: Callable[[pd.DataFrame], float],
    strata_col: str = "opacity_stratum",
    low_label: str = LOW_OPACITY_STRATUM,
    high_label: str = HIGH_OPACITY_STRATUM,
    cluster_col: str = "event_observation_id",
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, object]:
    """Bootstrap the difference in a statistic between high- and low-opacity strata."""
    low_frame = frame.loc[frame[strata_col] == low_label].copy()
    high_frame = frame.loc[frame[strata_col] == high_label].copy()
    if low_frame.empty or high_frame.empty:
        return {}

    low_cluster_col = cluster_col if cluster_col in low_frame.columns else None
    high_cluster_col = cluster_col if cluster_col in high_frame.columns else None
    if low_cluster_col is None or low_frame[low_cluster_col].isna().all():
        low_frame["_bootstrap_cluster"] = np.arange(len(low_frame))
        low_cluster_col = "_bootstrap_cluster"
    if high_cluster_col is None or high_frame[high_cluster_col].isna().all():
        high_frame["_bootstrap_cluster"] = np.arange(len(high_frame))
        high_cluster_col = "_bootstrap_cluster"

    low_clusters = low_frame[low_cluster_col].dropna().unique()
    high_clusters = high_frame[high_cluster_col].dropna().unique()
    if len(low_clusters) == 0 or len(high_clusters) == 0:
        return {}

    rng = np.random.default_rng(random_state)
    low_cluster_frames = {
        cluster: group.copy()
        for cluster, group in low_frame.groupby(low_cluster_col, sort=False, dropna=False)
    }
    high_cluster_frames = {
        cluster: group.copy()
        for cluster, group in high_frame.groupby(high_cluster_col, sort=False, dropna=False)
    }

    def _resample_clusters(
        cluster_frames: dict[object, pd.DataFrame],
        clusters: np.ndarray,
    ) -> pd.DataFrame:
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [cluster_frames[cluster] for cluster in sampled]
        return pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()

    observed = float(statistic(high_frame) - statistic(low_frame))
    boot = []
    for _ in range(int(n_bootstrap)):
        low_sample = _resample_clusters(low_cluster_frames, low_clusters)
        high_sample = _resample_clusters(high_cluster_frames, high_clusters)
        boot.append(float(statistic(high_sample) - statistic(low_sample)))
    interval = np.percentile(boot, [2.5, 97.5]).tolist()
    return {
        "high_label": high_label,
        "low_label": low_label,
        "observed_gap": observed,
        "ci_95": [float(interval[0]), float(interval[1])],
        "n_bootstrap": int(n_bootstrap),
    }


def build_calibration_curve_table(
    scored_predictions: pd.DataFrame,
    *,
    model_names: list[str] | None = None,
    strata: list[str] | None = None,
    n_bins: int = 6,
) -> pd.DataFrame:
    """Construct calibration-curve tables for selected models and strata."""
    model_names = model_names or ["baseline_history", PRIMARY_MODEL]
    strata = strata or ["overall", LOW_OPACITY_STRATUM, HIGH_OPACITY_STRATUM]
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        model_df = scored_predictions.loc[scored_predictions["model"] == model_name].copy()
        if model_df.empty:
            continue
        for stratum_label in strata:
            subset = (
                model_df.copy()
                if stratum_label == "overall"
                else model_df.loc[model_df["opacity_stratum"] == stratum_label].copy()
            )
            bins = build_quantile_bins(subset["label"], subset["score"], n_bins=n_bins)
            for _, row in bins.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "opacity_stratum": stratum_label,
                        "bin": int(row["bin"]),
                        "n": int(row["n"]),
                        "mean_score": float(row["mean_score"]),
                        "event_rate": float(row["event_rate"]),
                        "abs_gap": float(row["abs_gap"]),
                    }
                )
    return pd.DataFrame(rows)


def build_all_file_calibration_curve_table(
    scored_predictions: pd.DataFrame,
    *,
    model_names: list[str],
    group_labels: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Construct calibration curves for all-file predictions."""
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        model_df = scored_predictions.loc[scored_predictions["model"] == model_name].copy()
        if model_df.empty:
            continue
        for label in group_labels:
            subset = (
                model_df.copy()
                if label == "overall"
                else model_df.loc[model_df["opacity_stratum"] == label].copy()
            )
            bins = build_quantile_bins(subset["label"], subset["score"], n_bins=n_bins)
            for _, row in bins.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "opacity_stratum": label,
                        "bin": int(row["bin"]),
                        "n": int(row["n"]),
                        "mean_score": float(row["mean_score"]),
                        "event_rate": float(row["event_rate"]),
                        "abs_gap": float(row["abs_gap"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_quantification_limits(
    opacity_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    disagreement_summary: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Plot event rate, forecast error, and model disagreement across opacity strata."""
    if opacity_summary.empty or calibration_summary.empty or disagreement_summary.empty:
        return

    strata = OPACITY_STRATA
    event_rate = (
        opacity_summary.set_index("opacity_stratum").reindex(strata)["event_rate"].astype(float)
    )
    primary_error = (
        calibration_summary.loc[
            (calibration_summary["model"] == PRIMARY_MODEL)
            & (calibration_summary["opacity_stratum"].isin(strata))
        ]
        .set_index("opacity_stratum")
        .reindex(strata)["absolute_error_mean"]
        .astype(float)
    )
    score_range = (
        disagreement_summary.set_index("opacity_stratum")
        .reindex(strata)["mean_score_range"]
        .astype(float)
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    axes[0].bar(strata, event_rate.values, color="#4C78A8")
    axes[0].set_title("Future event rate")
    axes[0].set_ylabel("Rate")

    axes[1].bar(strata, primary_error.values, color="#F58518")
    axes[1].set_title("Absolute forecast error")
    axes[1].set_ylabel("Mean |y - p|")

    axes[2].bar(strata, score_range.values, color="#54A24B")
    axes[2].set_title("Cross-model disagreement")
    axes[2].set_ylabel("Mean score range")

    for ax in axes:
        ax.set_xlabel("Opacity stratum")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_calibration_curves(calibration_curve: pd.DataFrame, *, output_path: Path) -> None:
    """Plot low-, high-, and overall calibration curves for key models."""
    if calibration_curve.empty:
        return
    model_names = [name for name in ["baseline_history", PRIMARY_MODEL] if name in calibration_curve["model"].unique()]
    if not model_names:
        return

    fig, axes = plt.subplots(1, len(model_names), figsize=(6.2 * len(model_names), 4.5))
    if len(model_names) == 1:
        axes = [axes]

    palette = {
        "overall": "#4C78A8",
        LOW_OPACITY_STRATUM: "#72B7B2",
        HIGH_OPACITY_STRATUM: "#E45756",
    }
    for ax, model_name in zip(axes, model_names):
        subset = calibration_curve.loc[calibration_curve["model"] == model_name].copy()
        for stratum in ["overall", LOW_OPACITY_STRATUM, HIGH_OPACITY_STRATUM]:
            curve = subset.loc[subset["opacity_stratum"] == stratum].copy()
            if curve.empty:
                continue
            ax.plot(
                curve["mean_score"],
                curve["event_rate"],
                marker="o",
                label=stratum,
                color=palette[stratum],
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


def summarise_quantification_limits(
    scored_predictions: pd.DataFrame,
    disagreement_frame: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    *,
    n_bins: int = 6,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, object]:
    """Build a compact summary for the quantification-limit diagnostics."""
    summary: dict[str, object] = {
        "n_files": int(disagreement_frame.shape[0]) if not disagreement_frame.empty else 0,
        "n_pairs": int(disagreement_frame["pair_id"].nunique()) if not disagreement_frame.empty else 0,
        "n_events": (
            int(disagreement_frame["event_observation_id"].nunique())
            if not disagreement_frame.empty
            else 0
        ),
        "n_repos": int(disagreement_frame["repo"].nunique()) if not disagreement_frame.empty else 0,
        "primary_model": PRIMARY_MODEL,
        "opacity_strata": {},
        "primary_model_calibration": {},
        "model_disagreement": {},
    }
    if disagreement_frame.empty or calibration_summary.empty:
        return summary

    opacity_summary = summarise_opacity_strata(disagreement_frame)
    for _, row in opacity_summary.iterrows():
        summary["opacity_strata"][str(row["opacity_stratum"])] = {
            "n_files": int(row["n_files"]),
            "n_pairs": int(row["n_pairs"]),
            "n_events": int(row["n_events"]),
            "n_repos": int(row["n_repos"]),
            "event_rate": float(row["event_rate"]),
            "mean_composite": float(row["mean_composite"]),
            "median_composite": float(row["median_composite"]),
        }

    primary_rows = calibration_summary.loc[
        calibration_summary["model"] == PRIMARY_MODEL
    ].copy()
    for stratum in ["overall"] + OPACITY_STRATA:
        row = primary_rows.loc[primary_rows["opacity_stratum"] == stratum]
        if row.empty:
            continue
        record = row.iloc[0]
        summary["primary_model_calibration"][stratum] = {
            "n_files": int(record["n_files"]),
            "event_rate": float(record["event_rate"]),
            "mean_score": float(record["mean_score"]),
            "calibration_gap": float(record["calibration_gap"]),
            "absolute_error_mean": float(record["absolute_error_mean"]),
            "brier_score": float(record["brier_score"]),
            "ece": float(record["ece"]),
            "roc_auc": None if pd.isna(record["roc_auc"]) else float(record["roc_auc"]),
            "average_precision": (
                None
                if pd.isna(record["average_precision"])
                else float(record["average_precision"])
            ),
        }

    primary_predictions = scored_predictions.loc[
        scored_predictions["model"] == PRIMARY_MODEL
    ].copy()
    primary_predictions["absolute_error"] = (
        primary_predictions["label"].astype(float)
        - pd.to_numeric(primary_predictions["score"], errors="coerce")
    ).abs()
    summary["primary_model_calibration"]["high_minus_low_absolute_error"] = bootstrap_stratum_gap(
        primary_predictions,
        statistic=lambda frame: float(pd.to_numeric(frame["absolute_error"], errors="coerce").mean()),
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    summary["primary_model_calibration"]["high_minus_low_ece"] = bootstrap_stratum_gap(
        primary_predictions,
        statistic=lambda frame: expected_calibration_error(
            frame["label"], frame["score"], n_bins=n_bins
        ),
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    disagreement_summary = summarise_disagreement_by_stratum(disagreement_frame)
    for stratum in ["overall"] + OPACITY_STRATA:
        row = disagreement_summary.loc[disagreement_summary["opacity_stratum"] == stratum]
        if row.empty:
            continue
        record = row.iloc[0]
        summary["model_disagreement"][stratum] = {
            "n_files": int(record["n_files"]),
            "event_rate": float(record["event_rate"]),
            "mean_score_std": float(record["mean_score_std"]),
            "median_score_std": float(record["median_score_std"]),
            "mean_score_range": float(record["mean_score_range"]),
            "median_score_range": float(record["median_score_range"]),
            "mean_history_vs_composite_gap": float(record["mean_history_vs_composite_gap"]),
            "mean_structure_vs_composite_gap": float(record["mean_structure_vs_composite_gap"]),
        }

    summary["model_disagreement"]["high_minus_low_score_range"] = bootstrap_stratum_gap(
        disagreement_frame,
        statistic=lambda frame: float(pd.to_numeric(frame["score_range"], errors="coerce").mean()),
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    summary["model_disagreement"]["high_minus_low_history_vs_composite_gap"] = (
        bootstrap_stratum_gap(
            disagreement_frame,
            statistic=lambda frame: float(
                pd.to_numeric(frame["history_vs_composite_gap"], errors="coerce").mean()
            ),
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
    )

    return summary
