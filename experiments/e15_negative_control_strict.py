"""Experiment 15: strict security-vs-bugfix negative control."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.analysis.negative_control import (
    augment_security_file_dataset,
    build_security_file_dataset,
    collect_ordinary_bugfix_candidates,
    evaluate_negative_control_prediction,
    match_security_to_bugfix_pairs,
    summarise_negative_control_pairs,
    summarise_security_event_level_deltas,
)
from cku_sim.analysis.predictive_validation import (
    plot_model_comparison,
    plot_pooled_roc_curves,
)
from cku_sim.core.config import Config, DEFAULT_CORPUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _plot_delta_boxplot(pairs: pd.DataFrame, output_path: Path) -> None:
    if pairs.empty:
        return

    order = (
        pairs.groupby("repo")["delta_composite"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [pairs.loc[pairs["repo"] == repo, "delta_composite"].values for repo in order]
    ax.boxplot(data, tick_labels=order, patch_artist=True)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Security - ordinary bug-fix composite opacity")
    ax.set_xlabel("Repository")
    ax.set_title("Strict pre-fix opacity comparison: security fixes versus ordinary bug fixes")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_delta_histogram(pairs: pd.DataFrame, output_path: Path) -> None:
    if pairs.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairs["delta_composite"], bins=24, color="#4C78A8", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Security - ordinary bug-fix composite opacity")
    ax.set_ylabel("Matched file pairs")
    ax.set_title("Distribution of strict security-vs-bugfix opacity deltas")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _summarise_match_quality(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    return (
        pairs.groupby("repo", as_index=False)
        .agg(
            n_pairs=("delta_composite", "size"),
            same_suffix_share=("same_suffix_match", "mean"),
            same_subsystem_share=("same_subsystem_match", "mean"),
            same_top_level_share=("same_top_level_match", "mean"),
            median_loc_ratio=("loc_ratio", "median"),
            median_delta_log_prior_touches_total=("delta_log_prior_touches_total", "median"),
        )
        .sort_values(["same_subsystem_share", "n_pairs"], ascending=[False, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 15: strict security-fix versus ordinary bug-fix controls"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated repo names to include",
    )
    parser.add_argument(
        "--security-e06-subdir",
        type=str,
        default="e06_file_case_control__expanded_advisory_event",
        help="Results subdirectory containing experiment 6 pair_level.parquet",
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=20,
        help="Minimum LOC for security and bug-fix files",
    )
    parser.add_argument(
        "--max-changed-source-files",
        type=int,
        default=6,
        help="Maximum changed source files allowed in a strict negative-control bug-fix commit",
    )
    parser.add_argument(
        "--max-bugfix-commits",
        type=int,
        default=600,
        help="Maximum accepted ordinary bug-fix commits per repository",
    )
    parser.add_argument(
        "--max-log-loc-gap",
        type=float,
        default=0.75,
        help="Maximum allowed log-LOC gap between security and bug-fix matches",
    )
    parser.add_argument(
        "--max-log-touch-gap",
        type=float,
        default=1.10,
        help="Maximum allowed log prior-touch gap between security and bug-fix matches",
    )
    parser.add_argument(
        "--max-directory-depth-gap",
        type=int,
        default=2,
        help="Maximum allowed directory-depth gap between security and bug-fix matches",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=None,
        help="Optional results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    pairs_path = config.results_dir / args.security_e06_subdir / "pair_level.parquet"
    if not pairs_path.exists():
        logger.error("Security pair data not found: %s", pairs_path)
        return

    results_subdir = args.results_subdir or f"e15_negative_control_strict__{args.security_e06_subdir}"
    results_dir = config.results_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    selected: set[str] | None = None
    if args.repos:
        selected = {name.strip() for name in args.repos.split(",") if name.strip()}
        corpus = [entry for entry in corpus if entry.name in selected]

    security_pairs = pd.read_parquet(pairs_path)
    security_df = build_security_file_dataset(security_pairs)
    if selected is not None:
        security_df = security_df.loc[security_df["repo"].isin(selected)].copy()
    if security_df.empty:
        logger.error("No security observations available after repository filtering.")
        return

    corpus_by_name = {entry.name: entry for entry in corpus}
    repo_names = sorted(set(security_df["repo"]))
    repo_paths = {
        repo: config.raw_dir / repo
        for repo in repo_names
        if repo in corpus_by_name and (config.raw_dir / repo).exists()
    }
    if not repo_paths:
        logger.error("No local repository checkouts available for the selected security dataset.")
        return

    security_df = augment_security_file_dataset(
        security_df.loc[security_df["repo"].isin(repo_paths)].copy(),
        repo_paths,
    )

    bugfix_frames: list[pd.DataFrame] = []
    for repo in repo_names:
        if repo not in repo_paths:
            continue
        repo_security = security_df.loc[security_df["repo"] == repo]
        excluded_commits = set(repo_security["security_commit"].astype(str))
        bugfix_df = collect_ordinary_bugfix_candidates(
            repo_paths[repo],
            corpus_by_name[repo],
            excluded_commits=excluded_commits,
            max_changed_source_files=args.max_changed_source_files,
            max_bugfix_commits=args.max_bugfix_commits,
        )
        if bugfix_df.empty:
            logger.info("Skipping %s: no usable strict bug-fix candidates", repo)
            continue
        bugfix_frames.append(bugfix_df)
        logger.info(
            "  %s: %d bug-fix candidates across %d commits",
            repo,
            len(bugfix_df),
            bugfix_df["bugfix_commit"].nunique(),
        )

    if not bugfix_frames:
        logger.error("No strict ordinary bug-fix candidates were generated.")
        return

    bugfix_candidate_df = pd.concat(bugfix_frames, ignore_index=True)
    matched_pairs = match_security_to_bugfix_pairs(
        security_df,
        bugfix_candidate_df,
        repo_paths,
        min_loc=args.min_loc,
        require_same_suffix=True,
        require_same_subsystem=True,
        max_log_loc_gap=args.max_log_loc_gap,
        max_log_touch_gap=args.max_log_touch_gap,
        max_directory_depth_gap=args.max_directory_depth_gap,
    )
    if matched_pairs.empty:
        logger.error("No matched strict security-vs-bugfix pairs were generated.")
        return

    repo_summaries: list[dict[str, object]] = []
    for repo, group in matched_pairs.groupby("repo"):
        summary = summarise_negative_control_pairs(group)
        summary["repo"] = repo
        repo_summaries.append(summary)

    repo_summary_df = pd.DataFrame(repo_summaries).sort_values(
        by="median_delta_composite",
        ascending=False,
    )
    event_summary_df = (
        matched_pairs.groupby(
            ["repo", "security_commit", "security_event_id", "security_ground_truth_policy"],
            as_index=False,
        )
        .agg(
            bugfix_commit=("bugfix_commit", "first"),
            n_pairs=("delta_composite", "size"),
            mean_delta_composite=("delta_composite", "mean"),
            median_delta_composite=("delta_composite", "median"),
            positive_share=("delta_composite", lambda s: float((s > 0).mean())),
        )
        .sort_values(["repo", "mean_delta_composite"], ascending=[True, False])
    )
    match_quality_df = _summarise_match_quality(matched_pairs)
    overall_summary = {
        "pair_level": summarise_negative_control_pairs(matched_pairs),
        "event_level": summarise_security_event_level_deltas(event_summary_df),
        "matching": {
            "require_same_suffix": True,
            "require_same_subsystem": True,
            "max_log_loc_gap": float(args.max_log_loc_gap),
            "max_log_touch_gap": float(args.max_log_touch_gap),
            "max_directory_depth_gap": int(args.max_directory_depth_gap),
        },
    }

    dataset, predictions, fold_metrics, prediction_summary = evaluate_negative_control_prediction(
        matched_pairs
    )

    matched_pairs.to_parquet(results_dir / "security_vs_bugfix_pairs.parquet")
    matched_pairs.to_csv(results_dir / "security_vs_bugfix_pairs.csv", index=False)
    bugfix_candidate_df.to_csv(results_dir / "bugfix_candidate_pool.csv", index=False)
    repo_summary_df.to_csv(results_dir / "repo_summary.csv", index=False)
    event_summary_df.to_csv(results_dir / "event_summary.csv", index=False)
    match_quality_df.to_csv(results_dir / "match_quality.csv", index=False)
    dataset.to_parquet(results_dir / "classification_dataset.parquet")
    dataset.to_csv(results_dir / "classification_dataset.csv", index=False)
    predictions.to_parquet(results_dir / "heldout_predictions.parquet")
    predictions.to_csv(results_dir / "heldout_predictions.csv", index=False)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump({**overall_summary, "prediction": prediction_summary}, handle, indent=2)

    _plot_delta_boxplot(matched_pairs, results_dir / "delta_by_repo.pdf")
    _plot_delta_boxplot(matched_pairs, results_dir / "delta_by_repo.png")
    _plot_delta_histogram(matched_pairs, results_dir / "delta_histogram.pdf")
    _plot_delta_histogram(matched_pairs, results_dir / "delta_histogram.png")
    plot_model_comparison(prediction_summary, results_dir / "model_comparison.pdf")
    plot_model_comparison(prediction_summary, results_dir / "model_comparison.png")
    plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.pdf")
    plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.png")

    logger.info(
        "Strict negative control: %d pairs across %d security events in %d repos",
        overall_summary["pair_level"]["n_pairs"],
        overall_summary["pair_level"]["n_security_events"],
        overall_summary["pair_level"]["n_repos"],
    )
    logger.info(
        "Median delta %.4f, same-subsystem share %.3f, same-suffix share %.3f",
        overall_summary["pair_level"]["median_delta_composite"],
        overall_summary["pair_level"].get("same_subsystem_share", float("nan")),
        overall_summary["pair_level"].get("same_suffix_share", float("nan")),
    )


if __name__ == "__main__":
    main()
