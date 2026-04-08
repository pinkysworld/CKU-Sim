"""Experiment 6: file-level case-control analysis of vulnerability-fixing commits.

For each security-fix event available locally, this experiment compares the pre-fix
opacity of touched source files against matched untouched files from the same parent
snapshot. The `--ground-truth-policy` flag selects the event-definition policy.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.analysis.file_level_case_control import (
    GROUND_TRUTH_POLICIES,
    describe_ground_truth_policy,
    match_case_control_pairs,
    normalise_github_slug,
    results_subdir_for_ground_truth_policy,
    summarise_case_control_pairs,
    summarise_commit_level_deltas,
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
    ax.set_ylabel("Case - control composite opacity")
    ax.set_xlabel("Repository")
    ax.set_title("Pre-fix file opacity in vulnerability-fixing commits")
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
    ax.set_xlabel("Case - control composite opacity")
    ax.set_ylabel("Matched file pairs")
    ax.set_title("Distribution of pre-fix opacity deltas")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 6: file-level case-control study")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated repo names to include (default: all eligible corpus repos)",
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=20,
        help="Minimum LOC for case/control source files",
    )
    parser.add_argument(
        "--ground-truth-policy",
        type=str,
        default="nvd_commit_refs",
        choices=sorted(GROUND_TRUTH_POLICIES),
        help="Ground-truth event policy",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=None,
        help="Optional results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    if args.repos:
        selected = {name.strip() for name in args.repos.split(",") if name.strip()}
        corpus = [entry for entry in corpus if entry.name in selected]

    results_subdir = args.results_subdir or results_subdir_for_ground_truth_policy(
        args.ground_truth_policy
    )
    results_dir = config.results_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.processed_dir / "nvd_cache"
    policy_meta = describe_ground_truth_policy(args.ground_truth_policy)

    logger.info("=" * 60)
    logger.info("Experiment 6: File-level vulnerability case-control analysis")
    logger.info("=" * 60)
    logger.info("Ground-truth policy: %s", policy_meta["label"])

    pair_frames: list[pd.DataFrame] = []
    repo_summaries: list[dict[str, object]] = []

    for entry in corpus:
        repo_path = config.raw_dir / entry.name
        if not repo_path.exists():
            logger.info("Skipping %s: repository checkout not available", entry.name)
            continue
        if (
            args.ground_truth_policy in {"nvd_commit_refs", "strict_nvd_event"}
            and normalise_github_slug(entry.git_url) is None
        ):
            logger.info("Skipping %s: non-GitHub repository", entry.name)
            continue

        logger.info("Analysing %s...", entry.name)
        pairs = match_case_control_pairs(
            repo_path,
            entry,
            cache_dir,
            min_loc=args.min_loc,
            ground_truth_policy=args.ground_truth_policy,
        )
        if pairs.empty:
            logger.info("  %s: no usable matched pairs", entry.name)
            continue

        summary = summarise_case_control_pairs(pairs)
        summary["repo"] = entry.name
        summary["category"] = entry.category
        pair_frames.append(pairs)
        repo_summaries.append(summary)

        logger.info(
            "  %s: %d pairs across %d commits, median delta %.4f, positive share %.3f",
            entry.name,
            summary["n_pairs"],
            summary["n_commits"],
            summary["median_delta_composite"],
            summary["positive_share"],
        )

    if not pair_frames:
        logger.error("No usable matched pairs were generated.")
        return

    pairs_df = pd.concat(pair_frames, ignore_index=True)
    repo_summary_df = pd.DataFrame(repo_summaries).sort_values(
        by="median_delta_composite",
        ascending=False,
    )
    commit_summary_df = (
        pairs_df.groupby(["repo", "commit", "cve_ids"], as_index=False)
        .agg(
            n_pairs=("delta_composite", "size"),
            event_id=("event_id", "first"),
            ground_truth_policy=("ground_truth_policy", "first"),
            mean_delta_composite=("delta_composite", "mean"),
            median_delta_composite=("delta_composite", "median"),
            positive_share=("delta_composite", lambda s: float((s > 0).mean())),
        )
        .sort_values(["repo", "mean_delta_composite"], ascending=[True, False])
    )
    overall_summary = {
        "pair_level": summarise_case_control_pairs(pairs_df),
        "commit_level": summarise_commit_level_deltas(commit_summary_df),
    }

    pairs_df.to_parquet(results_dir / "pair_level.parquet")
    pairs_df.to_csv(results_dir / "pair_level.csv", index=False)
    repo_summary_df.to_csv(results_dir / "repo_summary.csv", index=False)
    commit_summary_df.to_csv(results_dir / "commit_summary.csv", index=False)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)

    _plot_delta_boxplot(pairs_df, results_dir / "delta_by_repo.pdf")
    _plot_delta_boxplot(pairs_df, results_dir / "delta_by_repo.png")
    _plot_delta_histogram(pairs_df, results_dir / "delta_histogram.pdf")
    _plot_delta_histogram(pairs_df, results_dir / "delta_histogram.png")

    logger.info("\n" + "=" * 40)
    logger.info("KEY RESULTS")
    logger.info("=" * 40)
    logger.info(
        "Pair level: %d pairs across %d commits in %d repos",
        overall_summary["pair_level"]["n_pairs"],
        overall_summary["pair_level"]["n_commits"],
        overall_summary["pair_level"]["n_repos"],
    )
    logger.info(
        "Pair-level composite delta: mean=%.4f median=%.4f positive_share=%.3f",
        overall_summary["pair_level"]["mean_delta_composite"],
        overall_summary["pair_level"]["median_delta_composite"],
        overall_summary["pair_level"]["positive_share"],
    )
    pair_bootstrap = overall_summary["pair_level"].get("bootstrap_primary_cluster", {})
    if pair_bootstrap:
        median_ci = pair_bootstrap.get("median_delta_composite_ci")
        if isinstance(median_ci, list) and len(median_ci) == 2:
            logger.info(
                "Pair-level clustered 95%% CI for median delta (%s): [%.4f, %.4f]",
                pair_bootstrap.get("cluster_col"),
                median_ci[0],
                median_ci[1],
            )
    if "wilcoxon_pvalue_greater" in overall_summary["pair_level"]:
        logger.info(
            "Pair-level Wilcoxon one-sided p=%.4g, sign-test p=%.4g, rank-biserial=%.4f",
            overall_summary["pair_level"]["wilcoxon_pvalue_greater"],
            overall_summary["pair_level"].get("sign_test_pvalue_greater", float("nan")),
            overall_summary["pair_level"].get("rank_biserial_effect", float("nan")),
        )
    logger.info(
        "Commit-event level: %d event rows across %d unique commits in %d repos, mean delta %.4f median %.4f positive_share %.3f",
        overall_summary["commit_level"]["n_commit_events"],
        overall_summary["commit_level"]["n_unique_commits"],
        overall_summary["commit_level"]["n_repos"],
        overall_summary["commit_level"]["mean_delta_composite"],
        overall_summary["commit_level"]["median_delta_composite"],
        overall_summary["commit_level"]["positive_share"],
    )
    commit_bootstrap = overall_summary["commit_level"].get("bootstrap_primary_cluster", {})
    if commit_bootstrap:
        median_ci = commit_bootstrap.get("median_delta_composite_ci")
        if isinstance(median_ci, list) and len(median_ci) == 2:
            logger.info(
                "Commit-event clustered 95%% CI for median delta (%s): [%.4f, %.4f]",
                commit_bootstrap.get("cluster_col"),
                median_ci[0],
                median_ci[1],
            )
    if "wilcoxon_pvalue_greater" in overall_summary["commit_level"]:
        logger.info(
            "Commit-level Wilcoxon one-sided p=%.4g, sign-test p=%.4g, rank-biserial=%.4f",
            overall_summary["commit_level"]["wilcoxon_pvalue_greater"],
            overall_summary["commit_level"].get("sign_test_pvalue_greater", float("nan")),
            overall_summary["commit_level"].get("rank_biserial_effect", float("nan")),
        )

    logger.info("Top repositories by median composite delta:")
    for _, row in repo_summary_df.head(10).iterrows():
        logger.info(
            "  %s: pairs=%d median_delta=%.4f positive_share=%.3f",
            row["repo"],
            row["n_pairs"],
            row["median_delta_composite"],
            row["positive_share"],
        )

    logger.info("Experiment 6 complete.")


if __name__ == "__main__":
    main()
