"""Experiment 9: security-fix files versus ordinary bug-fix negative controls."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.analysis.negative_control import (
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
    ax.set_title("Pre-fix opacity: security fixes versus ordinary bug fixes")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_delta_histogram(pairs: pd.DataFrame, output_path: Path) -> None:
    if pairs.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairs["delta_composite"], bins=24, color="#E45756", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Security - ordinary bug-fix composite opacity")
    ax.set_ylabel("Matched file pairs")
    ax.set_title("Distribution of security-vs-bugfix opacity deltas")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 9: security-fix files versus ordinary bug-fix controls"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated repo names to include (default: all repos in the security dataset)",
    )
    parser.add_argument(
        "--security-e06-subdir",
        type=str,
        default="e06_file_case_control__strict_nvd_event",
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
        default=8,
        help="Maximum changed source files allowed in a negative-control bug-fix commit",
    )
    parser.add_argument(
        "--max-bugfix-commits",
        type=int,
        default=800,
        help="Maximum accepted ordinary bug-fix commits per repository",
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
        logger.error(
            "Security pair data not found. Run experiment 6 first:\n"
            "  python -m experiments.e06_file_level_case_control --config experiments/config.yaml"
        )
        return

    results_subdir = args.results_subdir or (
        "e09_negative_control_bugfix"
        if args.security_e06_subdir == "e06_file_case_control__strict_nvd_event"
        else f"e09_negative_control_bugfix__{args.security_e06_subdir}"
    )
    results_dir = config.results_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    if args.repos:
        selected = {name.strip() for name in args.repos.split(",") if name.strip()}
        corpus = [entry for entry in corpus if entry.name in selected]

    security_pairs = pd.read_parquet(pairs_path)
    security_df = build_security_file_dataset(security_pairs)
    if args.repos:
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
        logger.error("No local raw repositories available for the selected security dataset.")
        return

    logger.info("=" * 60)
    logger.info("Experiment 9: Security fixes versus ordinary bug-fix controls")
    logger.info("=" * 60)
    logger.info("Security source dataset: %s", args.security_e06_subdir)

    bugfix_frames: list[pd.DataFrame] = []
    for repo in repo_names:
        if repo not in repo_paths:
            logger.info("Skipping %s: no local repository checkout", repo)
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
            logger.info("Skipping %s: no usable ordinary bug-fix candidates", repo)
            continue
        logger.info(
            "  %s: %d bug-fix file candidates across %d commits",
            repo,
            len(bugfix_df),
            bugfix_df["bugfix_commit"].nunique(),
        )
        bugfix_frames.append(bugfix_df)

    if not bugfix_frames:
        logger.error("No ordinary bug-fix candidates were generated.")
        return

    bugfix_candidate_df = pd.concat(bugfix_frames, ignore_index=True)
    matched_pairs = match_security_to_bugfix_pairs(
        security_df.loc[security_df["repo"].isin(repo_paths)].copy(),
        bugfix_candidate_df,
        repo_paths,
        min_loc=args.min_loc,
    )
    if matched_pairs.empty:
        logger.error("No matched security-vs-bugfix pairs were generated.")
        return

    repo_summaries: list[dict[str, object]] = []
    for repo, group in matched_pairs.groupby("repo"):
        summary = summarise_negative_control_pairs(group)
        summary["repo"] = repo
        repo_summaries.append(summary)
        logger.info(
            "  %s: %d pairs, median delta %.4f, positive share %.3f",
            repo,
            summary["n_pairs"],
            summary["median_delta_composite"],
            summary["positive_share"],
        )

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
    overall_summary = {
        "pair_level": summarise_negative_control_pairs(matched_pairs),
        "event_level": summarise_security_event_level_deltas(event_summary_df),
    }

    dataset, predictions, fold_metrics, prediction_summary = evaluate_negative_control_prediction(
        matched_pairs
    )

    matched_pairs.to_parquet(results_dir / "security_vs_bugfix_pairs.parquet")
    matched_pairs.to_csv(results_dir / "security_vs_bugfix_pairs.csv", index=False)
    bugfix_candidate_df.to_csv(results_dir / "bugfix_candidate_pool.csv", index=False)
    repo_summary_df.to_csv(results_dir / "repo_summary.csv", index=False)
    event_summary_df.to_csv(results_dir / "event_summary.csv", index=False)
    dataset.to_parquet(results_dir / "classification_dataset.parquet")
    dataset.to_csv(results_dir / "classification_dataset.csv", index=False)
    predictions.to_parquet(results_dir / "heldout_predictions.parquet")
    predictions.to_csv(results_dir / "heldout_predictions.csv", index=False)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(
            {
                **overall_summary,
                "prediction": prediction_summary,
            },
            handle,
            indent=2,
        )

    _plot_delta_boxplot(matched_pairs, results_dir / "delta_by_repo.pdf")
    _plot_delta_boxplot(matched_pairs, results_dir / "delta_by_repo.png")
    _plot_delta_histogram(matched_pairs, results_dir / "delta_histogram.pdf")
    _plot_delta_histogram(matched_pairs, results_dir / "delta_histogram.png")
    plot_model_comparison(prediction_summary, results_dir / "model_comparison.pdf")
    plot_model_comparison(prediction_summary, results_dir / "model_comparison.png")
    plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.pdf")
    plot_pooled_roc_curves(predictions, results_dir / "pooled_roc.png")

    logger.info("\n" + "=" * 40)
    logger.info("KEY RESULTS")
    logger.info("=" * 40)
    logger.info(
        "Pair level: %d matched security-vs-bugfix pairs across %d security commits in %d repos",
        overall_summary["pair_level"]["n_pairs"],
        overall_summary["pair_level"]["n_security_commits"],
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
        "Event level: %d security events across %d repos, mean delta %.4f median %.4f positive_share %.3f",
        overall_summary["event_level"]["n_security_events"],
        overall_summary["event_level"]["n_repos"],
        overall_summary["event_level"]["mean_delta_composite"],
        overall_summary["event_level"]["median_delta_composite"],
        overall_summary["event_level"]["positive_share"],
    )
    event_bootstrap = overall_summary["event_level"].get("bootstrap_primary_cluster", {})
    if event_bootstrap:
        median_ci = event_bootstrap.get("median_delta_composite_ci")
        if isinstance(median_ci, list) and len(median_ci) == 2:
            logger.info(
                "Event-level clustered 95%% CI for median delta (%s): [%.4f, %.4f]",
                event_bootstrap.get("cluster_col"),
                median_ci[0],
                median_ci[1],
            )
    if "wilcoxon_pvalue_greater" in overall_summary["event_level"]:
        logger.info(
            "Event-level Wilcoxon one-sided p=%.4g, sign-test p=%.4g, rank-biserial=%.4f",
            overall_summary["event_level"]["wilcoxon_pvalue_greater"],
            overall_summary["event_level"].get("sign_test_pvalue_greater", float("nan")),
            overall_summary["event_level"].get("rank_biserial_effect", float("nan")),
        )
    baseline = prediction_summary["models"]["baseline_size"]
    plus_composite = prediction_summary["models"]["baseline_plus_composite"]
    logger.info(
        "Held-out classification AUC: baseline=%.3f baseline+composite=%.3f",
        baseline["roc_auc"],
        plus_composite["roc_auc"],
    )
    logger.info(
        "Held-out pairwise accuracy: baseline=%.3f baseline+composite=%.3f",
        baseline["pairwise_accuracy"],
        plus_composite["pairwise_accuracy"],
    )


if __name__ == "__main__":
    main()
