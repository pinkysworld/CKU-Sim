"""Experiment 17: screening audit for ordinary bug-fix controls."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from cku_sim.analysis.negative_control import (
    screen_bugfix_control_commits,
    summarise_bugfix_control_screen,
)
from cku_sim.core.config import Config, DEFAULT_CORPUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 17: screening audit for ordinary bug-fix controls"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--e15-subdir",
        type=str,
        default="e15_negative_control_strict__expanded_advisory__light6",
        help="Results subdirectory containing strict negative-control outputs",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default=None,
        help="Optional results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    results_subdir = args.results_subdir or f"e17_bugfix_control_audit__{args.e15_subdir}"
    results_dir = config.results_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = config.results_dir / args.e15_subdir / "security_vs_bugfix_pairs.parquet"
    if not pairs_path.exists():
        logger.error("Strict negative-control pairs not found: %s", pairs_path)
        return

    pairs_df = pd.read_parquet(pairs_path)
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    repo_paths = {
        entry.name: config.raw_dir / entry.name
        for entry in corpus
        if (config.raw_dir / entry.name).exists()
    }
    corpus_by_name = {entry.name: entry for entry in corpus}

    screened_df = screen_bugfix_control_commits(
        pairs_df,
        repo_paths,
        corpus_by_name,
    )
    if screened_df.empty:
        logger.error("No bug-fix controls were available for screening.")
        return

    repo_summary = (
        screened_df.groupby(["repo", "review_decision"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("repo")
    )
    summary = summarise_bugfix_control_screen(screened_df)
    summary["e15_subdir"] = args.e15_subdir

    screened_df.to_csv(results_dir / "screened_controls.csv", index=False)
    repo_summary.to_csv(results_dir / "repo_summary.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Bug-fix control audit: %d reviewed controls, accept=%d ambiguous=%d reject=%d",
        summary["n_reviewed"],
        summary["review_counts"].get("accept", 0),
        summary["review_counts"].get("ambiguous", 0),
        summary["review_counts"].get("reject", 0),
    )


if __name__ == "__main__":
    main()
