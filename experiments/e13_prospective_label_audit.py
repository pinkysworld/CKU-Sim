"""Experiment 13: reviewed label audit for the prospective file-level panel."""

from __future__ import annotations

import argparse
import json
import logging

from cku_sim.analysis.label_audit import (
    apply_review_decisions,
    classify_ground_truth_source,
    compile_source_summary,
    enrich_audit_frame,
    sample_stratified_audit_rows,
    summarise_reviewed_audit,
)
from cku_sim.core.config import Config, DEFAULT_CORPUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 13: reviewed label audit for the prospective panel"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--e12-subdir",
        type=str,
        default="e12_prospective_file_panel__curated15_h730_l10_t5",
        help="Results subdirectory containing experiment 12 audit outputs",
    )
    parser.add_argument(
        "--audit-input",
        type=str,
        default="audit_full.csv",
        help="Audit CSV to review from the experiment 12 output directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120,
        help="Number of distinct event observations to review when sampling is enabled",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="stratified",
        choices=["stratified", "all", "existing"],
        help="Sampling strategy applied before review",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="ground_truth_source",
        choices=["ground_truth_source", "source_family"],
        help="Column used to stratify the audit sample",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for stratified sampling",
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
    corpus_by_name = {entry.name: entry for entry in corpus}
    repo_paths = {
        entry.name: config.raw_dir / entry.name
        for entry in corpus
        if (config.raw_dir / entry.name).exists()
    }

    input_dir = config.results_dir / args.e12_subdir
    audit_path = input_dir / args.audit_input
    if not audit_path.exists():
        logger.error("Audit input not found: %s", audit_path)
        return

    results_subdir = args.results_subdir or f"e13_prospective_label_audit__{args.e12_subdir}"
    results_dir = config.results_dir / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    audit_df = pd.read_csv(audit_path)
    if "source_family" not in audit_df.columns and "ground_truth_source" in audit_df.columns:
        audit_df["source_family"] = audit_df["ground_truth_source"].map(classify_ground_truth_source)

    if args.sampling == "all":
        review_input_df = audit_df.copy()
    elif args.sampling == "existing":
        review_input_df = audit_df.drop_duplicates(subset=["event_observation_id"]).copy()
    else:
        review_input_df = sample_stratified_audit_rows(
            audit_df,
            sample_size=args.sample_size,
            stratify_by=args.stratify_by,
            random_seed=args.random_seed,
        )

    enriched_df = enrich_audit_frame(review_input_df, repo_paths, corpus_by_name)
    reviewed_df = apply_review_decisions(enriched_df)
    summary = summarise_reviewed_audit(reviewed_df)
    source_summary_df = compile_source_summary(reviewed_df, group_col="ground_truth_source")
    family_summary_df = compile_source_summary(reviewed_df, group_col="source_family")
    summary["sampling"] = {
        "strategy": args.sampling,
        "audit_input": args.audit_input,
        "requested_sample_size": int(args.sample_size),
        "realised_sample_size": int(len(reviewed_df)),
        "stratify_by": args.stratify_by if args.sampling == "stratified" else None,
        "random_seed": int(args.random_seed),
    }

    review_input_df.to_csv(results_dir / "review_input.csv", index=False)
    reviewed_df.to_csv(results_dir / "reviewed_sample.csv", index=False)
    source_summary_df.to_csv(results_dir / "source_summary.csv", index=False)
    family_summary_df.to_csv(results_dir / "source_family_summary.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("=" * 60)
    logger.info("Experiment 13: Prospective label audit")
    logger.info("=" * 60)
    logger.info("Reviewed %d event observations from %s", summary["n_reviewed"], args.e12_subdir)
    logger.info(
        "Overall review: accept=%d ambiguous=%d reject=%d",
        summary["overall_review"]["accept"],
        summary["overall_review"]["ambiguous"],
        summary["overall_review"]["reject"],
    )
    logger.info(
        "Event-commit review: accept=%d ambiguous=%d reject=%d",
        summary["event_commit_review"]["accept"],
        summary["event_commit_review"]["ambiguous"],
        summary["event_commit_review"]["reject"],
    )
    logger.info(
        "File-touch review: accept=%d ambiguous=%d reject=%d",
        summary["file_review"]["accept"],
        summary["file_review"]["ambiguous"],
        summary["file_review"]["reject"],
    )


if __name__ == "__main__":
    main()
