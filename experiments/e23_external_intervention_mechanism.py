"""Experiment 23: external audited intervention study for CKU mechanism closure."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cku_sim.analysis.audited_panel import load_audited_security_table, load_repo_paths
from cku_sim.analysis.intervention_mechanism import (
    PRIMARY_EXTERNAL_MECHANISM_REPOS,
    build_anchor_observations,
    build_audited_intervention_pool,
    build_maintenance_pool,
    build_matched_intervention_pairs,
    build_security_label_lookup,
    fit_frozen_score_models,
    load_intervention_audit_table,
    score_anchor_observations,
    seed_refactoring_intervention_audit_table,
    summarise_intervention_mechanism,
    summarise_pairs_to_did,
)
from cku_sim.core.config import Config, DEFAULT_CORPUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _config_from_path(path: str | None) -> Config:
    if path:
        return Config.from_yaml(path)
    return Config()


def _default_until_date(horizon_days: int) -> str:
    today = pd.Timestamp.now(tz="UTC").normalize()
    return (today - pd.Timedelta(days=horizon_days)).date().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 23: external audited intervention study for CKU mechanism closure"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.external_holdout_expanded.yaml",
        help="Path to the external holdout config",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        default="data/results/e20_external_replication__expanded7_no_gitea__audited_v1/train_file_level_dataset.parquet",
        help="Frozen train dataset used to fit score models",
    )
    parser.add_argument(
        "--security-audit-path",
        type=str,
        default="data/processed/security_event_file_audit_curated.csv",
        help="Accepted security audit table used for future labels",
    )
    parser.add_argument(
        "--intervention-audit-path",
        type=str,
        default="data/processed/refactoring_intervention_audit_curated.csv",
        help="Curated intervention audit table for accepted structural refactors",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default=",".join(PRIMARY_EXTERNAL_MECHANISM_REPOS),
        help="Comma-separated external repos to include",
    )
    parser.add_argument(
        "--since",
        type=str,
        default="2014-01-01",
        help="Inclusive start date for intervention and control candidate scans",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Inclusive end date for candidate scans; defaults to analysis_date minus horizon",
    )
    parser.add_argument("--horizon-days", type=int, default=730)
    parser.add_argument("--min-loc", type=int, default=5)
    parser.add_argument(
        "--max-rows-per-repo",
        type=int,
        default=20,
        help="Maximum accepted intervention rows per repo when seeding the audit table",
    )
    parser.add_argument(
        "--repo-row-caps",
        type=str,
        default=None,
        help="Optional comma-separated repo=row_cap overrides for intervention seeding",
    )
    parser.add_argument(
        "--max-control-commits-per-repo",
        type=int,
        default=250,
        help="Maximum evenly spaced maintenance commits to materialise per repo",
    )
    parser.add_argument(
        "--seed-audit",
        action="store_true",
        help="Refresh the curated intervention audit table before the run",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Cluster bootstrap resamples for the pooled DiD endpoints",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for clustered bootstrap intervals",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e23_external_intervention_mechanism__expanded7_no_gitea__audited_v1",
        help="Output subdirectory under data/results",
    )
    args = parser.parse_args()

    config = _config_from_path(args.config)
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    repo_paths = load_repo_paths(config, corpus)
    selected_repos = [value.strip() for value in args.repos.split(",") if value.strip()]
    repo_row_caps = {}
    if args.repo_row_caps:
        for item in args.repo_row_caps.split(","):
            if "=" not in item:
                continue
            repo, raw_cap = item.split("=", 1)
            repo = repo.strip()
            raw_cap = raw_cap.strip()
            if not repo or not raw_cap:
                continue
            repo_row_caps[repo] = int(raw_cap)
    until = args.until or _default_until_date(args.horizon_days)

    intervention_audit_path = Path(args.intervention_audit_path)
    intervention_audit_path.parent.mkdir(parents=True, exist_ok=True)
    if args.seed_audit or not intervention_audit_path.exists():
        audit = seed_refactoring_intervention_audit_table(
            repo_paths,
            corpus,
            repos=selected_repos,
            since=args.since,
            until=until,
            max_rows_per_repo=args.max_rows_per_repo,
            repo_row_caps=repo_row_caps,
            min_loc=args.min_loc,
        )
        audit.to_csv(intervention_audit_path, index=False)
        logger.info(
            "Seeded intervention audit table with %d accepted rows at %s",
            len(audit),
            intervention_audit_path,
        )

    intervention_audit = load_intervention_audit_table(intervention_audit_path)
    intervention_audit = intervention_audit.loc[
        intervention_audit["repo"].isin(selected_repos)
    ].copy()

    security_audit = load_audited_security_table(Path(args.security_audit_path))
    security_audit = security_audit.loc[security_audit["repo"].isin(selected_repos)].copy()

    interventions = build_audited_intervention_pool(
        repo_paths,
        corpus,
        intervention_audit,
        min_loc=args.min_loc,
    )
    maintenance_pool = build_maintenance_pool(
        repo_paths,
        corpus,
        repos=selected_repos,
        since=args.since,
        until=until,
        min_loc=args.min_loc,
        max_commits_per_repo=args.max_control_commits_per_repo,
    )
    matched_pairs = build_matched_intervention_pairs(interventions, maintenance_pool)
    security_lookup = build_security_label_lookup(security_audit, repo_paths)
    anchor_rows = build_anchor_observations(
        matched_pairs,
        interventions,
        maintenance_pool,
        security_lookup=security_lookup,
        horizon_days=args.horizon_days,
    )

    train_dataset_path = Path(args.train_dataset_path)
    train_dataset = (
        pd.read_csv(train_dataset_path)
        if train_dataset_path.suffix.lower() == ".csv"
        else pd.read_parquet(train_dataset_path)
    )
    fitted_models = fit_frozen_score_models(train_dataset)
    scored_anchor_rows = score_anchor_observations(anchor_rows, fitted_models)
    pair_did = summarise_pairs_to_did(scored_anchor_rows)
    repo_summary, summary = summarise_intervention_mechanism(
        interventions,
        matched_pairs,
        pair_did,
        intervention_audit=intervention_audit,
        maintenance_pool=maintenance_pool,
        n_boot=args.n_bootstrap,
        seed=args.random_state,
    )

    summary["parameters"] = {
        "config": args.config,
        "train_dataset_path": args.train_dataset_path,
        "security_audit_path": args.security_audit_path,
        "intervention_audit_path": args.intervention_audit_path,
        "repos": selected_repos,
        "since": args.since,
        "until": until,
        "horizon_days": args.horizon_days,
        "min_loc": args.min_loc,
        "max_rows_per_repo": args.max_rows_per_repo,
        "repo_row_caps": repo_row_caps,
        "max_control_commits_per_repo": args.max_control_commits_per_repo,
        "n_bootstrap": args.n_bootstrap,
        "random_state": args.random_state,
    }

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    intervention_audit.to_csv(results_dir / "accepted_intervention_audit.csv", index=False)
    security_audit.to_csv(results_dir / "accepted_security_audit.csv", index=False)
    interventions.to_parquet(results_dir / "intervention_pool.parquet")
    interventions.to_csv(results_dir / "intervention_pool.csv", index=False)
    maintenance_pool.to_parquet(results_dir / "maintenance_pool.parquet")
    maintenance_pool.to_csv(results_dir / "maintenance_pool.csv", index=False)
    matched_pairs.to_csv(results_dir / "matched_pairs.csv", index=False)
    anchor_rows.to_parquet(results_dir / "anchor_observations.parquet")
    anchor_rows.to_csv(results_dir / "anchor_observations.csv", index=False)
    scored_anchor_rows.to_parquet(results_dir / "scored_anchor_observations.parquet")
    scored_anchor_rows.to_csv(results_dir / "scored_anchor_observations.csv", index=False)
    pair_did.to_csv(results_dir / "pair_did.csv", index=False)
    repo_summary.to_csv(results_dir / "repo_summary.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("=" * 60)
    logger.info("Experiment 23: external intervention mechanism study")
    logger.info("=" * 60)
    logger.info(
        "Accepted interventions: %d rows across %d repos",
        len(intervention_audit),
        intervention_audit["repo"].nunique() if not intervention_audit.empty else 0,
    )
    logger.info("Matched pairs: %d", len(matched_pairs))
    logger.info(
        "Composite DiD mean: %.4f",
        summary.get("pooled_endpoints", {})
        .get("delta_composite_did", {})
        .get("observed_mean", float("nan")),
    )
    logger.info(
        "Score-range DiD mean: %.4f",
        summary.get("pooled_endpoints", {})
        .get("delta_score_range_did", {})
        .get("observed_mean", float("nan")),
    )
    logger.info(
        "Absolute-error DiD mean: %.4f",
        summary.get("pooled_endpoints", {})
        .get("delta_absolute_error_did", {})
        .get("observed_mean", float("nan")),
    )
    logger.info(
        "Full CKU mechanism gate: %s",
        summary.get("gating", {}).get("full_cku_confirmation_gate", False),
    )


if __name__ == "__main__":
    main()
