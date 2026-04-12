"""Experiment 26: security-file-enriched external intervention mechanism expansion."""

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
    filter_material_interventions,
    fit_frozen_score_models,
    merge_intervention_audit_frames,
    score_anchor_observations,
    seed_security_file_refactoring_intervention_audit_table,
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


def _parse_repo_row_caps(raw_value: str | None) -> dict[str, int]:
    repo_row_caps: dict[str, int] = {}
    if not raw_value:
        return repo_row_caps
    for item in raw_value.split(","):
        if "=" not in item:
            continue
        repo, raw_cap = item.split("=", 1)
        repo = repo.strip()
        raw_cap = raw_cap.strip()
        if repo and raw_cap:
            repo_row_caps[repo] = int(raw_cap)
    return repo_row_caps


def _default_repos() -> str:
    return ",".join(PRIMARY_EXTERNAL_MECHANISM_REPOS + ["go-gitea-gitea"])


def _ci_below_zero(payload: dict[str, object]) -> bool:
    if not payload:
        return False
    return float(payload.get("ci_hi", 0.0)) < 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 26: security-file-enriched external intervention mechanism expansion"
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
        help="Accepted security audit table used for future labels and enrichment seeding",
    )
    parser.add_argument(
        "--base-intervention-audit-path",
        type=str,
        default="data/processed/refactoring_intervention_audit_curated.csv",
        help="Existing curated intervention audit used as the baseline intervention set",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default=_default_repos(),
        help="Comma-separated external repos to include",
    )
    parser.add_argument("--since", type=str, default="2014-01-01")
    parser.add_argument("--until", type=str, default="2024-04-12")
    parser.add_argument("--horizon-days", type=int, default=1825)
    parser.add_argument("--min-loc", type=int, default=5)
    parser.add_argument("--min-structure-drops", type=int, default=0)
    parser.add_argument("--max-enrichment-rows-per-repo", type=int, default=40)
    parser.add_argument("--max-enrichment-rows-per-file", type=int, default=4)
    parser.add_argument(
        "--repo-row-caps",
        type=str,
        default=(
            "django-django=40,traefik-traefik=40,prometheus-prometheus=40,"
            "fastapi-fastapi=60,psf-requests=60,pallets-flask=40,scrapy-scrapy=40,"
            "go-gitea-gitea=60"
        ),
    )
    parser.add_argument("--max-control-commits-per-repo", type=int, default=300)
    parser.add_argument("--max-control-files-per-commit", type=int, default=3)
    parser.add_argument(
        "--prefetch-commits",
        action="store_true",
        help="Prefetch commit objects before materialization; disabled by default for partial-clone stability",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e26_external_intervention_securityfile_enriched__mix8_h1825_v1",
    )
    args = parser.parse_args()

    config = _config_from_path(args.config)
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    repo_paths = load_repo_paths(config, corpus)
    selected_repos = [value.strip() for value in args.repos.split(",") if value.strip()]
    repo_row_caps = _parse_repo_row_caps(args.repo_row_caps)

    security_audit = load_audited_security_table(Path(args.security_audit_path))
    security_audit = security_audit.loc[security_audit["repo"].isin(selected_repos)].copy()

    base_intervention_audit = pd.read_csv(args.base_intervention_audit_path)
    base_intervention_audit = base_intervention_audit.loc[
        (base_intervention_audit["review_decision"] == "accept")
        & (base_intervention_audit["repo"].isin(selected_repos))
    ].copy()

    enrichment_audit = seed_security_file_refactoring_intervention_audit_table(
        repo_paths,
        corpus,
        security_audit,
        repos=selected_repos,
        since=args.since,
        until=args.until,
        max_rows_per_repo=args.max_enrichment_rows_per_repo,
        repo_row_caps=repo_row_caps,
        max_rows_per_file=args.max_enrichment_rows_per_file,
        min_loc=args.min_loc,
        prefetch_commits=args.prefetch_commits,
    )
    expanded_audit = merge_intervention_audit_frames(base_intervention_audit, enrichment_audit)
    expanded_audit = expanded_audit.loc[expanded_audit["repo"].isin(selected_repos)].copy()

    interventions = build_audited_intervention_pool(
        repo_paths,
        corpus,
        expanded_audit,
        min_loc=args.min_loc,
        prefetch_commits=args.prefetch_commits,
    )
    interventions = filter_material_interventions(
        interventions,
        min_structure_drops=args.min_structure_drops,
    )

    maintenance_pool = build_maintenance_pool(
        repo_paths,
        corpus,
        repos=selected_repos,
        since=args.since,
        until=args.until,
        min_loc=args.min_loc,
        max_commits_per_repo=args.max_control_commits_per_repo,
        max_files_per_commit=args.max_control_files_per_commit,
        prefetch_commits=args.prefetch_commits,
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
        intervention_audit=expanded_audit,
        maintenance_pool=maintenance_pool,
        n_boot=args.n_bootstrap,
        seed=args.random_state,
    )

    pooled = summary["pooled_endpoints"]
    summary["gating"]["sharper_underprediction_gate"] = bool(
        _ci_below_zero(pooled["delta_underprediction_loss_did"]["bootstrap_by_intervention"])
        and _ci_below_zero(pooled["delta_underprediction_loss_did"]["bootstrap_by_repo"])
    )
    summary["gating"]["sharper_positive_log_gate"] = bool(
        _ci_below_zero(pooled["delta_positive_log_loss_did"]["bootstrap_by_intervention"])
        and _ci_below_zero(pooled["delta_positive_log_loss_did"]["bootstrap_by_repo"])
    )
    summary["gating"]["sharper_observable_mechanism_gate"] = bool(
        summary["gating"]["sharper_underprediction_gate"]
        and summary["gating"]["sharper_positive_log_gate"]
    )
    summary["parameters"] = {
        "config": args.config,
        "train_dataset_path": args.train_dataset_path,
        "security_audit_path": args.security_audit_path,
        "base_intervention_audit_path": args.base_intervention_audit_path,
        "repos": selected_repos,
        "since": args.since,
        "until": args.until,
        "horizon_days": args.horizon_days,
        "min_loc": args.min_loc,
        "min_structure_drops": args.min_structure_drops,
        "max_enrichment_rows_per_repo": args.max_enrichment_rows_per_repo,
        "max_enrichment_rows_per_file": args.max_enrichment_rows_per_file,
        "repo_row_caps": repo_row_caps,
        "max_control_commits_per_repo": args.max_control_commits_per_repo,
        "max_control_files_per_commit": args.max_control_files_per_commit,
        "prefetch_commits": bool(args.prefetch_commits),
        "n_bootstrap": args.n_bootstrap,
        "random_state": args.random_state,
    }
    summary["audit_breakdown"] = {
        "base_rows": int(len(base_intervention_audit)),
        "enrichment_rows": int(len(enrichment_audit)),
        "expanded_rows": int(len(expanded_audit)),
        "expanded_repos": int(expanded_audit["repo"].nunique()) if not expanded_audit.empty else 0,
    }

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    base_intervention_audit.to_csv(results_dir / "base_intervention_audit.csv", index=False)
    enrichment_audit.to_csv(results_dir / "security_file_enrichment_audit.csv", index=False)
    expanded_audit.to_csv(results_dir / "expanded_intervention_audit.csv", index=False)
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

    logger.info(
        "Security-file-enriched intervention study: repos=%d interventions=%d pairs=%d positives=%d sharper=%s",
        len(selected_repos),
        len(interventions),
        len(matched_pairs),
        summary["n_positive_pair_rows"],
        summary["gating"]["sharper_observable_mechanism_gate"],
    )


if __name__ == "__main__":
    main()
