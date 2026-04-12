"""Experiment 25: long-horizon external intervention mechanism follow-up."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cku_sim.analysis.audited_panel import load_audited_security_table, load_repo_paths
from cku_sim.analysis.intervention_mechanism import (
    build_security_label_lookup,
    filter_material_interventions,
    fit_frozen_score_models,
    merge_intervention_audit_frames,
    relabel_anchor_observations,
    score_anchor_observations,
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


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ci_below_zero(payload: dict[str, object]) -> bool:
    if not payload:
        return False
    return float(payload.get("ci_hi", 0.0)) < 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 25: long-horizon external intervention mechanism follow-up"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.external_holdout_expanded.yaml",
        help="Path to the external holdout config",
    )
    parser.add_argument(
        "--e23-subdir",
        type=str,
        default="e23_external_intervention_mechanism__expanded7_no_gitea__audited_v1",
        help="Source e23 result directory used as the frozen intervention artifact set",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e25_external_intervention_long_horizon__expanded7_no_gitea__h1825_v1",
        help="Output subdirectory under data/results",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        default="data/results/e20_external_replication__expanded7_no_gitea__audited_v1/train_file_level_dataset.parquet",
        help="Frozen train dataset used to fit the score models",
    )
    parser.add_argument(
        "--security-audit-path",
        type=str,
        default="data/processed/security_event_file_audit_curated.csv",
        help="Accepted security audit table used to rebuild future labels",
    )
    parser.add_argument("--horizon-days", type=int, default=1825)
    parser.add_argument("--min-structure-drops", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = _config_from_path(args.config)
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    repo_paths = load_repo_paths(config, corpus)
    security_audit = load_audited_security_table(Path(args.security_audit_path))
    security_lookup = build_security_label_lookup(security_audit, repo_paths)
    train_dataset_path = Path(args.train_dataset_path)
    train_dataset = (
        pd.read_csv(train_dataset_path)
        if train_dataset_path.suffix.lower() == ".csv"
        else pd.read_parquet(train_dataset_path)
    )
    fitted_models = fit_frozen_score_models(train_dataset)

    source_dir = config.results_dir / args.e23_subdir
    if not source_dir.exists():
        raise SystemExit(f"Missing frozen e23 result directory: {source_dir}")

    interventions = filter_material_interventions(
        _load_frame(source_dir / "intervention_pool.parquet"),
        min_structure_drops=args.min_structure_drops,
    )
    matched_pairs = _load_frame(source_dir / "matched_pairs.csv")
    anchor_rows = _load_frame(source_dir / "anchor_observations.parquet")
    maintenance_pool = _load_frame(source_dir / "maintenance_pool.parquet")
    intervention_audit = merge_intervention_audit_frames(
        pd.DataFrame(columns=[]),
        _load_frame(source_dir / "accepted_intervention_audit.csv"),
    )

    keep_ids = set(interventions["intervention_id"].astype(str).tolist()) if not interventions.empty else set()
    if keep_ids:
        matched_pairs = matched_pairs.loc[
            matched_pairs["intervention_id"].astype(str).isin(keep_ids)
        ].copy()
        anchor_rows = anchor_rows.loc[anchor_rows["intervention_id"].astype(str).isin(keep_ids)].copy()
        audit_keys = interventions[["repo", "intervention_commit", "file_path"]].drop_duplicates()
        intervention_audit = intervention_audit.merge(
            audit_keys,
            on=["repo", "intervention_commit", "file_path"],
            how="inner",
        )
    else:
        matched_pairs = matched_pairs.iloc[0:0].copy()
        anchor_rows = anchor_rows.iloc[0:0].copy()
        intervention_audit = intervention_audit.iloc[0:0].copy()

    relabelled_anchor_rows = relabel_anchor_observations(
        anchor_rows,
        security_lookup=security_lookup,
        horizon_days=args.horizon_days,
    )
    scored_anchor_rows = score_anchor_observations(relabelled_anchor_rows, fitted_models)
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

    pooled = summary["pooled_endpoints"]
    sharper_underprediction_gate = bool(
        _ci_below_zero(pooled["delta_underprediction_loss_did"]["bootstrap_by_intervention"])
        and _ci_below_zero(pooled["delta_underprediction_loss_did"]["bootstrap_by_repo"])
    )
    sharper_positive_log_gate = bool(
        _ci_below_zero(pooled["delta_positive_log_loss_did"]["bootstrap_by_intervention"])
        and _ci_below_zero(pooled["delta_positive_log_loss_did"]["bootstrap_by_repo"])
    )
    summary["gating"]["sharper_underprediction_gate"] = sharper_underprediction_gate
    summary["gating"]["sharper_positive_log_gate"] = sharper_positive_log_gate
    summary["gating"]["sharper_observable_mechanism_gate"] = bool(
        sharper_underprediction_gate and sharper_positive_log_gate
    )
    summary["parameters"] = {
        "config": args.config,
        "source_e23_subdir": args.e23_subdir,
        "train_dataset_path": args.train_dataset_path,
        "security_audit_path": args.security_audit_path,
        "horizon_days": args.horizon_days,
        "min_structure_drops": args.min_structure_drops,
        "n_bootstrap": args.n_bootstrap,
        "random_state": args.random_state,
    }

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    intervention_audit.to_csv(results_dir / "accepted_intervention_audit.csv", index=False)
    interventions.to_parquet(results_dir / "intervention_pool.parquet")
    interventions.to_csv(results_dir / "intervention_pool.csv", index=False)
    matched_pairs.to_csv(results_dir / "matched_pairs.csv", index=False)
    relabelled_anchor_rows.to_parquet(results_dir / "anchor_observations.parquet")
    relabelled_anchor_rows.to_csv(results_dir / "anchor_observations.csv", index=False)
    scored_anchor_rows.to_parquet(results_dir / "scored_anchor_observations.parquet")
    scored_anchor_rows.to_csv(results_dir / "scored_anchor_observations.csv", index=False)
    pair_did.to_csv(results_dir / "pair_did.csv", index=False)
    repo_summary.to_csv(results_dir / "repo_summary.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Long-horizon intervention mechanism: horizon=%d pairs=%d sharper-gate=%s",
        args.horizon_days,
        len(matched_pairs),
        summary["gating"]["sharper_observable_mechanism_gate"],
    )


if __name__ == "__main__":
    main()
