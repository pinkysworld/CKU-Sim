"""Experiment 23: external audited intervention study for CKU mechanism closure."""

from __future__ import annotations

import argparse
import json
import logging
import re
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
    load_intervention_audit_table,
    merge_intervention_audit_frames,
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


REPO_SUMMARY_COLUMNS = [
    "repo",
    "accepted_intervention_rows",
    "n_interventions_materialised",
    "n_maintenance_rows",
    "n_matched_pairs",
    "n_anchor_rows",
    "n_scored_anchor_rows",
    "n_pair_did_rows",
    "completed_at",
]

CHECKPOINT_FRAME_FILES = {
    "intervention_audit": "accepted_intervention_audit.csv",
    "interventions": "intervention_pool.parquet",
    "maintenance_pool": "maintenance_pool.parquet",
    "matched_pairs": "matched_pairs.csv",
    "anchor_rows": "anchor_observations.parquet",
    "scored_anchor_rows": "scored_anchor_observations.parquet",
    "pair_did": "pair_did.csv",
}

CHECKPOINT_REQUIRED_KEYS = tuple(CHECKPOINT_FRAME_FILES.keys())


def _config_from_path(path: str | None) -> Config:
    if path:
        return Config.from_yaml(path)
    return Config()


def _default_until_date(horizon_days: int) -> str:
    today = pd.Timestamp.now(tz="UTC").normalize()
    return (today - pd.Timedelta(days=horizon_days)).date().isoformat()


def _parse_repo_row_caps(raw_value: str | None) -> dict[str, int]:
    repo_row_caps = {}
    if not raw_value:
        return repo_row_caps
    for item in raw_value.split(","):
        if "=" not in item:
            continue
        repo, raw_cap = item.split("=", 1)
        repo = repo.strip()
        raw_cap = raw_cap.strip()
        if not repo or not raw_cap:
            continue
        repo_row_caps[repo] = int(raw_cap)
    return repo_row_caps


def _repo_slug(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", repo)


def _timestamp_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _checkpoint_root(config: Config, args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return config.processed_dir / f"{args.results_subdir}__checkpoints"


def _repo_checkpoint_paths(checkpoint_root: Path, repo: str) -> dict[str, Path]:
    repo_dir = checkpoint_root / "repos" / _repo_slug(repo)
    paths = {"repo_dir": repo_dir, "repo_summary": repo_dir / "repo_summary.json"}
    for key, filename in CHECKPOINT_FRAME_FILES.items():
        paths[key] = repo_dir / filename
    return paths


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty and not len(normalized.columns):
        normalized = pd.DataFrame({"_empty": []})
    return normalized


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalise_frame(frame)
    if path.suffix.lower() == ".parquet":
        normalized.to_parquet(path, index=False)
        return
    normalized.to_csv(path, index=False)


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "_empty" in frame.columns:
        frame = frame.drop(columns=["_empty"])
    return frame


def _repo_checkpoint_complete(checkpoint_root: Path, repo: str) -> bool:
    paths = _repo_checkpoint_paths(checkpoint_root, repo)
    return all(paths[key].exists() for key in CHECKPOINT_REQUIRED_KEYS)


def _load_manifest(checkpoint_root: Path, *, results_subdir: str, selected_repos: list[str]) -> dict[str, object]:
    manifest_path = checkpoint_root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    else:
        manifest = {}
    manifest["results_subdir"] = results_subdir
    manifest["selected_repos"] = selected_repos
    manifest.setdefault("completed_repos", [])
    manifest.setdefault("repo_status", {})
    manifest["last_updated"] = _timestamp_now()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def _update_manifest(
    checkpoint_root: Path,
    *,
    manifest: dict[str, object],
    repo: str,
    repo_summary: dict[str, object],
) -> None:
    completed_repos = set(manifest.get("completed_repos", []))
    completed_repos.add(repo)
    manifest["completed_repos"] = sorted(completed_repos)
    repo_status = manifest.get("repo_status", {})
    repo_status[repo] = repo_summary
    manifest["repo_status"] = repo_status
    manifest["last_updated"] = _timestamp_now()
    with open(checkpoint_root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _update_repo_summary_table(checkpoint_root: Path, repo_summary: dict[str, object]) -> None:
    summary_path = checkpoint_root / "repo_summary_rows.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(columns=REPO_SUMMARY_COLUMNS)
    incoming = pd.DataFrame([repo_summary], columns=REPO_SUMMARY_COLUMNS)
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=["repo"], keep="last")
    merged = merged.sort_values("repo").reset_index(drop=True)
    merged.to_csv(summary_path, index=False)


def _merge_repo_audit_into_curated_file(
    audit_path: Path,
    repo_audit: pd.DataFrame,
) -> pd.DataFrame:
    existing = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    merged = merge_intervention_audit_frames(existing, repo_audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(audit_path, index=False)
    return merged


def _load_or_seed_repo_intervention_audit(
    *,
    repo: str,
    repo_paths: dict[str, Path],
    corpus: list,
    audit_path: Path,
    since: str,
    until: str,
    max_rows_per_repo: int,
    repo_row_caps: dict[str, int],
    min_loc: int,
    seed_audit: bool,
) -> pd.DataFrame:
    existing = load_intervention_audit_table(audit_path, accepted_only=False)
    repo_existing = existing.loc[existing["repo"] == repo].copy()
    if seed_audit or repo_existing.empty:
        repo_seed = seed_refactoring_intervention_audit_table(
            repo_paths,
            corpus,
            repos=[repo],
            since=since,
            until=until,
            max_rows_per_repo=max_rows_per_repo,
            repo_row_caps=repo_row_caps,
            min_loc=min_loc,
        )
        merged = _merge_repo_audit_into_curated_file(audit_path, repo_seed)
        repo_existing = merged.loc[merged["repo"] == repo].copy()
    repo_existing = repo_existing.loc[repo_existing["review_decision"] == "accept"].copy()
    return repo_existing.reset_index(drop=True)


def _run_repo_checkpoint(
    *,
    repo: str,
    config: Config,
    repo_paths: dict[str, Path],
    corpus: list,
    intervention_audit_path: Path,
    security_lookup: dict[str, dict[str, list[dict[str, object]]]],
    fitted_models: dict[str, dict[str, object]],
    since: str,
    until: str,
    horizon_days: int,
    min_loc: int,
    min_structure_drops: int,
    max_rows_per_repo: int,
    repo_row_caps: dict[str, int],
    max_control_commits_per_repo: int,
    seed_audit: bool,
    checkpoint_root: Path,
    manifest: dict[str, object],
) -> None:
    repo_audit = _load_or_seed_repo_intervention_audit(
        repo=repo,
        repo_paths=repo_paths,
        corpus=corpus,
        audit_path=intervention_audit_path,
        since=since,
        until=until,
        max_rows_per_repo=max_rows_per_repo,
        repo_row_caps=repo_row_caps,
        min_loc=min_loc,
        seed_audit=seed_audit,
    )
    logger.info("%s: accepted intervention audit rows=%d", repo, len(repo_audit))
    interventions = build_audited_intervention_pool(
        repo_paths,
        corpus,
        repo_audit,
        min_loc=min_loc,
    )
    interventions = filter_material_interventions(
        interventions,
        min_structure_drops=min_structure_drops,
    )
    logger.info("%s: materialised interventions=%d", repo, len(interventions))
    maintenance_pool = build_maintenance_pool(
        repo_paths,
        corpus,
        repos=[repo],
        since=since,
        until=until,
        min_loc=min_loc,
        max_commits_per_repo=max_control_commits_per_repo,
    )
    logger.info("%s: maintenance pool rows=%d", repo, len(maintenance_pool))
    matched_pairs = build_matched_intervention_pairs(interventions, maintenance_pool)
    logger.info("%s: matched intervention pairs=%d", repo, len(matched_pairs))
    anchor_rows = build_anchor_observations(
        matched_pairs,
        interventions,
        maintenance_pool,
        security_lookup=security_lookup,
        horizon_days=horizon_days,
    )
    scored_anchor_rows = score_anchor_observations(anchor_rows, fitted_models)
    pair_did = summarise_pairs_to_did(scored_anchor_rows)

    checkpoint_paths = _repo_checkpoint_paths(checkpoint_root, repo)
    _write_frame(repo_audit, checkpoint_paths["intervention_audit"])
    _write_frame(interventions, checkpoint_paths["interventions"])
    _write_frame(maintenance_pool, checkpoint_paths["maintenance_pool"])
    _write_frame(matched_pairs, checkpoint_paths["matched_pairs"])
    _write_frame(anchor_rows, checkpoint_paths["anchor_rows"])
    _write_frame(scored_anchor_rows, checkpoint_paths["scored_anchor_rows"])
    _write_frame(pair_did, checkpoint_paths["pair_did"])

    repo_summary = {
        "repo": repo,
        "accepted_intervention_rows": int(len(repo_audit)),
        "n_interventions_materialised": int(len(interventions)),
        "n_maintenance_rows": int(len(maintenance_pool)),
        "n_matched_pairs": int(len(matched_pairs)),
        "n_anchor_rows": int(len(anchor_rows)),
        "n_scored_anchor_rows": int(len(scored_anchor_rows)),
        "n_pair_did_rows": int(len(pair_did)),
        "completed_at": _timestamp_now(),
    }
    with open(checkpoint_paths["repo_summary"], "w") as handle:
        json.dump(repo_summary, handle, indent=2, sort_keys=True)
    _update_repo_summary_table(checkpoint_root, repo_summary)
    _update_manifest(checkpoint_root, manifest=manifest, repo=repo, repo_summary=repo_summary)
    logger.info("%s: checkpoint artifacts written to %s", repo, checkpoint_paths["repo_dir"])


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame for frame in frames if frame is not None and not frame.empty]
    if not materialized:
        return pd.DataFrame()
    return pd.concat(materialized, ignore_index=True)


def _collect_checkpoint_frames(checkpoint_root: Path, repos: list[str]) -> dict[str, pd.DataFrame]:
    missing = [repo for repo in repos if not _repo_checkpoint_complete(checkpoint_root, repo)]
    if missing:
        raise SystemExit(f"Missing repo checkpoints for finalize: {', '.join(missing)}")
    collected: dict[str, list[pd.DataFrame]] = {key: [] for key in CHECKPOINT_REQUIRED_KEYS}
    for repo in repos:
        checkpoint_paths = _repo_checkpoint_paths(checkpoint_root, repo)
        for key in CHECKPOINT_REQUIRED_KEYS:
            collected[key].append(_read_frame(checkpoint_paths[key]))
    return {key: _concat_frames(frames) for key, frames in collected.items()}


def _finalize_from_checkpoints(
    *,
    config: Config,
    args: argparse.Namespace,
    checkpoint_root: Path,
    selected_repos: list[str],
    security_audit: pd.DataFrame,
    until: str,
    repo_row_caps: dict[str, int],
) -> None:
    collected = _collect_checkpoint_frames(checkpoint_root, selected_repos)
    interventions = filter_material_interventions(
        collected["interventions"],
        min_structure_drops=args.min_structure_drops,
    )
    maintenance_pool = collected["maintenance_pool"]
    matched_pairs = collected["matched_pairs"]
    anchor_rows = collected["anchor_rows"]
    scored_anchor_rows = collected["scored_anchor_rows"]
    pair_did = collected["pair_did"]
    intervention_audit = merge_intervention_audit_frames(
        pd.DataFrame(columns=[]),
        collected["intervention_audit"],
    )
    keep_ids = set(interventions["intervention_id"].astype(str).tolist()) if not interventions.empty else set()
    if keep_ids:
        matched_pairs = matched_pairs.loc[
            matched_pairs["intervention_id"].astype(str).isin(keep_ids)
        ].copy()
        anchor_rows = anchor_rows.loc[
            anchor_rows["intervention_id"].astype(str).isin(keep_ids)
        ].copy()
        scored_anchor_rows = scored_anchor_rows.loc[
            scored_anchor_rows["intervention_id"].astype(str).isin(keep_ids)
        ].copy()
        pair_did = pair_did.loc[pair_did["intervention_id"].astype(str).isin(keep_ids)].copy()
        audit_keys = interventions[["repo", "intervention_commit", "file_path"]].drop_duplicates()
        intervention_audit = intervention_audit.merge(
            audit_keys,
            on=["repo", "intervention_commit", "file_path"],
            how="inner",
        )
    else:
        matched_pairs = matched_pairs.iloc[0:0].copy()
        anchor_rows = anchor_rows.iloc[0:0].copy()
        scored_anchor_rows = scored_anchor_rows.iloc[0:0].copy()
        pair_did = pair_did.iloc[0:0].copy()
        intervention_audit = intervention_audit.iloc[0:0].copy()

    if not anchor_rows.empty:
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
    checkpoint_repo_summary_path = checkpoint_root / "repo_summary_rows.csv"
    checkpoint_generation_note = None
    parameter_control_cap: int | None = args.max_control_commits_per_repo
    if checkpoint_repo_summary_path.exists():
        checkpoint_generation_note = (
            "Repo checkpoints may have been generated incrementally with repo-specific "
            "control caps; use checkpoint_repo_summary_rows.csv as the realised run ledger."
        )
        parameter_control_cap = None
    summary["parameters"] = {
        "config": args.config,
        "train_dataset_path": args.train_dataset_path,
        "security_audit_path": args.security_audit_path,
        "intervention_audit_path": args.intervention_audit_path,
        "checkpoint_dir": str(checkpoint_root),
        "repos": selected_repos,
        "since": args.since,
        "until": until,
        "horizon_days": args.horizon_days,
        "min_loc": args.min_loc,
        "max_rows_per_repo": args.max_rows_per_repo,
        "repo_row_caps": repo_row_caps,
        "min_structure_drops": args.min_structure_drops,
        "max_control_commits_per_repo": parameter_control_cap,
        "checkpoint_generation_note": checkpoint_generation_note,
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
    repo_summary_rows_path = checkpoint_repo_summary_path
    if repo_summary_rows_path.exists():
        checkpoint_repo_summary = pd.read_csv(repo_summary_rows_path)
        checkpoint_repo_summary.to_csv(results_dir / "checkpoint_repo_summary_rows.csv", index=False)
    manifest_path = checkpoint_root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        with open(results_dir / "checkpoint_manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Wrote final pooled e23 results to %s", results_dir)


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
        "--min-structure-drops",
        type=int,
        default=0,
        help="Require an intervention to reduce at least this many structural dimensions",
    )
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
        help="Refresh repo-scoped audit rows before the run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip repos with completed checkpoints and continue from unfinished repos",
    )
    parser.add_argument(
        "--skip-existing-repo-checkpoints",
        action="store_true",
        help="Skip repos that already have complete checkpoint artifacts",
    )
    parser.add_argument(
        "--finalize-from-checkpoints",
        action="store_true",
        help="Build the final pooled results directory from completed repo checkpoints",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional checkpoint directory; defaults under data/processed",
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
    repo_row_caps = _parse_repo_row_caps(args.repo_row_caps)
    until = args.until or _default_until_date(args.horizon_days)
    checkpoint_root = _checkpoint_root(config, args)
    manifest = _load_manifest(
        checkpoint_root,
        results_subdir=args.results_subdir,
        selected_repos=selected_repos,
    )

    intervention_audit_path = Path(args.intervention_audit_path)
    intervention_audit_path.parent.mkdir(parents=True, exist_ok=True)
    security_audit = load_audited_security_table(Path(args.security_audit_path))
    security_audit = security_audit.loc[security_audit["repo"].isin(selected_repos)].copy()

    if args.finalize_from_checkpoints:
        _finalize_from_checkpoints(
            config=config,
            args=args,
            checkpoint_root=checkpoint_root,
            selected_repos=selected_repos,
            security_audit=security_audit,
            until=until,
            repo_row_caps=repo_row_caps,
        )
        return

    train_dataset_path = Path(args.train_dataset_path)
    train_dataset = (
        pd.read_csv(train_dataset_path)
        if train_dataset_path.suffix.lower() == ".csv"
        else pd.read_parquet(train_dataset_path)
    )
    fitted_models = fit_frozen_score_models(train_dataset)
    security_lookup = build_security_label_lookup(security_audit, repo_paths)

    for repo in selected_repos:
        if (args.resume or args.skip_existing_repo_checkpoints) and _repo_checkpoint_complete(
            checkpoint_root, repo
        ):
            logger.info("Skipping %s because its checkpoint artifacts are already complete", repo)
            continue
        logger.info("Materialising repo checkpoint for %s", repo)
        _run_repo_checkpoint(
            repo=repo,
            config=config,
            repo_paths=repo_paths,
            corpus=corpus,
            intervention_audit_path=intervention_audit_path,
            security_lookup=security_lookup,
            fitted_models=fitted_models,
            since=args.since,
            until=until,
            horizon_days=args.horizon_days,
            min_loc=args.min_loc,
            min_structure_drops=args.min_structure_drops,
            max_rows_per_repo=args.max_rows_per_repo,
            repo_row_caps=repo_row_caps,
            max_control_commits_per_repo=args.max_control_commits_per_repo,
            seed_audit=args.seed_audit,
            checkpoint_root=checkpoint_root,
            manifest=manifest,
        )

    _finalize_from_checkpoints(
        config=config,
        args=args,
        checkpoint_root=checkpoint_root,
        selected_repos=selected_repos,
        security_audit=security_audit,
        until=until,
        repo_row_caps=repo_row_caps,
    )


if __name__ == "__main__":
    main()
