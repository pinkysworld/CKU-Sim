"""Experiment 21: audited security-vs-bugfix negative control."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cku_sim.analysis.audited_negative_control import (
    build_bugfix_pool_from_e15_and_audit,
    build_conditional_logit_dataset,
    build_security_pool_from_e15,
    fit_conditional_logit_models,
    match_audited_security_to_bugfix,
    summarise_audited_negative_control,
)
from cku_sim.analysis.audited_panel import (
    load_bugfix_control_audit_table,
    seed_bugfix_control_audit_table,
    summarise_audit_table,
)
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _summarise_pool(frame: pd.DataFrame, *, id_col: str, file_col: str) -> dict[str, object]:
    if frame.empty:
        return {"n_rows": 0, "n_repos": 0, "n_ids": 0, "n_files": 0}
    return {
        "n_rows": int(len(frame)),
        "n_repos": int(frame["repo"].nunique()),
        "n_ids": int(frame[id_col].nunique()),
        "n_files": int(frame[file_col].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 21: audited security-vs-bugfix negative control"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--e15-subdir",
        type=str,
        default="e15_negative_control_strict__expanded_advisory__light6",
    )
    parser.add_argument(
        "--e17-subdir",
        type=str,
        default="e17_bugfix_control_audit__e15_light6",
    )
    parser.add_argument(
        "--bugfix-audit-path",
        type=str,
        default="data/processed/ordinary_bugfix_control_audit_curated.csv",
    )
    parser.add_argument(
        "--seed-audit",
        action="store_true",
        help="Refresh the curated ordinary bug-fix audit table from the screened controls",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e21_security_negative_control__audited",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    e15_pairs_path = config.results_dir / args.e15_subdir / "security_vs_bugfix_pairs.parquet"
    screened_controls_path = config.results_dir / args.e17_subdir / "screened_controls.csv"
    if not e15_pairs_path.exists():
        logger.error("Strict negative-control pairs not found: %s", e15_pairs_path)
        return
    if not screened_controls_path.exists():
        logger.error("Screened bug-fix controls not found: %s", screened_controls_path)
        return

    bugfix_audit_path = Path(args.bugfix_audit_path)
    bugfix_audit_path.parent.mkdir(parents=True, exist_ok=True)
    screened_controls = pd.read_csv(screened_controls_path)
    if args.seed_audit or not bugfix_audit_path.exists():
        bugfix_audit = seed_bugfix_control_audit_table(screened_controls)
        bugfix_audit.to_csv(bugfix_audit_path, index=False)
        logger.info(
            "Seeded ordinary bug-fix audit table with %d rows at %s",
            len(bugfix_audit),
            bugfix_audit_path,
        )

    bugfix_audit = load_bugfix_control_audit_table(bugfix_audit_path)
    e15_pairs = pd.read_parquet(e15_pairs_path)
    security_pool = build_security_pool_from_e15(e15_pairs, supported_only=True)
    bugfix_pool = build_bugfix_pool_from_e15_and_audit(e15_pairs, bugfix_audit)
    matched_pairs = match_audited_security_to_bugfix(security_pool, bugfix_pool)
    conditional_logit = build_conditional_logit_dataset(matched_pairs)
    logit_summary = fit_conditional_logit_models(conditional_logit)

    accepted_screen = screened_controls.loc[screened_controls["review_decision"] == "accept"].copy()
    summary = {
        "security_pool": _summarise_pool(
            security_pool,
            id_col="security_event_id",
            file_col="security_file",
        ),
        "bugfix_audit": summarise_audit_table(bugfix_audit, id_col="fixed_commit"),
        "bugfix_pool": _summarise_pool(
            bugfix_pool.rename(
                columns={"bugfix_commit": "fixed_commit", "bugfix_file": "file_path"}
            ),
            id_col="fixed_commit",
            file_col="file_path",
        ),
        "matched_pairs": summarise_audited_negative_control(matched_pairs),
        "conditional_logit": logit_summary,
        "gating": {
            "accepted_bugfix_controls_ge_100": bool(len(accepted_screen) >= 100),
            "accepted_bugfix_controls_have_no_security_ids": bool(
                not accepted_screen["has_security_ids"].astype(bool).any()
            ),
            "accepted_bugfix_controls_have_no_security_keyword_signal": bool(
                not accepted_screen["has_security_keyword_signal"].astype(bool).any()
            ),
            "accepted_bugfix_controls_have_no_security_adjacent_signal": bool(
                not accepted_screen["has_security_adjacent_signal"].astype(bool).any()
            ),
        },
        "parameters": {
            "e15_subdir": args.e15_subdir,
            "e17_subdir": args.e17_subdir,
            "bugfix_audit_path": str(bugfix_audit_path),
        },
    }

    repo_summary = (
        matched_pairs.groupby("repo", as_index=False)
        .agg(
            n_pairs=("pair_id", "size"),
            n_security_events=("security_event_id", "nunique"),
            median_delta_composite=("delta_composite", "median"),
            mean_delta_composite=("delta_composite", "mean"),
        )
        .sort_values(["n_pairs", "median_delta_composite"], ascending=[False, False])
        if not matched_pairs.empty
        else pd.DataFrame()
    )

    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    bugfix_audit.to_csv(results_dir / "accepted_bugfix_audit.csv", index=False)
    screened_controls.to_csv(results_dir / "screened_bugfix_controls.csv", index=False)
    security_pool.to_csv(results_dir / "security_pool.csv", index=False)
    bugfix_pool.to_csv(results_dir / "bugfix_pool.csv", index=False)
    matched_pairs.to_parquet(results_dir / "matched_pairs.parquet")
    matched_pairs.to_csv(results_dir / "matched_pairs.csv", index=False)
    conditional_logit.to_csv(results_dir / "conditional_logit_dataset.csv", index=False)
    repo_summary.to_csv(results_dir / "repo_summary.csv", index=False)
    with open(results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Audited negative control: %d matched pairs across %d repos",
        summary["matched_pairs"].get("n_pairs", 0),
        summary["matched_pairs"].get("n_repos", 0),
    )
    if logit_summary:
        logger.info(
            "Conditional logit composite p-value: %.4f",
            logit_summary["baseline_plus_composite"]["composite_pvalue"],
        )


if __name__ == "__main__":
    main()
