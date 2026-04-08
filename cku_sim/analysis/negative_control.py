"""Negative-control comparison between security fixes and ordinary bug fixes."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy import stats

from cku_sim.analysis.bootstrap import clustered_delta_bootstrap
from cku_sim.analysis.file_level_case_control import (
    _get_commit_epoch,
    _get_first_parent,
    _get_metrics_for_snapshot_file,
    _list_changed_source_files,
    _list_source_files_at_commit,
    _run_git,
    extract_security_ids,
)
from cku_sim.analysis.predictive_validation import evaluate_leave_one_repo_out
from cku_sim.core.config import CorpusEntry

logger = logging.getLogger(__name__)

BUGFIX_GREP_PATTERNS = [
    r"^fix",
    r"^fixed",
    r"^bugfix",
    r"^correct",
    r"^resolve",
    r"^prevent",
    r"^avoid",
    r"crash",
    r"regression",
    r"leak",
    r"hang",
    r"deadlock",
    r"panic",
    r"assert",
    r"null",
]
BUGFIX_SUBJECT_RE = re.compile(
    r"^(?:fix(?:es|ed)?|bugfix|correct(?:s|ed)?|resolve[sd]?|prevent(?:s|ed)?|avoid(?:s|ed)?)\b"
    r"|\b(?:bug|crash|regression|leak|hang|deadlock|panic|assert(?:ion)?|null)\b",
    re.IGNORECASE,
)
MERGE_OR_REVERT_RE = re.compile(r"^(?:merge|revert)\b", re.IGNORECASE)
SECURITY_EXCLUDE_RE = re.compile(
    r"\b(?:security|vulnerability|exploit|overflow|use-after-free|uaf|oob|"
    r"out[- ]of[- ]bounds|xss|rce|cwe-\d+|dos|denial[- ]of[- ]service)\b",
    re.IGNORECASE,
)
META_REFERENCE_EXCLUDE_RE = re.compile(
    r"\b(?:ai|gpt|llm|hallucinat\w*|machine-generated|model-generated)\b",
    re.IGNORECASE,
)


def build_security_file_dataset(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Project experiment-6 pairs onto the security-fix file side only."""
    rows: list[dict[str, object]] = []
    for _, row in pairs_df.iterrows():
        security_file = str(row["case_file"])
        rows.append(
            {
                "repo": row["repo"],
                "security_commit": row["commit"],
                "security_parent": row.get("parent"),
                "security_event_id": row.get("event_id", row.get("cve_ids", row["commit"])),
                "security_ground_truth_policy": row.get(
                    "ground_truth_policy",
                    "nvd_commit_refs",
                ),
                "security_ground_truth_source": row.get("ground_truth_source", "nvd_ref"),
                "security_ids": row.get("cve_ids", ""),
                "security_file": security_file,
                "security_suffix": Path(security_file).suffix.lower(),
                "security_loc": float(row["case_loc"]),
                "security_size_bytes": float(row["case_size_bytes"]),
                "security_ci_gzip": float(row["case_ci_gzip"]),
                "security_entropy": float(row["case_entropy"]),
                "security_cc_density": float(row["case_cc_density"]),
                "security_halstead": float(row["case_halstead"]),
                "security_composite": float(row["case_composite"]),
            }
        )

    return pd.DataFrame(rows)


def collect_ordinary_bugfix_candidates(
    repo_path: Path,
    entry: CorpusEntry,
    *,
    excluded_commits: set[str],
    max_changed_source_files: int = 8,
    max_bugfix_commits: int = 800,
) -> pd.DataFrame:
    """Collect conservative non-security bug-fix file candidates from local history."""
    cmd = ["log", "--all", "--no-merges", "--regexp-ignore-case"]
    for pattern in BUGFIX_GREP_PATTERNS:
        cmd.extend(["--grep", pattern])
    cmd.extend(["--format=%H%x1f%s%x1f%B%x1e"])
    proc = _run_git(repo_path, cmd)
    if proc.returncode != 0:
        return pd.DataFrame()

    snapshot_file_cache: dict[str, dict[str, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    accepted_commits = 0

    for block in proc.stdout.split("\x1e"):
        block = block.strip()
        if not block:
            continue
        parts = block.split("\x1f", 2)
        if len(parts) != 3:
            continue
        commit, subject, body = parts
        subject = subject.strip()
        body = body.strip()
        full_message = f"{subject}\n{body}".strip()

        if commit in excluded_commits:
            continue
        if MERGE_OR_REVERT_RE.match(subject):
            continue
        if not BUGFIX_SUBJECT_RE.search(subject):
            continue
        if extract_security_ids(full_message):
            continue
        if SECURITY_EXCLUDE_RE.search(full_message):
            continue
        if META_REFERENCE_EXCLUDE_RE.search(full_message):
            continue

        parent = _get_first_parent(repo_path, commit)
        if parent is None:
            continue

        changed_files = _list_changed_source_files(repo_path, commit, entry.source_extensions)
        if not changed_files or len(changed_files) > max_changed_source_files:
            continue

        if parent not in snapshot_file_cache:
            snapshot_file_cache[parent] = _list_source_files_at_commit(
                repo_path,
                parent,
                entry.source_extensions,
            )
        files_at_parent = snapshot_file_cache[parent]
        if not files_at_parent:
            continue

        epoch = _get_commit_epoch(repo_path, commit)
        added_any = False
        for path_str in changed_files:
            file_info = files_at_parent.get(path_str)
            if file_info is None:
                continue
            rows.append(
                {
                    "repo": entry.name,
                    "bugfix_commit": commit,
                    "bugfix_parent": parent,
                    "bugfix_subject": subject,
                    "bugfix_file": path_str,
                    "bugfix_suffix": str(file_info["suffix"]),
                    "bugfix_size_bytes": int(file_info["size"]),
                    "bugfix_epoch": epoch,
                }
            )
            added_any = True

        if added_any:
            accepted_commits += 1
            if accepted_commits >= max_bugfix_commits:
                break

    return pd.DataFrame(rows)


def _match_one_bugfix_candidate(
    security_row: dict[str, object],
    bugfix_pool: list[dict[str, object]],
    repo_path: Path,
    metrics_cache: dict[tuple[str, str], object | None],
    used_bugfix_keys: set[tuple[str, str]],
    *,
    min_loc: int,
) -> tuple[dict[str, object], object] | None:
    security_suffix = str(security_row["security_suffix"])
    security_size = float(security_row["security_size_bytes"])
    security_epoch = int(security_row["security_epoch"])

    same_suffix = [
        item
        for item in bugfix_pool
        if item["bugfix_suffix"] == security_suffix
        and (item["bugfix_commit"], item["bugfix_file"]) not in used_bugfix_keys
    ]
    fallback = [
        item
        for item in bugfix_pool
        if (item["bugfix_commit"], item["bugfix_file"]) not in used_bugfix_keys
    ]

    for pool in (same_suffix, fallback):
        if not pool:
            continue
        ranked = sorted(
            pool,
            key=lambda item: (
                abs(math.log1p(float(item["bugfix_size_bytes"])) - math.log1p(security_size)),
                abs(int(item["bugfix_epoch"]) - security_epoch),
                str(item["bugfix_file"]),
            ),
        )
        for candidate in ranked:
            metrics = _get_metrics_for_snapshot_file(
                repo_path,
                str(candidate["bugfix_parent"]),
                str(candidate["bugfix_file"]),
                metrics_cache,
            )
            if metrics is None or metrics.total_loc < min_loc:
                continue
            return candidate, metrics

    return None


def match_security_to_bugfix_pairs(
    security_df: pd.DataFrame,
    bugfix_candidate_df: pd.DataFrame,
    repo_paths: dict[str, Path],
    *,
    min_loc: int = 20,
) -> pd.DataFrame:
    """Match security-fix files to conservative ordinary bug-fix files."""
    if security_df.empty or bugfix_candidate_df.empty:
        return pd.DataFrame()

    commit_epoch_cache: dict[tuple[str, str], int] = {}
    repo_metrics_caches: dict[str, dict[tuple[str, str], object | None]] = defaultdict(dict)
    rows: list[dict[str, object]] = []

    security_df = security_df.copy()
    epochs: list[int] = []
    for _, row in security_df.iterrows():
        repo = str(row["repo"])
        commit = str(row["security_commit"])
        key = (repo, commit)
        if key not in commit_epoch_cache:
            commit_epoch_cache[key] = _get_commit_epoch(repo_paths[repo], commit)
        epochs.append(commit_epoch_cache[key])
    security_df["security_epoch"] = epochs

    for repo, repo_security in security_df.groupby("repo"):
        repo_path = repo_paths[str(repo)]
        repo_bugfix = bugfix_candidate_df.loc[bugfix_candidate_df["repo"] == repo].copy()
        if repo_bugfix.empty:
            continue

        bugfix_pool = repo_bugfix.to_dict(orient="records")
        metrics_cache = repo_metrics_caches[str(repo)]
        used_bugfix_keys: set[tuple[str, str]] = set()

        repo_security = repo_security.sort_values(
            by=["security_epoch", "security_size_bytes", "security_file"],
            ascending=[True, True, True],
        )

        for _, security_row in repo_security.iterrows():
            security_dict = security_row.to_dict()
            match = _match_one_bugfix_candidate(
                security_dict,
                bugfix_pool,
                repo_path,
                metrics_cache,
                used_bugfix_keys,
                min_loc=min_loc,
            )
            if match is None:
                continue

            bugfix_row, bugfix_metrics = match
            bugfix_key = (str(bugfix_row["bugfix_commit"]), str(bugfix_row["bugfix_file"]))
            used_bugfix_keys.add(bugfix_key)

            rows.append(
                {
                    "repo": repo,
                    "security_commit": security_dict["security_commit"],
                    "security_parent": security_dict["security_parent"],
                    "security_event_id": security_dict["security_event_id"],
                    "security_ids": security_dict["security_ids"],
                    "security_ground_truth_policy": security_dict["security_ground_truth_policy"],
                    "security_ground_truth_source": security_dict["security_ground_truth_source"],
                    "security_epoch": security_dict["security_epoch"],
                    "security_file": security_dict["security_file"],
                    "security_suffix": security_dict["security_suffix"],
                    "security_size_bytes": security_dict["security_size_bytes"],
                    "security_loc": security_dict["security_loc"],
                    "security_composite": security_dict["security_composite"],
                    "security_ci_gzip": security_dict["security_ci_gzip"],
                    "security_entropy": security_dict["security_entropy"],
                    "security_cc_density": security_dict["security_cc_density"],
                    "security_halstead": security_dict["security_halstead"],
                    "bugfix_commit": bugfix_row["bugfix_commit"],
                    "bugfix_parent": bugfix_row["bugfix_parent"],
                    "bugfix_subject": bugfix_row["bugfix_subject"],
                    "bugfix_epoch": bugfix_row["bugfix_epoch"],
                    "bugfix_file": bugfix_row["bugfix_file"],
                    "bugfix_suffix": bugfix_row["bugfix_suffix"],
                    "bugfix_size_bytes": bugfix_metrics.total_bytes,
                    "bugfix_loc": bugfix_metrics.total_loc,
                    "bugfix_composite": bugfix_metrics.composite_score,
                    "bugfix_ci_gzip": bugfix_metrics.ci_gzip,
                    "bugfix_entropy": bugfix_metrics.shannon_entropy,
                    "bugfix_cc_density": bugfix_metrics.cyclomatic_density,
                    "bugfix_halstead": bugfix_metrics.halstead_volume,
                    "loc_ratio": (
                        float(security_dict["security_loc"]) / bugfix_metrics.total_loc
                        if bugfix_metrics.total_loc
                        else math.nan
                    ),
                    "delta_composite": (
                        float(security_dict["security_composite"]) - bugfix_metrics.composite_score
                    ),
                    "delta_ci_gzip": (
                        float(security_dict["security_ci_gzip"]) - bugfix_metrics.ci_gzip
                    ),
                    "delta_entropy": (
                        float(security_dict["security_entropy"]) - bugfix_metrics.shannon_entropy
                    ),
                    "delta_cc_density": (
                        float(security_dict["security_cc_density"])
                        - bugfix_metrics.cyclomatic_density
                    ),
                    "delta_halstead": (
                        float(security_dict["security_halstead"]) - bugfix_metrics.halstead_volume
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarise_negative_control_pairs(pairs: pd.DataFrame) -> dict[str, object]:
    """Summarise security-minus-bugfix matched pairs."""
    if pairs.empty:
        return {
            "n_pairs": 0,
            "n_security_commits": 0,
            "n_bugfix_commits": 0,
            "n_repos": 0,
        }

    delta = pairs["delta_composite"].dropna()
    nonzero = delta[delta != 0]
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())

    summary: dict[str, object] = {
        "n_pairs": int(len(pairs)),
        "n_security_commits": int(pairs["security_commit"].nunique()),
        "n_bugfix_commits": int(pairs["bugfix_commit"].nunique()),
        "n_repos": int(pairs["repo"].nunique()),
        "n_security_events": int(pairs["security_event_id"].nunique()),
        "mean_delta_composite": float(delta.mean()),
        "median_delta_composite": float(delta.median()),
        "positive_share": float((delta > 0).mean()),
        "mean_security_loc": float(pairs["security_loc"].mean()),
        "mean_bugfix_loc": float(pairs["bugfix_loc"].mean()),
        "median_loc_ratio": float(pairs["loc_ratio"].median()),
    }
    if "security_ground_truth_policy" in pairs.columns:
        summary["security_ground_truth_policy"] = str(
            pairs["security_ground_truth_policy"].iloc[0]
        )

    if len(nonzero) >= 1:
        sign_test = stats.binomtest(positive, positive + negative, p=0.5, alternative="greater")
        summary["sign_test_positive_count"] = positive
        summary["sign_test_negative_count"] = negative
        summary["sign_test_pvalue_greater"] = float(sign_test.pvalue)
        summary["rank_biserial_effect"] = float((positive - negative) / (positive + negative))

    if len(delta) >= 3:
        try:
            wilcoxon = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
            summary["wilcoxon_statistic"] = float(wilcoxon.statistic)
            summary["wilcoxon_pvalue_greater"] = float(wilcoxon.pvalue)
        except ValueError:
            pass

    summary["bootstrap_primary_cluster"] = clustered_delta_bootstrap(
        pairs,
        cluster_col="security_event_id",
    )
    summary["bootstrap_repo_cluster"] = clustered_delta_bootstrap(
        pairs,
        cluster_col="repo",
    )

    return summary


def summarise_security_event_level_deltas(event_summary: pd.DataFrame) -> dict[str, object]:
    """Summarise security-event mean deltas against ordinary bug-fix controls."""
    if event_summary.empty:
        return {
            "n_security_events": 0,
            "n_security_commits": 0,
            "n_bugfix_commits": 0,
            "n_repos": 0,
        }

    delta = event_summary["mean_delta_composite"].dropna()
    nonzero = delta[delta != 0]
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())

    summary: dict[str, object] = {
        "n_security_events": int(len(event_summary)),
        "n_security_commits": int(event_summary["security_commit"].nunique()),
        "n_bugfix_commits": int(event_summary["bugfix_commit"].nunique()),
        "n_repos": int(event_summary["repo"].nunique()),
        "mean_delta_composite": float(delta.mean()),
        "median_delta_composite": float(delta.median()),
        "positive_share": float((delta > 0).mean()),
    }
    if "security_ground_truth_policy" in event_summary.columns:
        summary["security_ground_truth_policy"] = str(
            event_summary["security_ground_truth_policy"].iloc[0]
        )

    if len(nonzero) >= 1:
        sign_test = stats.binomtest(positive, positive + negative, p=0.5, alternative="greater")
        summary["sign_test_positive_count"] = positive
        summary["sign_test_negative_count"] = negative
        summary["sign_test_pvalue_greater"] = float(sign_test.pvalue)
        summary["rank_biserial_effect"] = float((positive - negative) / (positive + negative))

    if len(delta) >= 3:
        try:
            wilcoxon = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
            summary["wilcoxon_statistic"] = float(wilcoxon.statistic)
            summary["wilcoxon_pvalue_greater"] = float(wilcoxon.pvalue)
        except ValueError:
            pass

    summary["bootstrap_primary_cluster"] = clustered_delta_bootstrap(
        event_summary,
        cluster_col="security_event_id",
        delta_col="mean_delta_composite",
    )
    summary["bootstrap_repo_cluster"] = clustered_delta_bootstrap(
        event_summary,
        cluster_col="repo",
        delta_col="mean_delta_composite",
    )

    return summary


def build_negative_control_prediction_dataset(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Convert matched security-vs-bugfix pairs into a file-level classification set."""
    rows: list[dict[str, object]] = []
    for pair_id, row in pairs_df.reset_index(drop=True).iterrows():
        common = {
            "pair_id": int(pair_id),
            "repo": row["repo"],
            "security_event_id": row["security_event_id"],
            "security_ground_truth_policy": row["security_ground_truth_policy"],
        }

        rows.append(
            {
                **common,
                "commit": row["security_commit"],
                "label": 1,
                "kind": "security",
                "file_path": row["security_file"],
                "suffix": row["security_suffix"] or "<none>",
                "loc": float(row["security_loc"]),
                "size_bytes": float(row["security_size_bytes"]),
                "ci_gzip": float(row["security_ci_gzip"]),
                "shannon_entropy": float(row["security_entropy"]),
                "cyclomatic_density": float(row["security_cc_density"]),
                "halstead_volume": float(row["security_halstead"]),
                "composite_score": float(row["security_composite"]),
            }
        )
        rows.append(
            {
                **common,
                "commit": row["bugfix_commit"],
                "label": 0,
                "kind": "bugfix",
                "file_path": row["bugfix_file"],
                "suffix": row["bugfix_suffix"] or "<none>",
                "loc": float(row["bugfix_loc"]),
                "size_bytes": float(row["bugfix_size_bytes"]),
                "ci_gzip": float(row["bugfix_ci_gzip"]),
                "shannon_entropy": float(row["bugfix_entropy"]),
                "cyclomatic_density": float(row["bugfix_cc_density"]),
                "halstead_volume": float(row["bugfix_halstead"]),
                "composite_score": float(row["bugfix_composite"]),
            }
        )

    dataset = pd.DataFrame(rows)
    dataset["log_loc"] = dataset["loc"].map(lambda value: math.log1p(value))
    dataset["log_size_bytes"] = dataset["size_bytes"].map(lambda value: math.log1p(value))
    dataset["suffix"] = dataset["suffix"].replace("", "<none>")
    return dataset


def evaluate_negative_control_prediction(
    pairs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run leave-one-repository-out security-vs-bugfix classification."""
    dataset = build_negative_control_prediction_dataset(pairs_df)
    predictions, fold_metrics, summary = evaluate_leave_one_repo_out(dataset)
    summary["task"] = "security_vs_ordinary_bugfix"
    return dataset, predictions, fold_metrics, summary
