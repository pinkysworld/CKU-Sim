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
from cku_sim.analysis.prospective_file_panel import _history_features_for_file
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
BUGFIX_AUDIT_SUSPICION_RE = re.compile(
    r"\b(?:auth|permission|credential|token|cookie|session|csrf|cors|secret|"
    r"encryp|decrypt|ssl|tls|x509|certificate|header|access control)\b",
    re.IGNORECASE,
)


def _subsystem_key(path_str: str, *, depth: int = 2) -> str:
    parts = Path(path_str).parts[:-1]
    if not parts:
        return "<root>"
    return "/".join(parts[: max(1, depth)])


def _top_level_key(path_str: str) -> str:
    parts = Path(path_str).parts[:-1]
    return parts[0] if parts else "<root>"


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
                "security_directory_depth": max(0, len(Path(security_file).parts) - 1),
                "security_subsystem_key": _subsystem_key(security_file),
                "security_top_level_key": _top_level_key(security_file),
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


def augment_security_file_dataset(
    security_df: pd.DataFrame,
    repo_paths: dict[str, Path],
) -> pd.DataFrame:
    """Attach historical context for stricter security-vs-bugfix matching."""
    if security_df.empty:
        return security_df

    augmented = security_df.copy()
    history_cache_by_repo: dict[str, dict[tuple[str, str], list[dict[str, object]]]] = defaultdict(dict)
    prior_touches_total: list[float] = []
    prior_touches_365d: list[float] = []
    total_churn: list[float] = []
    churn_365d: list[float] = []
    file_age_days: list[float] = []

    for _, row in augmented.iterrows():
        repo = str(row["repo"])
        repo_path = repo_paths.get(repo)
        parent = str(row.get("security_parent") or "")
        file_path = str(row["security_file"])
        if repo_path is None or not parent:
            prior_touches_total.append(math.nan)
            prior_touches_365d.append(math.nan)
            total_churn.append(math.nan)
            churn_365d.append(math.nan)
            file_age_days.append(math.nan)
            continue

        epoch = _get_commit_epoch(repo_path, str(row["security_commit"]))
        history = _history_features_for_file(
            repo_path,
            parent,
            epoch,
            file_path,
            history_cache_by_repo[repo],
        )
        prior_touches_total.append(float(history["prior_touches_total"]))
        prior_touches_365d.append(float(history["prior_touches_365d"]))
        total_churn.append(float(history["total_churn"]))
        churn_365d.append(float(history["churn_365d"]))
        file_age_days.append(float(history["file_age_days"]))

    augmented["security_prior_touches_total"] = prior_touches_total
    augmented["security_prior_touches_365d"] = prior_touches_365d
    augmented["security_total_churn"] = total_churn
    augmented["security_churn_365d"] = churn_365d
    augmented["security_file_age_days"] = file_age_days
    return augmented


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
    history_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
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
            history = _history_features_for_file(
                repo_path,
                parent,
                epoch,
                path_str,
                history_cache,
            )
            rows.append(
                {
                    "repo": entry.name,
                    "bugfix_commit": commit,
                    "bugfix_parent": parent,
                    "bugfix_subject": subject,
                    "bugfix_file": path_str,
                    "bugfix_suffix": str(file_info["suffix"]),
                    "bugfix_size_bytes": int(file_info["size"]),
                    "bugfix_directory_depth": max(0, len(Path(path_str).parts) - 1),
                    "bugfix_subsystem_key": _subsystem_key(path_str),
                    "bugfix_top_level_key": _top_level_key(path_str),
                    "bugfix_loc": int(file_info.get("loc", 0) or 0),
                    "bugfix_prior_touches_total": int(history["prior_touches_total"]),
                    "bugfix_prior_touches_365d": int(history["prior_touches_365d"]),
                    "bugfix_total_churn": int(history["total_churn"]),
                    "bugfix_churn_365d": int(history["churn_365d"]),
                    "bugfix_file_age_days": float(history["file_age_days"]),
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
    require_same_suffix: bool = False,
    require_same_subsystem: bool = False,
    max_log_loc_gap: float | None = None,
    max_log_touch_gap: float | None = None,
    max_directory_depth_gap: int | None = None,
) -> tuple[dict[str, object], object] | None:
    security_suffix = str(security_row["security_suffix"])
    security_size = float(security_row["security_size_bytes"])
    security_epoch = int(security_row["security_epoch"])
    security_loc = float(security_row.get("security_loc", 0.0) or 0.0)
    security_subsystem = str(security_row.get("security_subsystem_key", ""))
    security_top_level = str(security_row.get("security_top_level_key", ""))
    security_depth = int(security_row.get("security_directory_depth", 0) or 0)
    security_touches = float(security_row.get("security_prior_touches_total", 0.0) or 0.0)

    available = [
        item
        for item in bugfix_pool
        if (item["bugfix_commit"], item["bugfix_file"]) not in used_bugfix_keys
    ]

    filtered: list[dict[str, object]] = []
    for candidate in available:
        same_suffix = candidate["bugfix_suffix"] == security_suffix
        same_subsystem = str(candidate.get("bugfix_subsystem_key", "")) == security_subsystem
        if require_same_suffix and not same_suffix:
            continue
        if require_same_subsystem and not same_subsystem:
            continue

        loc_gap = abs(math.log1p(float(candidate.get("bugfix_loc", 0.0) or 0.0)) - math.log1p(security_loc))
        touch_gap = abs(
            math.log1p(float(candidate.get("bugfix_prior_touches_total", 0.0) or 0.0))
            - math.log1p(security_touches)
        )
        depth_gap = abs(int(candidate.get("bugfix_directory_depth", 0) or 0) - security_depth)
        if max_log_loc_gap is not None and loc_gap > max_log_loc_gap:
            continue
        if max_log_touch_gap is not None and touch_gap > max_log_touch_gap:
            continue
        if max_directory_depth_gap is not None and depth_gap > max_directory_depth_gap:
            continue
        filtered.append(candidate)

    ranked = sorted(
        filtered or available,
        key=lambda item: (
            str(item.get("bugfix_subsystem_key", "")) != security_subsystem,
            str(item.get("bugfix_top_level_key", "")) != security_top_level,
            item["bugfix_suffix"] != security_suffix,
            abs(
                int(item.get("bugfix_directory_depth", 0) or 0) - security_depth
            ),
            abs(math.log1p(float(item.get("bugfix_loc", 0.0) or 0.0)) - math.log1p(security_loc)),
            abs(
                math.log1p(float(item.get("bugfix_prior_touches_total", 0.0) or 0.0))
                - math.log1p(security_touches)
            ),
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
    require_same_suffix: bool = False,
    require_same_subsystem: bool = False,
    max_log_loc_gap: float | None = None,
    max_log_touch_gap: float | None = None,
    max_directory_depth_gap: int | None = None,
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
                require_same_suffix=require_same_suffix,
                require_same_subsystem=require_same_subsystem,
                max_log_loc_gap=max_log_loc_gap,
                max_log_touch_gap=max_log_touch_gap,
                max_directory_depth_gap=max_directory_depth_gap,
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
                    "security_directory_depth": security_dict.get("security_directory_depth"),
                    "security_subsystem_key": security_dict.get("security_subsystem_key"),
                    "security_top_level_key": security_dict.get("security_top_level_key"),
                    "security_prior_touches_total": security_dict.get("security_prior_touches_total"),
                    "security_prior_touches_365d": security_dict.get("security_prior_touches_365d"),
                    "security_total_churn": security_dict.get("security_total_churn"),
                    "security_churn_365d": security_dict.get("security_churn_365d"),
                    "security_file_age_days": security_dict.get("security_file_age_days"),
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
                    "bugfix_directory_depth": bugfix_row.get("bugfix_directory_depth"),
                    "bugfix_subsystem_key": bugfix_row.get("bugfix_subsystem_key"),
                    "bugfix_top_level_key": bugfix_row.get("bugfix_top_level_key"),
                    "bugfix_size_bytes": bugfix_metrics.total_bytes,
                    "bugfix_loc": bugfix_metrics.total_loc,
                    "bugfix_prior_touches_total": bugfix_row.get("bugfix_prior_touches_total"),
                    "bugfix_prior_touches_365d": bugfix_row.get("bugfix_prior_touches_365d"),
                    "bugfix_total_churn": bugfix_row.get("bugfix_total_churn"),
                    "bugfix_churn_365d": bugfix_row.get("bugfix_churn_365d"),
                    "bugfix_file_age_days": bugfix_row.get("bugfix_file_age_days"),
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
                    "same_suffix_match": int(
                        str(security_dict["security_suffix"]) == str(bugfix_row.get("bugfix_suffix"))
                    ),
                    "same_subsystem_match": int(
                        str(security_dict.get("security_subsystem_key", ""))
                        == str(bugfix_row.get("bugfix_subsystem_key", ""))
                    ),
                    "same_top_level_match": int(
                        str(security_dict.get("security_top_level_key", ""))
                        == str(bugfix_row.get("bugfix_top_level_key", ""))
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
                    "delta_log_prior_touches_total": (
                        math.log1p(float(security_dict.get("security_prior_touches_total", 0.0) or 0.0))
                        - math.log1p(float(bugfix_row.get("bugfix_prior_touches_total", 0.0) or 0.0))
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
    if "same_suffix_match" in pairs.columns:
        summary["same_suffix_share"] = float(pairs["same_suffix_match"].mean())
    if "same_subsystem_match" in pairs.columns:
        summary["same_subsystem_share"] = float(pairs["same_subsystem_match"].mean())
    if "same_top_level_match" in pairs.columns:
        summary["same_top_level_share"] = float(pairs["same_top_level_match"].mean())
    if "delta_log_prior_touches_total" in pairs.columns:
        summary["median_delta_log_prior_touches_total"] = float(
            pairs["delta_log_prior_touches_total"].median()
        )
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


def screen_bugfix_control_commits(
    pairs_df: pd.DataFrame,
    repo_paths: dict[str, Path],
    corpus_by_name: dict[str, CorpusEntry],
) -> pd.DataFrame:
    """Review ordinary bug-fix controls for residual security-related signal."""
    if pairs_df.empty:
        return pd.DataFrame()

    deduped = (
        pairs_df.sort_values(["repo", "bugfix_commit", "bugfix_file"])
        .drop_duplicates(subset=["repo", "bugfix_commit"])
        .copy()
    )
    rows: list[dict[str, object]] = []
    changed_cache: dict[tuple[str, str], list[str]] = {}

    for _, row in deduped.iterrows():
        repo = str(row["repo"])
        repo_path = repo_paths.get(repo)
        entry = corpus_by_name.get(repo)
        if repo_path is None or entry is None:
            continue

        bugfix_commit = str(row["bugfix_commit"])
        bugfix_file = str(row["bugfix_file"])
        cache_key = (repo, bugfix_commit)
        if cache_key not in changed_cache:
            changed_cache[cache_key] = _list_changed_source_files(
                repo_path,
                bugfix_commit,
                entry.source_extensions,
            )
        changed_files = changed_cache[cache_key]

        body_proc = _run_git(repo_path, ["show", "-s", "--format=%B", bugfix_commit])
        commit_message = body_proc.stdout.strip() if body_proc.returncode == 0 else ""
        full_message = commit_message or str(row.get("bugfix_subject", ""))
        security_ids = extract_security_ids(full_message)
        has_security_signal = bool(SECURITY_EXCLUDE_RE.search(full_message))
        has_suspicion_signal = bool(BUGFIX_AUDIT_SUSPICION_RE.search(full_message))
        bugfix_file_in_changed_files = bugfix_file in changed_files

        if security_ids or has_security_signal:
            review_decision = "reject"
            review_notes = "Commit message retains an explicit security identifier or a direct security keyword."
        elif not bugfix_file_in_changed_files:
            review_decision = "reject"
            review_notes = "Matched bug-fix file is not present in the changed-source file list."
        elif has_suspicion_signal:
            review_decision = "ambiguous"
            review_notes = "Commit message is not explicitly security-labelled, but it contains security-adjacent terms."
        else:
            review_decision = "accept"
            review_notes = "Commit message and changed-file context remain consistent with an ordinary bug-fix control."

        rows.append(
            {
                "repo": repo,
                "bugfix_commit": bugfix_commit,
                "bugfix_subject": str(row.get("bugfix_subject", "")),
                "bugfix_file": bugfix_file,
                "matched_security_event_id": str(row.get("security_event_id", "")),
                "matched_security_commit": str(row.get("security_commit", "")),
                "matched_security_file": str(row.get("security_file", "")),
                "bugfix_file_in_changed_files": int(bugfix_file_in_changed_files),
                "changed_source_files_count": int(len(changed_files)),
                "changed_source_files_sample": ";".join(changed_files[:10]),
                "commit_message_full": commit_message,
                "has_security_ids": int(bool(security_ids)),
                "has_security_keyword_signal": int(has_security_signal),
                "has_security_adjacent_signal": int(has_suspicion_signal),
                "review_decision": review_decision,
                "review_notes": review_notes,
            }
        )

    return pd.DataFrame(rows)


def summarise_bugfix_control_screen(screened_df: pd.DataFrame) -> dict[str, object]:
    """Summarise the ordinary bug-fix control screening audit."""
    if screened_df.empty:
        return {"n_reviewed": 0}

    decision_counts = (
        screened_df["review_decision"].value_counts().reindex(["accept", "ambiguous", "reject"], fill_value=0)
    )
    return {
        "n_reviewed": int(len(screened_df)),
        "review_counts": {str(key): int(value) for key, value in decision_counts.items()},
        "accept_rate": float((screened_df["review_decision"] == "accept").mean()),
        "non_reject_rate": float(screened_df["review_decision"].isin({"accept", "ambiguous"}).mean()),
        "bugfix_file_in_changed_files_rate": float(screened_df["bugfix_file_in_changed_files"].mean()),
        "security_keyword_signal_rate": float(screened_df["has_security_keyword_signal"].mean()),
        "security_adjacent_signal_rate": float(screened_df["has_security_adjacent_signal"].mean()),
        "by_repo": (
            screened_df.groupby(["repo", "review_decision"]).size().unstack(fill_value=0).reset_index().to_dict("records")
        ),
    }
