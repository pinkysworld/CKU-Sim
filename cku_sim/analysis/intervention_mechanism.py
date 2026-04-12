"""Audited external intervention study for CKU mechanism closure."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cku_sim.analysis.audited_panel import (
    _load_commit_timestamp,
    _lookup_snapshot_history_features,
    _snapshot_history_index,
)
from cku_sim.analysis.file_level_case_control import (
    _get_commit_epoch,
    _get_metrics_for_snapshot_file,
    _list_changed_source_files,
    _list_source_files_at_commit,
    _run_git,
    extract_security_ids,
)
from cku_sim.analysis.negative_control import (
    BUGFIX_SUBJECT_RE,
    MERGE_OR_REVERT_RE,
    SECURITY_EXCLUDE_RE,
)
from cku_sim.analysis.prospective_file_panel import (
    PROSPECTIVE_MODEL_SPECS,
    _build_pipeline,
)
from cku_sim.core.config import CorpusEntry

logger = logging.getLogger(__name__)

INTERVENTION_AUDIT_COLUMNS = [
    "repo",
    "anchor_tag",
    "anchor_commit",
    "intervention_commit",
    "intervention_parent",
    "file_path",
    "commit_date",
    "intervention_type",
    "keyword_basis",
    "review_decision",
    "reviewer",
    "notes",
]

PRIMARY_EXTERNAL_MECHANISM_REPOS = [
    "django-django",
    "traefik-traefik",
    "prometheus-prometheus",
    "fastapi-fastapi",
    "psf-requests",
    "pallets-flask",
    "scrapy-scrapy",
]

PRIMARY_INTERVENTION_MODEL = "baseline_plus_composite"

REFACTOR_KEYWORD_PATTERNS = {
    "cleanup": re.compile(r"\bcleanup\b", re.IGNORECASE),
    "dedup": re.compile(r"\bdedup(?:licate|lication)?\b", re.IGNORECASE),
    "extract": re.compile(r"\bextract(?:ion|ed|ing)?\b", re.IGNORECASE),
    "modular": re.compile(r"\bmodular(?:ity|ize|ized|ization)?\b", re.IGNORECASE),
    "refactor": re.compile(r"\brefactor(?:ing|ed)?\b", re.IGNORECASE),
    "rework": re.compile(r"\brework(?:ed|ing)?\b", re.IGNORECASE),
    "simplify": re.compile(r"\bsimplif(?:y|ied|ies|ication)\b", re.IGNORECASE),
    "split": re.compile(r"\bsplit(?:ting)?\b", re.IGNORECASE),
}

RELEASE_EXCLUDE_RE = re.compile(
    r"\b(?:release|prepare release|version bump|bump(?:ed)? version|changelog|"
    r"newsfile|release note|bump to v?\d)\b",
    re.IGNORECASE,
)
BUGFIX_EXCLUDE_RE = re.compile(
    r"\b(?:fix(?:es|ed)?|bug(?:fix)?|regression|crash|panic|leak|correct(?:s|ed)?|"
    r"resolve[sd]?|prevent(?:s|ed)?|avoid(?:s|ed)?|error|failure|issue)\b",
    re.IGNORECASE,
)


def _normalise_keyword_basis(message: str) -> tuple[str, str]:
    matches = [
        label
        for label, pattern in REFACTOR_KEYWORD_PATTERNS.items()
        if pattern.search(message)
    ]
    if not matches:
        return "", ""
    matches = sorted(dict.fromkeys(matches))
    return matches[0], ";".join(matches)


def _depth_bucket(depth: object) -> int:
    try:
        value = int(depth)
    except (TypeError, ValueError):
        value = 0
    return min(max(value, 0), 4)


def _timestamp_from_epoch(epoch: object) -> pd.Timestamp:
    return pd.Timestamp(int(epoch), unit="s", tz="UTC")


def _screen_commit_message(
    subject: str,
    body: str,
    *,
    require_refactor_keyword: bool,
    exclude_refactor_keyword: bool,
) -> tuple[bool, str, str]:
    full_message = f"{subject}\n{body}".strip()
    intervention_type, keyword_basis = _normalise_keyword_basis(full_message)
    has_refactor_keyword = bool(keyword_basis)

    if MERGE_OR_REVERT_RE.match(subject):
        return False, "", ""
    if extract_security_ids(full_message):
        return False, "", ""
    if SECURITY_EXCLUDE_RE.search(full_message):
        return False, "", ""
    if RELEASE_EXCLUDE_RE.search(full_message):
        return False, "", ""
    if BUGFIX_SUBJECT_RE.search(subject) or BUGFIX_EXCLUDE_RE.search(full_message):
        return False, "", ""
    if require_refactor_keyword and not has_refactor_keyword:
        return False, "", ""
    if exclude_refactor_keyword and has_refactor_keyword:
        return False, "", ""
    return True, intervention_type, keyword_basis


def _scan_commit_metadata(
    repo_path: Path,
    *,
    since: str,
    until: str,
    grep_patterns: list[str] | None = None,
    pathspecs: list[str] | None = None,
) -> pd.DataFrame:
    cmd = [
        "log",
        "--all",
        "--no-merges",
        f"--since={since}",
        f"--until={until}",
        "--format=%H%x1f%P%x1f%ct%x1f%s%x1f%B%x1e",
    ]
    if grep_patterns:
        cmd.insert(3, "--regexp-ignore-case")
        for pattern in grep_patterns:
            cmd.extend(["--grep", pattern])
    if pathspecs:
        cmd.append("--")
        cmd.extend(pathspecs)
    proc = _run_git(repo_path, cmd)
    if proc.returncode != 0:
        logger.warning("git log scan failed for %s", repo_path.name)
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for block in proc.stdout.split("\x1e"):
        block = block.strip()
        if not block:
            continue
        parts = block.split("\x1f", 4)
        if len(parts) != 5:
            continue
        commit, parents, epoch_s, subject, body = parts
        parent_list = [value.strip() for value in parents.split() if value.strip()]
        if len(parent_list) != 1:
            continue
        try:
            epoch = int(epoch_s)
        except ValueError:
            continue
        rows.append(
            {
                "commit": commit.strip(),
                "parent": parent_list[0],
                "epoch": epoch,
                "commit_date": _timestamp_from_epoch(epoch).isoformat(),
                "subject": subject.strip(),
                "body": body.strip(),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["commit"]).sort_values(
        ["epoch", "commit"]
    ).reset_index(drop=True)


def _filter_commit_metadata(
    metadata: pd.DataFrame,
    *,
    require_refactor_keyword: bool,
    exclude_refactor_keyword: bool,
) -> pd.DataFrame:
    if metadata.empty:
        return metadata
    rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        keep, intervention_type, keyword_basis = _screen_commit_message(
            str(row.get("subject", "")),
            str(row.get("body", "")),
            require_refactor_keyword=require_refactor_keyword,
            exclude_refactor_keyword=exclude_refactor_keyword,
        )
        if not keep:
            continue
        payload = row.to_dict()
        payload["intervention_type"] = intervention_type
        payload["keyword_basis"] = keyword_basis
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["epoch", "commit"]).reset_index(drop=True)


def _select_evenly_spaced(frame: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    indices = np.linspace(0, len(frame) - 1, num=max_rows, dtype=int)
    unique_indices = sorted(dict.fromkeys(int(value) for value in indices))
    return frame.iloc[unique_indices].reset_index(drop=True)


def _prefetch_commit_objects(
    repo_path: Path,
    commits: list[str],
    *,
    batch_size: int = 10,
) -> None:
    unique_commits = [value for value in dict.fromkeys(commits) if value]
    if not unique_commits:
        return
    for start in range(0, len(unique_commits), batch_size):
        batch = unique_commits[start : start + batch_size]
        proc = _run_git(
            repo_path,
            ["fetch", "--quiet", "--filter=blob:limit=10m", "origin", *batch],
        )
        if proc.returncode != 0:
            logger.warning(
                "Prefetch failed for %s batch starting at %s: %s",
                repo_path.name,
                batch[0],
                proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
            )
            return


def _describe_anchor_tag(
    repo_path: Path,
    commit: str,
    tag_cache: dict[str, str],
) -> str:
    if commit in tag_cache:
        return tag_cache[commit]
    proc = _run_git(repo_path, ["describe", "--tags", "--abbrev=0", commit])
    tag_cache[commit] = proc.stdout.strip() if proc.returncode == 0 else ""
    return tag_cache[commit]


def _encode_anchor_features(prefix: str, metrics, history: dict[str, object]) -> dict[str, object]:
    return {
        f"{prefix}loc": float(metrics.total_loc),
        f"{prefix}size_bytes": float(metrics.total_bytes),
        f"{prefix}ci_gzip": float(metrics.ci_gzip),
        f"{prefix}shannon_entropy": float(metrics.shannon_entropy),
        f"{prefix}cyclomatic_density": float(metrics.cyclomatic_density),
        f"{prefix}halstead_volume": float(metrics.halstead_volume),
        f"{prefix}composite_score": float(metrics.composite_score),
        f"{prefix}directory_depth": float(history["directory_depth"]),
        f"{prefix}prior_touches_total": float(history["prior_touches_total"]),
        f"{prefix}prior_touches_365d": float(history["prior_touches_365d"]),
        f"{prefix}total_churn": float(history["total_churn"]),
        f"{prefix}churn_365d": float(history["churn_365d"]),
        f"{prefix}author_count_total": float(history["author_count_total"]),
        f"{prefix}author_count_365d": float(history["author_count_365d"]),
        f"{prefix}file_age_days": float(history["file_age_days"]),
        f"{prefix}latest_touch_days": float(history["latest_touch_days"]),
    }


def _build_single_commit_file_row(
    repo_path: Path,
    entry: CorpusEntry,
    *,
    commit: str,
    parent: str,
    epoch: int,
    subject: str,
    file_path: str,
    intervention_type: str,
    keyword_basis: str,
    min_loc: int,
    metrics_cache: dict[tuple[str, str], object | None],
    tree_cache: dict[str, dict[str, dict[str, object]]],
    history_cache: dict[str, dict[str, dict[str, object]]],
    epoch_cache: dict[str, int],
    tag_cache: dict[str, str],
    parent_epoch: int | None = None,
    pre_history_index: dict[str, dict[str, object]] | None = None,
    post_history_index: dict[str, dict[str, object]] | None = None,
) -> dict[str, object] | None:
    if parent not in tree_cache:
        tree_cache[parent] = _list_source_files_at_commit(
            repo_path,
            parent,
            entry.source_extensions,
        )
    if commit not in tree_cache:
        tree_cache[commit] = _list_source_files_at_commit(
            repo_path,
            commit,
            entry.source_extensions,
        )
    if file_path not in tree_cache[parent] or file_path not in tree_cache[commit]:
        return None

    pre_metrics = _get_metrics_for_snapshot_file(repo_path, parent, file_path, metrics_cache)
    post_metrics = _get_metrics_for_snapshot_file(repo_path, commit, file_path, metrics_cache)
    if pre_metrics is None or post_metrics is None:
        return None
    if pre_metrics.total_loc < min_loc or post_metrics.total_loc < min_loc:
        return None

    if parent_epoch is None:
        if parent not in epoch_cache:
            epoch_cache[parent] = _get_commit_epoch(repo_path, parent)
        parent_epoch = int(epoch_cache[parent])
    if pre_history_index is None:
        pre_history_index = _snapshot_history_index(
            repo_path,
            parent,
            int(parent_epoch),
            entry.source_extensions,
            history_cache,
            snapshot_paths=[file_path],
        )
    if post_history_index is None:
        post_history_index = _snapshot_history_index(
            repo_path,
            commit,
            epoch,
            entry.source_extensions,
            history_cache,
            snapshot_paths=[file_path],
        )
    pre_history = _lookup_snapshot_history_features(pre_history_index, file_path)
    post_history = _lookup_snapshot_history_features(post_history_index, file_path)
    anchor_tag = _describe_anchor_tag(repo_path, parent, tag_cache)

    row = {
        "candidate_id": f"{entry.name}:{commit}:{file_path}",
        "repo": entry.name,
        "anchor_tag": anchor_tag,
        "anchor_commit": parent,
        "intervention_commit": commit,
        "intervention_parent": parent,
        "commit_epoch": int(epoch),
        "commit_date": _timestamp_from_epoch(epoch).isoformat(),
        "anchor_epoch": int(parent_epoch),
        "anchor_date": _timestamp_from_epoch(parent_epoch).isoformat(),
        "subject": subject,
        "file_path": file_path,
        "suffix": Path(file_path).suffix.lower() or "<none>",
        "directory_depth_bucket": _depth_bucket(pre_history["directory_depth"]),
        "intervention_type": intervention_type,
        "keyword_basis": keyword_basis,
    }
    row.update(_encode_anchor_features("pre_", pre_metrics, pre_history))
    row.update(_encode_anchor_features("post_", post_metrics, post_history))
    row["pre_log_loc"] = float(math.log1p(row["pre_loc"]))
    row["pre_log_prior_touches_total"] = float(math.log1p(row["pre_prior_touches_total"]))
    return row


def build_commit_file_pool(
    repo_path: Path,
    entry: CorpusEntry,
    commit_metadata: pd.DataFrame,
    *,
    min_loc: int,
    max_files_per_commit: int = 1,
    prefetch_commits: bool = True,
) -> pd.DataFrame:
    if commit_metadata.empty:
        return pd.DataFrame()

    metrics_cache: dict[tuple[str, str], object | None] = {}
    tree_cache: dict[str, dict[str, dict[str, object]]] = {}
    history_cache: dict[str, dict[str, dict[str, object]]] = {}
    epoch_cache: dict[str, int] = {}
    tag_cache: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    if prefetch_commits:
        _prefetch_commit_objects(
            repo_path,
            [
                str(value)
                for value in pd.unique(
                    pd.concat([commit_metadata["commit"], commit_metadata["parent"]], ignore_index=True)
                )
                if str(value)
            ],
        )

    for _, row in commit_metadata.iterrows():
        commit = str(row["commit"])
        parent = str(row["parent"])
        changed_files = _list_changed_source_files(repo_path, commit, entry.source_extensions)
        if not changed_files:
            continue
        if parent not in tree_cache:
            tree_cache[parent] = _list_source_files_at_commit(
                repo_path,
                parent,
                entry.source_extensions,
            )
        if commit not in tree_cache:
            tree_cache[commit] = _list_source_files_at_commit(
                repo_path,
                commit,
                entry.source_extensions,
            )
        candidate_files = [
            file_path
            for file_path in sorted(changed_files)
            if file_path in tree_cache[parent] and file_path in tree_cache[commit]
        ]
        if max_files_per_commit:
            candidate_files = candidate_files[: max_files_per_commit]
        if not candidate_files:
            continue
        if parent not in epoch_cache:
            epoch_cache[parent] = _get_commit_epoch(repo_path, parent)
        parent_epoch = int(epoch_cache[parent])
        pre_history_index = _snapshot_history_index(
            repo_path,
            parent,
            parent_epoch,
            entry.source_extensions,
            history_cache,
            snapshot_paths=candidate_files,
        )
        post_history_index = _snapshot_history_index(
            repo_path,
            commit,
            int(row["epoch"]),
            entry.source_extensions,
            history_cache,
            snapshot_paths=candidate_files,
        )
        for file_path in candidate_files:
            materialised = _build_single_commit_file_row(
                repo_path,
                entry,
                commit=commit,
                parent=parent,
                epoch=int(row["epoch"]),
                subject=str(row.get("subject", "")),
                file_path=file_path,
                intervention_type=str(row.get("intervention_type", "")),
                keyword_basis=str(row.get("keyword_basis", "")),
                min_loc=min_loc,
                metrics_cache=metrics_cache,
                tree_cache=tree_cache,
                history_cache=history_cache,
                epoch_cache=epoch_cache,
                tag_cache=tag_cache,
                parent_epoch=parent_epoch,
                pre_history_index=pre_history_index,
                post_history_index=post_history_index,
            )
            if materialised is not None:
                rows.append(materialised)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(
        subset=["repo", "intervention_commit", "file_path"]
    ).sort_values(["commit_epoch", "intervention_commit", "file_path"]).reset_index(drop=True)


def build_seed_commit_file_pool(
    repo_path: Path,
    entry: CorpusEntry,
    commit_metadata: pd.DataFrame,
    *,
    min_loc: int,
    max_files_per_commit: int = 1,
    prefetch_commits: bool = True,
) -> pd.DataFrame:
    if commit_metadata.empty:
        return pd.DataFrame()

    metrics_cache: dict[tuple[str, str], object | None] = {}
    tree_cache: dict[str, dict[str, dict[str, object]]] = {}
    tag_cache: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    if prefetch_commits:
        _prefetch_commit_objects(
            repo_path,
            [
                str(value)
                for value in pd.unique(
                    pd.concat([commit_metadata["commit"], commit_metadata["parent"]], ignore_index=True)
                )
                if str(value)
            ],
        )

    for _, row in commit_metadata.iterrows():
        commit = str(row["commit"])
        parent = str(row["parent"])
        changed_files = _list_changed_source_files(repo_path, commit, entry.source_extensions)
        if not changed_files:
            continue
        if parent not in tree_cache:
            tree_cache[parent] = _list_source_files_at_commit(
                repo_path,
                parent,
                entry.source_extensions,
            )
        if commit not in tree_cache:
            tree_cache[commit] = _list_source_files_at_commit(
                repo_path,
                commit,
                entry.source_extensions,
            )
        candidate_files = [
            file_path
            for file_path in sorted(changed_files)
            if file_path in tree_cache[parent] and file_path in tree_cache[commit]
        ]
        if max_files_per_commit:
            candidate_files = candidate_files[: max_files_per_commit]
        for file_path in candidate_files:
            pre_metrics = _get_metrics_for_snapshot_file(
                repo_path, parent, file_path, metrics_cache
            )
            post_metrics = _get_metrics_for_snapshot_file(
                repo_path, commit, file_path, metrics_cache
            )
            if pre_metrics is None or post_metrics is None:
                continue
            if pre_metrics.total_loc < min_loc or post_metrics.total_loc < min_loc:
                continue
            rows.append(
                {
                    "repo": entry.name,
                    "anchor_tag": _describe_anchor_tag(repo_path, parent, tag_cache),
                    "anchor_commit": parent,
                    "intervention_commit": commit,
                    "intervention_parent": parent,
                    "commit_epoch": int(row["epoch"]),
                    "commit_date": _timestamp_from_epoch(int(row["epoch"])).isoformat(),
                    "subject": str(row.get("subject", "")),
                    "file_path": file_path,
                    "intervention_type": str(row.get("intervention_type", "")),
                    "keyword_basis": str(row.get("keyword_basis", "")),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(
        subset=["repo", "intervention_commit", "file_path"]
    ).sort_values(["commit_epoch", "intervention_commit", "file_path"]).reset_index(drop=True)


def seed_refactoring_intervention_audit_table(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    *,
    repos: list[str] | None = None,
    since: str,
    until: str,
    max_rows_per_repo: int = 20,
    repo_row_caps: dict[str, int] | None = None,
    min_loc: int = 5,
    reviewer: str = "initial_curation",
) -> pd.DataFrame:
    allowed_repos = set(repos or [])
    repo_row_caps = repo_row_caps or {}
    rows: list[pd.DataFrame] = []
    for entry in corpus:
        if allowed_repos and entry.name not in allowed_repos:
            continue
        repo_path = repo_paths.get(entry.name)
        if repo_path is None or not repo_path.exists():
            continue
        repo_cap = int(repo_row_caps.get(entry.name, max_rows_per_repo))
        if repo_cap <= 0:
            continue
        metadata = _scan_commit_metadata(
            repo_path,
            since=since,
            until=until,
            grep_patterns=["refactor", "cleanup", "simplify", "extract", "modular", "split", "dedup", "rework"],
        )
        metadata = _filter_commit_metadata(
            metadata,
            require_refactor_keyword=True,
            exclude_refactor_keyword=False,
        )
        if metadata.empty:
            continue
        metadata = _select_evenly_spaced(
            metadata,
            max_rows=max(repo_cap * 6, repo_cap),
        )
        pool = build_seed_commit_file_pool(
            repo_path,
            entry,
            metadata,
            min_loc=min_loc,
            max_files_per_commit=1,
        )
        if pool.empty:
            continue
        pool = _select_evenly_spaced(pool, max_rows=repo_cap)
        repo_audit = pd.DataFrame(
            [
                {
                    "repo": row["repo"],
                    "anchor_tag": row["anchor_tag"],
                    "anchor_commit": row["anchor_commit"],
                    "intervention_commit": row["intervention_commit"],
                    "intervention_parent": row["intervention_parent"],
                    "file_path": row["file_path"],
                    "commit_date": row["commit_date"],
                    "intervention_type": row["intervention_type"] or "refactor",
                    "keyword_basis": row["keyword_basis"],
                    "review_decision": "accept",
                    "reviewer": reviewer,
                    "notes": "Accepted under the deterministic structural-refactoring screen.",
                }
                for _, row in pool.iterrows()
            ],
            columns=INTERVENTION_AUDIT_COLUMNS,
        )
        rows.append(repo_audit)
    if not rows:
        return pd.DataFrame(columns=INTERVENTION_AUDIT_COLUMNS)
    audit = pd.concat(rows, ignore_index=True)
    audit = audit.drop_duplicates(
        subset=["repo", "intervention_commit", "file_path"]
    ).sort_values(["repo", "commit_date", "intervention_commit", "file_path"])
    return audit.reset_index(drop=True)


def seed_security_file_refactoring_intervention_audit_table(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    security_audit: pd.DataFrame,
    *,
    repos: list[str] | None = None,
    since: str,
    until: str,
    max_rows_per_repo: int = 40,
    repo_row_caps: dict[str, int] | None = None,
    max_rows_per_file: int = 4,
    min_loc: int = 5,
    reviewer: str = "security_file_enrichment",
    prefetch_commits: bool = False,
) -> pd.DataFrame:
    """Seed additional audited intervention rows by scanning refactors on security-linked files."""
    allowed_repos = set(repos or [])
    repo_row_caps = repo_row_caps or {}
    security_subset = security_audit.copy()
    if "review_decision" in security_subset.columns:
        security_subset = security_subset.loc[security_subset["review_decision"] == "accept"].copy()
    security_subset = security_subset.loc[security_subset["file_path"].notna()].copy()
    if allowed_repos:
        security_subset = security_subset.loc[security_subset["repo"].isin(allowed_repos)].copy()
    if security_subset.empty:
        return pd.DataFrame(columns=INTERVENTION_AUDIT_COLUMNS)

    entry_map = {entry.name: entry for entry in corpus}
    rows: list[pd.DataFrame] = []
    for repo, repo_security in security_subset.groupby("repo", sort=False):
        entry = entry_map.get(str(repo))
        repo_path = repo_paths.get(str(repo))
        if entry is None or repo_path is None or not repo_path.exists():
            continue
        repo_cap = int(repo_row_caps.get(str(repo), max_rows_per_repo))
        if repo_cap <= 0:
            continue

        candidate_rows: list[dict[str, object]] = []
        for file_path, file_security in repo_security.groupby("file_path", sort=False):
            published_ts = pd.to_datetime(file_security["published_at"], utc=True, errors="coerce")
            valid_published = published_ts.dropna()
            file_until = until
            if not valid_published.empty:
                candidate_until = (valid_published.min() - pd.Timedelta(days=1)).date().isoformat()
                if candidate_until < since:
                    continue
                file_until = min(until, candidate_until)
            metadata = _scan_commit_metadata(
                repo_path,
                since=since,
                until=file_until,
                grep_patterns=[
                    "refactor",
                    "cleanup",
                    "simplify",
                    "extract",
                    "modular",
                    "split",
                    "dedup",
                    "rework",
                ],
                pathspecs=[str(file_path)],
            )
            metadata = _filter_commit_metadata(
                metadata,
                require_refactor_keyword=True,
                exclude_refactor_keyword=False,
            )
            if metadata.empty:
                continue
            metadata = _select_evenly_spaced(
                metadata,
                max_rows=max(max_rows_per_file * 3, max_rows_per_file),
            )
            for _, meta_row in metadata.iterrows():
                payload = meta_row.to_dict()
                payload["file_path"] = str(file_path)
                candidate_rows.append(payload)

        if not candidate_rows:
            continue
        candidate_df = (
            pd.DataFrame(candidate_rows)
            .drop_duplicates(subset=["commit", "parent", "file_path"])
            .sort_values(["epoch", "commit", "file_path"])
            .reset_index(drop=True)
        )
        if prefetch_commits:
            _prefetch_commit_objects(
                repo_path,
                [
                    str(value)
                    for value in pd.unique(
                        pd.concat([candidate_df["commit"], candidate_df["parent"]], ignore_index=True)
                    )
                    if str(value)
                ],
            )
        metrics_cache: dict[tuple[str, str], object | None] = {}
        tree_cache: dict[str, dict[str, dict[str, object]]] = {}
        history_cache: dict[str, dict[str, dict[str, object]]] = {}
        epoch_cache: dict[str, int] = {}
        tag_cache: dict[str, str] = {}
        materialised_rows: list[dict[str, object]] = []
        for _, row in candidate_df.iterrows():
            materialised = _build_single_commit_file_row(
                repo_path,
                entry,
                commit=str(row["commit"]),
                parent=str(row["parent"]),
                epoch=int(row["epoch"]),
                subject=str(row.get("subject", "")),
                file_path=str(row["file_path"]),
                intervention_type=str(row.get("intervention_type", "")),
                keyword_basis=str(row.get("keyword_basis", "")),
                min_loc=min_loc,
                metrics_cache=metrics_cache,
                tree_cache=tree_cache,
                history_cache=history_cache,
                epoch_cache=epoch_cache,
                tag_cache=tag_cache,
            )
            if materialised is None:
                continue
            materialised_rows.append(
                {
                    "repo": materialised["repo"],
                    "anchor_tag": materialised["anchor_tag"],
                    "anchor_commit": materialised["anchor_commit"],
                    "intervention_commit": materialised["intervention_commit"],
                    "intervention_parent": materialised["intervention_parent"],
                    "file_path": materialised["file_path"],
                    "commit_date": materialised["commit_date"],
                    "intervention_type": materialised["intervention_type"] or "refactor",
                    "keyword_basis": materialised["keyword_basis"],
                    "review_decision": "accept",
                    "reviewer": reviewer,
                    "notes": (
                        "Accepted under the deterministic security-file refactoring screen "
                        "before the first audited security event on this file."
                    ),
                }
            )
        if not materialised_rows:
            continue
        repo_audit = pd.DataFrame(materialised_rows, columns=INTERVENTION_AUDIT_COLUMNS)
        repo_audit = _select_evenly_spaced(repo_audit, max_rows=repo_cap)
        rows.append(repo_audit)

    if not rows:
        return pd.DataFrame(columns=INTERVENTION_AUDIT_COLUMNS)
    audit = pd.concat(rows, ignore_index=True)
    audit = audit.drop_duplicates(
        subset=["repo", "intervention_commit", "file_path"]
    ).sort_values(["repo", "commit_date", "intervention_commit", "file_path"])
    return audit.reset_index(drop=True)


def load_intervention_audit_table(
    audit_path: Path,
    *,
    accepted_only: bool = True,
) -> pd.DataFrame:
    if not audit_path.exists():
        return pd.DataFrame(columns=INTERVENTION_AUDIT_COLUMNS)
    audit = pd.read_csv(audit_path)
    for column in INTERVENTION_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""
    if accepted_only:
        audit = audit.loc[audit["review_decision"] == "accept"].copy()
    audit = audit.drop_duplicates(subset=["repo", "intervention_commit", "file_path"])
    return audit.sort_values(["repo", "commit_date", "intervention_commit", "file_path"]).reset_index(drop=True)


def merge_intervention_audit_frames(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame in (incoming, existing):
        if frame is None:
            continue
        if frame.empty:
            continue
        normalized = frame.copy()
        for column in INTERVENTION_AUDIT_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = ""
        frames.append(normalized[INTERVENTION_AUDIT_COLUMNS].copy())
    if not frames:
        return pd.DataFrame(columns=INTERVENTION_AUDIT_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["repo", "intervention_commit", "file_path"])
    return merged.sort_values(
        ["repo", "commit_date", "intervention_commit", "file_path"]
    ).reset_index(drop=True)


def summarise_intervention_audit_table(audit: pd.DataFrame) -> dict[str, object]:
    if audit.empty:
        return {
            "n_rows": 0,
            "n_repos": 0,
            "n_anchor_tags": 0,
            "n_intervention_commits": 0,
            "review_breakdown": {},
            "intervention_type_breakdown": {},
        }
    return {
        "n_rows": int(len(audit)),
        "n_repos": int(audit["repo"].nunique()),
        "n_anchor_tags": int(audit["anchor_tag"].replace("", pd.NA).dropna().nunique()),
        "n_intervention_commits": int(
            audit["intervention_commit"].replace("", pd.NA).dropna().nunique()
        ),
        "review_breakdown": audit["review_decision"].value_counts().to_dict(),
        "intervention_type_breakdown": audit["intervention_type"].value_counts().to_dict(),
    }


def build_audited_intervention_pool(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    audit: pd.DataFrame,
    *,
    min_loc: int = 5,
    prefetch_commits: bool = True,
) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()

    entry_map = {entry.name: entry for entry in corpus}
    metrics_cache: dict[tuple[str, str], object | None] = {}
    rows: list[dict[str, object]] = []
    repo_tree_cache: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    repo_history_cache: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    repo_epoch_cache: dict[str, dict[str, int]] = {}
    repo_tag_cache: dict[str, dict[str, str]] = {}

    for repo, repo_audit in audit.groupby("repo", sort=False):
        repo_path = repo_paths.get(str(repo))
        if repo_path is None:
            continue
        parent_commits = repo_audit["intervention_parent"].fillna(repo_audit["anchor_commit"])
        if prefetch_commits:
            _prefetch_commit_objects(
                repo_path,
                [
                    str(value)
                    for value in pd.unique(
                        pd.concat([repo_audit["intervention_commit"], parent_commits], ignore_index=True)
                    )
                    if str(value)
                ],
            )

    for _, audit_row in audit.iterrows():
        repo = str(audit_row["repo"])
        entry = entry_map.get(repo)
        repo_path = repo_paths.get(repo)
        if entry is None or repo_path is None:
            continue
        tree_cache = repo_tree_cache.setdefault(repo, {})
        history_cache = repo_history_cache.setdefault(repo, {})
        epoch_cache = repo_epoch_cache.setdefault(repo, {})
        tag_cache = repo_tag_cache.setdefault(repo, {})
        commit = str(audit_row["intervention_commit"])
        parent = str(audit_row["intervention_parent"] or audit_row["anchor_commit"])
        commit_ts = _load_commit_timestamp(repo_path, commit)
        if commit_ts is None:
            continue
        materialised = _build_single_commit_file_row(
            repo_path,
            entry,
            commit=commit,
            parent=parent,
            epoch=int(commit_ts.timestamp()),
            subject="",
            file_path=str(audit_row["file_path"]),
            intervention_type=str(audit_row.get("intervention_type", "")),
            keyword_basis=str(audit_row.get("keyword_basis", "")),
            min_loc=min_loc,
            metrics_cache=metrics_cache,
            tree_cache=tree_cache,
            history_cache=history_cache,
            epoch_cache=epoch_cache,
            tag_cache=tag_cache,
        )
        if materialised is None:
            continue
        rows.append(materialised)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["intervention_id"] = [
        f"{row['repo']}::{row['intervention_commit']}::{row['file_path']}"
        for _, row in frame.iterrows()
    ]
    return frame.sort_values(["repo", "commit_epoch", "intervention_commit", "file_path"]).reset_index(drop=True)


def annotate_intervention_simplification_strength(interventions: pd.DataFrame) -> pd.DataFrame:
    """Annotate intervention rows with simple pre/post structural-reduction counts."""
    if interventions.empty:
        return interventions.copy()
    frame = interventions.copy()
    reduction_metrics = [
        ("composite_score", "delta_composite_score"),
        ("ci_gzip", "delta_ci_gzip"),
        ("halstead_volume", "delta_halstead_volume"),
        ("cyclomatic_density", "delta_cyclomatic_density"),
    ]
    reduction_flags: list[str] = []
    for metric, delta_col in reduction_metrics:
        pre_col = f"pre_{metric}"
        post_col = f"post_{metric}"
        flag_col = f"reduced_{metric}"
        frame[delta_col] = pd.to_numeric(frame[post_col], errors="coerce") - pd.to_numeric(
            frame[pre_col], errors="coerce"
        )
        frame[flag_col] = (pd.to_numeric(frame[delta_col], errors="coerce") < 0).astype(int)
        reduction_flags.append(flag_col)
    frame["structure_drop_count"] = frame[reduction_flags].sum(axis=1).astype(int)
    return frame


def filter_material_interventions(
    interventions: pd.DataFrame,
    *,
    min_structure_drops: int = 0,
) -> pd.DataFrame:
    """Restrict to interventions that reduce enough structural dimensions."""
    annotated = annotate_intervention_simplification_strength(interventions)
    if min_structure_drops <= 0 or annotated.empty:
        return annotated
    return (
        annotated.loc[annotated["structure_drop_count"] >= int(min_structure_drops)]
        .copy()
        .reset_index(drop=True)
    )


def build_maintenance_pool(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    *,
    repos: list[str] | None = None,
    since: str,
    until: str,
    min_loc: int = 5,
    max_commits_per_repo: int = 250,
    max_files_per_commit: int = 1,
    prefetch_commits: bool = True,
) -> pd.DataFrame:
    allowed_repos = set(repos or [])
    rows: list[pd.DataFrame] = []
    for entry in corpus:
        if allowed_repos and entry.name not in allowed_repos:
            continue
        repo_path = repo_paths.get(entry.name)
        if repo_path is None or not repo_path.exists():
            continue
        metadata = _scan_commit_metadata(repo_path, since=since, until=until)
        metadata = _filter_commit_metadata(
            metadata,
            require_refactor_keyword=False,
            exclude_refactor_keyword=True,
        )
        if metadata.empty:
            continue
        metadata = _select_evenly_spaced(metadata, max_rows=max_commits_per_repo)
        pool = build_commit_file_pool(
            repo_path,
            entry,
            metadata,
            min_loc=min_loc,
            max_files_per_commit=max_files_per_commit,
            prefetch_commits=prefetch_commits,
        )
        if pool.empty:
            continue
        pool = pool.rename(
            columns={
                "candidate_id": "maintenance_id",
                "anchor_tag": "maintenance_anchor_tag",
                "anchor_commit": "maintenance_anchor_commit",
                "intervention_commit": "maintenance_commit",
                "intervention_parent": "maintenance_parent",
                "commit_epoch": "maintenance_epoch",
                "commit_date": "maintenance_date",
                "anchor_epoch": "maintenance_anchor_epoch",
                "anchor_date": "maintenance_anchor_date",
                "subject": "maintenance_subject",
                "intervention_type": "maintenance_type",
                "keyword_basis": "maintenance_keyword_basis",
                "directory_depth_bucket": "maintenance_depth_bucket",
                "pre_loc": "pre_loc",
                "pre_size_bytes": "pre_size_bytes",
                "pre_ci_gzip": "pre_ci_gzip",
                "pre_shannon_entropy": "pre_shannon_entropy",
                "pre_cyclomatic_density": "pre_cyclomatic_density",
                "pre_halstead_volume": "pre_halstead_volume",
                "pre_composite_score": "pre_composite_score",
                "pre_directory_depth": "pre_directory_depth",
                "pre_prior_touches_total": "pre_prior_touches_total",
                "pre_prior_touches_365d": "pre_prior_touches_365d",
                "pre_total_churn": "pre_total_churn",
                "pre_churn_365d": "pre_churn_365d",
                "pre_author_count_total": "pre_author_count_total",
                "pre_author_count_365d": "pre_author_count_365d",
                "pre_file_age_days": "pre_file_age_days",
                "pre_latest_touch_days": "pre_latest_touch_days",
                "post_loc": "post_loc",
                "post_size_bytes": "post_size_bytes",
                "post_ci_gzip": "post_ci_gzip",
                "post_shannon_entropy": "post_shannon_entropy",
                "post_cyclomatic_density": "post_cyclomatic_density",
                "post_halstead_volume": "post_halstead_volume",
                "post_composite_score": "post_composite_score",
                "post_directory_depth": "post_directory_depth",
                "post_prior_touches_total": "post_prior_touches_total",
                "post_prior_touches_365d": "post_prior_touches_365d",
                "post_total_churn": "post_total_churn",
                "post_churn_365d": "post_churn_365d",
                "post_author_count_total": "post_author_count_total",
                "post_author_count_365d": "post_author_count_365d",
                "post_file_age_days": "post_file_age_days",
                "post_latest_touch_days": "post_latest_touch_days",
                "pre_log_loc": "pre_log_loc",
                "pre_log_prior_touches_total": "pre_log_prior_touches_total",
            }
        )
        rows.append(pool)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    return frame.sort_values(["repo", "maintenance_epoch", "maintenance_commit", "file_path"]).reset_index(drop=True)


def build_security_label_lookup(
    security_audit: pd.DataFrame,
    repo_paths: dict[str, Path],
) -> dict[str, dict[str, list[dict[str, object]]]]:
    lookup: dict[str, dict[str, list[dict[str, object]]]] = {}
    fixed_commit_cache: dict[tuple[str, str], pd.Timestamp | None] = {}
    for _, row in security_audit.iterrows():
        repo = str(row["repo"])
        file_path = str(row["file_path"])
        ts = pd.to_datetime(row.get("published_at"), utc=True, errors="coerce")
        if pd.isna(ts):
            repo_path = repo_paths.get(repo)
            fixed_commit = str(row.get("fixed_commit", ""))
            key = (repo, fixed_commit)
            if key not in fixed_commit_cache:
                fixed_commit_cache[key] = (
                    _load_commit_timestamp(repo_path, fixed_commit)
                    if repo_path is not None and fixed_commit
                    else None
                )
            ts = fixed_commit_cache[key]
        if ts is None or pd.isna(ts):
            continue
        lookup.setdefault(repo, {}).setdefault(file_path, []).append(
            {
                "published_at": pd.Timestamp(ts),
                "advisory_id": str(row.get("advisory_id", "")),
            }
        )
    for repo_map in lookup.values():
        for file_path, items in repo_map.items():
            repo_map[file_path] = sorted(items, key=lambda item: item["published_at"])
    return lookup


def lookup_future_security_events(
    lookup: dict[str, dict[str, list[dict[str, object]]]],
    *,
    repo: str,
    file_path: str,
    anchor_date: object,
    horizon_days: int,
) -> tuple[int, str]:
    repo_items = lookup.get(repo, {}).get(file_path, [])
    if not repo_items:
        return 0, ""
    anchor_ts = pd.to_datetime(anchor_date, utc=True, errors="coerce")
    if pd.isna(anchor_ts):
        return 0, ""
    end_ts = anchor_ts + pd.Timedelta(days=horizon_days)
    event_ids = [
        str(item["advisory_id"])
        for item in repo_items
        if anchor_ts < item["published_at"] <= end_ts
    ]
    event_ids = sorted(dict.fromkeys(value for value in event_ids if value))
    return (1 if event_ids else 0), ";".join(event_ids)


def select_best_control_candidate(
    intervention_row: pd.Series,
    maintenance_pool: pd.DataFrame,
    *,
    used_controls: set[tuple[str, str, str]] | None = None,
    windows_days: tuple[int, ...] = (90, 180, 365, 730),
) -> dict[str, object] | None:
    if maintenance_pool.empty:
        return None
    used_controls = used_controls or set()
    repo = str(intervention_row["repo"])
    suffix = str(intervention_row["suffix"])
    depth_bucket = int(intervention_row["directory_depth_bucket"])
    target_log_loc = float(intervention_row["pre_log_loc"])
    target_log_touches = float(intervention_row["pre_log_prior_touches_total"])
    target_epoch = int(intervention_row["commit_epoch"])
    target_file = str(intervention_row["file_path"])

    repo_controls = maintenance_pool.loc[
        (maintenance_pool["repo"] == repo)
        & (maintenance_pool["suffix"] == suffix)
        & (maintenance_pool["maintenance_depth_bucket"] == depth_bucket)
        & (maintenance_pool["file_path"] != target_file)
    ].copy()
    if repo_controls.empty:
        return None

    repo_controls = repo_controls.loc[
        [
            (
                str(row["repo"]),
                str(row["maintenance_commit"]),
                str(row["file_path"]),
            )
            not in used_controls
            for _, row in repo_controls.iterrows()
        ]
    ].copy()
    if repo_controls.empty:
        return None

    repo_controls["date_gap_days"] = (
        (repo_controls["maintenance_epoch"].astype(float) - float(target_epoch)).abs() / 86400.0
    )
    repo_controls["match_distance"] = (
        (repo_controls["pre_log_loc"].astype(float) - target_log_loc).abs()
        + (repo_controls["pre_log_prior_touches_total"].astype(float) - target_log_touches).abs()
    )
    for window in windows_days:
        window_controls = repo_controls.loc[repo_controls["date_gap_days"] <= float(window)].copy()
        if window_controls.empty:
            continue
        window_controls = window_controls.sort_values(
            ["match_distance", "date_gap_days", "maintenance_commit", "file_path"]
        )
        match = window_controls.iloc[0].to_dict()
        match["match_window_days"] = int(window)
        return match
    return None


def build_matched_intervention_pairs(
    interventions: pd.DataFrame,
    maintenance_pool: pd.DataFrame,
) -> pd.DataFrame:
    if interventions.empty or maintenance_pool.empty:
        return pd.DataFrame()
    used_controls: set[tuple[str, str, str]] = set()
    rows: list[dict[str, object]] = []
    for _, intervention in interventions.sort_values(
        ["repo", "commit_epoch", "intervention_commit", "file_path"]
    ).iterrows():
        match = select_best_control_candidate(
            intervention,
            maintenance_pool,
            used_controls=used_controls,
        )
        if match is None:
            continue
        used_controls.add(
            (
                str(match["repo"]),
                str(match["maintenance_commit"]),
                str(match["file_path"]),
            )
        )
        rows.append(
            {
                "intervention_id": str(intervention["intervention_id"]),
                "repo": str(intervention["repo"]),
                "intervention_commit": str(intervention["intervention_commit"]),
                "intervention_file_path": str(intervention["file_path"]),
                "intervention_epoch": int(intervention["commit_epoch"]),
                "maintenance_id": str(match["maintenance_id"]),
                "maintenance_commit": str(match["maintenance_commit"]),
                "maintenance_file_path": str(match["file_path"]),
                "maintenance_epoch": int(match["maintenance_epoch"]),
                "match_window_days": int(match["match_window_days"]),
                "match_distance": float(match["match_distance"]),
                "date_gap_days": float(match["date_gap_days"]),
            }
        )
    return pd.DataFrame(rows)


def _anchor_feature_payload(row: pd.Series | dict[str, object], prefix: str) -> dict[str, object]:
    payload = {
        "loc": float(row[f"{prefix}loc"]),
        "size_bytes": float(row[f"{prefix}size_bytes"]),
        "ci_gzip": float(row[f"{prefix}ci_gzip"]),
        "shannon_entropy": float(row[f"{prefix}shannon_entropy"]),
        "cyclomatic_density": float(row[f"{prefix}cyclomatic_density"]),
        "halstead_volume": float(row[f"{prefix}halstead_volume"]),
        "composite_score": float(row[f"{prefix}composite_score"]),
        "directory_depth": float(row[f"{prefix}directory_depth"]),
        "prior_touches_total": float(row[f"{prefix}prior_touches_total"]),
        "prior_touches_365d": float(row[f"{prefix}prior_touches_365d"]),
        "total_churn": float(row[f"{prefix}total_churn"]),
        "churn_365d": float(row[f"{prefix}churn_365d"]),
        "author_count_total": float(row[f"{prefix}author_count_total"]),
        "author_count_365d": float(row[f"{prefix}author_count_365d"]),
        "file_age_days": float(row[f"{prefix}file_age_days"]),
        "latest_touch_days": float(row[f"{prefix}latest_touch_days"]),
    }
    payload["log_loc"] = float(math.log1p(payload["loc"]))
    payload["log_size_bytes"] = float(math.log1p(payload["size_bytes"]))
    payload["log_prior_touches_total"] = float(math.log1p(payload["prior_touches_total"]))
    payload["log_prior_touches_365d"] = float(math.log1p(payload["prior_touches_365d"]))
    payload["log_total_churn"] = float(math.log1p(payload["total_churn"]))
    payload["log_churn_365d"] = float(math.log1p(payload["churn_365d"]))
    return payload


def build_anchor_observations(
    matched_pairs: pd.DataFrame,
    interventions: pd.DataFrame,
    maintenance_pool: pd.DataFrame,
    *,
    security_lookup: dict[str, dict[str, list[dict[str, object]]]],
    horizon_days: int,
) -> pd.DataFrame:
    if matched_pairs.empty:
        return pd.DataFrame()
    interventions_by_id = {
        str(row["intervention_id"]): row for _, row in interventions.iterrows()
    }
    maintenance_by_id = {
        str(row["maintenance_id"]): row for _, row in maintenance_pool.iterrows()
    }
    rows: list[dict[str, object]] = []
    for _, pair in matched_pairs.iterrows():
        intervention = interventions_by_id.get(str(pair["intervention_id"]))
        maintenance = maintenance_by_id.get(str(pair["maintenance_id"]))
        if intervention is None or maintenance is None:
            continue
        for role, source_row in (("intervention", intervention), ("control", maintenance)):
            row_prefix = "" if role == "intervention" else "maintenance_"
            for period, prefix in (("pre", "pre_"), ("post", "post_")):
                anchor_commit = (
                    str(source_row[f"{row_prefix}anchor_commit"])
                    if row_prefix
                    else str(source_row["anchor_commit"])
                )
                anchor_date = (
                    str(source_row[f"{row_prefix}anchor_date"])
                    if row_prefix
                    else str(source_row["anchor_date"])
                )
                if period == "post":
                    anchor_commit = (
                        str(source_row[f"{row_prefix}commit"])
                        if row_prefix
                        else str(source_row["intervention_commit"])
                    )
                    anchor_date = (
                        str(source_row[f"{row_prefix}date"])
                        if row_prefix
                        else str(source_row["commit_date"])
                    )
                label, future_events = lookup_future_security_events(
                    security_lookup,
                    repo=str(source_row["repo"]),
                    file_path=str(source_row["file_path"]),
                    anchor_date=anchor_date,
                    horizon_days=horizon_days,
                )
                anchor_row = {
                    "intervention_id": str(pair["intervention_id"]),
                    "repo": str(source_row["repo"]),
                    "role": role,
                    "period": period,
                    "anchor_commit": anchor_commit,
                    "anchor_date": anchor_date,
                    "file_path": str(source_row["file_path"]),
                    "suffix": str(source_row["suffix"]),
                    "label": int(label),
                    "future_event_ids": future_events,
                    "match_window_days": int(pair["match_window_days"]),
                    "match_distance": float(pair["match_distance"]),
                }
                anchor_row.update(_anchor_feature_payload(source_row, prefix))
                rows.append(anchor_row)
    return pd.DataFrame(rows)


def relabel_anchor_observations(
    anchor_rows: pd.DataFrame,
    *,
    security_lookup: dict[str, dict[str, list[dict[str, object]]]],
    horizon_days: int,
) -> pd.DataFrame:
    """Recompute future-event labels for saved anchor rows under a new horizon."""
    if anchor_rows.empty:
        return anchor_rows.copy()
    relabelled = anchor_rows.copy()
    labels: list[int] = []
    future_event_ids: list[str] = []
    for row in relabelled.itertuples(index=False):
        label, event_ids = lookup_future_security_events(
            security_lookup,
            repo=str(row.repo),
            file_path=str(row.file_path),
            anchor_date=row.anchor_date,
            horizon_days=horizon_days,
        )
        labels.append(int(label))
        future_event_ids.append(str(event_ids))
    relabelled["label"] = labels
    relabelled["future_event_ids"] = future_event_ids
    return relabelled


def fit_frozen_score_models(
    train_dataset: pd.DataFrame,
    *,
    model_specs: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, dict[str, object]]:
    model_specs = model_specs or PROSPECTIVE_MODEL_SPECS
    fitted: dict[str, dict[str, object]] = {}
    for model_name, spec in model_specs.items():
        feature_cols = spec["numeric"] + spec["categorical"]
        pipeline = _build_pipeline(spec["numeric"], spec["categorical"])
        pipeline.fit(train_dataset[feature_cols], train_dataset["label"])
        fitted[model_name] = {
            "pipeline": pipeline,
            "feature_cols": feature_cols,
        }
    return fitted


def score_anchor_observations(
    anchor_rows: pd.DataFrame,
    fitted_models: dict[str, dict[str, object]],
    *,
    primary_model: str = PRIMARY_INTERVENTION_MODEL,
) -> pd.DataFrame:
    if anchor_rows.empty:
        return anchor_rows
    scored = anchor_rows.copy()
    score_cols: list[str] = []
    for model_name, payload in fitted_models.items():
        feature_cols = list(payload["feature_cols"])
        pipeline = payload["pipeline"]
        score_col = f"score_{model_name}"
        scored[score_col] = pipeline.predict_proba(scored[feature_cols])[:, 1]
        score_cols.append(score_col)
    scored["primary_model_score"] = scored[f"score_{primary_model}"]
    scored["score_range"] = scored[score_cols].max(axis=1) - scored[score_cols].min(axis=1)
    scored["absolute_error"] = (scored["label"].astype(float) - scored["primary_model_score"]).abs()
    scored["underprediction_loss"] = (
        scored["label"].astype(float) * (1.0 - scored["primary_model_score"].astype(float))
    )
    clipped_score = scored["primary_model_score"].astype(float).clip(lower=1e-6, upper=1.0)
    scored["positive_log_loss"] = 0.0
    positive_mask = scored["label"].astype(float) == 1.0
    scored.loc[positive_mask, "positive_log_loss"] = -np.log(clipped_score.loc[positive_mask])
    return scored


def summarise_pairs_to_did(scored_anchor_rows: pd.DataFrame) -> pd.DataFrame:
    if scored_anchor_rows.empty:
        return pd.DataFrame()
    value_cols = [
        metric
        for metric in (
            "composite_score",
            "score_range",
            "absolute_error",
            "underprediction_loss",
            "positive_log_loss",
        )
        if metric in scored_anchor_rows.columns
    ]
    if not value_cols:
        return pd.DataFrame()
    pivot = scored_anchor_rows.pivot_table(
        index=["intervention_id", "repo"],
        columns=["role", "period"],
        values=value_cols,
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()
    pivot.columns = [
        f"{metric}_{role}_{period}" for metric, role, period in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index()
    for metric in value_cols:
        pivot[f"delta_{metric}_intervention"] = (
            pivot[f"{metric}_intervention_post"] - pivot[f"{metric}_intervention_pre"]
        )
        pivot[f"delta_{metric}_control"] = (
            pivot[f"{metric}_control_post"] - pivot[f"{metric}_control_pre"]
        )
        pivot[f"did_{metric}"] = (
            pivot[f"delta_{metric}_intervention"] - pivot[f"delta_{metric}_control"]
        )
    return pivot


def _bootstrap_endpoint(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    n_boot: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    if frame.empty or cluster_col not in frame.columns or value_col not in frame.columns:
        return {}
    grouped: list[np.ndarray] = []
    for _, group in frame.groupby(cluster_col, sort=False):
        values = group[value_col].dropna().to_numpy(dtype=float)
        if len(values):
            grouped.append(values)
    if len(grouped) < 2:
        return {}

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        sample = np.concatenate([grouped[idx] for idx in sampled])
        means[i] = float(np.mean(sample))
    observed = float(frame[value_col].dropna().mean())
    return {
        "cluster_col": cluster_col,
        "value_col": value_col,
        "n_clusters": int(len(grouped)),
        "n_boot": int(n_boot),
        "confidence_level": float(confidence_level),
        "observed_mean": observed,
        "ci_lo": float(np.quantile(means, alpha / 2.0)),
        "ci_hi": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def summarise_intervention_mechanism(
    interventions: pd.DataFrame,
    matched_pairs: pd.DataFrame,
    pair_did: pd.DataFrame,
    *,
    intervention_audit: pd.DataFrame,
    maintenance_pool: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    repo_summary = pd.DataFrame()
    if not pair_did.empty:
        repo_agg: dict[str, tuple[str, str]] = {
            "n_interventions": ("intervention_id", "nunique"),
        }
        for column_name, label in (
            ("did_composite_score", "mean_did_composite"),
            ("did_score_range", "mean_did_score_range"),
            ("did_absolute_error", "mean_did_absolute_error"),
            ("did_underprediction_loss", "mean_did_underprediction_loss"),
            ("did_positive_log_loss", "mean_did_positive_log_loss"),
        ):
            if column_name in pair_did.columns:
                repo_agg[label] = (column_name, "mean")
        repo_summary = (
            pair_did.groupby("repo", as_index=False)
            .agg(**repo_agg)
            .sort_values(["n_interventions", "repo"], ascending=[False, True])
        )

    pooled = {}
    for value_col, summary_key in (
        ("did_composite_score", "delta_composite_did"),
        ("did_score_range", "delta_score_range_did"),
        ("did_absolute_error", "delta_absolute_error_did"),
        ("did_underprediction_loss", "delta_underprediction_loss_did"),
        ("did_positive_log_loss", "delta_positive_log_loss_did"),
    ):
        observed_mean = float("nan")
        if not pair_did.empty and value_col in pair_did.columns:
            observed_mean = float(pair_did[value_col].mean())
        pooled[summary_key] = {
            "observed_mean": observed_mean,
            "bootstrap_by_intervention": _bootstrap_endpoint(
                pair_did,
                cluster_col="intervention_id",
                value_col=value_col,
                n_boot=n_boot,
                seed=seed,
            ),
            "bootstrap_by_repo": _bootstrap_endpoint(
                pair_did,
                cluster_col="repo",
                value_col=value_col,
                n_boot=n_boot,
                seed=seed,
            ),
        }

    def _ci_below_zero(payload: dict[str, object]) -> bool:
        if not payload:
            return False
        return float(payload.get("ci_hi", math.nan)) < 0.0

    full_gate = all(
        _ci_below_zero(pooled[key]["bootstrap_by_intervention"])
        and _ci_below_zero(pooled[key]["bootstrap_by_repo"])
        for key in (
            "delta_composite_did",
            "delta_score_range_did",
            "delta_absolute_error_did",
        )
    )

    summary = {
        "intervention_audit": summarise_intervention_audit_table(intervention_audit),
        "n_interventions_materialised": int(len(interventions)),
        "n_intervention_repos": int(interventions["repo"].nunique()) if not interventions.empty else 0,
        "n_matched_pairs": int(len(matched_pairs)),
        "n_pair_repos": int(pair_did["repo"].nunique()) if not pair_did.empty else 0,
        "maintenance_pool_rows": int(len(maintenance_pool)),
        "n_positive_pair_rows": int(
            pair_did["did_underprediction_loss"].abs().gt(0).sum()
        )
        if not pair_did.empty and "did_underprediction_loss" in pair_did.columns
        else 0,
        "pooled_endpoints": pooled,
        "gating": {
            "accepted_intervention_rows_ge_100": bool(len(intervention_audit) >= 100),
            "accepted_intervention_repos_ge_5": bool(intervention_audit["repo"].nunique() >= 5)
            if not intervention_audit.empty
            else False,
            "django_present": bool(
                not intervention_audit.loc[intervention_audit["repo"] == "django-django"].empty
            ),
            "traefik_present": bool(
                not intervention_audit.loc[intervention_audit["repo"] == "traefik-traefik"].empty
            ),
            "prometheus_present": bool(
                not intervention_audit.loc[
                    intervention_audit["repo"] == "prometheus-prometheus"
                ].empty
            ),
            "bootstrap_composite_present": bool(
                pooled["delta_composite_did"]["bootstrap_by_intervention"]
                and pooled["delta_composite_did"]["bootstrap_by_repo"]
            ),
            "bootstrap_score_range_present": bool(
                pooled["delta_score_range_did"]["bootstrap_by_intervention"]
                and pooled["delta_score_range_did"]["bootstrap_by_repo"]
            ),
            "bootstrap_absolute_error_present": bool(
                pooled["delta_absolute_error_did"]["bootstrap_by_intervention"]
                and pooled["delta_absolute_error_did"]["bootstrap_by_repo"]
            ),
            "bootstrap_underprediction_present": bool(
                pooled["delta_underprediction_loss_did"]["bootstrap_by_intervention"]
                and pooled["delta_underprediction_loss_did"]["bootstrap_by_repo"]
            ),
            "bootstrap_positive_log_loss_present": bool(
                pooled["delta_positive_log_loss_did"]["bootstrap_by_intervention"]
                and pooled["delta_positive_log_loss_did"]["bootstrap_by_repo"]
            ),
            "underprediction_direct_gate": bool(
                _ci_below_zero(pooled["delta_underprediction_loss_did"]["bootstrap_by_intervention"])
                and _ci_below_zero(pooled["delta_underprediction_loss_did"]["bootstrap_by_repo"])
            ),
            "full_cku_confirmation_gate": bool(full_gate),
        },
    }
    return repo_summary, summary
