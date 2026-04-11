"""Audited all-file prospective panel utilities."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from cku_sim.analysis.bootstrap import clustered_delta_bootstrap
from cku_sim.analysis.file_level_case_control import (
    compute_file_opacity_from_text,
    _get_metrics_for_snapshot_file,
    _list_source_files_at_commit,
    _run_git,
    should_include_source_path,
)
from cku_sim.analysis.forward_panel import _to_utc_timestamp
from cku_sim.analysis.prospective_file_panel import (
    PROSPECTIVE_MODEL_SPECS,
    WITHIN_PROJECT_BASELINE_FEATURES,
    _supports_primary_event_label,
)
from cku_sim.core.config import Config, CorpusEntry

logger = logging.getLogger(__name__)

PROMISOR_RETRY_HINTS = (
    "promisor remote",
    "could not fetch",
    "missing blob object",
)

AUDITED_SECURITY_COLUMNS = [
    "repo",
    "snapshot_tag",
    "snapshot_commit",
    "advisory_id",
    "fixed_commit",
    "file_path",
    "published_at",
    "severity_label",
    "source_family",
    "mapping_basis",
    "review_decision",
    "reviewer",
    "notes",
]

AUDITED_BUGFIX_COLUMNS = [
    "repo",
    "snapshot_tag",
    "snapshot_commit",
    "advisory_id",
    "fixed_commit",
    "file_path",
    "published_at",
    "severity_label",
    "source_family",
    "mapping_basis",
    "review_decision",
    "reviewer",
    "notes",
]

PRIMARY_BASELINE_MODEL = "baseline_history_plus_structure"
PRIMARY_PLUS_MODEL = "baseline_plus_composite"
PRIMARY_EXTERNAL_CANDIDATES = [
    "django-django",
    "python-cpython",
    "psf-requests",
    "prometheus-prometheus",
    "go-gitea-gitea",
    "scrapy-scrapy",
    "pallets-flask",
    "fastapi-fastapi",
    "gin-gonic-gin",
    "caddyserver-caddy",
]


def _run_git_bytes(
    repo_path: Path,
    args: list[str],
    *,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    def _invoke(env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            input=stdin,
            capture_output=True,
            env=env,
        )

    env = os.environ.copy()
    env.setdefault("GIT_NO_LAZY_FETCH", "1")
    proc = _invoke(env)
    stderr_text = proc.stderr.decode("utf-8", errors="replace").lower() if proc.stderr else ""
    if proc.returncode == 0 or not any(hint in stderr_text for hint in PROMISOR_RETRY_HINTS):
        return proc

    retry_env = os.environ.copy()
    retry_env.pop("GIT_NO_LAZY_FETCH", None)
    retry = _invoke(retry_env)
    if retry.returncode == 0:
        logger.info(
            "Retried git byte command with promisor lazy fetch for %s",
            repo_path.name,
        )
        return retry
    return retry if retry.stdout else proc


def _normalise_file_path(path: object) -> str:
    return str(path or "").strip()


def _normalise_source_family(source_family: object, mapping_basis: object) -> str:
    if source_family not in {None, "", "nan"}:
        return str(source_family)
    mapping = str(mapping_basis or "")
    if "reference" in mapping:
        return "reference_only"
    if "explicit" in mapping:
        return "explicit_only"
    return ""


def explode_event_catalog_to_audit_rows(
    event_catalog: pd.DataFrame,
    *,
    reviewer: str,
    accept_supported_only: bool = True,
) -> pd.DataFrame:
    """Explode an event catalog into file-level audit rows using changed-source-file mappings."""
    if event_catalog.empty:
        return pd.DataFrame(columns=AUDITED_SECURITY_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in event_catalog.iterrows():
        changed_files = [
            _normalise_file_path(value)
            for value in str(row.get("changed_source_files", "")).split(";")
            if _normalise_file_path(value)
        ]
        if not changed_files:
            continue
        source_family = _normalise_source_family(row.get("source_family"), row.get("source"))
        is_supported = _supports_primary_event_label(row.get("source", ""))
        review_decision = "accept" if (is_supported or not accept_supported_only) else "ambiguous"
        notes = (
            "Retained from supported advisory/event catalog and seeded for researcher review."
            if review_decision == "accept"
            else "Retained for robustness only because the mapping lacks explicit identifier or reference support."
        )
        mapping_basis = str(row.get("source", ""))
        for file_path in changed_files:
            rows.append(
                {
                    "repo": str(row.get("repo", "")),
                    "snapshot_tag": str(row.get("snapshot_tag", "")),
                    "snapshot_commit": str(row.get("snapshot_commit", "")),
                    "advisory_id": str(row.get("event_id", "")),
                    "fixed_commit": str(row.get("fixed_commit", "")),
                    "file_path": file_path,
                    "published_at": str(row.get("published", "")),
                    "severity_label": str(row.get("severity_label", "UNKNOWN")),
                    "source_family": source_family,
                    "mapping_basis": mapping_basis,
                    "review_decision": review_decision,
                    "reviewer": reviewer,
                    "notes": notes,
                }
            )
    audit = pd.DataFrame(rows, columns=AUDITED_SECURITY_COLUMNS)
    if audit.empty:
        return audit
    audit = audit.drop_duplicates(
        subset=["repo", "snapshot_tag", "snapshot_commit", "advisory_id", "fixed_commit", "file_path"]
    ).sort_values(["repo", "snapshot_tag", "advisory_id", "file_path"])
    return audit.reset_index(drop=True)


def seed_security_audit_table(
    event_catalog_paths: list[Path],
    *,
    reviewer: str = "initial_curation",
) -> pd.DataFrame:
    """Seed a combined curated security audit table from one or more event catalogs."""
    frames: list[pd.DataFrame] = []
    for path in event_catalog_paths:
        if not path.exists():
            continue
        catalog = pd.read_csv(path)
        frame = explode_event_catalog_to_audit_rows(catalog, reviewer=reviewer)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=AUDITED_SECURITY_COLUMNS)
    audit = pd.concat(frames, ignore_index=True)
    audit = audit.drop_duplicates(
        subset=["repo", "snapshot_tag", "snapshot_commit", "advisory_id", "fixed_commit", "file_path"]
    )
    return audit.sort_values(["repo", "snapshot_tag", "advisory_id", "file_path"]).reset_index(drop=True)


def load_audited_security_table(
    audit_path: Path,
    *,
    accepted_only: bool = True,
    supported_only: bool = True,
) -> pd.DataFrame:
    """Load the curated security audit table with the primary-policy filter applied."""
    if not audit_path.exists():
        return pd.DataFrame(columns=AUDITED_SECURITY_COLUMNS)
    audit = pd.read_csv(audit_path)
    for column in AUDITED_SECURITY_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""
    audit["file_path"] = audit["file_path"].map(_normalise_file_path)
    audit["source_family"] = audit["source_family"].fillna("")
    audit["review_decision"] = audit["review_decision"].fillna("")
    if accepted_only:
        audit = audit.loc[audit["review_decision"] == "accept"].copy()
    if supported_only:
        audit = audit.loc[
            audit["source_family"].isin(
                {
                    "explicit_only",
                    "reference_only",
                    "explicit_plus_reference",
                    "explicit_plus_range",
                    "reference_plus_range",
                    "explicit_plus_reference_plus_range",
                }
            )
        ].copy()
    audit = audit.drop_duplicates(
        subset=["repo", "snapshot_tag", "snapshot_commit", "advisory_id", "fixed_commit", "file_path"]
    )
    return audit.reset_index(drop=True)


def seed_bugfix_control_audit_table(
    screened_controls: pd.DataFrame,
    *,
    reviewer_default: str = "initial_curation",
) -> pd.DataFrame:
    """Convert the accepted strict negative-control screen into a reusable bug-fix audit table."""
    if screened_controls.empty:
        return pd.DataFrame(columns=AUDITED_BUGFIX_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in screened_controls.iterrows():
        rows.append(
            {
                "repo": str(row.get("repo", "")),
                "snapshot_tag": "",
                "snapshot_commit": "",
                "advisory_id": str(row.get("matched_security_event_id", "")),
                "fixed_commit": str(row.get("bugfix_commit", "")),
                "file_path": str(row.get("bugfix_file", "")),
                "published_at": "",
                "severity_label": "NON_SECURITY_BUGFIX",
                "source_family": "ordinary_bugfix",
                "mapping_basis": "strict_bugfix_screen",
                "review_decision": str(row.get("review_decision", "")) or "accept",
                "reviewer": reviewer_default,
                "notes": str(row.get("review_notes", "")),
            }
        )
    audit = pd.DataFrame(rows, columns=AUDITED_BUGFIX_COLUMNS)
    audit = audit.drop_duplicates(subset=["repo", "fixed_commit", "file_path"])
    return audit.sort_values(["repo", "fixed_commit", "file_path"]).reset_index(drop=True)


def load_bugfix_control_audit_table(
    audit_path: Path,
    *,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Load the curated ordinary bug-fix control audit table."""
    if not audit_path.exists():
        return pd.DataFrame(columns=AUDITED_BUGFIX_COLUMNS)
    audit = pd.read_csv(audit_path)
    for column in AUDITED_BUGFIX_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""
    if accepted_only:
        audit = audit.loc[audit["review_decision"] == "accept"].copy()
    audit["file_path"] = audit["file_path"].map(_normalise_file_path)
    audit = audit.drop_duplicates(subset=["repo", "fixed_commit", "file_path"])
    return audit.reset_index(drop=True)


def summarise_audit_table(audit: pd.DataFrame, *, id_col: str = "advisory_id") -> dict[str, object]:
    """Summarise an audit table for experiment metadata."""
    if audit.empty:
        return {
            "n_rows": 0,
            "n_repos": 0,
            "n_snapshots": 0,
            "n_ids": 0,
            "review_breakdown": {},
            "source_family_breakdown": {},
        }
    return {
        "n_rows": int(len(audit)),
        "n_repos": int(audit["repo"].nunique()),
        "n_snapshots": int(audit["snapshot_tag"].replace("", pd.NA).dropna().nunique()),
        "n_ids": int(audit[id_col].replace("", pd.NA).dropna().nunique()),
        "review_breakdown": audit["review_decision"].value_counts().to_dict(),
        "source_family_breakdown": audit["source_family"].value_counts().to_dict(),
    }


def build_holdout_screen(
    audit: pd.DataFrame,
    *,
    candidate_repos: list[str] | None = None,
    min_snapshots: int = 3,
    min_events: int = 5,
) -> pd.DataFrame:
    """Build a deterministic external-holdout screening table."""
    candidate_repos = candidate_repos or PRIMARY_EXTERNAL_CANDIDATES
    rows: list[dict[str, object]] = []
    for repo in candidate_repos:
        subset = audit.loc[audit["repo"] == repo].copy()
        rows.append(
            {
                "repo": repo,
                "n_rows": int(len(subset)),
                "n_snapshots": int(subset["snapshot_tag"].replace("", pd.NA).dropna().nunique()),
                "n_events": int(subset["advisory_id"].replace("", pd.NA).dropna().nunique()),
                "n_files": int(subset["file_path"].replace("", pd.NA).dropna().nunique()),
                "eligible_holdout": bool(
                    subset["snapshot_tag"].replace("", pd.NA).dropna().nunique() >= min_snapshots
                    and subset["advisory_id"].replace("", pd.NA).dropna().nunique() >= min_events
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["eligible_holdout", "n_events", "n_snapshots"], ascending=[False, False, False])
    return frame.reset_index(drop=True)


def _encode_all_file_row(
    *,
    repo: str,
    snapshot_tag: str,
    snapshot_commit: str,
    snapshot_date: str,
    snapshot_key: str,
    file_path: str,
    label: int,
    metrics,
    history: dict[str, object],
    advisory_ids: list[str],
    fixed_commits: list[str],
    published_values: list[str],
    severity_labels: list[str],
    source_families: list[str],
    mapping_bases: list[str],
) -> dict[str, object]:
    advisory_ids = sorted({str(value) for value in advisory_ids if str(value)})
    fixed_commits = sorted({str(value) for value in fixed_commits if str(value)})
    published_values = sorted({str(value) for value in published_values if str(value)})
    severity_labels = sorted({str(value) for value in severity_labels if str(value)})
    source_families = sorted({str(value) for value in source_families if str(value)})
    mapping_bases = sorted({str(value) for value in mapping_bases if str(value)})
    event_observation_id = (
        f"{snapshot_key}:{'|'.join(advisory_ids)}:{file_path}"
        if advisory_ids
        else f"{snapshot_key}:NEG:{file_path}"
    )
    return {
        "repo": repo,
        "snapshot_tag": snapshot_tag,
        "snapshot_commit": snapshot_commit,
        "snapshot_date": snapshot_date,
        "snapshot_key": snapshot_key,
        "event_observation_id": event_observation_id,
        "label": int(label),
        "file_path": file_path,
        "suffix": Path(file_path).suffix.lower() or "<none>",
        "loc": float(metrics.total_loc),
        "size_bytes": float(metrics.total_bytes),
        "ci_gzip": float(metrics.ci_gzip),
        "shannon_entropy": float(metrics.shannon_entropy),
        "cyclomatic_density": float(metrics.cyclomatic_density),
        "halstead_volume": float(metrics.halstead_volume),
        "composite_score": float(metrics.composite_score),
        "directory_depth": float(history["directory_depth"]),
        "prior_touches_total": float(history["prior_touches_total"]),
        "prior_touches_365d": float(history["prior_touches_365d"]),
        "total_churn": float(history["total_churn"]),
        "churn_365d": float(history["churn_365d"]),
        "author_count_total": float(history["author_count_total"]),
        "author_count_365d": float(history["author_count_365d"]),
        "file_age_days": float(history["file_age_days"]),
        "latest_touch_days": float(history["latest_touch_days"]),
        "advisory_ids": ";".join(advisory_ids),
        "fixed_commits": ";".join(fixed_commits),
        "published_at": ";".join(published_values),
        "severity_labels": ";".join(severity_labels),
        "source_families": ";".join(source_families),
        "mapping_bases": ";".join(mapping_bases),
    }


def _normalise_history_path(path: str) -> str:
    path = str(path).strip()
    if "=>" not in path:
        return path
    if "{" in path and "}" in path:
        prefix, rest = path.split("{", 1)
        inner, suffix = rest.split("}", 1)
        parts = inner.split("=>")
        if len(parts) == 2:
            return f"{prefix}{parts[1].strip()}{suffix}"
    return path.split("=>")[-1].strip()


def _history_pathspecs(source_extensions: list[str]) -> list[str]:
    """Restrict history scans to in-scope source extensions at the git layer."""
    specs: list[str] = []
    for ext in source_extensions:
        ext = str(ext).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        specs.append(f":(glob)**/*{ext}")
    return sorted(set(specs))


def _snapshot_history_index(
    repo_path: Path,
    snapshot_commit: str,
    snapshot_epoch: int,
    source_extensions: list[str],
    history_cache: dict[str, dict[str, dict[str, object]]],
    snapshot_paths: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    if snapshot_commit in history_cache:
        return history_cache[snapshot_commit]

    history_pathspecs = (
        sorted({str(path).strip() for path in (snapshot_paths or []) if str(path).strip()})
        or _history_pathspecs(source_extensions)
    )
    proc = _run_git(
        repo_path,
        [
            "log",
            "--no-renames",
            "--numstat",
            "--format=%x1e%ct%x1f%an",
            snapshot_commit,
            "--",
            *history_pathspecs,
        ],
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        history_cache[snapshot_commit] = {}
        return {}
    if proc.returncode != 0:
        logger.warning(
            "History index for %s at %s completed with git status %s; using available stdout despite stderr: %s",
            repo_path.name,
            snapshot_commit,
            proc.returncode,
            proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
        )

    index: dict[str, dict[str, object]] = {}
    for block in proc.stdout.split("\x1e"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if not lines or "\x1f" not in lines[0]:
            continue
        epoch_s, author = lines[0].split("\x1f", 1)
        try:
            epoch = int(epoch_s)
        except ValueError:
            continue
        within_365 = 0 <= snapshot_epoch - epoch <= 365 * 86400
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            path = _normalise_history_path(parts[2])
            if not should_include_source_path(path, source_extensions):
                continue
            added = int(parts[0]) if parts[0].isdigit() else 0
            deleted = int(parts[1]) if parts[1].isdigit() else 0
            churn = added + deleted
            record = index.setdefault(
                path,
                {
                    "prior_touches_total": 0,
                    "prior_touches_365d": 0,
                    "total_churn": 0,
                    "churn_365d": 0,
                    "author_total": set(),
                    "author_365": set(),
                    "first_epoch": epoch,
                    "last_epoch": epoch,
                },
            )
            record["prior_touches_total"] += 1
            record["total_churn"] += churn
            record["author_total"].add(author)
            record["first_epoch"] = min(int(record["first_epoch"]), epoch)
            record["last_epoch"] = max(int(record["last_epoch"]), epoch)
            if within_365:
                record["prior_touches_365d"] += 1
                record["churn_365d"] += churn
                record["author_365"].add(author)

    final_index: dict[str, dict[str, object]] = {}
    for path, record in index.items():
        first_epoch = int(record["first_epoch"])
        last_epoch = int(record["last_epoch"])
        final_index[path] = {
            "directory_depth": int(max(0, len(Path(path).parts) - 1)),
            "prior_touches_total": int(record["prior_touches_total"]),
            "prior_touches_365d": int(record["prior_touches_365d"]),
            "total_churn": int(record["total_churn"]),
            "churn_365d": int(record["churn_365d"]),
            "author_count_total": int(len(record["author_total"])),
            "author_count_365d": int(len(record["author_365"])),
            "file_age_days": float((snapshot_epoch - first_epoch) / 86400.0)
            if first_epoch
            else float("nan"),
            "latest_touch_days": float((snapshot_epoch - last_epoch) / 86400.0)
            if last_epoch
            else float("nan"),
        }
    history_cache[snapshot_commit] = final_index
    return final_index


def _lookup_snapshot_history_features(
    history_index: dict[str, dict[str, object]],
    file_path: str,
) -> dict[str, object]:
    if file_path in history_index:
        return dict(history_index[file_path])
    return {
        "directory_depth": int(max(0, len(Path(file_path).parts) - 1)),
        "prior_touches_total": 0,
        "prior_touches_365d": 0,
        "total_churn": 0,
        "churn_365d": 0,
        "author_count_total": 0,
        "author_count_365d": 0,
        "file_age_days": float("nan"),
        "latest_touch_days": float("nan"),
    }


def _load_commit_timestamp(repo_path: Path, commit: str) -> pd.Timestamp | None:
    proc = _run_git(repo_path, ["show", "-s", "--format=%cI", commit])
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    try:
        return _to_utc_timestamp(text)
    except Exception:
        return None


def _audited_snapshots_for_repo(
    repo_path: Path,
    repo_audit: pd.DataFrame,
    *,
    min_snapshot_date: pd.Timestamp,
    max_snapshot_date: pd.Timestamp,
    max_tags: int,
    min_tag_gap_days: int,
) -> list[dict[str, object]]:
    """Return audited snapshot entries, preferring exact audited tags over generic sampling."""
    if repo_audit.empty:
        return []

    snapshots: list[dict[str, object]] = []
    grouped = (
        repo_audit[["snapshot_tag", "snapshot_commit"]]
        .drop_duplicates()
        .sort_values(["snapshot_tag", "snapshot_commit"])
    )
    for _, row in grouped.iterrows():
        snapshot_tag = str(row["snapshot_tag"])
        snapshot_commit = str(row["snapshot_commit"])
        if not snapshot_tag or not snapshot_commit:
            continue
        snapshot_date = _load_commit_timestamp(repo_path, snapshot_commit)
        if snapshot_date is None:
            continue
        if snapshot_date < min_snapshot_date or snapshot_date > max_snapshot_date:
            continue
        snapshots.append(
            {
                "tag": snapshot_tag,
                "commit": snapshot_commit,
                "date": snapshot_date.isoformat(),
            }
        )

    if not snapshots:
        return []

    snapshots.sort(key=lambda item: _to_utc_timestamp(item["date"]))
    filtered: list[dict[str, object]] = []
    min_gap = pd.Timedelta(days=min_tag_gap_days)
    for snapshot in snapshots:
        snapshot_date = _to_utc_timestamp(snapshot["date"])
        if filtered and snapshot_date - _to_utc_timestamp(filtered[-1]["date"]) < min_gap:
            continue
        filtered.append(snapshot)

    if max_tags and len(filtered) > max_tags:
        filtered = filtered[-max_tags:]
    return filtered


def _batch_fetch_blob_texts(
    repo_path: Path,
    blob_ids: list[str],
) -> dict[str, str]:
    """Fetch blob contents in one Git batch call keyed by blob SHA."""
    unique_blob_ids = [blob_id for blob_id in dict.fromkeys(blob_ids) if blob_id]
    if not unique_blob_ids:
        return {}

    proc = _run_git_bytes(
        repo_path,
        ["cat-file", "--batch"],
        stdin="".join(f"{blob_id}\n" for blob_id in unique_blob_ids).encode("utf-8"),
    )
    if proc.returncode != 0 and not proc.stdout:
        logger.warning(
            "Blob batch fetch failed for %s: %s",
            repo_path.name,
            proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1]
            if proc.stderr
            else "",
        )
        return {}

    output = proc.stdout
    cursor = 0
    texts: dict[str, str] = {}
    while cursor < len(output):
        line_end = output.find(b"\n", cursor)
        if line_end < 0:
            break
        header = output[cursor:line_end].decode("utf-8", errors="replace").strip()
        cursor = line_end + 1
        if not header:
            continue
        parts = header.split()
        if len(parts) >= 2 and parts[1] == "missing":
            continue
        if len(parts) < 3:
            continue
        blob_id, object_type, size_str = parts[0], parts[1], parts[2]
        if object_type != "blob":
            continue
        try:
            size = int(size_str)
        except ValueError:
            continue
        blob = output[cursor : cursor + size]
        cursor += size
        if cursor < len(output) and output[cursor : cursor + 1] == b"\n":
            cursor += 1
        texts[blob_id] = blob.decode("utf-8", errors="replace")
    return texts


def build_audited_all_file_panel(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    config: Config,
    security_audit: pd.DataFrame,
    *,
    repos: list[str] | None = None,
    max_tags: int = 5,
    min_tag_gap_days: int = 365,
    horizon_days: int = 730,
    lookback_years: int = 10,
    min_loc: int = 5,
) -> pd.DataFrame:
    """Build an audited all-file prospective panel for all eligible files at sampled snapshots."""
    if security_audit.empty:
        return pd.DataFrame()

    allowed_repos = set(repos or [])
    analysis_date = pd.Timestamp.now(tz="UTC")
    max_snapshot_date = analysis_date - pd.Timedelta(days=horizon_days)
    min_snapshot_date = max_snapshot_date - pd.Timedelta(days=365 * lookback_years)
    rows: list[dict[str, object]] = []

    for entry in corpus:
        if allowed_repos and entry.name not in allowed_repos:
            continue
        repo_path = repo_paths.get(entry.name)
        if repo_path is None or not repo_path.exists():
            continue

        repo_audit = security_audit.loc[security_audit["repo"] == entry.name].copy()
        if repo_audit.empty:
            continue

        snapshots = _audited_snapshots_for_repo(
            repo_path,
            repo_audit,
            min_snapshot_date=min_snapshot_date,
            max_snapshot_date=max_snapshot_date,
            max_tags=max_tags,
            min_tag_gap_days=min_tag_gap_days,
        )
        if not snapshots:
            continue
        logger.info(
            "Building audited all-file panel for %s across %d sampled snapshots",
            entry.name,
            len(snapshots),
        )

        blob_metrics_cache: dict[str, object | None] = {}
        history_cache: dict[str, dict[str, dict[str, object]]] = {}
        for snapshot in snapshots:
            snapshot_tag = str(snapshot["tag"])
            snapshot_commit = str(snapshot["commit"])
            snapshot_date = _to_utc_timestamp(snapshot["date"])
            snapshot_epoch = int(snapshot_date.timestamp())
            snapshot_key = f"{entry.name}:{snapshot_tag}"

            snapshot_audit = repo_audit.loc[
                (repo_audit["snapshot_tag"] == snapshot_tag)
                & (repo_audit["snapshot_commit"] == snapshot_commit)
            ].copy()
            if snapshot_audit.empty:
                continue
            logger.info(
                "Snapshot %s@%s: %d audited positive file mappings",
                entry.name,
                snapshot_tag,
                len(snapshot_audit),
            )

            positives = (
                snapshot_audit.groupby("file_path", as_index=False)
                .agg(
                    advisory_ids=("advisory_id", lambda s: sorted({str(v) for v in s if str(v)})),
                    fixed_commits=("fixed_commit", lambda s: sorted({str(v) for v in s if str(v)})),
                    published_at=("published_at", lambda s: sorted({str(v) for v in s if str(v)})),
                    severity_labels=("severity_label", lambda s: sorted({str(v) for v in s if str(v)})),
                    source_families=("source_family", lambda s: sorted({str(v) for v in s if str(v)})),
                    mapping_bases=("mapping_basis", lambda s: sorted({str(v) for v in s if str(v)})),
                )
            )
            positive_map = {
                str(row["file_path"]): row
                for _, row in positives.iterrows()
            }

            files_at_snapshot = _list_source_files_at_commit(
                repo_path,
                snapshot_commit,
                entry.source_extensions,
            )
            if not files_at_snapshot:
                continue
            missing_blob_ids = sorted(
                {
                    str(metadata.get("blob_sha", ""))
                    for metadata in files_at_snapshot.values()
                    if str(metadata.get("blob_sha", "")) and str(metadata.get("blob_sha", "")) not in blob_metrics_cache
                }
            )
            logger.info(
                "Snapshot %s@%s: %d eligible source files, %d uncached blobs",
                entry.name,
                snapshot_tag,
                len(files_at_snapshot),
                len(missing_blob_ids),
            )
            if missing_blob_ids:
                for blob_id, text in _batch_fetch_blob_texts(repo_path, missing_blob_ids).items():
                    blob_metrics_cache[blob_id] = compute_file_opacity_from_text(
                        text,
                        name=blob_id,
                        snapshot_id=blob_id,
                    )
                for blob_id in missing_blob_ids:
                    blob_metrics_cache.setdefault(blob_id, None)
            history_index = _snapshot_history_index(
                repo_path,
                snapshot_commit,
                snapshot_epoch,
                entry.source_extensions,
                history_cache,
                snapshot_paths=sorted(files_at_snapshot),
            )
            for metadata in files_at_snapshot.values():
                metadata["directory_depth"] = max(0, len(Path(str(metadata["path"])).parts) - 1)

            for file_path, metadata in sorted(files_at_snapshot.items()):
                blob_id = str(metadata.get("blob_sha", ""))
                metrics = blob_metrics_cache.get(blob_id)
                if not blob_id:
                    metrics = _get_metrics_for_snapshot_file(
                        repo_path,
                        snapshot_commit,
                        file_path,
                        {},
                    )
                if metrics is None or metrics.total_loc < min_loc:
                    continue
                history = _lookup_snapshot_history_features(history_index, file_path)
                positive = positive_map.get(file_path)
                rows.append(
                    _encode_all_file_row(
                        repo=entry.name,
                        snapshot_tag=snapshot_tag,
                        snapshot_commit=snapshot_commit,
                        snapshot_date=snapshot_date.isoformat(),
                        snapshot_key=snapshot_key,
                        file_path=file_path,
                        label=1 if positive is not None else 0,
                        metrics=metrics,
                        history=history,
                        advisory_ids=[] if positive is None else list(positive["advisory_ids"]),
                        fixed_commits=[] if positive is None else list(positive["fixed_commits"]),
                        published_values=[] if positive is None else list(positive["published_at"]),
                        severity_labels=[] if positive is None else list(positive["severity_labels"]),
                        source_families=[] if positive is None else list(positive["source_families"]),
                        mapping_bases=[] if positive is None else list(positive["mapping_bases"]),
                    )
                )
    dataset = pd.DataFrame(rows)
    if dataset.empty:
        return dataset
    dataset["log_loc"] = np.log1p(dataset["loc"])
    dataset["log_size_bytes"] = np.log1p(dataset["size_bytes"])
    dataset["log_prior_touches_total"] = np.log1p(dataset["prior_touches_total"])
    dataset["log_prior_touches_365d"] = np.log1p(dataset["prior_touches_365d"])
    dataset["log_total_churn"] = np.log1p(dataset["total_churn"])
    dataset["log_churn_365d"] = np.log1p(dataset["churn_365d"])
    return dataset.sort_values(["repo", "snapshot_tag", "file_path"]).reset_index(drop=True)


def summarise_all_file_panel(dataset: pd.DataFrame) -> dict[str, object]:
    """Summarise the audited all-file prospective panel."""
    if dataset.empty:
        return {
            "n_files": 0,
            "n_positive_files": 0,
            "n_positive_events": 0,
            "n_repos": 0,
            "n_snapshots": 0,
        }
    positives = dataset.loc[dataset["label"] == 1].copy()
    return {
        "n_files": int(len(dataset)),
        "n_positive_files": int(len(positives)),
        "n_negative_files": int((dataset["label"] == 0).sum()),
        "positive_rate": float(dataset["label"].mean()),
        "n_positive_events": int(
            positives["advisory_ids"]
            .str.split(";")
            .explode()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        ),
        "n_repos": int(dataset["repo"].nunique()),
        "n_snapshots": int(dataset["snapshot_key"].nunique()),
        "source_family_breakdown": (
            positives["source_families"]
            .str.split(";")
            .explode()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        ),
    }


def summarise_repo_panel(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarise all-file dataset coverage by repository."""
    if dataset.empty:
        return pd.DataFrame()
    return (
        dataset.groupby("repo", as_index=False)
        .agg(
            n_files=("file_path", "size"),
            n_positive_files=("label", "sum"),
            n_snapshots=("snapshot_key", "nunique"),
            positive_rate=("label", "mean"),
        )
        .sort_values(["n_positive_files", "n_files"], ascending=False)
    )


def evaluate_all_file_leave_one_repo_out(
    dataset: pd.DataFrame,
    model_specs: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Leave-one-repo-out predictive validation for all-file audited panels."""
    from sklearn.model_selection import LeaveOneGroupOut
    from cku_sim.analysis.prospective_file_panel import _build_pipeline, _score_predictions

    model_specs = model_specs or PROSPECTIVE_MODEL_SPECS
    if dataset.empty or dataset["repo"].nunique() < 2 or dataset["label"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    logo = LeaveOneGroupOut()
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for train_idx, test_idx in logo.split(dataset, dataset["label"], groups=dataset["repo"]):
        train = dataset.iloc[train_idx].copy()
        test = dataset.iloc[test_idx].copy()
        held_out_repo = str(test["repo"].iloc[0])
        for model_name, spec in model_specs.items():
            feature_cols = spec["numeric"] + spec["categorical"]
            pipeline = _build_pipeline(spec["numeric"], spec["categorical"])
            pipeline.fit(train[feature_cols], train["label"])
            y_score = pipeline.predict_proba(test[feature_cols])[:, 1]
            fold_pred = test[
                ["repo", "snapshot_tag", "snapshot_key", "event_observation_id", "label", "file_path"]
            ].copy()
            fold_pred["model"] = model_name
            fold_pred["score"] = y_score
            prediction_frames.append(fold_pred)
            metrics = _score_predictions(test["label"], y_score)
            metrics["model"] = model_name
            metrics["held_out_repo"] = held_out_repo
            metrics["n_files"] = int(len(test))
            metrics["n_positive_files"] = int(test["label"].sum())
            fold_rows.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    fold_metrics = pd.DataFrame(fold_rows)
    summary: dict[str, object] = {
        "n_files": int(len(dataset)),
        "n_positive_files": int(dataset["label"].sum()),
        "n_repos": int(dataset["repo"].nunique()),
        "n_snapshots": int(dataset["snapshot_key"].nunique()),
        "models": {},
    }
    if predictions.empty:
        return predictions, fold_metrics, summary

    from cku_sim.analysis.prospective_file_panel import _score_predictions

    for model_name in model_specs:
        model_preds = predictions.loc[predictions["model"] == model_name].copy()
        model_folds = fold_metrics.loc[fold_metrics["model"] == model_name].copy()
        overall = _score_predictions(model_preds["label"], model_preds["score"].to_numpy())
        overall["macro_roc_auc"] = float(model_folds["roc_auc"].mean())
        overall["macro_average_precision"] = float(model_folds["average_precision"].mean())
        overall["macro_brier_score"] = float(model_folds["brier_score"].mean())
        summary["models"][model_name] = overall
    return predictions, fold_metrics, summary


def evaluate_all_file_external_holdout(
    train_dataset: pd.DataFrame,
    holdout_dataset: pd.DataFrame,
    model_specs: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Frozen external validation for audited all-file panels."""
    from cku_sim.analysis.prospective_file_panel import _build_pipeline, _score_predictions

    model_specs = model_specs or PROSPECTIVE_MODEL_SPECS
    summary: dict[str, object] = {
        "n_train_files": int(len(train_dataset)),
        "n_train_positive_files": int(train_dataset["label"].sum()) if not train_dataset.empty else 0,
        "n_train_repos": int(train_dataset["repo"].nunique()) if not train_dataset.empty else 0,
        "n_holdout_files": int(len(holdout_dataset)),
        "n_holdout_positive_files": int(holdout_dataset["label"].sum()) if not holdout_dataset.empty else 0,
        "n_holdout_repos": int(holdout_dataset["repo"].nunique()) if not holdout_dataset.empty else 0,
        "models": {},
    }
    if train_dataset.empty or holdout_dataset.empty:
        return pd.DataFrame(), pd.DataFrame(), summary

    prediction_frames: list[pd.DataFrame] = []
    repo_rows: list[dict[str, object]] = []
    for model_name, spec in model_specs.items():
        feature_cols = spec["numeric"] + spec["categorical"]
        pipeline = _build_pipeline(spec["numeric"], spec["categorical"])
        pipeline.fit(train_dataset[feature_cols], train_dataset["label"])
        y_score = pipeline.predict_proba(holdout_dataset[feature_cols])[:, 1]

        pred = holdout_dataset[
            ["repo", "snapshot_tag", "snapshot_key", "event_observation_id", "label", "file_path"]
        ].copy()
        pred["model"] = model_name
        pred["score"] = y_score
        prediction_frames.append(pred)
        summary["models"][model_name] = _score_predictions(holdout_dataset["label"], y_score)
        for repo, group in pred.groupby("repo"):
            metrics = _score_predictions(group["label"], group["score"].to_numpy())
            metrics["repo"] = str(repo)
            metrics["model"] = model_name
            metrics["n_files"] = int(len(group))
            metrics["n_positive_files"] = int(group["label"].sum())
            repo_rows.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    repo_metrics = pd.DataFrame(repo_rows)
    return predictions, repo_metrics, summary


def fit_repo_fixed_effect_models_all_file(dataset: pd.DataFrame) -> dict[str, object]:
    """Repository fixed-effects models for the audited all-file panel."""
    if dataset.empty or dataset["repo"].nunique() < 2 or dataset["label"].nunique() < 2:
        return {}
    numeric_cols = WITHIN_PROJECT_BASELINE_FEATURES + ["cyclomatic_density", "halstead_volume", "composite_score"]
    work = dataset.copy()
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in numeric_cols:
        if work[col].dropna().empty:
            work[col] = 0.0
        else:
            work[col] = work[col].fillna(work[col].median())
    repo_dummies = pd.get_dummies(work["repo"], prefix="repo", drop_first=True)
    y = work["label"].astype(float)
    groups = work["event_observation_id"]

    baseline_features = WITHIN_PROJECT_BASELINE_FEATURES + ["cyclomatic_density", "halstead_volume"]
    x_base = pd.concat([work[baseline_features], repo_dummies], axis=1)
    x_base = x_base.loc[:, x_base.nunique(dropna=False) > 1]
    x_base = sm.add_constant(x_base.astype(float), has_constant="add")
    x_plus = pd.concat([x_base, work[["composite_score"]]], axis=1)

    try:
        base_model = sm.GLM(y, x_base, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
        plus_model = sm.GLM(y, x_plus, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
    except Exception as exc:
        logger.warning("All-file repo fixed-effects models failed: %s", exc)
        return {}

    lr_stat = max(0.0, 2.0 * (plus_model.llf - base_model.llf))
    lr_pvalue = float(stats.chi2.sf(lr_stat, df=1))
    conf = plus_model.conf_int().loc["composite_score"]
    return {
        "baseline_history_plus_structure_repo_fixed_effects": {
            "log_likelihood": float(base_model.llf),
            "aic": float(base_model.aic),
        },
        "baseline_plus_composite_repo_fixed_effects": {
            "log_likelihood": float(plus_model.llf),
            "aic": float(plus_model.aic),
            "composite_coef": float(plus_model.params["composite_score"]),
            "composite_pvalue": float(plus_model.pvalues["composite_score"]),
            "composite_ci_lo": float(conf[0]),
            "composite_ci_hi": float(conf[1]),
            "lr_statistic_vs_baseline": float(lr_stat),
            "lr_pvalue_vs_baseline": lr_pvalue,
        },
    }


def summarise_external_replication(summary: dict[str, object]) -> dict[str, object]:
    """Extract the frozen primary-model comparison for paper-facing reporting."""
    models = summary.get("models", {})
    baseline = models.get(PRIMARY_BASELINE_MODEL, {})
    plus = models.get(PRIMARY_PLUS_MODEL, {})
    if not baseline or not plus:
        return {}
    return {
        "primary_baseline_model": PRIMARY_BASELINE_MODEL,
        "primary_plus_model": PRIMARY_PLUS_MODEL,
        "roc_auc_lift": float(plus["roc_auc"] - baseline["roc_auc"]),
        "average_precision_lift": float(plus["average_precision"] - baseline["average_precision"]),
        "brier_lift": float(plus["brier_score"] - baseline["brier_score"]),
    }


def load_repo_paths(config: Config, corpus: list[CorpusEntry]) -> dict[str, Path]:
    """Resolve available local repo paths for a corpus."""
    return {
        entry.name: config.raw_dir / entry.name
        for entry in corpus
        if (config.raw_dir / entry.name).exists()
    }


def split_corpora_for_external_replication(
    train_corpus: list[CorpusEntry],
    holdout_corpus: list[CorpusEntry],
    *,
    candidate_repos: list[str] | None = None,
) -> tuple[list[CorpusEntry], list[CorpusEntry]]:
    """Freeze a disjoint train/holdout split for audited external replication."""
    candidate_set = set(candidate_repos or PRIMARY_EXTERNAL_CANDIDATES)
    filtered_train = [entry for entry in train_corpus if entry.name not in candidate_set]
    filtered_holdout = [entry for entry in holdout_corpus if entry.name in candidate_set]
    train_names = {entry.name for entry in filtered_train}
    filtered_holdout = [entry for entry in filtered_holdout if entry.name not in train_names]
    return filtered_train, filtered_holdout
