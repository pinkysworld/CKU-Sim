"""Prospective matched file-level panel analysis for later security-fix involvement."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cku_sim.analysis.bootstrap import clustered_delta_bootstrap
from cku_sim.analysis.file_level_case_control import (
    _extract_explicit_id_event_candidates,
    _extract_nvd_event_candidates,
    _get_commit_epoch,
    _get_first_parent,
    _get_metrics_for_snapshot_file,
    _list_changed_source_files,
    _list_source_files_at_commit,
    _run_git,
    _select_primary_event_commit,
    _canonicalise_event_candidate_map,
    cache_file_for_cpe,
    compute_file_opacity_from_text,
    describe_ground_truth_policy,
    normalise_github_slug,
)
from cku_sim.analysis.forward_panel import _future_events_for_snapshot, _to_utc_timestamp
from cku_sim.analysis.predictive_validation import pairwise_accuracy
from cku_sim.collectors.osv_collector import (
    build_osv_alias_map,
    extract_osv_event_candidates,
    fetch_osv_records_for_repo,
    repo_url_variants,
)
from cku_sim.core.config import Config, CorpusEntry
from cku_sim.core.opacity import StructuralOpacity

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    "UNKNOWN": 0,
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}
CVSS_V3_RE = re.compile(r"^CVSS:3\.[01]/")

PROSPECTIVE_GROUND_TRUTH_METADATA = {
    "expanded_advisory_plus_explicit": {
        "label": "Expanded advisory plus explicit-ID future events",
        "description": (
            "Future security events combine NVD-linked fixing commits, OSV-linked advisories, "
            "and locally explicit CVE/GHSA-tagged fixes; external publication timestamps are "
            "used when available and otherwise the fixing-commit date is used."
        ),
    }
}

PROSPECTIVE_MODEL_SPECS = {
    "baseline_size": {
        "numeric": ["log_loc", "log_size_bytes", "directory_depth"],
        "categorical": ["suffix"],
    },
    "baseline_history": {
        "numeric": [
            "log_loc",
            "log_size_bytes",
            "directory_depth",
            "log_prior_touches_total",
            "log_prior_touches_365d",
            "log_total_churn",
            "log_churn_365d",
            "author_count_total",
            "author_count_365d",
            "file_age_days",
        ],
        "categorical": ["suffix"],
    },
    "baseline_history_plus_structure": {
        "numeric": [
            "log_loc",
            "log_size_bytes",
            "directory_depth",
            "log_prior_touches_total",
            "log_prior_touches_365d",
            "log_total_churn",
            "log_churn_365d",
            "author_count_total",
            "author_count_365d",
            "file_age_days",
            "cyclomatic_density",
            "halstead_volume",
        ],
        "categorical": ["suffix"],
    },
    "baseline_plus_composite": {
        "numeric": [
            "log_loc",
            "log_size_bytes",
            "directory_depth",
            "log_prior_touches_total",
            "log_prior_touches_365d",
            "log_total_churn",
            "log_churn_365d",
            "author_count_total",
            "author_count_365d",
            "file_age_days",
            "composite_score",
        ],
        "categorical": ["suffix"],
    },
}

WITHIN_PROJECT_BASELINE_FEATURES = [
    "log_loc",
    "log_size_bytes",
    "directory_depth",
    "file_age_days",
    "log_prior_touches_total",
]


def prospective_policy_metadata(policy: str) -> dict[str, str]:
    if policy not in PROSPECTIVE_GROUND_TRUTH_METADATA:
        raise ValueError(f"Unknown prospective ground-truth policy: {policy}")
    return dict(PROSPECTIVE_GROUND_TRUTH_METADATA[policy])


def _safe_timestamp(value: object | None) -> pd.Timestamp | None:
    if value in {None, "", "nan"}:
        return None
    try:
        return _to_utc_timestamp(value)
    except Exception:
        return None


def _load_nvd_items(entry: CorpusEntry, cache_dir: Path) -> list[dict]:
    if not entry.cpe_id:
        return []
    cache_file = cache_file_for_cpe(entry.cpe_id, cache_dir)
    if not cache_file.exists():
        return []
    try:
        return json.loads(cache_file.read_text())
    except Exception as exc:
        logger.warning("Could not read NVD cache for %s: %s", entry.name, exc)
        return []


def _extract_nvd_published_epochs(vuln_items: list[dict]) -> dict[str, int]:
    published_by_event: dict[str, int] = {}
    for item in vuln_items:
        cve = item.get("cve", {})
        event_id = str(cve.get("id", "")).upper()
        if not event_id:
            continue
        published = _safe_timestamp(cve.get("published"))
        if published is None:
            continue
        published_by_event[event_id] = int(published.timestamp())
    return published_by_event


def _normalise_severity_label(value: object | None) -> str | None:
    if value is None:
        return None
    label = str(value).strip().upper()
    if not label:
        return None
    if label == "MODERATE":
        return "MEDIUM"
    if label in SEVERITY_ORDER:
        return label
    return None


def _severity_label_from_score(score: float | None) -> str | None:
    if score is None or math.isnan(score):
        return None
    if score == 0.0:
        return "NONE"
    if score < 4.0:
        return "LOW"
    if score < 7.0:
        return "MEDIUM"
    if score < 9.0:
        return "HIGH"
    return "CRITICAL"


def _cvss_round_up(value: float) -> float:
    return math.ceil(value * 10.0 - 1e-10) / 10.0


def _cvss_v3_score_from_vector(vector: str) -> float | None:
    vector = vector.strip()
    if not CVSS_V3_RE.match(vector):
        return None

    parts = {}
    for item in vector.split("/")[1:]:
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        parts[key] = value
    required = {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}
    if not required.issubset(parts):
        return None

    av = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
    ac = {"L": 0.77, "H": 0.44}
    ui = {"N": 0.85, "R": 0.62}
    cia = {"H": 0.56, "L": 0.22, "N": 0.0}
    pr_u = {"N": 0.85, "L": 0.62, "H": 0.27}
    pr_c = {"N": 0.85, "L": 0.68, "H": 0.50}

    scope = parts["S"]
    if scope not in {"U", "C"}:
        return None

    try:
        av_score = av[parts["AV"]]
        ac_score = ac[parts["AC"]]
        ui_score = ui[parts["UI"]]
        c_score = cia[parts["C"]]
        i_score = cia[parts["I"]]
        a_score = cia[parts["A"]]
        pr_score = (pr_c if scope == "C" else pr_u)[parts["PR"]]
    except KeyError:
        return None

    iss = 1.0 - ((1.0 - c_score) * (1.0 - i_score) * (1.0 - a_score))
    if scope == "U":
        impact = 6.42 * iss
    else:
        impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)
    exploitability = 8.22 * av_score * ac_score * pr_score * ui_score
    if impact <= 0:
        return 0.0
    if scope == "U":
        return _cvss_round_up(min(impact + exploitability, 10.0))
    return _cvss_round_up(min(1.08 * (impact + exploitability), 10.0))


def _extract_nvd_severity(vuln_items: list[dict]) -> dict[str, dict[str, object]]:
    severity_by_event: dict[str, dict[str, object]] = {}
    metric_order = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")
    for item in vuln_items:
        cve = item.get("cve", {})
        event_id = str(cve.get("id", "")).upper()
        if not event_id:
            continue
        metrics = cve.get("metrics", {})
        selected: dict | None = None
        for metric_key in metric_order:
            candidates = metrics.get(metric_key, [])
            if not candidates:
                continue
            primary = next(
                (entry for entry in candidates if str(entry.get("type", "")).upper() == "PRIMARY"),
                None,
            )
            selected = primary or candidates[0]
            if selected is not None:
                break
        if selected is None:
            continue
        cvss_data = selected.get("cvssData", {})
        score = cvss_data.get("baseScore")
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = math.nan
        label = _normalise_severity_label(selected.get("baseSeverity")) or _severity_label_from_score(
            score_value
        )
        severity_by_event[event_id] = {
            "score": score_value,
            "label": label or "UNKNOWN",
            "source": "nvd_metrics",
        }
    return severity_by_event


def _extract_osv_record_severity(record: dict) -> dict[str, object]:
    scores: list[float] = []
    labels: list[str] = []

    for item in record.get("severity", []) or []:
        score_value = item.get("score")
        if isinstance(score_value, (int, float)):
            scores.append(float(score_value))
            continue
        if isinstance(score_value, str):
            try:
                scores.append(float(score_value))
                continue
            except ValueError:
                parsed = _cvss_v3_score_from_vector(score_value)
                if parsed is not None:
                    scores.append(parsed)

    top_db_label = _normalise_severity_label((record.get("database_specific") or {}).get("severity"))
    if top_db_label:
        labels.append(top_db_label)
    for affected in record.get("affected", []) or []:
        label = _normalise_severity_label((affected.get("database_specific") or {}).get("severity"))
        if label:
            labels.append(label)

    score = max(scores) if scores else math.nan
    label = None
    if labels:
        label = max(labels, key=lambda item: SEVERITY_ORDER.get(item, -1))
    if label is None:
        label = _severity_label_from_score(score)

    return {
        "score": score,
        "label": label or "UNKNOWN",
        "source": "osv_severity",
    }


def _is_high_critical_event(severity_score: object, severity_label: object) -> bool:
    try:
        score = float(severity_score)
    except (TypeError, ValueError):
        score = math.nan
    label = _normalise_severity_label(severity_label) or "UNKNOWN"
    return bool((not math.isnan(score) and score >= 7.0) or label in {"HIGH", "CRITICAL"})


def _build_expanded_future_events(
    repo_path: Path,
    entry: CorpusEntry,
    *,
    nvd_cache_dir: Path,
    osv_cache_dir: Path,
    osv_rate_limit: float,
    osv_query_batch_size: int,
) -> pd.DataFrame:
    expected_slug = normalise_github_slug(entry.git_url)
    vuln_items = _load_nvd_items(entry, nvd_cache_dir)
    nvd_event_candidates = _extract_nvd_event_candidates(vuln_items, expected_slug)
    nvd_published_epochs = _extract_nvd_published_epochs(vuln_items)
    nvd_severity_by_event = _extract_nvd_severity(vuln_items)
    explicit_event_candidates = _extract_explicit_id_event_candidates(repo_path)

    try:
        osv_records = fetch_osv_records_for_repo(
            repo_path,
            entry.git_url,
            osv_cache_dir,
            rate_limit=osv_rate_limit,
            batch_size=osv_query_batch_size,
        )
    except Exception as exc:  # pragma: no cover - network failures are environment-specific
        logger.warning("OSV fetch failed for %s: %s", entry.name, exc)
        osv_records = []

    alias_map = build_osv_alias_map(osv_records)
    repo_urls = repo_url_variants(entry.git_url)
    osv_event_candidates = extract_osv_event_candidates(osv_records, repo_urls)
    nvd_event_candidates = _canonicalise_event_candidate_map(nvd_event_candidates, alias_map)
    explicit_event_candidates = _canonicalise_event_candidate_map(
        explicit_event_candidates,
        alias_map,
    )
    nvd_published_epochs = {
        alias_map.get(event_id, event_id): epoch
        for event_id, epoch in nvd_published_epochs.items()
    }

    event_state: dict[str, dict[str, object]] = {}

    def ensure(event_id: str) -> dict[str, object]:
        return event_state.setdefault(
            event_id,
            {
                "candidate_commits": set(),
                "sources": set(),
                "aliases": {event_id},
                "published_epochs": [],
                "event_date_sources": set(),
                "severity_scores": [],
                "severity_labels": set(),
                "severity_sources": set(),
            },
        )

    for event_id, commits in nvd_event_candidates.items():
        state = ensure(event_id)
        state["candidate_commits"].update(commits)
        state["sources"].add("nvd_ref")
        if event_id in nvd_published_epochs:
            state["published_epochs"].append(nvd_published_epochs[event_id])
            state["event_date_sources"].add("nvd_published")
        severity_payload = nvd_severity_by_event.get(event_id)
        if severity_payload is not None:
            score = severity_payload.get("score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = math.nan
            if not math.isnan(score_value):
                state["severity_scores"].append(score_value)
            label = _normalise_severity_label(severity_payload.get("label"))
            if label:
                state["severity_labels"].add(label)
            state["severity_sources"].add(str(severity_payload.get("source", "nvd_metrics")))

    for event_id, commits in explicit_event_candidates.items():
        state = ensure(event_id)
        state["candidate_commits"].update(commits)
        state["sources"].add("explicit_id")

    records_by_event: dict[str, list[dict]] = {}
    for record in osv_records:
        event_id = alias_map.get(str(record.get("id", "")).upper(), str(record.get("id", "")).upper())
        records_by_event.setdefault(event_id, []).append(record)

    for event_id, payload in osv_event_candidates.items():
        state = ensure(event_id)
        state["candidate_commits"].update(payload.get("commits", set()))
        state["sources"].update(payload.get("sources", set()))
        for record in records_by_event.get(event_id, []):
            state["aliases"].update(
                str(token).upper()
                for token in [record.get("id"), *record.get("aliases", [])]
                if token
            )
            published = _safe_timestamp(record.get("published"))
            if published is not None:
                state["published_epochs"].append(int(published.timestamp()))
                state["event_date_sources"].add("osv_published")
            severity_payload = _extract_osv_record_severity(record)
            score = severity_payload.get("score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = math.nan
            if not math.isnan(score_value):
                state["severity_scores"].append(score_value)
            label = _normalise_severity_label(severity_payload.get("label"))
            if label:
                state["severity_labels"].add(label)
            state["severity_sources"].add(str(severity_payload.get("source", "osv_severity")))

    metadata_cache: dict[str, tuple[str, tuple[str, ...], int] | None] = {}
    rows: list[dict[str, object]] = []
    for event_id, state in sorted(event_state.items()):
        selected_commit = _select_primary_event_commit(
            repo_path,
            entry,
            set(state["candidate_commits"]),
            metadata_cache,
        )
        if selected_commit is None:
            continue
        fixed_epoch = _get_commit_epoch(repo_path, selected_commit)
        published_epoch = (
            min(int(value) for value in state["published_epochs"])
            if state["published_epochs"]
            else fixed_epoch
        )
        event_date_source = (
            "+".join(sorted(state["event_date_sources"]))
            if state["event_date_sources"]
            else "fixed_commit_epoch"
        )
        severity_score = max(state["severity_scores"]) if state["severity_scores"] else math.nan
        severity_label = (
            max(state["severity_labels"], key=lambda item: SEVERITY_ORDER.get(item, -1))
            if state["severity_labels"]
            else (_severity_label_from_score(severity_score) or "UNKNOWN")
        )
        severity_source = (
            "+".join(sorted(state["severity_sources"]))
            if state["severity_sources"]
            else "unavailable"
        )
        rows.append(
            {
                "repo": entry.name,
                "event_id": event_id,
                "published_epoch": int(published_epoch),
                "published": pd.Timestamp(published_epoch, unit="s", tz="UTC").isoformat(),
                "fixed_commit": selected_commit,
                "fixed_epoch": int(fixed_epoch),
                "aliases": ";".join(sorted(str(item) for item in state["aliases"] if item)),
                "source": "+".join(sorted(str(item) for item in state["sources"] if item)),
                "event_date_source": event_date_source,
                "severity_score": severity_score,
                "severity_label": severity_label,
                "severity_source": severity_source,
            }
        )

    return pd.DataFrame(rows)


def _build_snapshot_case_map(
    repo_path: Path,
    snapshot_commit: str,
    files_at_snapshot: dict[str, dict[str, object]],
    future_events: pd.DataFrame,
    entry: CorpusEntry,
    changed_cache: dict[str, list[str]],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    file_to_event: dict[str, dict[str, object]] = {}
    event_rows: list[dict[str, object]] = []

    future_events = future_events.sort_values(["published_epoch", "event_id"])
    for _, event in future_events.iterrows():
        fixed_commit = str(event["fixed_commit"])
        if fixed_commit not in changed_cache:
            changed_cache[fixed_commit] = _list_changed_source_files(
                repo_path,
                fixed_commit,
                entry.source_extensions,
            )
        changed_files = [
            path
            for path in changed_cache[fixed_commit]
            if path in files_at_snapshot
        ]
        event_rows.append(
            {
                **event.to_dict(),
                "snapshot_commit": snapshot_commit,
                "changed_source_files_count": int(len(changed_files)),
                "changed_source_files": ";".join(changed_files),
            }
        )
        for path in changed_files:
            file_to_event.setdefault(path, event.to_dict())

    return file_to_event, pd.DataFrame(event_rows)


def _parse_file_history_log(text: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for block in text.split("\x1e"):
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
        churn = 0
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            if parts[0].isdigit():
                churn += int(parts[0])
            if parts[1].isdigit():
                churn += int(parts[1])
        entries.append({"epoch": epoch, "author": author, "churn": churn})
    return entries


def _file_history_entries(
    repo_path: Path,
    snapshot_commit: str,
    path_str: str,
    history_cache: dict[tuple[str, str], list[dict[str, object]]],
) -> list[dict[str, object]]:
    key = (snapshot_commit, path_str)
    if key in history_cache:
        return history_cache[key]

    proc = _run_git(
        repo_path,
        [
            "log",
            "--follow",
            "--numstat",
            "--format=%x1e%ct%x1f%an",
            snapshot_commit,
            "--",
            path_str,
        ],
    )
    if proc.returncode != 0:
        history_cache[key] = []
        return []
    history_cache[key] = _parse_file_history_log(proc.stdout)
    return history_cache[key]


def _history_features_for_file(
    repo_path: Path,
    snapshot_commit: str,
    snapshot_epoch: int,
    path_str: str,
    history_cache: dict[tuple[str, str], list[dict[str, object]]],
) -> dict[str, object]:
    entries = _file_history_entries(repo_path, snapshot_commit, path_str, history_cache)
    within_365 = [
        entry
        for entry in entries
        if 0 <= snapshot_epoch - int(entry["epoch"]) <= 365 * 86400
    ]
    total_touches = len(entries)
    touches_365 = len(within_365)
    total_churn = int(sum(int(entry["churn"]) for entry in entries))
    churn_365 = int(sum(int(entry["churn"]) for entry in within_365))
    author_total = len({str(entry["author"]) for entry in entries})
    author_365 = len({str(entry["author"]) for entry in within_365})
    file_age_days = math.nan
    latest_touch_days = math.nan
    if entries:
        epochs = [int(entry["epoch"]) for entry in entries]
        file_age_days = float((snapshot_epoch - min(epochs)) / 86400.0)
        latest_touch_days = float((snapshot_epoch - max(epochs)) / 86400.0)

    return {
        "directory_depth": int(max(0, len(Path(path_str).parts) - 1)),
        "prior_touches_total": int(total_touches),
        "prior_touches_365d": int(touches_365),
        "total_churn": int(total_churn),
        "churn_365d": int(churn_365),
        "author_count_total": int(author_total),
        "author_count_365d": int(author_365),
        "file_age_days": file_age_days,
        "latest_touch_days": latest_touch_days,
    }


def _select_prospective_control(
    repo_path: Path,
    snapshot_commit: str,
    case_file: dict[str, object],
    files_at_snapshot: dict[str, dict[str, object]],
    excluded_paths: set[str],
    used_controls: set[str],
    metrics_cache: dict[tuple[str, str], StructuralOpacity | None],
    history_cache: dict[tuple[str, str], list[dict[str, object]]],
    *,
    snapshot_epoch: int,
    min_loc: int,
) -> tuple[dict[str, object], StructuralOpacity, dict[str, object]] | None:
    candidate_files = [
        item
        for path_str, item in files_at_snapshot.items()
        if path_str not in excluded_paths and path_str not in used_controls
    ]
    ranked = sorted(
        candidate_files,
        key=lambda item: (
            str(item["suffix"]) != str(case_file["suffix"]),
            abs(int(item.get("directory_depth", 0)) - int(case_file.get("directory_depth", 0))),
            abs(math.log1p(int(item["size"])) - math.log1p(int(case_file["size"]))),
            str(item["path"]),
        ),
    )
    for candidate in ranked:
        control_metrics = _get_metrics_for_snapshot_file(
            repo_path,
            snapshot_commit,
            str(candidate["path"]),
            metrics_cache,
        )
        if control_metrics is None or control_metrics.total_loc < min_loc:
            continue
        control_history = _history_features_for_file(
            repo_path,
            snapshot_commit,
            snapshot_epoch,
            str(candidate["path"]),
            history_cache,
        )
        return candidate, control_metrics, control_history
    return None


def _encode_file_row(
    metrics: StructuralOpacity,
    history: dict[str, object],
    *,
    path_str: str,
) -> dict[str, object]:
    return {
        "file_path": path_str,
        "suffix": Path(path_str).suffix.lower() or "<none>",
        "size_bytes": float(metrics.total_bytes),
        "loc": float(metrics.total_loc),
        "ci_gzip": float(metrics.ci_gzip),
        "shannon_entropy": float(metrics.shannon_entropy),
        "cyclomatic_density": float(metrics.cyclomatic_density),
        "halstead_volume": float(metrics.halstead_volume),
        "composite_score": float(metrics.composite_score),
        **history,
    }


def build_prospective_file_panel(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    config: Config,
    *,
    max_tags: int = 3,
    min_tag_gap_days: int = 365,
    horizon_days: int = 730,
    lookback_years: int = 10,
    min_loc: int = 20,
    ground_truth_policy: str = "expanded_advisory_plus_explicit",
    severity_band: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from cku_sim.analysis.forward_panel import sample_release_snapshots

    if ground_truth_policy not in PROSPECTIVE_GROUND_TRUTH_METADATA:
        raise ValueError(f"Unknown ground-truth policy: {ground_truth_policy}")
    if severity_band not in {"all", "high_critical"}:
        raise ValueError(f"Unknown severity band: {severity_band}")

    analysis_date = pd.Timestamp.now(tz="UTC")
    max_snapshot_date = analysis_date - pd.Timedelta(days=horizon_days)
    min_snapshot_date = max_snapshot_date - pd.Timedelta(days=365 * lookback_years)
    ancestor_cache: dict[tuple[str, str, str], bool] = {}
    pair_rows: list[dict[str, object]] = []
    event_catalog_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    nvd_cache_dir = config.processed_dir / "nvd_cache"
    osv_cache_dir = config.processed_dir / "osv_cache"

    for entry in corpus:
        repo_path = repo_paths.get(entry.name)
        if repo_path is None or not repo_path.exists():
            logger.info("Skipping %s: repository checkout not available", entry.name)
            continue

        snapshots = sample_release_snapshots(
            repo_path,
            max_tags=max_tags,
            min_gap_days=min_tag_gap_days,
            min_date=min_snapshot_date,
            max_date=max_snapshot_date,
        )
        if len(snapshots) < 1:
            logger.info("Skipping %s: no fully observed release tags", entry.name)
            continue

        events = _build_expanded_future_events(
            repo_path,
            entry,
            nvd_cache_dir=nvd_cache_dir,
            osv_cache_dir=osv_cache_dir,
            osv_rate_limit=config.osv_rate_limit,
            osv_query_batch_size=config.osv_query_batch_size,
        )
        if events.empty:
            logger.info("Skipping %s: no usable future security events", entry.name)
            continue
        total_events = int(len(events))
        known_severity_events = int(
            (
                events["severity_label"].fillna("UNKNOWN").ne("UNKNOWN")
                | events["severity_score"].notna()
            ).sum()
        )
        if severity_band == "high_critical":
            events = events.loc[
                [
                    _is_high_critical_event(row["severity_score"], row["severity_label"])
                    for _, row in events.iterrows()
                ]
            ].copy()
        if events.empty:
            logger.info(
                "Skipping %s: no usable future security events remain after severity filter %s "
                "(known severity %d/%d)",
                entry.name,
                severity_band,
                known_severity_events,
                total_events,
            )
            continue
        logger.info(
            "Prospective panel: %s with %d snapshots and %d candidate events (%s; known severity %d/%d)",
            entry.name,
            len(snapshots),
            len(events),
            severity_band,
            known_severity_events,
            total_events,
        )

        metrics_cache: dict[tuple[str, str], StructuralOpacity | None] = {}
        history_cache: dict[tuple[str, str], list[dict[str, object]]] = {}
        changed_cache: dict[str, list[str]] = {}
        repo_pair_start = len(pair_rows)

        for snapshot in snapshots:
            snapshot_tag = str(snapshot["tag"])
            snapshot_commit = str(snapshot["commit"])
            snapshot_date = _to_utc_timestamp(snapshot["date"])
            snapshot_epoch = int(snapshot_date.timestamp())

            future_events = _future_events_for_snapshot(
                repo_path,
                snapshot_commit,
                snapshot_epoch,
                events,
                horizon_days=horizon_days,
                ancestor_cache=ancestor_cache,
            )
            if future_events.empty:
                logger.info(
                    "  %s %s: no future security events within %d days",
                    entry.name,
                    snapshot_tag,
                    horizon_days,
                )
                continue

            files_at_snapshot = _list_source_files_at_commit(
                repo_path,
                snapshot_commit,
                entry.source_extensions,
            )
            if not files_at_snapshot:
                continue
            for metadata in files_at_snapshot.values():
                metadata["directory_depth"] = max(0, len(Path(str(metadata["path"])).parts) - 1)

            case_map, snapshot_event_catalog = _build_snapshot_case_map(
                repo_path,
                snapshot_commit,
                files_at_snapshot,
                future_events,
                entry,
                changed_cache,
            )
            if not case_map:
                logger.info(
                    "  %s %s: future events found, but no case files present at snapshot",
                    entry.name,
                    snapshot_tag,
                )
                continue

            if not snapshot_event_catalog.empty:
                snapshot_event_catalog["snapshot_tag"] = snapshot_tag
                snapshot_event_catalog["snapshot_date"] = snapshot_date.isoformat()
                snapshot_event_catalog["snapshot_key"] = f"{entry.name}:{snapshot_tag}"
                event_catalog_frames.append(snapshot_event_catalog)
            logger.info(
                "  %s %s: %d future events, %d candidate case files",
                entry.name,
                snapshot_tag,
                int(future_events["event_id"].nunique()),
                len(case_map),
            )

            used_controls: set[str] = set()
            excluded_paths = set(case_map)
            for case_path, event in sorted(
                case_map.items(),
                key=lambda item: (int(item[1]["published_epoch"]), item[0]),
            ):
                case_file = files_at_snapshot.get(case_path)
                if case_file is None:
                    continue

                case_metrics = _get_metrics_for_snapshot_file(
                    repo_path,
                    snapshot_commit,
                    case_path,
                    metrics_cache,
                )
                if case_metrics is None or case_metrics.total_loc < min_loc:
                    continue

                case_history = _history_features_for_file(
                    repo_path,
                    snapshot_commit,
                    snapshot_epoch,
                    case_path,
                    history_cache,
                )
                control_match = _select_prospective_control(
                    repo_path,
                    snapshot_commit,
                    case_file,
                    files_at_snapshot,
                    excluded_paths,
                    used_controls,
                    metrics_cache,
                    history_cache,
                    snapshot_epoch=snapshot_epoch,
                    min_loc=min_loc,
                )
                if control_match is None:
                    continue

                control_file, control_metrics, control_history = control_match
                control_path = str(control_file["path"])
                used_controls.add(control_path)
                pair_id = len(pair_rows)
                snapshot_key = f"{entry.name}:{snapshot_tag}"
                event_observation_id = f"{snapshot_key}:{event['event_id']}"

                case_payload = _encode_file_row(case_metrics, case_history, path_str=case_path)
                control_payload = _encode_file_row(
                    control_metrics,
                    control_history,
                    path_str=control_path,
                )

                row = {
                    "pair_id": int(pair_id),
                    "repo": entry.name,
                    "snapshot_tag": snapshot_tag,
                    "snapshot_commit": snapshot_commit,
                    "snapshot_date": snapshot_date.isoformat(),
                    "snapshot_key": snapshot_key,
                    "event_id": str(event["event_id"]),
                    "event_observation_id": event_observation_id,
                    "ground_truth_policy": ground_truth_policy,
                    "ground_truth_source": str(event["source"]),
                    "ground_truth_aliases": str(event["aliases"]),
                    "event_published": str(event["published"]),
                    "event_date_source": str(event["event_date_source"]),
                    "severity_band": severity_band,
                    "severity_score": float(event["severity_score"])
                    if pd.notna(event["severity_score"])
                    else math.nan,
                    "severity_label": str(event["severity_label"]),
                    "severity_source": str(event["severity_source"]),
                    "fixed_commit": str(event["fixed_commit"]),
                    "case_file": case_path,
                    "control_file": control_path,
                }
                for key, value in case_payload.items():
                    row[f"case_{key}"] = value
                for key, value in control_payload.items():
                    row[f"control_{key}"] = value

                row["delta_composite"] = float(
                    case_payload["composite_score"] - control_payload["composite_score"]
                )
                row["delta_ci_gzip"] = float(case_payload["ci_gzip"] - control_payload["ci_gzip"])
                row["delta_entropy"] = float(
                    case_payload["shannon_entropy"] - control_payload["shannon_entropy"]
                )
                row["delta_cc_density"] = float(
                    case_payload["cyclomatic_density"] - control_payload["cyclomatic_density"]
                )
                row["delta_halstead"] = float(
                    case_payload["halstead_volume"] - control_payload["halstead_volume"]
                )
                row["delta_log_prior_touches_365d"] = float(
                    math.log1p(case_payload["prior_touches_365d"])
                    - math.log1p(control_payload["prior_touches_365d"])
                )
                row["delta_log_churn_365d"] = float(
                    math.log1p(case_payload["churn_365d"])
                    - math.log1p(control_payload["churn_365d"])
                )
                pair_rows.append(row)

                commit_subject_proc = _run_git(
                    repo_path,
                    ["show", "-s", "--format=%s", str(event["fixed_commit"])],
                )
                audit_rows.append(
                    {
                        "repo": entry.name,
                        "snapshot_tag": snapshot_tag,
                        "snapshot_date": snapshot_date.isoformat(),
                        "event_id": str(event["event_id"]),
                        "event_observation_id": event_observation_id,
                        "event_published": str(event["published"]),
                        "event_date_source": str(event["event_date_source"]),
                        "severity_band": severity_band,
                        "severity_score": float(event["severity_score"])
                        if pd.notna(event["severity_score"])
                        else math.nan,
                        "severity_label": str(event["severity_label"]),
                        "severity_source": str(event["severity_source"]),
                        "ground_truth_source": str(event["source"]),
                        "ground_truth_aliases": str(event["aliases"]),
                        "fixed_commit": str(event["fixed_commit"]),
                        "commit_subject": commit_subject_proc.stdout.strip()
                        if commit_subject_proc.returncode == 0
                        else "",
                        "case_file": case_path,
                        "control_file": control_path,
                        "changed_source_files_count": int(
                            len(changed_cache.get(str(event["fixed_commit"]), []))
                        ),
                        "changed_source_files_sample": ";".join(
                            changed_cache.get(str(event["fixed_commit"]), [])[:10]
                        ),
                        "review_decision": "",
                        "review_notes": "",
                    }
                )
        repo_pairs = len(pair_rows) - repo_pair_start
        logger.info("  %s: retained %d matched file pairs", entry.name, repo_pairs)

    pairs = pd.DataFrame(pair_rows)
    event_catalog = (
        pd.concat(event_catalog_frames, ignore_index=True)
        if event_catalog_frames
        else pd.DataFrame()
    )
    audit_df = pd.DataFrame(audit_rows)
    return pairs, event_catalog, audit_df


def build_prospective_prediction_dataset(pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in pairs_df.iterrows():
        common = {
            "pair_id": int(row["pair_id"]),
            "repo": row["repo"],
            "snapshot_tag": row["snapshot_tag"],
            "snapshot_key": row["snapshot_key"],
            "event_id": row["event_id"],
            "event_observation_id": row["event_observation_id"],
            "ground_truth_policy": row["ground_truth_policy"],
            "ground_truth_source": row["ground_truth_source"],
        }
        for kind, label in (("case", 1), ("control", 0)):
            rows.append(
                {
                    **common,
                    "label": label,
                    "kind": kind,
                    "file_path": row[f"{kind}_file_path"],
                    "suffix": row[f"{kind}_suffix"] or "<none>",
                    "loc": float(row[f"{kind}_loc"]),
                    "size_bytes": float(row[f"{kind}_size_bytes"]),
                    "ci_gzip": float(row[f"{kind}_ci_gzip"]),
                    "shannon_entropy": float(row[f"{kind}_shannon_entropy"]),
                    "cyclomatic_density": float(row[f"{kind}_cyclomatic_density"]),
                    "halstead_volume": float(row[f"{kind}_halstead_volume"]),
                    "composite_score": float(row[f"{kind}_composite_score"]),
                    "directory_depth": float(row[f"{kind}_directory_depth"]),
                    "prior_touches_total": float(row[f"{kind}_prior_touches_total"]),
                    "prior_touches_365d": float(row[f"{kind}_prior_touches_365d"]),
                    "total_churn": float(row[f"{kind}_total_churn"]),
                    "churn_365d": float(row[f"{kind}_churn_365d"]),
                    "author_count_total": float(row[f"{kind}_author_count_total"]),
                    "author_count_365d": float(row[f"{kind}_author_count_365d"]),
                    "file_age_days": float(row[f"{kind}_file_age_days"]),
                    "latest_touch_days": float(row[f"{kind}_latest_touch_days"]),
                }
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
    return dataset


def _build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def _score_predictions(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
    }


def evaluate_leave_one_repo_out(
    dataset: pd.DataFrame,
    model_specs: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    model_specs = model_specs or PROSPECTIVE_MODEL_SPECS
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
                ["pair_id", "repo", "snapshot_tag", "event_id", "label", "kind", "file_path"]
            ].copy()
            fold_pred["model"] = model_name
            fold_pred["score"] = y_score
            prediction_frames.append(fold_pred)

            fold_metrics = _score_predictions(test["label"], y_score)
            fold_metrics["pairwise_accuracy"] = pairwise_accuracy(fold_pred, "score")
            fold_metrics["model"] = model_name
            fold_metrics["held_out_repo"] = held_out_repo
            fold_metrics["n_files"] = int(len(test))
            fold_metrics["n_pairs"] = int(test["pair_id"].nunique())
            fold_rows.append(fold_metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    fold_metrics_df = pd.DataFrame(fold_rows)
    summary: dict[str, object] = {
        "n_files": int(len(dataset)),
        "n_pairs": int(dataset["pair_id"].nunique()),
        "n_events": int(dataset["event_observation_id"].nunique()),
        "n_repos": int(dataset["repo"].nunique()),
        "models": {},
    }
    if predictions.empty:
        return predictions, fold_metrics_df, summary

    for model_name in model_specs:
        model_preds = predictions.loc[predictions["model"] == model_name].copy()
        model_folds = fold_metrics_df.loc[fold_metrics_df["model"] == model_name].copy()
        overall = _score_predictions(model_preds["label"], model_preds["score"].to_numpy())
        overall["pairwise_accuracy"] = pairwise_accuracy(model_preds, "score")
        overall["macro_roc_auc"] = float(model_folds["roc_auc"].mean())
        overall["macro_average_precision"] = float(model_folds["average_precision"].mean())
        overall["macro_brier_score"] = float(model_folds["brier_score"].mean())
        overall["macro_pairwise_accuracy"] = float(model_folds["pairwise_accuracy"].mean())
        summary["models"][model_name] = overall

    return predictions, fold_metrics_df, summary


def fit_repo_fixed_effect_models(dataset: pd.DataFrame) -> dict[str, object]:
    if dataset.empty or dataset["repo"].nunique() < 2 or dataset["label"].nunique() < 2:
        return {}

    numeric_cols = WITHIN_PROJECT_BASELINE_FEATURES + ["composite_score"]
    work = dataset.copy()
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in numeric_cols:
        work[col] = work[col].fillna(work[col].median())

    repo_dummies = pd.get_dummies(work["repo"], prefix="repo", drop_first=True)
    y = work["label"].astype(float)
    groups = work["event_observation_id"]

    x_base = pd.concat([work[WITHIN_PROJECT_BASELINE_FEATURES], repo_dummies], axis=1)
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
        logger.warning("Repo fixed-effect models failed: %s", exc)
        return {}

    lr_stat = max(0.0, 2.0 * (plus_model.llf - base_model.llf))
    lr_pvalue = float(stats.chi2.sf(lr_stat, df=1))
    conf = plus_model.conf_int().loc["composite_score"]
    return {
        "baseline_repo_fixed_effects": {
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


def summarise_prospective_pairs(pairs: pd.DataFrame) -> dict[str, object]:
    if pairs.empty:
        return {"n_pairs": 0, "n_events": 0, "n_repos": 0, "n_snapshots": 0}

    delta = pairs["delta_composite"].dropna()
    summary = {
        "n_pairs": int(len(pairs)),
        "n_events": int(pairs["event_observation_id"].nunique()),
        "n_repos": int(pairs["repo"].nunique()),
        "n_snapshots": int(pairs["snapshot_key"].nunique()),
        "mean_delta_composite": float(delta.mean()),
        "median_delta_composite": float(delta.median()),
        "positive_share": float((delta > 0).mean()),
        "mean_case_loc": float(pairs["case_loc"].mean()),
        "mean_control_loc": float(pairs["control_loc"].mean()),
        "ground_truth_policy": str(pairs["ground_truth_policy"].iloc[0]),
        "severity_band": str(pairs["severity_band"].iloc[0]) if "severity_band" in pairs else "all",
        "severity_label_breakdown": (
            pairs["severity_label"].fillna("UNKNOWN").value_counts().to_dict()
            if "severity_label" in pairs
            else {}
        ),
    }
    summary.update(prospective_policy_metadata(summary["ground_truth_policy"]))
    if len(delta) >= 3:
        try:
            wilcoxon = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
            summary["wilcoxon_pvalue_greater"] = float(wilcoxon.pvalue)
            summary["wilcoxon_statistic"] = float(wilcoxon.statistic)
        except ValueError:
            pass
    bootstrap_primary = clustered_delta_bootstrap(
        pairs,
        cluster_col="event_observation_id",
        delta_col="delta_composite",
    )
    if bootstrap_primary:
        summary["bootstrap_primary_cluster"] = bootstrap_primary
    bootstrap_repo = clustered_delta_bootstrap(
        pairs,
        cluster_col="repo",
        delta_col="delta_composite",
    )
    if bootstrap_repo:
        summary["bootstrap_repo_cluster"] = bootstrap_repo
    return summary


def summarise_repo_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    return (
        pairs.groupby("repo", as_index=False)
        .agg(
            n_pairs=("pair_id", "size"),
            n_events=("event_observation_id", "nunique"),
            n_snapshots=("snapshot_key", "nunique"),
            mean_delta_composite=("delta_composite", "mean"),
            median_delta_composite=("delta_composite", "median"),
            positive_share=("delta_composite", lambda series: float((series > 0).mean())),
        )
        .sort_values(["median_delta_composite", "mean_delta_composite"], ascending=False)
    )


def sample_audit_rows(audit_df: pd.DataFrame, *, sample_size: int = 40, random_seed: int = 42) -> pd.DataFrame:
    if audit_df.empty:
        return audit_df
    deduped = audit_df.drop_duplicates(subset=["event_observation_id"]).copy()
    sample_size = min(sample_size, len(deduped))
    return deduped.sample(n=sample_size, random_state=random_seed).sort_values(
        ["repo", "snapshot_date", "event_id"]
    )


def plot_model_comparison(summary: dict[str, object], output_path: Path) -> None:
    if not summary.get("models"):
        return
    models = list(summary["models"].keys())
    roc_aucs = [summary["models"][name]["roc_auc"] for name in models]
    pair_accs = [summary["models"][name]["pairwise_accuracy"] for name in models]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(models, roc_aucs, color=["#7A9E9F", "#4C78A8", "#72B7B2", "#F58518"])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Pooled ROC AUC")
    axes[0].set_title("Held-out discrimination")

    axes[1].bar(models, pair_accs, color=["#7A9E9F", "#4C78A8", "#72B7B2", "#F58518"])
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Pairwise ranking accuracy")
    axes[1].set_title("Held-out pair ranking")

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_repo_deltas(repo_summary: pd.DataFrame, output_path: Path) -> None:
    if repo_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(repo_summary["repo"], repo_summary["median_delta_composite"], color="#4C78A8")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Median case-control delta")
    ax.set_title("Prospective file-level opacity delta by repository")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
