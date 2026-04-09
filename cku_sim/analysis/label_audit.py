"""Helpers for reviewing prospective file-level label quality."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from cku_sim.analysis.file_level_case_control import _list_changed_source_files
from cku_sim.core.config import CorpusEntry

REVIEW_ORDER = {"reject": 0, "ambiguous": 1, "accept": 2}
SECURITY_SIGNAL_RE = re.compile(
    r"\b("
    r"cve-\d{4}-\d+|ghsa-|security|overflow|out[- ]of[- ]bounds|use[- ]after[- ]free|"
    r"uaf|double free|null pointer|denial[- ]of[- ]service|dos|traversal|auth|"
    r"credential|x509|ssl|tls|escape|ownership|header|buffer|infinite loop"
    r")\b",
    re.IGNORECASE,
)


MANUAL_EVENT_OVERRIDES: dict[tuple[str, str, str], dict[str, str]] = {
    (
        "curl",
        "CVE-2024-2379",
        "aedbbdf18e689a5eee8dc39600914f5eda6c409c",
    ): {
        "event_commit_review": "ambiguous",
        "overall_review": "ambiguous",
        "review_notes": "OSV-range-only mapping; commit message fixes error handling but does not itself name the vulnerability.",
    },
    (
        "curl",
        "OSV-2025-657",
        "c294d0abc5b62b8068af391576be87b821d2ff8b",
    ): {
        "event_commit_review": "ambiguous",
        "overall_review": "ambiguous",
        "review_notes": "OSV-only event with a plausible denial-of-service style fix, but the commit message does not explicitly indicate a security context.",
    },
    (
        "openssh",
        "CVE-2021-41617",
        "bf944e3794eff5413f2df1ef37cddf96918c6bde",
    ): {
        "event_commit_review": "ambiguous",
        "overall_review": "ambiguous",
        "review_notes": "OSV-range-only mapping; the selected commit is terse and does not by itself make the security relevance explicit.",
    },
}


def _default_review_note(row: pd.Series) -> str:
    source = str(row.get("ground_truth_source", ""))
    if "explicit_id" in source:
        return "Commit message or body links the fix explicitly to a vulnerability identifier."
    if "nvd_ref" in source or "osv_ref" in source:
        return "Advisory record resolves to an explicit fixing commit reference."
    if "osv_range" in source:
        return "Advisory record resolves the event to a fixing commit via an affected range."
    return "Event label inherited from the prospective advisory mapping."


def enrich_audit_frame(
    audit_df: pd.DataFrame,
    repo_paths: dict[str, Path],
    corpus_by_name: dict[str, CorpusEntry],
) -> pd.DataFrame:
    """Attach review-oriented context to the audit sample."""
    rows: list[dict[str, object]] = []
    changed_cache: dict[tuple[str, str], list[str]] = {}

    for _, row in audit_df.iterrows():
        repo = str(row["repo"])
        repo_path = repo_paths.get(repo)
        entry = corpus_by_name.get(repo)
        if repo_path is None or entry is None:
            continue

        fixed_commit = str(row["fixed_commit"])
        cache_key = (repo, fixed_commit)
        if cache_key not in changed_cache:
            changed_cache[cache_key] = _list_changed_source_files(
                repo_path,
                fixed_commit,
                entry.source_extensions,
            )
        changed_files = changed_cache[cache_key]

        commit_body = ""
        commit_proc = None
        try:
            import subprocess

            commit_proc = subprocess.run(
                ["git", "-C", str(repo_path), "show", "-s", "--format=%B", fixed_commit],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception:
            commit_proc = None
        if commit_proc is not None and commit_proc.returncode == 0:
            commit_body = commit_proc.stdout.strip()

        case_file = str(row["case_file"])
        source = str(row["ground_truth_source"])
        case_in_changed = case_file in changed_files
        multi_file = len(changed_files) > 1
        subject = str(row.get("commit_subject", ""))
        body_signal = bool(SECURITY_SIGNAL_RE.search(commit_body or subject))
        message_mentions_event = str(row["event_id"]).upper() in (commit_body or "").upper()

        enriched = dict(row)
        enriched.update(
            {
                "changed_source_files_full": ";".join(changed_files),
                "changed_source_files_review_count": int(len(changed_files)),
                "case_file_in_changed_files": int(case_in_changed),
                "multi_file_fix": int(multi_file),
                "commit_message_full": commit_body,
                "commit_mentions_event_id": int(message_mentions_event),
                "message_has_security_signal": int(body_signal),
                "source_has_explicit_id": int("explicit_id" in source),
                "source_has_nvd_ref": int("nvd_ref" in source),
                "source_has_osv_ref": int("osv_ref" in source),
                "source_has_osv_range": int("osv_range" in source),
            }
        )
        rows.append(enriched)

    return pd.DataFrame(rows)


def apply_review_decisions(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Apply a conservative manual review policy to the enriched audit frame."""
    if audit_df.empty:
        return audit_df

    reviewed = audit_df.copy()
    reviewed["event_commit_review"] = "ambiguous"
    reviewed["file_review"] = "ambiguous"
    reviewed["overall_review"] = "ambiguous"
    reviewed["review_notes"] = ""

    for idx, row in reviewed.iterrows():
        source = str(row.get("ground_truth_source", ""))
        case_in_changed = bool(int(row.get("case_file_in_changed_files", 0)))
        changed_count = int(row.get("changed_source_files_review_count", 0))
        message_mentions_event = bool(int(row.get("commit_mentions_event_id", 0)))

        event_commit_review = "ambiguous"
        if any(token in source for token in ("explicit_id", "nvd_ref", "osv_ref")) or message_mentions_event:
            event_commit_review = "accept"

        if case_in_changed and changed_count <= 3:
            file_review = "accept"
        elif case_in_changed:
            file_review = "ambiguous"
        else:
            file_review = "reject"

        overall_review = "accept"
        if event_commit_review == "reject" or file_review == "reject":
            overall_review = "reject"
        elif "ambiguous" in {event_commit_review, file_review}:
            overall_review = "ambiguous"

        reviewed.at[idx, "event_commit_review"] = event_commit_review
        reviewed.at[idx, "file_review"] = file_review
        reviewed.at[idx, "overall_review"] = overall_review
        reviewed.at[idx, "review_notes"] = _default_review_note(row)

    for key, override in MANUAL_EVENT_OVERRIDES.items():
        repo, event_id, fixed_commit = key
        mask = (
            reviewed["repo"].astype(str).eq(repo)
            & reviewed["event_id"].astype(str).eq(event_id)
            & reviewed["fixed_commit"].astype(str).eq(fixed_commit)
        )
        for column, value in override.items():
            reviewed.loc[mask, column] = value

    return reviewed


def summarise_reviewed_audit(reviewed_df: pd.DataFrame) -> dict[str, object]:
    """Summarise reviewed audit outcomes."""
    if reviewed_df.empty:
        return {"n_reviewed": 0}

    def counts(column: str) -> dict[str, int]:
        values = reviewed_df[column].astype(str).value_counts().to_dict()
        return {key: int(values.get(key, 0)) for key in ("accept", "ambiguous", "reject")}

    overall_counts = counts("overall_review")
    event_counts = counts("event_commit_review")
    file_counts = counts("file_review")
    n = int(len(reviewed_df))

    return {
        "n_reviewed": n,
        "overall_review": overall_counts,
        "event_commit_review": event_counts,
        "file_review": file_counts,
        "overall_accept_rate": overall_counts["accept"] / n,
        "overall_non_reject_rate": (overall_counts["accept"] + overall_counts["ambiguous"]) / n,
        "event_commit_accept_rate": event_counts["accept"] / n,
        "event_commit_non_reject_rate": (event_counts["accept"] + event_counts["ambiguous"]) / n,
        "file_accept_rate": file_counts["accept"] / n,
        "file_non_reject_rate": (file_counts["accept"] + file_counts["ambiguous"]) / n,
        "case_file_touch_confirmed_rate": float(reviewed_df["case_file_in_changed_files"].mean()),
        "source_breakdown": (
            reviewed_df.groupby("ground_truth_source").size().sort_values(ascending=False).to_dict()
        ),
        "overall_by_source": (
            reviewed_df.groupby(["ground_truth_source", "overall_review"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def compile_source_summary(reviewed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review outcomes by event-source pattern."""
    if reviewed_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for source, group in reviewed_df.groupby("ground_truth_source", sort=False):
        summary = summarise_reviewed_audit(group)
        rows.append(
            {
                "ground_truth_source": source,
                "n_reviewed": int(len(group)),
                "overall_accept": int(summary["overall_review"]["accept"]),
                "overall_ambiguous": int(summary["overall_review"]["ambiguous"]),
                "overall_reject": int(summary["overall_review"]["reject"]),
                "event_accept": int(summary["event_commit_review"]["accept"]),
                "file_accept": int(summary["file_review"]["accept"]),
                "case_file_touch_confirmed_rate": float(summary["case_file_touch_confirmed_rate"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["n_reviewed", "ground_truth_source"], ascending=[False, True])
