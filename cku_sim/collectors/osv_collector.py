"""Fetch and cache repository-linked vulnerability records from the OSV API."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

OSV_QUERY_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vuln_id}"
DEFAULT_BATCH_SIZE = 100

CVE_ID_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
GHSA_ID_RE = re.compile(
    r"^GHSA-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}$",
    re.IGNORECASE,
)
HEX_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
GITHUB_COMMIT_URL_RE = re.compile(r"^(https?://[^/]+/[^/]+/[^/]+)/commit/([0-9a-fA-F]{7,40})")


def osv_query_repo_url(git_url: str) -> str:
    """Convert a remote URL into the HTTPS repository URL expected by OSV."""
    url = git_url.strip()
    if not url:
        return ""

    if url.startswith("git@"):
        host_path = url[4:]
        if ":" not in host_path:
            return ""
        host, path = host_path.split(":", 1)
        canonical = f"https://{host}/{path}"
    elif url.startswith("ssh://"):
        parsed = urlparse(url)
        if not parsed.hostname:
            return ""
        canonical = f"https://{parsed.hostname}{parsed.path}"
    elif "://" in url:
        parsed = urlparse(url)
        if not parsed.netloc:
            return ""
        canonical = f"https://{parsed.netloc}{parsed.path}"
    else:
        return ""

    if canonical.endswith(".git"):
        canonical = canonical[:-4]
    return canonical.rstrip("/")


def normalise_repo_url(url: str) -> str:
    """Normalise a repository URL for matching across sources."""
    return osv_query_repo_url(url).lower()


def repo_url_variants(git_url: str) -> set[str]:
    """Return normalised repository URL variants with and without a .git suffix."""
    canonical = osv_query_repo_url(git_url)
    if not canonical:
        return set()
    return {canonical.lower(), f"{canonical}.git".lower()}


def list_repo_tags(repo_path: Path) -> list[str]:
    """List tags available in a local git checkout."""
    proc = subprocess.run(
        ["git", "-C", str(repo_path), "tag"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        logger.warning("Unable to list tags for %s", repo_path)
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def canonicalise_osv_event_id(record: dict) -> str:
    """Choose a stable event identifier, preferring CVE then GHSA aliases."""
    candidates = [str(record.get("id", "")), *[str(alias) for alias in record.get("aliases", [])]]
    cve_ids = sorted({value.upper() for value in candidates if CVE_ID_RE.fullmatch(value.upper())})
    if cve_ids:
        return cve_ids[0]

    ghsa_ids = sorted(
        {value.upper() for value in candidates if GHSA_ID_RE.fullmatch(value.upper())}
    )
    if ghsa_ids:
        return ghsa_ids[0]

    return str(record.get("id", "UNKNOWN")).upper()


def build_osv_alias_map(records: list[dict]) -> dict[str, str]:
    """Map all OSV record aliases to the canonical event identifier."""
    alias_map: dict[str, str] = {}
    for record in records:
        canonical = canonicalise_osv_event_id(record)
        for token in [str(record.get("id", "")), *[str(alias) for alias in record.get("aliases", [])]]:
            if token:
                alias_map[token.upper()] = canonical
        alias_map[canonical] = canonical
    return alias_map


def extract_osv_event_candidates(
    records: list[dict],
    repo_urls: set[str],
) -> dict[str, dict[str, set[str]]]:
    """Extract fixed-commit candidates keyed by canonical vulnerability identifier."""
    candidates: dict[str, dict[str, set[str]]] = {}

    for record in records:
        event_id = canonicalise_osv_event_id(record)
        commits, sources = extract_fixed_commits_from_osv_record(record, repo_urls)
        if not commits:
            continue

        entry = candidates.setdefault(event_id, {"commits": set(), "sources": set()})
        entry["commits"].update(commits)
        entry["sources"].update(sources)

    return candidates


def extract_fixed_commits_from_osv_record(
    record: dict,
    repo_urls: set[str],
) -> tuple[set[str], set[str]]:
    """Extract fixed revision candidates from OSV affected ranges and references."""
    commits: set[str] = set()
    sources: set[str] = set()

    for affected in record.get("affected", []):
        for range_item in affected.get("ranges", []):
            if str(range_item.get("type", "")).upper() != "GIT":
                continue
            repo = normalise_repo_url(str(range_item.get("repo", "")))
            if repo and repo not in repo_urls:
                continue

            for event in range_item.get("events", []):
                fixed = str(event.get("fixed", "")).strip()
                if HEX_SHA_RE.fullmatch(fixed):
                    commits.add(fixed.lower())
                    sources.add("osv_range")

    for reference in record.get("references", []):
        commit = extract_commit_from_reference(reference.get("url", ""), repo_urls)
        if commit:
            commits.add(commit)
            sources.add("osv_ref")

    return commits, sources


def extract_commit_from_reference(url: str, repo_urls: set[str]) -> str | None:
    """Parse a commit hash from a repository-matching advisory reference URL."""
    match = GITHUB_COMMIT_URL_RE.match(url.strip())
    if not match:
        return None

    repo = normalise_repo_url(match.group(1))
    if repo not in repo_urls:
        return None

    commit = match.group(2).lower()
    if not HEX_SHA_RE.fullmatch(commit):
        return None
    return commit


def fetch_osv_records_for_repo(
    repo_path: Path,
    git_url: str,
    cache_dir: Path,
    *,
    rate_limit: float = 0.1,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Fetch or load cached OSV vulnerability records for a repository."""
    repo_url = osv_query_repo_url(git_url)
    if not repo_url:
        return []

    tags = list_repo_tags(repo_path)
    if not tags:
        logger.info("No tags available for OSV query in %s", repo_path.name)
        return []

    repo_cache_dir = cache_dir / _safe_repo_key(repo_url)
    repo_cache_dir.mkdir(parents=True, exist_ok=True)

    vuln_ids = _load_or_query_repo_index(
        repo_url,
        tags,
        repo_cache_dir / "index.json",
        rate_limit=rate_limit,
        batch_size=batch_size,
    )
    if not vuln_ids:
        return []

    records: list[dict] = []
    for idx, vuln_id in enumerate(vuln_ids):
        if idx > 0 and rate_limit > 0:
            time.sleep(rate_limit)
        record = _load_or_fetch_vuln_record(vuln_id, repo_cache_dir / "vulns")
        if record:
            records.append(record)

    return records


def _load_or_query_repo_index(
    repo_url: str,
    tags: list[str],
    cache_path: Path,
    *,
    rate_limit: float,
    batch_size: int,
) -> list[str]:
    tags_signature = _tags_signature(tags)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("repo_url") == repo_url
            and cached.get("tags_signature") == tags_signature
            and isinstance(cached.get("vulnerability_ids"), list)
        ):
            logger.info("Loading cached OSV index for %s", repo_url)
            return [str(item) for item in cached["vulnerability_ids"]]

    vuln_ids: set[str] = set()
    batch_size = max(1, int(batch_size))

    for batch_index, start in enumerate(range(0, len(tags), batch_size)):
        if batch_index > 0 and rate_limit > 0:
            time.sleep(rate_limit)

        batch = tags[start : start + batch_size]
        payload = {
            "queries": [
                {
                    "package": {
                        "name": repo_url,
                        "ecosystem": "GIT",
                    },
                    "version": tag,
                }
                for tag in batch
            ]
        }
        logger.info(
            "Querying OSV for %s tags %d-%d/%d",
            repo_url,
            start + 1,
            min(start + len(batch), len(tags)),
            len(tags),
        )
        response = requests.post(OSV_QUERY_BATCH_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        for result in data.get("results", []):
            for vuln in result.get("vulns", []):
                vuln_id = str(vuln.get("id", "")).strip()
                if vuln_id:
                    vuln_ids.add(vuln_id)

    ordered = sorted(vuln_ids)
    cache_path.write_text(
        json.dumps(
            {
                "repo_url": repo_url,
                "n_tags": len(tags),
                "tags_signature": tags_signature,
                "vulnerability_ids": ordered,
            },
            indent=2,
        )
    )
    logger.info("Cached %d OSV vulnerability identifiers for %s", len(ordered), repo_url)
    return ordered


def _load_or_fetch_vuln_record(vuln_id: str, cache_dir: Path) -> dict | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{vuln_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    response = requests.get(OSV_VULN_URL.format(vuln_id=vuln_id), timeout=120)
    if response.status_code == 404:
        logger.warning("OSV record not found for %s", vuln_id)
        return None
    response.raise_for_status()
    data = response.json()
    cache_path.write_text(json.dumps(data, indent=2))
    return data


def _safe_repo_key(repo_url: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_url)
    digest = hashlib.sha1(repo_url.encode("utf-8")).hexdigest()[:12]
    return f"{collapsed[:80]}_{digest}"


def _tags_signature(tags: list[str]) -> str:
    payload = "\n".join(tags).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()
