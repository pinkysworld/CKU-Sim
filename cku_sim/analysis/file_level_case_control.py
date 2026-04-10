"""File-level matched case-control analysis for vulnerability-fixing commits."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from scipy import stats

from cku_sim.analysis.bootstrap import clustered_delta_bootstrap
from cku_sim.collectors.osv_collector import (
    build_osv_alias_map,
    extract_osv_event_candidates,
    fetch_osv_records_for_repo,
    repo_url_variants,
)
from cku_sim.core.config import CorpusEntry
from cku_sim.core.opacity import StructuralOpacity
from cku_sim.metrics.compressibility import compressibility_index

logger = logging.getLogger(__name__)

EXCLUDE_PARTS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".tox",
    ".eggs",
    "vendor",
    "third_party",
    "3rdparty",
    "external",
    "deps",
    "test",
    "tests",
    "testing",
    "t",
    "regress",
    "doc",
    "docs",
    "documentation",
    "man",
    "examples",
    "samples",
    "demo",
    "benchmarks",
}

TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[{}()\[\];,.<>!=+\-*/%&|^~?:#]")
DECISION_KEYWORDS = re.compile(
    r"\b(if|else\s+if|for|while|case|catch|switch)\b"
    r"|(\?\s)"
    r"|(&&|\|\|)"
)
OPERATORS = re.compile(
    r"(==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|->|::|"
    r"[+\-*/%=<>&|^~!?:;,.\[\]{}()])"
)
OPERANDS = re.compile(r"\b[A-Za-z_]\w*\b|\b\d+\.?\d*\b")
C_KEYWORDS = {
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "break",
    "continue",
    "return",
    "goto",
    "default",
    "typedef",
    "struct",
    "union",
    "enum",
    "sizeof",
    "static",
    "extern",
    "const",
    "volatile",
    "register",
    "auto",
    "void",
    "int",
    "char",
    "short",
    "long",
    "float",
    "double",
    "signed",
    "unsigned",
    "inline",
}
SECURITY_ID_RE = re.compile(
    r"\b(?:CVE-\d{4}-\d+|GHSA-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4})\b",
    re.IGNORECASE,
)
GROUND_TRUTH_POLICY_METADATA = {
    "nvd_commit_refs": {
        "label": "NVD commit-reference observations",
        "description": (
            "Each locally accessible NVD-linked fixing commit is treated as one observation."
        ),
    },
    "strict_nvd_event": {
        "label": "NVD event-collapsed observations",
        "description": (
            "Each NVD-linked vulnerability is mapped to one primary single-parent "
            "source-touching fixing commit."
        ),
    },
    "balanced_explicit_id_event": {
        "label": "Expanded explicit-ID observations",
        "description": (
            "The event-collapsed NVD set is augmented with locally explicit CVE/GHSA-tagged "
            "security-fix commits, with one primary fixing commit retained per event."
        ),
    },
    "expanded_advisory_event": {
        "label": "Expanded advisory observations",
        "description": (
            "The event-collapsed NVD set is augmented with OSV-linked repository advisories "
            "and locally explicit CVE/GHSA-tagged security-fix commits, with one primary "
            "fixing commit retained per event."
        ),
    },
}
GROUND_TRUTH_POLICIES = set(GROUND_TRUTH_POLICY_METADATA)


def describe_ground_truth_policy(policy: str) -> dict[str, str]:
    """Return metadata for a supported ground-truth policy."""
    if policy not in GROUND_TRUTH_POLICY_METADATA:
        raise ValueError(f"Unknown ground-truth policy: {policy}")
    return dict(GROUND_TRUTH_POLICY_METADATA[policy])


def results_subdir_for_ground_truth_policy(policy: str) -> str:
    """Return the default results subdirectory for a policy-specific e06 run."""
    if policy not in GROUND_TRUTH_POLICIES:
        raise ValueError(f"Unknown ground-truth policy: {policy}")
    if policy == "nvd_commit_refs":
        return "e06_file_case_control"
    return f"e06_file_case_control__{policy}"


@dataclass(frozen=True)
class GroundTruthEvent:
    """A labeled vulnerability-fix event tied to a single commit."""

    event_id: str
    commit: str
    vulnerability_ids: tuple[str, ...]
    source: str


def cache_file_for_cpe(cpe_id: str, cache_dir: Path) -> Path:
    """Return the cache path used by the NVD collector."""
    safe_name = cpe_id.replace(":", "_").replace("*", "ALL").replace("/", "_")
    return cache_dir / f"{safe_name}.json"


def normalise_github_slug(git_url: str) -> str | None:
    """Extract an owner/repo slug from a GitHub remote URL."""
    url = git_url.strip()
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if url.startswith(prefix):
            slug = url[len(prefix):]
            if slug.endswith(".git"):
                slug = slug[:-4]
            return slug.strip("/")
    return None


def extract_commit_refs_from_nvd_items(
    vuln_items: list[dict],
    expected_slug: str,
) -> dict[str, set[str]]:
    """Map fixing commits from NVD references to their linked CVE IDs."""
    commit_to_cves: dict[str, set[str]] = {}

    for item in vuln_items:
        cve = item.get("cve", {})
        cve_id = cve.get("id", "UNKNOWN")
        for ref in cve.get("references", []):
            url = ref.get("url", "")
            if "github.com/" not in url or "/commit/" not in url:
                continue

            after_host = url.split("github.com/", 1)[1]
            slug, commit = after_host.split("/commit/", 1)
            slug = slug.strip("/")
            commit = commit.split("?", 1)[0].split("#", 1)[0].strip("/")

            if slug != expected_slug:
                continue
            if len(commit) < 7 or any(ch not in "0123456789abcdefABCDEF" for ch in commit):
                continue

            commit_to_cves.setdefault(commit, set()).add(cve_id)

    return commit_to_cves


def extract_security_ids(text: str) -> set[str]:
    """Extract explicit vulnerability identifiers from commit text."""
    return {match.group(0).upper() for match in SECURITY_ID_RE.finditer(text or "")}


def should_include_source_path(path_str: str, extensions: list[str]) -> bool:
    """Return whether a path should be treated as an in-scope source file."""
    path = Path(path_str)
    if path.suffix.lower() not in {ext.lower() for ext in extensions}:
        return False
    return not (set(path.parts) & EXCLUDE_PARTS)


def count_loc_text(text: str) -> int:
    """Count non-blank, non-comment lines using the repo-level heuristic."""
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
            total += 1
    return total


def token_frequencies_text(text: str) -> Counter:
    """Token frequencies for a single source file."""
    return Counter(TOKEN_RE.findall(text))


def shannon_entropy_text(text: str) -> float:
    """Normalised Shannon entropy for a single source file."""
    freq = token_frequencies_text(text)
    total = sum(freq.values())
    if total == 0 or len(freq) <= 1:
        return 0.0

    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    max_entropy = math.log2(len(freq))
    return entropy / max_entropy if max_entropy else 0.0


def cyclomatic_density_text(text: str, total_loc: int) -> float:
    """Estimate cyclomatic density for a single source file."""
    if total_loc == 0:
        return 0.0

    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)

    decisions = len(DECISION_KEYWORDS.findall(text))
    return (1 + decisions) / total_loc


def halstead_volume_text(text: str) -> float:
    """Normalised Halstead volume for a single source file."""
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
    text = re.sub(r"'(?:[^'\\]|\\.)*'", "''", text)

    operators = OPERATORS.findall(text)
    operands = [token for token in OPERANDS.findall(text) if token not in C_KEYWORDS]

    eta = len(set(operators)) + len(set(operands))
    length = len(operators) + len(operands)
    if eta <= 1 or length == 0:
        return 0.0

    normalised = (math.log2(eta) - 8.0) / 10.0
    return max(0.0, min(1.0, normalised))


def compute_file_opacity_from_text(
    text: str,
    *,
    name: str,
    snapshot_id: str,
    weights: dict[str, float] | None = None,
) -> StructuralOpacity:
    """Compute the standard opacity bundle for a single file."""
    raw = text.encode("utf-8", errors="replace")
    total_loc = count_loc_text(text)

    opacity = StructuralOpacity(
        name=name,
        snapshot_id=snapshot_id,
        total_bytes=len(raw),
        total_loc=total_loc,
        num_files=1,
        ci_gzip=compressibility_index(raw, method="gzip") if raw else 0.0,
        ci_lzma=compressibility_index(raw, method="lzma") if raw else 0.0,
        ci_zstd=compressibility_index(raw, method="zstd") if raw else 0.0,
        shannon_entropy=shannon_entropy_text(text),
        cyclomatic_density=cyclomatic_density_text(text, total_loc),
        halstead_volume=halstead_volume_text(text),
    )
    opacity.compute_composite(weights)
    return opacity


def _run_git(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("GIT_NO_LAZY_FETCH", "1")
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _get_first_parent(repo_path: Path, commit: str) -> str | None:
    proc = _run_git(repo_path, ["rev-list", "--parents", "-n", "1", commit])
    if proc.returncode != 0:
        return None
    words = proc.stdout.strip().split()
    return words[1] if len(words) > 1 else None


def _get_commit_parents(repo_path: Path, commit: str) -> list[str]:
    proc = _run_git(repo_path, ["rev-list", "--parents", "-n", "1", commit])
    if proc.returncode != 0:
        return []
    words = proc.stdout.strip().split()
    return words[1:]


@lru_cache(maxsize=200_000)
def _cached_commit_epoch(repo_path_str: str, commit: str) -> int:
    proc = _run_git(Path(repo_path_str), ["show", "-s", "--format=%ct", commit])
    if proc.returncode != 0:
        return 2**63 - 1
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return 2**63 - 1


def _get_commit_epoch(repo_path: Path, commit: str) -> int:
    return _cached_commit_epoch(str(repo_path), commit)


def _list_source_files_at_commit(
    repo_path: Path,
    commit: str,
    extensions: list[str],
) -> dict[str, dict[str, object]]:
    proc = _run_git(repo_path, ["ls-tree", "-r", "-l", commit])
    if proc.returncode != 0:
        return {}

    files: dict[str, dict[str, object]] = {}
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        meta, path_str = line.split("\t", 1)
        if not should_include_source_path(path_str, extensions):
            continue

        meta_parts = meta.split()
        size = 0
        if len(meta_parts) >= 4 and meta_parts[3].isdigit():
            size = int(meta_parts[3])

        files[path_str] = {
            "path": path_str,
            "suffix": Path(path_str).suffix.lower(),
            "size": size,
        }

    return files


def _list_changed_source_files(
    repo_path: Path,
    commit: str,
    extensions: list[str],
) -> list[str]:
    proc = _run_git(
        repo_path,
        ["diff-tree", "--no-commit-id", "--name-only", "--diff-filter=AMR", "-r", commit],
    )
    if proc.returncode != 0:
        return []

    return [
        path.strip()
        for path in proc.stdout.splitlines()
        if path.strip() and should_include_source_path(path.strip(), extensions)
    ]


def _git_show_text(repo_path: Path, commit: str, path_str: str) -> str | None:
    proc = _run_git(repo_path, ["show", f"{commit}:{path_str}"])
    if proc.returncode != 0:
        return None
    return proc.stdout


def _get_metrics_for_snapshot_file(
    repo_path: Path,
    commit: str,
    path_str: str,
    metrics_cache: dict[tuple[str, str], StructuralOpacity | None],
) -> StructuralOpacity | None:
    key = (commit, path_str)
    if key in metrics_cache:
        return metrics_cache[key]

    text = _git_show_text(repo_path, commit, path_str)
    if text is None:
        metrics_cache[key] = None
        return None

    metrics_cache[key] = compute_file_opacity_from_text(
        text,
        name=path_str,
        snapshot_id=commit,
    )
    return metrics_cache[key]


def _select_control_file(
    repo_path: Path,
    parent: str,
    case_file: dict[str, object],
    candidate_files: list[dict[str, object]],
    metrics_cache: dict[tuple[str, str], StructuralOpacity | None],
    *,
    min_loc: int,
) -> tuple[dict[str, object], StructuralOpacity] | None:
    same_suffix = [item for item in candidate_files if item["suffix"] == case_file["suffix"]]
    pool = same_suffix or candidate_files

    ranked = sorted(
        pool,
        key=lambda item: (
            abs(math.log1p(int(item["size"])) - math.log1p(int(case_file["size"]))),
            item["path"],
        ),
    )

    for candidate in ranked:
        control_metrics = _get_metrics_for_snapshot_file(
            repo_path,
            parent,
            str(candidate["path"]),
            metrics_cache,
        )
        if control_metrics is None or control_metrics.total_loc < min_loc:
            continue
        return candidate, control_metrics

    return None


def _extract_nvd_event_candidates(
    vuln_items: list[dict],
    expected_slug: str | None,
) -> dict[str, set[str]]:
    """Collect NVD-linked commit candidates keyed by vulnerability identifier."""
    if expected_slug is None:
        return {}

    commit_to_cves = extract_commit_refs_from_nvd_items(vuln_items, expected_slug)
    cve_to_commits: dict[str, set[str]] = {}
    for commit, cve_ids in commit_to_cves.items():
        for cve_id in cve_ids:
            cve_to_commits.setdefault(cve_id, set()).add(commit)
    return cve_to_commits


def _extract_explicit_id_event_candidates(repo_path: Path) -> dict[str, set[str]]:
    """Collect commits whose subject or body explicitly names a vulnerability identifier."""
    proc = _run_git(
        repo_path,
        [
            "log",
            "--all",
            "--regexp-ignore-case",
            "--grep",
            "CVE-",
            "--grep",
            "GHSA-",
            "--format=%H%x1f%B%x1e",
        ],
    )
    if proc.returncode != 0:
        return {}

    event_to_commits: dict[str, set[str]] = {}
    for block in proc.stdout.split("\x1e"):
        block = block.strip()
        if not block or "\x1f" not in block:
            continue
        commit, message = block.split("\x1f", 1)
        security_ids = extract_security_ids(message)
        for security_id in security_ids:
            event_to_commits.setdefault(security_id, set()).add(commit.strip())

    return event_to_commits


def _canonicalise_event_candidate_map(
    candidates: dict[str, set[str]],
    alias_map: dict[str, str],
) -> dict[str, set[str]]:
    """Collapse event identifiers through an alias map."""
    if not alias_map:
        return {event_id.upper(): set(commits) for event_id, commits in candidates.items()}

    canonical: dict[str, set[str]] = {}
    for event_id, commits in candidates.items():
        key = alias_map.get(event_id.upper(), event_id.upper())
        canonical.setdefault(key, set()).update(commits)
    return canonical


def _select_primary_event_commit(
    repo_path: Path,
    entry: CorpusEntry,
    commits: set[str],
    metadata_cache: dict[str, tuple[str, tuple[str, ...], int] | None],
) -> str | None:
    """Choose the earliest single-parent source-touching commit for an event."""
    candidates: list[tuple[int, str]] = []

    for commit in commits:
        if commit not in metadata_cache:
            parents = _get_commit_parents(repo_path, commit)
            if len(parents) != 1:
                metadata_cache[commit] = None
            else:
                changed_files = tuple(
                    _list_changed_source_files(repo_path, commit, entry.source_extensions)
                )
                if not changed_files:
                    metadata_cache[commit] = None
                else:
                    metadata_cache[commit] = (
                        parents[0],
                        changed_files,
                        _get_commit_epoch(repo_path, commit),
                    )

        meta = metadata_cache[commit]
        if meta is None:
            continue
        _, _, epoch = meta
        candidates.append((epoch, commit))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def collect_ground_truth_events(
    repo_path: Path,
    entry: CorpusEntry,
    cache_dir: Path,
    *,
    ground_truth_policy: str = "nvd_commit_refs",
    osv_cache_dir: Path | None = None,
    osv_rate_limit: float = 0.1,
    osv_query_batch_size: int = 100,
) -> list[GroundTruthEvent]:
    """Collect event-labeled fixing commits under an explicit event-definition policy."""
    if ground_truth_policy not in GROUND_TRUTH_POLICIES:
        raise ValueError(f"Unknown ground_truth_policy: {ground_truth_policy}")

    expected_slug = normalise_github_slug(entry.git_url)
    vuln_items: list[dict] = []
    cache_file = cache_file_for_cpe(entry.cpe_id, cache_dir)
    if cache_file.exists():
        vuln_items = json.loads(cache_file.read_text())

    nvd_commit_to_cves = (
        extract_commit_refs_from_nvd_items(vuln_items, expected_slug) if expected_slug else {}
    )
    if ground_truth_policy == "nvd_commit_refs":
        return [
            GroundTruthEvent(
                event_id=";".join(sorted(cve_ids)),
                commit=commit,
                vulnerability_ids=tuple(sorted(cve_ids)),
                source="nvd_ref",
            )
            for commit, cve_ids in sorted(nvd_commit_to_cves.items())
        ]

    nvd_event_candidates = _extract_nvd_event_candidates(vuln_items, expected_slug)
    explicit_event_candidates = (
        _extract_explicit_id_event_candidates(repo_path)
        if ground_truth_policy in {"balanced_explicit_id_event", "expanded_advisory_event"}
        else {}
    )
    osv_event_candidates: dict[str, dict[str, set[str]]] = {}
    alias_map: dict[str, str] = {}
    if ground_truth_policy == "expanded_advisory_event" and osv_cache_dir is not None:
        repo_urls = repo_url_variants(entry.git_url)
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
        osv_event_candidates = extract_osv_event_candidates(osv_records, repo_urls)
        nvd_event_candidates = _canonicalise_event_candidate_map(nvd_event_candidates, alias_map)
        explicit_event_candidates = _canonicalise_event_candidate_map(
            explicit_event_candidates,
            alias_map,
        )

    all_event_ids = set(nvd_event_candidates)
    if ground_truth_policy in {"balanced_explicit_id_event", "expanded_advisory_event"}:
        all_event_ids |= set(explicit_event_candidates)
    if ground_truth_policy == "expanded_advisory_event":
        all_event_ids |= set(osv_event_candidates)

    metadata_cache: dict[str, tuple[str, tuple[str, ...], int] | None] = {}
    events: list[GroundTruthEvent] = []

    for event_id in sorted(all_event_ids):
        candidate_commits = set(nvd_event_candidates.get(event_id, set()))
        sources: set[str] = set()
        if candidate_commits:
            sources.add("nvd_ref")

        if ground_truth_policy in {"balanced_explicit_id_event", "expanded_advisory_event"}:
            explicit_commits = explicit_event_candidates.get(event_id, set())
            if explicit_commits:
                candidate_commits |= explicit_commits
                sources.add("explicit_id")

        if ground_truth_policy == "expanded_advisory_event":
            osv_entry = osv_event_candidates.get(event_id)
            if osv_entry:
                candidate_commits |= set(osv_entry.get("commits", set()))
                sources |= set(osv_entry.get("sources", set()))

        selected_commit = _select_primary_event_commit(
            repo_path,
            entry,
            candidate_commits,
            metadata_cache,
        )
        if selected_commit is None:
            continue

        events.append(
            GroundTruthEvent(
                event_id=event_id,
                commit=selected_commit,
                vulnerability_ids=(event_id,),
                source="+".join(sorted(sources)),
            )
        )

    return events


def match_case_control_pairs(
    repo_path: Path,
    entry: CorpusEntry,
    cache_dir: Path,
    *,
    min_loc: int = 20,
    ground_truth_policy: str = "nvd_commit_refs",
    osv_cache_dir: Path | None = None,
    osv_rate_limit: float = 0.1,
    osv_query_batch_size: int = 100,
) -> pd.DataFrame:
    """Build matched file-level pairs for one repository."""
    if (
        ground_truth_policy in {"nvd_commit_refs", "strict_nvd_event"}
        and normalise_github_slug(entry.git_url) is None
    ):
        logger.info("Skipping %s: non-GitHub remote", entry.name)
        return pd.DataFrame()

    if ground_truth_policy in {"nvd_commit_refs", "strict_nvd_event"}:
        cache_file = cache_file_for_cpe(entry.cpe_id, cache_dir)
        if not cache_file.exists():
            logger.info("Skipping %s: no local NVD cache", entry.name)
            return pd.DataFrame()

    events = collect_ground_truth_events(
        repo_path,
        entry,
        cache_dir,
        ground_truth_policy=ground_truth_policy,
        osv_cache_dir=osv_cache_dir,
        osv_rate_limit=osv_rate_limit,
        osv_query_batch_size=osv_query_batch_size,
    )
    if not events:
        logger.info("Skipping %s: no usable ground-truth events under %s", entry.name, ground_truth_policy)
        return pd.DataFrame()

    snapshot_file_cache: dict[str, dict[str, dict[str, object]]] = {}
    metrics_cache: dict[tuple[str, str], StructuralOpacity | None] = {}
    rows: list[dict[str, object]] = []

    for event in events:
        commit = event.commit
        parent = _get_first_parent(repo_path, commit)
        if parent is None:
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

        changed_files = _list_changed_source_files(repo_path, commit, entry.source_extensions)
        if not changed_files:
            continue

        changed_set = set(changed_files)
        used_controls: set[str] = set()

        for case_path in changed_files:
            case_file = files_at_parent.get(case_path)
            if case_file is None:
                continue

            case_metrics = _get_metrics_for_snapshot_file(
                repo_path,
                parent,
                case_path,
                metrics_cache,
            )
            if case_metrics is None or case_metrics.total_loc < min_loc:
                continue

            candidate_files = [
                item
                for path_str, item in files_at_parent.items()
                if path_str not in changed_set and path_str not in used_controls
            ]
            if not candidate_files:
                continue

            control_match = _select_control_file(
                repo_path,
                parent,
                case_file,
                candidate_files,
                metrics_cache,
                min_loc=min_loc,
            )
            if control_match is None:
                continue

            control_file, control_metrics = control_match
            control_path = str(control_file["path"])
            used_controls.add(control_path)

            rows.append(
                {
                    "repo": entry.name,
                    "commit": commit,
                    "parent": parent,
                    "event_id": event.event_id,
                    "ground_truth_source": event.source,
                    "ground_truth_policy": ground_truth_policy,
                    "cve_ids": ";".join(event.vulnerability_ids),
                    "case_file": case_path,
                    "control_file": control_path,
                    "case_size_bytes": case_metrics.total_bytes,
                    "control_size_bytes": control_metrics.total_bytes,
                    "case_loc": case_metrics.total_loc,
                    "control_loc": control_metrics.total_loc,
                    "loc_ratio": (
                        case_metrics.total_loc / control_metrics.total_loc
                        if control_metrics.total_loc
                        else math.nan
                    ),
                    "case_composite": case_metrics.composite_score,
                    "control_composite": control_metrics.composite_score,
                    "delta_composite": case_metrics.composite_score - control_metrics.composite_score,
                    "case_ci_gzip": case_metrics.ci_gzip,
                    "control_ci_gzip": control_metrics.ci_gzip,
                    "delta_ci_gzip": case_metrics.ci_gzip - control_metrics.ci_gzip,
                    "case_entropy": case_metrics.shannon_entropy,
                    "control_entropy": control_metrics.shannon_entropy,
                    "delta_entropy": case_metrics.shannon_entropy - control_metrics.shannon_entropy,
                    "case_cc_density": case_metrics.cyclomatic_density,
                    "control_cc_density": control_metrics.cyclomatic_density,
                    "delta_cc_density": (
                        case_metrics.cyclomatic_density - control_metrics.cyclomatic_density
                    ),
                    "case_halstead": case_metrics.halstead_volume,
                    "control_halstead": control_metrics.halstead_volume,
                    "delta_halstead": case_metrics.halstead_volume - control_metrics.halstead_volume,
                }
            )

    return pd.DataFrame(rows)


def summarise_case_control_pairs(pairs: pd.DataFrame) -> dict[str, object]:
    """Summarise matched pair results with conservative non-parametric tests."""
    if pairs.empty:
        return {
            "n_pairs": 0,
            "n_commits": 0,
            "n_repos": 0,
        }

    delta = pairs["delta_composite"].dropna()
    nonzero = delta[delta != 0]
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())

    summary: dict[str, object] = {
        "n_pairs": int(len(pairs)),
        "n_commits": int(pairs["commit"].nunique()),
        "n_repos": int(pairs["repo"].nunique()),
        "mean_delta_composite": float(delta.mean()),
        "median_delta_composite": float(delta.median()),
        "positive_share": float((delta > 0).mean()),
        "mean_case_loc": float(pairs["case_loc"].mean()),
        "mean_control_loc": float(pairs["control_loc"].mean()),
        "median_loc_ratio": float(pairs["loc_ratio"].median()),
    }
    if "event_id" in pairs.columns:
        summary["n_events"] = int(pairs["event_id"].nunique())
    if "ground_truth_policy" in pairs.columns:
        policy = str(pairs["ground_truth_policy"].iloc[0])
        summary["ground_truth_policy"] = policy
        summary.update(describe_ground_truth_policy(policy))

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

    primary_cluster_col = "event_id" if "event_id" in pairs.columns else "commit"
    summary["bootstrap_primary_cluster"] = clustered_delta_bootstrap(
        pairs,
        cluster_col=primary_cluster_col,
    )
    summary["bootstrap_repo_cluster"] = clustered_delta_bootstrap(
        pairs,
        cluster_col="repo",
    )

    return summary


def summarise_commit_level_deltas(commit_summary: pd.DataFrame) -> dict[str, object]:
    """Summarise commit-event mean deltas as a higher-level observation set."""
    if commit_summary.empty:
        return {
            "n_commit_events": 0,
            "n_unique_commits": 0,
            "n_repos": 0,
        }

    delta = commit_summary["mean_delta_composite"].dropna()
    nonzero = delta[delta != 0]
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())

    summary: dict[str, object] = {
        "n_commit_events": int(len(commit_summary)),
        "n_unique_commits": int(commit_summary["commit"].nunique()),
        "n_repos": int(commit_summary["repo"].nunique()),
        "mean_delta_composite": float(delta.mean()),
        "median_delta_composite": float(delta.median()),
        "positive_share": float((delta > 0).mean()),
    }
    if "event_id" in commit_summary.columns:
        summary["n_events"] = int(commit_summary["event_id"].nunique())
    if "ground_truth_policy" in commit_summary.columns:
        policy = str(commit_summary["ground_truth_policy"].iloc[0])
        summary["ground_truth_policy"] = policy
        summary.update(describe_ground_truth_policy(policy))

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

    primary_cluster_col = "event_id" if "event_id" in commit_summary.columns else "commit"
    summary["bootstrap_primary_cluster"] = clustered_delta_bootstrap(
        commit_summary,
        cluster_col=primary_cluster_col,
        delta_col="mean_delta_composite",
    )
    summary["bootstrap_repo_cluster"] = clustered_delta_bootstrap(
        commit_summary,
        cluster_col="repo",
        delta_col="mean_delta_composite",
    )

    return summary
