"""File-level matched case-control analysis for vulnerability-fixing commits."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy import stats

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
        env=env,
    )


def _get_first_parent(repo_path: Path, commit: str) -> str | None:
    proc = _run_git(repo_path, ["rev-list", "--parents", "-n", "1", commit])
    if proc.returncode != 0:
        return None
    words = proc.stdout.strip().split()
    return words[1] if len(words) > 1 else None


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


def match_case_control_pairs(
    repo_path: Path,
    entry: CorpusEntry,
    cache_dir: Path,
    *,
    min_loc: int = 20,
) -> pd.DataFrame:
    """Build matched file-level pairs for one repository."""
    expected_slug = normalise_github_slug(entry.git_url)
    if expected_slug is None:
        logger.info("Skipping %s: non-GitHub remote", entry.name)
        return pd.DataFrame()

    cache_file = cache_file_for_cpe(entry.cpe_id, cache_dir)
    if not cache_file.exists():
        logger.info("Skipping %s: no local NVD cache", entry.name)
        return pd.DataFrame()

    vuln_items = json.loads(cache_file.read_text())
    commit_to_cves = extract_commit_refs_from_nvd_items(vuln_items, expected_slug)
    if not commit_to_cves:
        logger.info("Skipping %s: no matching commit references", entry.name)
        return pd.DataFrame()

    snapshot_file_cache: dict[str, dict[str, dict[str, object]]] = {}
    metrics_cache: dict[tuple[str, str], StructuralOpacity | None] = {}
    rows: list[dict[str, object]] = []

    for commit, cve_ids in sorted(commit_to_cves.items()):
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
                    "cve_ids": ";".join(sorted(cve_ids)),
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

    return summary


def summarise_commit_level_deltas(commit_summary: pd.DataFrame) -> dict[str, object]:
    """Summarise commit-level mean deltas as a higher-level observation set."""
    if commit_summary.empty:
        return {
            "n_commits": 0,
            "n_repos": 0,
        }

    delta = commit_summary["mean_delta_composite"].dropna()
    nonzero = delta[delta != 0]
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())

    summary: dict[str, object] = {
        "n_commits": int(len(commit_summary)),
        "n_repos": int(commit_summary["repo"].nunique()),
        "mean_delta_composite": float(delta.mean()),
        "median_delta_composite": float(delta.median()),
        "positive_share": float((delta > 0).mean()),
    }

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

    return summary
