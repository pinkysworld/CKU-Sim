"""Discover and validate larger GitHub-based research corpora."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
DEFAULT_LANGUAGES = ["C", "C++", "Go", "Rust", "Java", "Python"]
LANGUAGE_EXTENSIONS = {
    "C": [".c", ".h"],
    "C++": [".cpp", ".hpp", ".cc", ".cxx", ".h", ".hh", ".hxx"],
    "Go": [".go"],
    "Rust": [".rs"],
    "Java": [".java"],
    "Python": [".py"],
    "JavaScript": [".js", ".mjs", ".cjs"],
    "TypeScript": [".ts", ".tsx"],
}
NON_SOFTWARE_RE = re.compile(
    r"\b(?:awesome|guide|books?|primer|tutorials?|interviews?|leetcode|"
    r"public-apis|algorithms?|roadmap|cheatsheet|top-charts|resources?|notes?)\b",
    re.IGNORECASE,
)
AI_APP_RE = re.compile(
    r"\b(?:ai|ml|openai|gpt|chatgpt|claude|gemini|llms?|ollama|autogpt|"
    r"nanogpt|langchain|stable-diffusion|diffusion|transformers?|whisper|rag|webui|"
    r"generative-ai|language-models?|large-language-models?|machine-learning|"
    r"deep-learning|neural)\b",
    re.IGNORECASE,
)
DESCRIPTION_REDACTIONS = [
    re.compile(r"\boh-my-[a-z][a-z-]*\b", re.IGNORECASE),
]


def sanitize_repository_text(value: object) -> str | None:
    """Normalize externally sourced text for publication-facing outputs."""
    if value is None:
        return None
    text = str(value)
    for pattern in DESCRIPTION_REDACTIONS:
        text = pattern.sub("a Rust workflow", text)
    return text


def infer_source_extensions(language: str | None) -> list[str]:
    """Infer likely source extensions from a primary language label."""
    if not language:
        return [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]
    return LANGUAGE_EXTENSIONS.get(language, [f".{language.lower()}"])


def slug_to_local_name(full_name: str) -> str:
    """Convert owner/repo into a filesystem-safe unique corpus name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", full_name.strip().lower())


def github_headers(token: str | None = None) -> dict[str, str]:
    """Build standard GitHub API headers."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "CKU-Sim/0.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def is_research_corpus_candidate(item: dict[str, object], *, min_size_kb: int = 500) -> bool:
    """Apply lightweight heuristics to exclude non-software repositories."""
    combined = " ".join(
        str(value or "")
        for value in [item.get("full_name"), item.get("description"), " ".join(item.get("topics", []))]
    )
    if NON_SOFTWARE_RE.search(combined):
        return False
    if AI_APP_RE.search(combined):
        return False
    size_kb = int(item.get("size_kb", 0) or 0)
    if size_kb < min_size_kb:
        return False
    return True


def search_top_repositories(
    language: str,
    *,
    per_language: int = 20,
    min_stars: int = 2000,
    token: str | None = None,
    max_pages: int = 3,
    pause_seconds: float = 1.0,
) -> list[dict[str, object]]:
    """Search GitHub for high-signal repositories in one language."""
    selected: list[dict[str, object]] = []
    page = 1
    while len(selected) < per_language and page <= max_pages:
        params = {
            "q": f"language:{language} stars:>={min_stars} archived:false fork:false mirror:false",
            "sort": "stars",
            "order": "desc",
            "per_page": min(100, per_language * 2),
            "page": page,
        }
        response = requests.get(
            GITHUB_SEARCH_API,
            headers=github_headers(token),
            params=params,
            timeout=120,
        )
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_epoch = int(response.headers.get("X-RateLimit-Reset", "0") or "0")
            wait_seconds = max(1, reset_epoch - int(time.time()))
            logger.warning(
                "GitHub search rate limit reached; waiting %ds before retry",
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items:
            break

        for item in items:
            selected.append(
                {
                    "full_name": str(item["full_name"]),
                    "name": slug_to_local_name(str(item["full_name"])),
                    "git_url": str(item["clone_url"]),
                    "html_url": str(item["html_url"]),
                    "description": sanitize_repository_text(item.get("description")),
                    "primary_language": item.get("language"),
                    "stars": int(item.get("stargazers_count", 0)),
                    "topics": item.get("topics", []),
                    "size_kb": int(item.get("size", 0)),
                    "updated_at": item.get("updated_at"),
                    "pushed_at": item.get("pushed_at"),
                    "default_branch": item.get("default_branch"),
                    "source_extensions": infer_source_extensions(item.get("language")),
                }
            )
            if not is_research_corpus_candidate(selected[-1]):
                selected.pop()
                continue
            if len(selected) >= per_language:
                break
        page += 1
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    return selected[:per_language]


def discover_large_corpus(
    *,
    languages: list[str] | None = None,
    per_language: int = 20,
    min_stars: int = 2000,
    token: str | None = None,
    max_pages: int = 3,
    pause_seconds: float = 1.0,
) -> list[dict[str, object]]:
    """Collect and deduplicate candidate repositories across languages."""
    languages = languages or DEFAULT_LANGUAGES
    discovered: dict[str, dict[str, object]] = {}

    for language in languages:
        logger.info("Searching GitHub for %s repositories...", language)
        for item in search_top_repositories(
            language,
            per_language=per_language,
            min_stars=min_stars,
            token=token,
            max_pages=max_pages,
            pause_seconds=pause_seconds,
        ):
            full_name = str(item["full_name"])
            if full_name not in discovered:
                discovered[full_name] = item

    rows = sorted(
        discovered.values(),
        key=lambda item: (-int(item.get("stars", 0)), str(item.get("full_name"))),
    )
    return rows


def count_remote_tags(git_url: str) -> int | None:
    """Count refs under refs/tags without cloning the repository."""
    proc = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", git_url],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return None
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


def validate_remote_tags(
    candidates: list[dict[str, object]],
    *,
    min_remote_tags: int = 10,
    pause_seconds: float = 0.1,
) -> list[dict[str, object]]:
    """Annotate candidates with remote tag counts and apply a minimum threshold."""
    validated: list[dict[str, object]] = []
    for idx, item in enumerate(candidates, start=1):
        tag_count = count_remote_tags(str(item["git_url"]))
        row = dict(item)
        row["remote_tag_count"] = tag_count
        logger.info(
            "Validating tags %d/%d: %s -> %s",
            idx,
            len(candidates),
            item["full_name"],
            "n/a" if tag_count is None else tag_count,
        )
        if tag_count is not None and tag_count >= min_remote_tags:
            validated.append(row)
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return validated


def build_manifest_entries(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    """Project GitHub discovery rows into CorpusEntry-compatible manifest rows."""
    entries: list[dict[str, object]] = []
    for item in candidates:
        entries.append(
            {
                "name": item["name"],
                "full_name": item["full_name"],
                "git_url": item["git_url"],
                "cpe_id": None,
                "category": "expanded",
                "primary_language": item.get("primary_language"),
                "stars": int(item.get("stars", 0)),
                "source_extensions": list(item.get("source_extensions", [])),
            }
        )
    return entries


def write_corpus_outputs(
    results_dir: Path,
    *,
    candidates: list[dict[str, object]],
    validated: list[dict[str, object]],
) -> None:
    """Write CSV, JSON, and YAML outputs for the expanded corpus."""
    import pandas as pd

    results_dir.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.DataFrame(candidates)
    validated_df = pd.DataFrame(validated)
    manifest = {"corpus": build_manifest_entries(validated)}

    candidate_df.to_csv(results_dir / "candidate_repositories.csv", index=False)
    validated_df.to_csv(results_dir / "validated_repositories.csv", index=False)
    (results_dir / "candidate_repositories.json").write_text(
        json.dumps(candidates, indent=2)
    )
    (results_dir / "validated_repositories.json").write_text(
        json.dumps(validated, indent=2)
    )
    (results_dir / "expanded_corpus_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    (results_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_candidates": len(candidates),
                "n_validated": len(validated),
                "languages": sorted(
                    {str(item.get("primary_language")) for item in validated if item.get("primary_language")}
                ),
                "top_repositories": [
                    {
                        "full_name": item["full_name"],
                        "stars": item.get("stars"),
                        "remote_tag_count": item.get("remote_tag_count"),
                    }
                    for item in validated[:20]
                ],
            },
            indent=2,
        )
    )


def default_github_token() -> str | None:
    """Return an optional token from the environment."""
    return os.environ.get("GITHUB_TOKEN") or None
