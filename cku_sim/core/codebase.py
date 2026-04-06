"""Codebase representation and source file collection."""

from __future__ import annotations

import io
import tarfile
from collections import Counter
from pathlib import Path


# Directories to always exclude
EXCLUDE_DIRS = {
    ".git", "node_modules", "__pycache__", ".tox", ".eggs",
    "vendor", "third_party", "3rdparty", "external", "deps",
    "test", "tests", "testing", "t", "regress",
    "doc", "docs", "documentation", "man",
    "examples", "samples", "demo", "benchmarks",
}


def collect_source_files(
    repo_path: Path,
    extensions: list[str] | None = None,
    subdirectory: str | None = None,
) -> list[Path]:
    """Collect source files from a repository checkout.

    Args:
        repo_path: Path to cloned repository root.
        extensions: File extensions to include (e.g. [".c", ".h"]).
        subdirectory: Optional subdirectory to restrict to (e.g. "net/").

    Returns:
        List of Paths to source files.
    """
    if extensions is None:
        extensions = [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]

    root = repo_path / subdirectory if subdirectory else repo_path
    if not root.exists():
        return []

    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip excluded directories
        parts = set(path.relative_to(repo_path).parts)
        if parts & EXCLUDE_DIRS:
            continue
        if path.suffix.lower() in extensions:
            files.append(path)

    return sorted(files)


def source_to_archive(files: list[Path], repo_path: Path) -> bytes:
    """Pack source files into a tar archive in memory.

    This produces a deterministic byte stream for compression-based metrics.
    Files are sorted by relative path for reproducibility.

    Args:
        files: List of source file paths.
        repo_path: Repository root (for computing relative paths).

    Returns:
        Tar archive as bytes.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for fpath in sorted(files):
            relpath = str(fpath.relative_to(repo_path))
            tar.add(fpath, arcname=relpath)
    return buf.getvalue()


def count_loc(files: list[Path]) -> int:
    """Count non-blank, non-comment lines across source files.

    Simple heuristic: counts lines that are non-empty after stripping
    and don't start with // or /*.  Good enough for normalisation.
    """
    total = 0
    for fpath in files:
        try:
            text = fpath.read_text(errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
                total += 1
    return total


def token_frequencies(files: list[Path]) -> Counter:
    """Compute token frequency distribution across source files.

    Uses a simple whitespace + punctuation tokeniser.
    For a more language-aware approach, tree-sitter can be integrated later.
    """
    import re

    TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[{}()\[\];,.<>!=+\-*/%&|^~?:#]")
    freq = Counter()

    for fpath in files:
        try:
            text = fpath.read_text(errors="replace")
        except Exception:
            continue
        freq.update(TOKEN_RE.findall(text))

    return freq
