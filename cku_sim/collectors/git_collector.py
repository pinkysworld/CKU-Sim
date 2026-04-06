"""Clone and manage Git repositories for analysis."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from git import Repo

from cku_sim.core.config import CorpusEntry

logger = logging.getLogger(__name__)


def clone_repo(entry: CorpusEntry, dest_dir: Path, shallow: bool = True) -> Path:
    """Clone a repository if not already present.

    Args:
        entry: Corpus entry with git_url.
        dest_dir: Parent directory for clones.
        shallow: If True, use depth=1 for faster cloning.

    Returns:
        Path to cloned repository.
    """
    repo_path = dest_dir / entry.name
    if repo_path.exists() and (repo_path / ".git").exists():
        logger.info(f"Repository {entry.name} already cloned at {repo_path}")
        return repo_path

    logger.info(f"Cloning {entry.name} from {entry.git_url}...")
    kwargs = {"depth": 1} if shallow else {}

    try:
        Repo.clone_from(entry.git_url, str(repo_path), **kwargs)
    except Exception as e:
        logger.error(f"Failed to clone {entry.name}: {e}")
        if repo_path.exists():
            shutil.rmtree(repo_path)
        raise

    logger.info(f"Cloned {entry.name} ({_dir_size_mb(repo_path):.1f} MB)")
    return repo_path


def clone_corpus(
    corpus: list[CorpusEntry], dest_dir: Path, shallow: bool = True
) -> dict[str, Path]:
    """Clone all repositories in the corpus.

    Returns:
        Dict mapping repo name to local path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for entry in corpus:
        try:
            paths[entry.name] = clone_repo(entry, dest_dir, shallow=shallow)
        except Exception:
            logger.warning(f"Skipping {entry.name} due to clone failure")
    return paths


def get_latest_tag(repo_path: Path) -> str | None:
    """Get the most recent tag from a repository."""
    try:
        repo = Repo(str(repo_path))
        tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
        return str(tags[-1]) if tags else None
    except Exception:
        return None


def get_head_hash(repo_path: Path) -> str:
    """Get the HEAD commit hash."""
    try:
        repo = Repo(str(repo_path))
        return repo.head.commit.hexsha[:12]
    except Exception:
        return "unknown"


def _dir_size_mb(path: Path) -> float:
    """Total size of directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)
