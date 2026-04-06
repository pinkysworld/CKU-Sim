"""Experiment 4: Temporal evolution of opacity across Git history.

Tracks how structural opacity changes over time for selected projects.
This is an illustrative analysis — not core to the paper's argument,
but provides nice figures showing opacity trends.

Usage:
    python -m experiments.e04_temporal_evolution --config experiments/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from git import Repo

from cku_sim.core.codebase import collect_source_files, source_to_archive, count_loc
from cku_sim.core.config import Config, DEFAULT_CORPUS, CorpusEntry
from cku_sim.metrics.compressibility import compressibility_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Projects for temporal analysis (smaller repos that are fast to traverse)
TEMPORAL_PROJECTS = ["curl", "redis", "zlib", "jq", "libsodium", "openssh"]


def sample_commits(repo_path: Path, n_samples: int = 30) -> list[tuple[str, str]]:
    """Sample N evenly-spaced commits from repository history.

    Returns:
        List of (commit_hash, ISO date) tuples, oldest first.
    """
    repo = Repo(str(repo_path))

    # Get all commits on default branch
    commits = list(repo.iter_commits(max_count=10_000))
    commits.reverse()  # oldest first

    if len(commits) <= n_samples:
        return [(c.hexsha[:12], c.committed_datetime.isoformat()) for c in commits]

    # Evenly sample
    indices = np.linspace(0, len(commits) - 1, n_samples, dtype=int)
    return [
        (commits[i].hexsha[:12], commits[i].committed_datetime.isoformat())
        for i in indices
    ]


def compute_temporal_opacity(
    repo_path: Path,
    entry: CorpusEntry,
    n_samples: int = 30,
) -> pd.DataFrame:
    """Compute compressibility index at sampled commits across history.

    Note: requires a full (non-shallow) clone.
    """
    repo = Repo(str(repo_path))
    samples = sample_commits(repo_path, n_samples)

    rows = []
    for commit_hash, date_str in samples:
        try:
            repo.git.checkout(commit_hash, force=True)
        except Exception as e:
            logger.warning(f"  Cannot checkout {commit_hash}: {e}")
            continue

        files = collect_source_files(
            repo_path,
            extensions=entry.source_extensions,
            subdirectory=entry.subdirectory,
        )

        if not files:
            continue

        archive = source_to_archive(files, repo_path)
        loc = count_loc(files)
        ci = compressibility_index(archive, "gzip")

        rows.append({
            "name": entry.name,
            "commit": commit_hash,
            "date": date_str,
            "num_files": len(files),
            "total_bytes": len(archive),
            "loc": loc,
            "ci_gzip": ci,
        })

        logger.debug(f"  {commit_hash} ({date_str[:10]}): {len(files)} files, CI={ci:.4f}")

    # Restore HEAD
    try:
        repo.git.checkout("-", force=True)
    except Exception:
        pass

    return pd.DataFrame(rows)


def plot_temporal(
    temporal_data: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot opacity evolution over time for multiple projects."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in temporal_data.items():
        if df.empty:
            continue
        dates = pd.to_datetime(df["date"])
        ax.plot(dates, df["ci_gzip"], "-o", markersize=3, label=name, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Compressibility Index (gzip)")
    ax.set_title("Structural Opacity Evolution Over Time")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Temporal evolution")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=30, help="Commits to sample per repo")
    parser.add_argument(
        "--projects", nargs="*", default=TEMPORAL_PROJECTS,
        help="Projects to analyse (must be in corpus)"
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    corpus = config.corpus if config.corpus else DEFAULT_CORPUS
    results_dir = config.results_dir / "e04_temporal"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment 4: Temporal Opacity Evolution")
    logger.info(f"Projects: {args.projects}")
    logger.info("=" * 60)

    logger.warning(
        "NOTE: This experiment requires FULL (non-shallow) clones.\n"
        "If you cloned with --depth=1 in Experiment 2, re-clone with:\n"
        "  git clone <url>  (without --depth)\n"
        "Or set shallow=False in the config."
    )

    corpus_map = {e.name: e for e in corpus}
    temporal_data = {}

    for project_name in args.projects:
        entry = corpus_map.get(project_name)
        if entry is None:
            logger.warning(f"Project {project_name} not in corpus, skipping")
            continue

        repo_path = config.raw_dir / entry.name
        if not repo_path.exists():
            logger.warning(f"Repo not cloned: {repo_path}, skipping")
            continue

        logger.info(f"Analysing {project_name}...")
        df = compute_temporal_opacity(repo_path, entry, n_samples=args.n_samples)
        temporal_data[project_name] = df

        if not df.empty:
            df.to_parquet(results_dir / f"{project_name}_temporal.parquet")
            df.to_csv(results_dir / f"{project_name}_temporal.csv", index=False)
            logger.info(f"  {project_name}: {len(df)} snapshots, "
                       f"CI range [{df['ci_gzip'].min():.4f}, {df['ci_gzip'].max():.4f}]")

    # Plot
    if temporal_data:
        plot_temporal(temporal_data, results_dir / "temporal_evolution.pdf")
        plot_temporal(temporal_data, results_dir / "temporal_evolution.png")
        logger.info("Figures saved.")

    logger.info("Experiment 4 complete.")


if __name__ == "__main__":
    main()
