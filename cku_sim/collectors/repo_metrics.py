"""Orchestrate metric extraction across a corpus of repositories."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cku_sim.core.codebase import (
    collect_source_files,
    count_loc,
    source_to_archive,
    token_frequencies,
)
from cku_sim.core.config import Config, CorpusEntry
from cku_sim.core.opacity import StructuralOpacity
from cku_sim.metrics.compressibility import compressibility_index
from cku_sim.metrics.cyclomatic import cyclomatic_density
from cku_sim.metrics.entropy import shannon_entropy_from_frequencies
from cku_sim.metrics.halstead import halstead_volume_normalised

logger = logging.getLogger(__name__)


def compute_opacity(
    repo_path: Path,
    entry: CorpusEntry,
    snapshot_id: str = "HEAD",
    config: Config | None = None,
) -> StructuralOpacity | None:
    """Compute all opacity metrics for a single repository snapshot.

    Args:
        repo_path: Path to cloned repository.
        entry: Corpus entry metadata.
        snapshot_id: Git ref identifier.
        config: Optional config for weights and algorithms.

    Returns:
        StructuralOpacity instance, or None on failure.
    """
    config = config or Config()

    files = collect_source_files(
        repo_path,
        extensions=entry.source_extensions,
        subdirectory=entry.subdirectory,
    )

    if not files:
        logger.warning(f"No source files found for {entry.name}")
        return None

    logger.info(f"Computing opacity for {entry.name}: {len(files)} files")

    # Source archive for compression metrics
    archive = source_to_archive(files, repo_path)
    loc = count_loc(files)

    # Compressibility indices
    ci_gzip = compressibility_index(archive, method="gzip")
    ci_lzma = compressibility_index(archive, method="lzma")
    ci_zstd = compressibility_index(archive, method="zstd")

    # Shannon entropy
    freqs = token_frequencies(files)
    entropy = shannon_entropy_from_frequencies(freqs)

    # Cyclomatic density
    cc_density = cyclomatic_density(files, loc)

    # Halstead volume (normalised)
    halstead = halstead_volume_normalised(files)

    opacity = StructuralOpacity(
        name=entry.name,
        snapshot_id=snapshot_id,
        total_bytes=len(archive),
        total_loc=loc,
        num_files=len(files),
        ci_gzip=ci_gzip,
        ci_lzma=ci_lzma,
        ci_zstd=ci_zstd,
        shannon_entropy=entropy,
        cyclomatic_density=cc_density,
        halstead_volume=halstead,
    )

    opacity.compute_composite(config.composite_weights)

    logger.info(
        f"  {entry.name}: CI(gzip)={ci_gzip:.4f}, entropy={entropy:.4f}, "
        f"CC/LOC={cc_density:.6f}, composite={opacity.composite_score:.4f}"
    )

    return opacity


def compute_corpus_opacity(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    config: Config | None = None,
) -> pd.DataFrame:
    """Compute opacity metrics for all repos in the corpus.

    Args:
        repo_paths: Dict mapping repo name to local path.
        corpus: List of corpus entries.
        config: Configuration.

    Returns:
        DataFrame with one row per repo, columns for all metrics.
    """
    from cku_sim.collectors.git_collector import get_head_hash

    config = config or Config()
    results = []

    for entry in corpus:
        path = repo_paths.get(entry.name)
        if path is None:
            logger.warning(f"No path for {entry.name}, skipping")
            continue

        snapshot_id = get_head_hash(path)
        opacity = compute_opacity(path, entry, snapshot_id, config)
        if opacity:
            row = opacity.to_dict()
            row["category"] = entry.category
            row["cpe_id"] = entry.cpe_id
            results.append(row)

    return pd.DataFrame(results)
