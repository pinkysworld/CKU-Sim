"""Experiment 11: build a larger external corpus manifest."""

from __future__ import annotations

import argparse
import logging

from cku_sim.collectors.github_corpus import (
    DEFAULT_LANGUAGES,
    default_github_token,
    discover_large_corpus,
    validate_remote_tags,
    write_corpus_outputs,
)
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 11: large corpus builder")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated languages to search",
    )
    parser.add_argument(
        "--per-language",
        type=int,
        default=24,
        help="Repositories to retain per language before deduplication",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=3000,
        help="Minimum GitHub stars for discovery",
    )
    parser.add_argument(
        "--min-remote-tags",
        type=int,
        default=10,
        help="Minimum remote tag count to include in the manifest",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Maximum GitHub search pages per language",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e11_large_corpus",
        help="Results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    results_dir = config.results_dir / args.results_subdir
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    token = default_github_token()

    logger.info("=" * 60)
    logger.info("Experiment 11: Large external corpus builder")
    logger.info("=" * 60)
    logger.info(
        "Languages=%s per_language=%d min_stars=%d min_remote_tags=%d",
        languages,
        args.per_language,
        args.min_stars,
        args.min_remote_tags,
    )

    candidates = discover_large_corpus(
        languages=languages,
        per_language=args.per_language,
        min_stars=args.min_stars,
        token=token,
        max_pages=args.max_pages,
    )
    validated = validate_remote_tags(
        candidates,
        min_remote_tags=args.min_remote_tags,
    )
    write_corpus_outputs(results_dir, candidates=candidates, validated=validated)

    logger.info(
        "Candidate repositories: %d, validated repositories: %d",
        len(candidates),
        len(validated),
    )
    logger.info("Manifest written to %s", results_dir / "expanded_corpus_manifest.yaml")


if __name__ == "__main__":
    main()
