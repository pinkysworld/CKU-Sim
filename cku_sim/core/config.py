"""Global configuration and defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CorpusEntry:
    """A single repository in the corpus."""

    name: str
    git_url: str
    cpe_id: str  # NVD CPE identifier, e.g. "cpe:2.3:a:openssl:openssl:*:*:*:*:*:*:*:*"
    category: str  # "high_opacity", "low_opacity", "mixed"
    subdirectory: str | None = None  # e.g. "net/" for Linux kernel subset
    source_extensions: list[str] = field(
        default_factory=lambda: [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]
    )


@dataclass
class Config:
    """Global configuration for CKU-Sim."""

    # Paths
    data_dir: Path = Path("data")
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # NVD
    nvd_api_key: str | None = None
    nvd_rate_limit: float = 6.0  # seconds between requests (0.6 with API key)

    # OSV
    osv_rate_limit: float = 0.1  # seconds between OSV requests
    osv_query_batch_size: int = 100

    # Metrics
    compression_algorithms: list[str] = field(
        default_factory=lambda: ["gzip", "lzma", "zstd"]
    )
    composite_weights: dict[str, float] = field(
        default_factory=lambda: {
            "compressibility": 0.35,
            "entropy": 0.25,
            "cyclomatic_density": 0.25,
            "halstead_volume": 0.15,
        }
    )

    # Simulation
    monte_carlo_runs: int = 10_000
    random_seed: int = 42
    portfolio_size: int = 100

    # Corpus
    corpus: list[CorpusEntry] = field(default_factory=list)

    def __post_init__(self):
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.data_dir / "results"
        for d in (self.raw_dir, self.processed_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        corpus_raw = raw.pop("corpus", [])
        corpus = [CorpusEntry(**entry) for entry in corpus_raw]

        # Handle path conversion
        if "data_dir" in raw:
            raw["data_dir"] = Path(raw["data_dir"])

        return cls(corpus=corpus, **raw)


# Default corpus
DEFAULT_CORPUS = [
    # High opacity (expected)
    CorpusEntry(
        name="openssl",
        git_url="https://github.com/openssl/openssl.git",
        cpe_id="cpe:2.3:a:openssl:openssl:*:*:*:*:*:*:*:*",
        category="high_opacity",
    ),
    CorpusEntry(
        name="ffmpeg",
        git_url="https://github.com/FFmpeg/FFmpeg.git",
        cpe_id="cpe:2.3:a:ffmpeg:ffmpeg:*:*:*:*:*:*:*:*",
        category="high_opacity",
    ),
    CorpusEntry(
        name="php-src",
        git_url="https://github.com/php/php-src.git",
        cpe_id="cpe:2.3:a:php:php:*:*:*:*:*:*:*:*",
        category="high_opacity",
    ),
    CorpusEntry(
        name="wireshark",
        git_url="https://gitlab.com/wireshark/wireshark.git",
        cpe_id="cpe:2.3:a:wireshark:wireshark:*:*:*:*:*:*:*:*",
        category="high_opacity",
    ),
    CorpusEntry(
        name="imagemagick",
        git_url="https://github.com/ImageMagick/ImageMagick.git",
        cpe_id="cpe:2.3:a:imagemagick:imagemagick:*:*:*:*:*:*:*:*",
        category="high_opacity",
    ),
    CorpusEntry(
        name="libxml2",
        git_url="https://github.com/GNOME/libxml2.git",
        cpe_id="cpe:2.3:a:xmlsoft:libxml2:*:*:*:*:*:*:*:*",
        category="high_opacity",
    ),
    CorpusEntry(
        name="linux-net",
        git_url="https://github.com/torvalds/linux.git",
        cpe_id="cpe:2.3:o:linux:linux_kernel:*:*:*:*:*:*:*:*",
        category="high_opacity",
        subdirectory="net/",
    ),
    CorpusEntry(
        name="samba",
        git_url="https://github.com/samba-team/samba.git",
        cpe_id="cpe:2.3:a:samba:samba:*:*:*:*:*:*:*:*",
        category="high_opacity",
    ),
    # Low opacity (expected)
    CorpusEntry(
        name="redis",
        git_url="https://github.com/redis/redis.git",
        cpe_id="cpe:2.3:a:redis:redis:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    CorpusEntry(
        name="sqlite",
        git_url="https://github.com/sqlite/sqlite.git",
        cpe_id="cpe:2.3:a:sqlite:sqlite:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    CorpusEntry(
        name="curl",
        git_url="https://github.com/curl/curl.git",
        cpe_id="cpe:2.3:a:haxx:curl:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    CorpusEntry(
        name="zlib",
        git_url="https://github.com/madler/zlib.git",
        cpe_id="cpe:2.3:a:zlib:zlib:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    CorpusEntry(
        name="musl",
        git_url="https://git.musl-libc.org/cgit/musl",
        cpe_id="cpe:2.3:a:musl-libc:musl:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    CorpusEntry(
        name="jq",
        git_url="https://github.com/jqlang/jq.git",
        cpe_id="cpe:2.3:a:jqlang:jq:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    CorpusEntry(
        name="libsodium",
        git_url="https://github.com/jedisct1/libsodium.git",
        cpe_id="cpe:2.3:a:jedisct1:libsodium:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    CorpusEntry(
        name="busybox",
        git_url="https://github.com/mirror/busybox.git",
        cpe_id="cpe:2.3:a:busybox:busybox:*:*:*:*:*:*:*:*",
        category="low_opacity",
    ),
    # Mixed / control
    CorpusEntry(
        name="cpython",
        git_url="https://github.com/python/cpython.git",
        cpe_id="cpe:2.3:a:python:python:*:*:*:*:*:*:*:*",
        category="mixed",
    ),
    CorpusEntry(
        name="nginx",
        git_url="https://github.com/nginx/nginx.git",
        cpe_id="cpe:2.3:a:f5:nginx:*:*:*:*:*:*:*:*",
        category="mixed",
    ),
    CorpusEntry(
        name="postgres",
        git_url="https://github.com/postgres/postgres.git",
        cpe_id="cpe:2.3:a:postgresql:postgresql:*:*:*:*:*:*:*:*",
        category="mixed",
    ),
    CorpusEntry(
        name="git",
        git_url="https://github.com/git/git.git",
        cpe_id="cpe:2.3:a:git-scm:git:*:*:*:*:*:*:*:*",
        category="mixed",
    ),
    CorpusEntry(
        name="openssh",
        git_url="https://github.com/openssh/openssh-portable.git",
        cpe_id="cpe:2.3:a:openbsd:openssh:*:*:*:*:*:*:*:*",
        category="mixed",
    ),
]
