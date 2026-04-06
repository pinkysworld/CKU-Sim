"""Structural opacity model — the central abstraction of CKU-Sim."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructuralOpacity:
    """Opacity metrics for a single codebase snapshot.

    Higher values = higher opacity = more CKU-relevant.
    The compressibility_index is inverted: raw CI is high for regular code,
    so opacity_ci = 1 - CI = compressed_size / raw_size.
    """

    name: str
    snapshot_id: str  # commit hash or tag

    # Raw metrics
    total_bytes: int = 0
    total_loc: int = 0
    num_files: int = 0

    # Opacity proxies (all normalised so higher = more opaque)
    ci_gzip: float = 0.0  # 1 - gzip compression ratio
    ci_lzma: float = 0.0
    ci_zstd: float = 0.0
    shannon_entropy: float = 0.0  # bits per token
    cyclomatic_density: float = 0.0  # total CC / LOC
    halstead_volume: float = 0.0  # normalised Halstead V

    # Composite
    composite_score: float = 0.0

    def compute_composite(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted composite opacity score.

        Args:
            weights: dict mapping metric name to weight. Defaults to equal weights.

        Returns:
            Composite opacity score (higher = more opaque).
        """
        if weights is None:
            weights = {
                "compressibility": 0.35,
                "entropy": 0.25,
                "cyclomatic_density": 0.25,
                "halstead_volume": 0.15,
            }

        # Use mean CI across algorithms for the compressibility component
        mean_ci = (self.ci_gzip + self.ci_lzma + self.ci_zstd) / 3.0

        components = {
            "compressibility": mean_ci,
            "entropy": self.shannon_entropy,
            "cyclomatic_density": self.cyclomatic_density,
            "halstead_volume": self.halstead_volume,
        }

        total_weight = sum(weights.get(k, 0) for k in components)
        if total_weight == 0:
            return 0.0

        self.composite_score = sum(
            weights.get(k, 0) * v for k, v in components.items()
        ) / total_weight

        return self.composite_score

    def to_dict(self) -> dict:
        """Serialise to dict for DataFrame construction."""
        return {
            "name": self.name,
            "snapshot_id": self.snapshot_id,
            "total_bytes": self.total_bytes,
            "total_loc": self.total_loc,
            "num_files": self.num_files,
            "ci_gzip": self.ci_gzip,
            "ci_lzma": self.ci_lzma,
            "ci_zstd": self.ci_zstd,
            "shannon_entropy": self.shannon_entropy,
            "cyclomatic_density": self.cyclomatic_density,
            "halstead_volume": self.halstead_volume,
            "composite_score": self.composite_score,
        }
