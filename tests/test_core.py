"""Unit tests for CKU-Sim core modules."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cku_sim.core.opacity import StructuralOpacity
from cku_sim.metrics.compressibility import compressibility_index
from cku_sim.metrics.entropy import shannon_entropy_from_frequencies
from cku_sim.metrics.cyclomatic import cyclomatic_complexity_file
from cku_sim.analysis.file_level_case_control import (
    compute_file_opacity_from_text,
    extract_commit_refs_from_nvd_items,
    should_include_source_path,
)
from cku_sim.simulation.scenario_generator import (
    generate_regular_source,
    generate_irregular_source,
    generate_spectrum,
)
from cku_sim.simulation.monte_carlo import InsuranceConfig, simulate_portfolio


class TestCompressibilityIndex:
    def test_empty_data(self):
        assert compressibility_index(b"", "gzip") == 0.0

    def test_regular_data_compresses_well(self):
        """Highly repetitive data should have low CI (compresses well)."""
        regular = b"int x = 0;\n" * 1000
        ci = compressibility_index(regular, "gzip")
        assert ci < 0.1  # should compress to < 10% of original

    def test_random_data_compresses_poorly(self):
        """Random data should have high CI."""
        rng = np.random.RandomState(42)
        random_data = rng.bytes(10_000)
        ci = compressibility_index(random_data, "gzip")
        assert ci > 0.9  # random data barely compresses

    def test_algorithms_agree(self):
        """Different compression algorithms should produce correlated results."""
        data = generate_irregular_source(10_000, seed=42)
        ci_gzip = compressibility_index(data, "gzip")
        ci_lzma = compressibility_index(data, "lzma")
        # Both should be in the same ballpark
        assert abs(ci_gzip - ci_lzma) < 0.3

    def test_ci_range(self):
        """CI should always be in [0, 1+epsilon]."""
        for data in [b"a" * 100, b"x" * 10_000, generate_regular_source(5000)]:
            ci = compressibility_index(data, "gzip")
            assert 0.0 <= ci <= 1.1  # slight > 1.0 possible for tiny data


class TestShannonEntropy:
    def test_uniform_distribution(self):
        """Uniform distribution should have entropy = 1.0 (normalised)."""
        freq = Counter({f"token_{i}": 100 for i in range(100)})
        entropy = shannon_entropy_from_frequencies(freq)
        assert abs(entropy - 1.0) < 0.001

    def test_single_token(self):
        """Single-token distribution should have entropy = 0."""
        freq = Counter({"only": 1000})
        entropy = shannon_entropy_from_frequencies(freq)
        assert entropy == 0.0

    def test_empty(self):
        assert shannon_entropy_from_frequencies(Counter()) == 0.0

    def test_skewed_lower_than_uniform(self):
        """Skewed distribution should have lower entropy than uniform."""
        uniform = Counter({f"t_{i}": 100 for i in range(50)})
        skewed = Counter({f"t_{i}": (1000 if i == 0 else 1) for i in range(50)})
        assert shannon_entropy_from_frequencies(skewed) < shannon_entropy_from_frequencies(uniform)


class TestCyclomaticComplexity:
    def test_simple_function(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write("int main() { return 0; }\n")
            f.flush()
            cc = cyclomatic_complexity_file(Path(f.name))
            assert cc == 1  # no decision points

    def test_function_with_branches(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write(
                "int foo(int x) {\n"
                "  if (x > 0) {\n"
                "    if (x > 10) return 2;\n"
                "    return 1;\n"
                "  }\n"
                "  return 0;\n"
                "}\n"
            )
            f.flush()
            cc = cyclomatic_complexity_file(Path(f.name))
            assert cc >= 3  # base + 2 ifs


class TestScenarioGenerator:
    def test_regular_smaller_than_irregular(self):
        """Regular source should compress better (lower CI) than irregular."""
        regular = generate_regular_source(20_000, seed=1)
        irregular = generate_irregular_source(20_000, seed=1)

        ci_reg = compressibility_index(regular, "gzip")
        ci_irreg = compressibility_index(irregular, "gzip")

        assert ci_reg < ci_irreg

    def test_spectrum_monotonic(self):
        """CI should roughly increase as regularity decreases across the spectrum."""
        samples = generate_spectrum(n_samples=10, size_bytes=10_000, base_seed=42)
        cis = [compressibility_index(data, "gzip") for _, data, _ in samples]
        # regularity=0 (first) is irregular → high CI; regularity=1 (last) is regular → low CI
        assert cis[0] > cis[-1]


class TestStructuralOpacity:
    def test_composite_score(self):
        o = StructuralOpacity(
            name="test", snapshot_id="abc",
            ci_gzip=0.5, ci_lzma=0.4, ci_zstd=0.45,
            shannon_entropy=0.8,
            cyclomatic_density=0.1,
            halstead_volume=0.6,
        )
        score = o.compute_composite()
        assert 0.0 < score < 1.0

    def test_to_dict(self):
        o = StructuralOpacity(name="x", snapshot_id="y")
        d = o.to_dict()
        assert d["name"] == "x"
        assert "ci_gzip" in d


class TestFileLevelCaseControlHelpers:
    def test_extract_commit_refs_filters_to_expected_slug(self):
        items = [
            {
                "cve": {
                    "id": "CVE-2026-0001",
                    "references": [
                        {"url": "https://github.com/example/repo/commit/abcdef1234567890"},
                        {"url": "https://github.com/other/repo/commit/1234567890abcdef"},
                    ],
                }
            },
            {
                "cve": {
                    "id": "CVE-2026-0002",
                    "references": [
                        {"url": "https://github.com/example/repo/commit/abcdef1234567890"},
                    ],
                }
            },
        ]

        refs = extract_commit_refs_from_nvd_items(items, "example/repo")

        assert sorted(refs) == ["abcdef1234567890"]
        assert refs["abcdef1234567890"] == {"CVE-2026-0001", "CVE-2026-0002"}

    def test_compute_file_opacity_distinguishes_irregular_source(self):
        regular = "int x = 0;\n" * 300
        irregular = "\n".join(
            f"int value_{i}(int x) {{ if (x > {i}) return x + {i}; return x - {i}; }}"
            for i in range(300)
        )

        regular_metrics = compute_file_opacity_from_text(regular, name="regular.c", snapshot_id="A")
        irregular_metrics = compute_file_opacity_from_text(
            irregular,
            name="irregular.c",
            snapshot_id="B",
        )

        assert irregular_metrics.ci_gzip > regular_metrics.ci_gzip
        assert irregular_metrics.cyclomatic_density > regular_metrics.cyclomatic_density
        assert irregular_metrics.halstead_volume >= regular_metrics.halstead_volume

    def test_should_include_source_path_excludes_tests_and_docs(self):
        extensions = [".c", ".h"]

        assert should_include_source_path("src/server.c", extensions)
        assert not should_include_source_path("tests/server_test.c", extensions)
        assert not should_include_source_path("docs/example.c", extensions)
        assert not should_include_source_path("src/server.py", extensions)


class TestMonteCarloSimulation:
    def test_basic_run(self):
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        cfg = InsuranceConfig(n_firms=4, n_simulations=100, seed=42)
        result = simulate_portfolio(scores, cfg)

        assert "aware_model" in result
        assert "blind_model" in result
        assert "gaps" in result
        assert result["aware_model"]["mean_loss"] > 0
        assert result["blind_model"]["mean_loss"] > 0

    def test_higher_opacity_more_loss(self):
        """Portfolio with higher opacity should have higher expected losses."""
        cfg = InsuranceConfig(n_firms=50, n_simulations=5_000, seed=42)

        low_scores = np.full(50, 0.1)
        high_scores = np.full(50, 0.9)

        low_result = simulate_portfolio(low_scores, cfg)
        high_result = simulate_portfolio(high_scores, cfg)

        assert high_result["aware_model"]["mean_loss"] > low_result["aware_model"]["mean_loss"]
