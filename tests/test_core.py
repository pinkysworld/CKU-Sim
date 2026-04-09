"""Unit tests for CKU-Sim core modules."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cku_sim.core.opacity import StructuralOpacity
from cku_sim.metrics.compressibility import compressibility_index
from cku_sim.metrics.entropy import shannon_entropy_from_frequencies
from cku_sim.metrics.cyclomatic import cyclomatic_complexity_file
from cku_sim.analysis.file_level_case_control import (
    compute_file_opacity_from_text,
    extract_commit_refs_from_nvd_items,
    extract_security_ids,
    should_include_source_path,
)
from cku_sim.analysis.predictive_validation import (
    build_prediction_dataset,
    pairwise_accuracy,
)
from cku_sim.analysis.policy_comparison import compile_policy_comparison
from cku_sim.analysis.bootstrap import clustered_delta_bootstrap, flatten_bootstrap_interval
from cku_sim.analysis.negative_control import (
    build_negative_control_prediction_dataset,
    build_security_file_dataset,
    summarise_negative_control_pairs,
)
from cku_sim.collectors.osv_collector import (
    build_osv_alias_map,
    extract_osv_event_candidates,
    repo_url_variants,
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

    def test_extract_security_ids_finds_cve_and_ghsa_tokens(self):
        text = "Fix GHSA-ab12-cd34-ef56 and CVE-2026-12345 in parser"
        ids = extract_security_ids(text)
        assert "CVE-2026-12345" in ids
        assert "GHSA-AB12-CD34-EF56" in ids


class TestOSVCollectorHelpers:
    def test_build_osv_alias_map_prefers_cve_alias(self):
        records = [
            {
                "id": "CURL-CVE-2023-46218",
                "aliases": ["CVE-2023-46218", "GHSA-1111-2222-3333"],
            }
        ]

        alias_map = build_osv_alias_map(records)

        assert alias_map["CURL-CVE-2023-46218"] == "CVE-2023-46218"
        assert alias_map["CVE-2023-46218"] == "CVE-2023-46218"
        assert alias_map["GHSA-1111-2222-3333"] == "CVE-2023-46218"

    def test_extract_osv_event_candidates_uses_git_ranges_and_refs(self):
        records = [
            {
                "id": "CURL-CVE-2023-46218",
                "aliases": ["CVE-2023-46218"],
                "affected": [
                    {
                        "ranges": [
                            {
                                "type": "GIT",
                                "repo": "https://github.com/curl/curl.git",
                                "events": [
                                    {"introduced": "abc1234"},
                                    {"fixed": "2b0994c29a721c91c572cff7808c572a24d251eb"},
                                ],
                            }
                        ]
                    }
                ],
                "references": [
                    {
                        "url": (
                            "https://github.com/curl/curl/commit/"
                            "172e54cda18412da73fd8eb4e444e8a5b371ca59"
                        )
                    }
                ],
            }
        ]

        candidates = extract_osv_event_candidates(
            records,
            repo_url_variants("https://github.com/curl/curl.git"),
        )

        assert list(candidates) == ["CVE-2023-46218"]
        assert candidates["CVE-2023-46218"]["commits"] == {
            "2b0994c29a721c91c572cff7808c572a24d251eb",
            "172e54cda18412da73fd8eb4e444e8a5b371ca59",
        }
        assert candidates["CVE-2023-46218"]["sources"] == {"osv_range", "osv_ref"}


class TestPredictiveValidationHelpers:
    def test_build_prediction_dataset_expands_pairs(self):
        pairs = pd.DataFrame(
            [
                {
                    "repo": "demo",
                    "commit": "abc123",
                    "cve_ids": "CVE-2026-0001",
                    "event_id": "CVE-2026-0001",
                    "ground_truth_policy": "strict_nvd_event",
                    "ground_truth_source": "nvd_ref",
                    "case_file": "src/a.c",
                    "control_file": "src/b.c",
                    "case_loc": 100,
                    "control_loc": 90,
                    "case_size_bytes": 1000,
                    "control_size_bytes": 900,
                    "case_ci_gzip": 0.6,
                    "control_ci_gzip": 0.5,
                    "case_entropy": 0.7,
                    "control_entropy": 0.6,
                    "case_cc_density": 0.1,
                    "control_cc_density": 0.05,
                    "case_halstead": 0.4,
                    "control_halstead": 0.3,
                    "case_composite": 0.55,
                    "control_composite": 0.45,
                }
            ]
        )

        dataset = build_prediction_dataset(pairs)

        assert len(dataset) == 2
        assert set(dataset["label"]) == {0, 1}
        assert set(dataset["kind"]) == {"case", "control"}
        assert set(dataset["suffix"]) == {".c"}
        assert set(dataset["event_id"]) == {"CVE-2026-0001"}
        assert set(dataset["ground_truth_policy"]) == {"strict_nvd_event"}

    def test_pairwise_accuracy_scores_case_above_control(self):
        predictions = pd.DataFrame(
            [
                {"pair_id": 0, "label": 1, "score": 0.8},
                {"pair_id": 0, "label": 0, "score": 0.3},
                {"pair_id": 1, "label": 1, "score": 0.2},
                {"pair_id": 1, "label": 0, "score": 0.7},
            ]
        )

        assert pairwise_accuracy(predictions, "score") == 0.5

    def test_compile_policy_comparison_aggregates_policy_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixtures = {
                "e06_file_case_control": {
                    "pair_level": {
                        "n_pairs": 10,
                        "n_commits": 6,
                        "n_repos": 2,
                        "n_events": 6,
                        "mean_delta_composite": 0.01,
                        "median_delta_composite": 0.005,
                        "positive_share": 0.6,
                        "wilcoxon_pvalue_greater": 0.04,
                    },
                    "commit_level": {
                        "n_commit_events": 6,
                        "n_unique_commits": 6,
                        "n_repos": 2,
                        "n_events": 6,
                        "mean_delta_composite": 0.02,
                        "median_delta_composite": 0.01,
                        "positive_share": 0.7,
                        "wilcoxon_pvalue_greater": 0.03,
                    },
                },
                "e07_predictive_validation": {
                    "n_files": 20,
                    "n_pairs": 10,
                    "n_commits": 6,
                    "n_repos": 2,
                    "models": {
                        "baseline_size": {
                            "roc_auc": 0.51,
                            "macro_roc_auc": 0.52,
                            "pairwise_accuracy": 0.54,
                            "macro_pairwise_accuracy": 0.55,
                        },
                        "baseline_plus_composite": {
                            "roc_auc": 0.57,
                            "macro_roc_auc": 0.58,
                            "pairwise_accuracy": 0.6,
                            "macro_pairwise_accuracy": 0.62,
                        },
                    },
                },
            }

            for dirname, payload in fixtures.items():
                target = root / dirname
                target.mkdir(parents=True, exist_ok=True)
                (target / "summary.json").write_text(json.dumps(payload))

            comparison = compile_policy_comparison(root, policies=["nvd_commit_refs"])

            assert list(comparison["policy"]) == ["nvd_commit_refs"]
            assert comparison.loc[0, "pair_n_pairs"] == 10
            assert comparison.loc[0, "pred_auc_lift"] == pytest.approx(0.06)
            assert comparison.loc[0, "pred_pairacc_lift"] == pytest.approx(0.06)


class TestNegativeControlHelpers:
    def test_build_security_file_dataset_projects_case_side(self):
        pairs = pd.DataFrame(
            [
                {
                    "repo": "demo",
                    "commit": "abc123",
                    "parent": "def456",
                    "event_id": "CVE-2026-0001",
                    "ground_truth_policy": "strict_nvd_event",
                    "ground_truth_source": "nvd_ref",
                    "cve_ids": "CVE-2026-0001",
                    "case_file": "src/a.c",
                    "case_loc": 100,
                    "case_size_bytes": 1000,
                    "case_ci_gzip": 0.6,
                    "case_entropy": 0.7,
                    "case_cc_density": 0.1,
                    "case_halstead": 0.4,
                    "case_composite": 0.55,
                }
            ]
        )

        security = build_security_file_dataset(pairs)

        assert len(security) == 1
        assert security.loc[0, "security_event_id"] == "CVE-2026-0001"
        assert security.loc[0, "security_ground_truth_policy"] == "strict_nvd_event"
        assert security.loc[0, "security_file"] == "src/a.c"

    def test_summarise_negative_control_pairs_reports_positive_effect(self):
        pairs = pd.DataFrame(
            [
                {
                    "repo": "demo",
                    "security_commit": "s1",
                    "security_event_id": "CVE-2026-1",
                    "bugfix_commit": "b1",
                    "security_loc": 100,
                    "bugfix_loc": 80,
                    "loc_ratio": 1.25,
                    "delta_composite": 0.05,
                    "security_ground_truth_policy": "strict_nvd_event",
                },
                {
                    "repo": "demo",
                    "security_commit": "s2",
                    "security_event_id": "CVE-2026-2",
                    "bugfix_commit": "b2",
                    "security_loc": 90,
                    "bugfix_loc": 100,
                    "loc_ratio": 0.9,
                    "delta_composite": 0.01,
                    "security_ground_truth_policy": "strict_nvd_event",
                },
            ]
        )

        summary = summarise_negative_control_pairs(pairs)

        assert summary["n_pairs"] == 2
        assert summary["n_security_events"] == 2
        assert summary["positive_share"] == 1.0
        assert summary["security_ground_truth_policy"] == "strict_nvd_event"

    def test_build_negative_control_prediction_dataset_expands_pairs(self):
        pairs = pd.DataFrame(
            [
                {
                    "repo": "demo",
                    "security_event_id": "CVE-2026-0001",
                    "security_ground_truth_policy": "strict_nvd_event",
                    "security_commit": "abc123",
                    "security_file": "src/a.c",
                    "security_suffix": ".c",
                    "security_loc": 100,
                    "security_size_bytes": 1000,
                    "security_ci_gzip": 0.6,
                    "security_entropy": 0.7,
                    "security_cc_density": 0.1,
                    "security_halstead": 0.4,
                    "security_composite": 0.55,
                    "bugfix_commit": "def456",
                    "bugfix_file": "src/b.c",
                    "bugfix_suffix": ".c",
                    "bugfix_loc": 90,
                    "bugfix_size_bytes": 900,
                    "bugfix_ci_gzip": 0.5,
                    "bugfix_entropy": 0.6,
                    "bugfix_cc_density": 0.05,
                    "bugfix_halstead": 0.3,
                    "bugfix_composite": 0.45,
                }
            ]
        )

        dataset = build_negative_control_prediction_dataset(pairs)

        assert len(dataset) == 2
        assert set(dataset["label"]) == {0, 1}
        assert set(dataset["kind"]) == {"security", "bugfix"}
        assert set(dataset["security_ground_truth_policy"]) == {"strict_nvd_event"}


class TestBootstrapHelpers:
    def test_clustered_delta_bootstrap_returns_intervals(self):
        frame = pd.DataFrame(
            [
                {"event_id": "a", "repo": "r1", "delta_composite": 0.1},
                {"event_id": "a", "repo": "r1", "delta_composite": 0.2},
                {"event_id": "b", "repo": "r2", "delta_composite": -0.1},
                {"event_id": "b", "repo": "r2", "delta_composite": 0.0},
            ]
        )

        summary = clustered_delta_bootstrap(
            frame,
            cluster_col="event_id",
            n_boot=200,
            seed=7,
        )

        assert summary["cluster_col"] == "event_id"
        assert summary["n_clusters"] == 2
        assert len(summary["median_delta_composite_ci"]) == 2
        assert len(summary["positive_share_ci"]) == 2

    def test_flatten_bootstrap_interval_emits_scalar_columns(self):
        flat = flatten_bootstrap_interval(
            {
                "cluster_col": "event_id",
                "n_clusters": 5,
                "n_boot": 2000,
                "confidence_level": 0.95,
                "mean_delta_composite_ci": [0.1, 0.2],
                "median_delta_composite_ci": [0.0, 0.3],
                "positive_share_ci": [0.4, 0.8],
            },
            prefix="pair_primary_bootstrap",
        )

        assert flat["pair_primary_bootstrap_cluster_col"] == "event_id"
        assert flat["pair_primary_bootstrap_median_delta_composite_ci_lo"] == 0.0
        assert flat["pair_primary_bootstrap_positive_share_ci_hi"] == 0.8


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
