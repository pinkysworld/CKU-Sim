"""Unit tests for CKU-Sim core modules."""

from __future__ import annotations

import json
import math
import os
import subprocess
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
    _run_git,
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
    augment_security_file_dataset,
    build_negative_control_prediction_dataset,
    build_security_file_dataset,
    summarise_bugfix_control_screen,
    summarise_negative_control_pairs,
)
from cku_sim.collectors.osv_collector import (
    build_osv_alias_map,
    extract_osv_event_candidates,
    repo_url_variants,
)
from cku_sim.collectors.github_corpus import (
    build_manifest_entries,
    infer_source_extensions,
    is_research_corpus_candidate,
    slug_to_local_name,
)
from cku_sim.analysis.forward_panel import summarise_forward_panel
from cku_sim.analysis.forward_panel import sample_release_snapshots
from cku_sim.analysis.prospective_file_panel import (
    _build_snapshot_case_map,
    _filter_events_for_ground_truth_policy,
    _cvss_v3_score_from_vector,
    _extract_osv_record_severity,
    build_prospective_prediction_dataset,
    evaluate_external_holdout,
    fit_repo_fixed_effect_models,
    sample_audit_rows,
    summarise_prospective_pairs,
)
from cku_sim.analysis.label_audit import (
    apply_review_decisions,
    classify_ground_truth_source,
    sample_stratified_audit_rows,
    summarise_reviewed_audit,
)
from cku_sim.analysis.quantification_limits import (
    assign_opacity_strata,
    brier_reliability,
    build_all_file_disagreement_frame,
    build_model_disagreement_frame,
    merge_all_file_prediction_diagnostics,
    expected_calibration_error,
    merge_prediction_diagnostics,
    summarise_all_file_calibration_by_stratum,
    summarise_calibration_by_stratum,
)
from cku_sim.analysis.audited_panel import (
    _history_pathspecs,
    _snapshot_history_index,
    _run_git_bytes,
    build_holdout_screen,
    explode_event_catalog_to_audit_rows,
    load_audited_security_table,
    split_corpora_for_external_replication,
    summarise_external_replication,
)
from cku_sim.analysis.intervention_mechanism import (
    build_security_label_lookup,
    load_intervention_audit_table,
    lookup_future_security_events,
    select_best_control_candidate,
    summarise_intervention_audit_table,
    summarise_pairs_to_did,
)
from cku_sim.analysis.audited_negative_control import (
    build_conditional_logit_dataset,
    fit_conditional_logit_models,
    match_audited_security_to_bugfix,
)
from cku_sim.core.config import CorpusEntry
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


class TestQuantificationLimits:
    def test_expected_calibration_error_distinguishes_better_scores(self):
        labels = pd.Series([0, 0, 1, 1], dtype=float)
        better_scores = pd.Series([0.0, 0.0, 1.0, 1.0], dtype=float)
        worse_scores = pd.Series([0.4, 0.4, 0.6, 0.6], dtype=float)

        better_ece = expected_calibration_error(labels, better_scores, n_bins=2)
        worse_ece = expected_calibration_error(labels, worse_scores, n_bins=2)

        assert better_ece < worse_ece

    def test_merge_prediction_diagnostics_and_disagreement_frame(self):
        dataset = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "repo": "repo-a",
                    "snapshot_tag": "v1.0",
                    "event_id": "CVE-2026-0001",
                    "event_observation_id": "evt-1",
                    "label": 1,
                    "kind": "case",
                    "file_path": "src/a.c",
                    "ground_truth_source": "explicit_id",
                    "ground_truth_source_family": "explicit_only",
                    "composite_score": 0.8,
                    "loc": 120,
                    "log_loc": np.log1p(120),
                    "directory_depth": 2,
                    "suffix": ".c",
                },
                {
                    "pair_id": "p2",
                    "repo": "repo-a",
                    "snapshot_tag": "v1.0",
                    "event_id": "CVE-2026-0002",
                    "event_observation_id": "evt-2",
                    "label": 0,
                    "kind": "control",
                    "file_path": "src/b.c",
                    "ground_truth_source": "explicit_id",
                    "ground_truth_source_family": "explicit_only",
                    "composite_score": 0.2,
                    "loc": 80,
                    "log_loc": np.log1p(80),
                    "directory_depth": 2,
                    "suffix": ".c",
                },
                {
                    "pair_id": "p3",
                    "repo": "repo-b",
                    "snapshot_tag": "v2.0",
                    "event_id": "CVE-2026-0003",
                    "event_observation_id": "evt-3",
                    "label": 1,
                    "kind": "case",
                    "file_path": "src/c.py",
                    "ground_truth_source": "reference_id",
                    "ground_truth_source_family": "reference_only",
                    "composite_score": 0.7,
                    "loc": 200,
                    "log_loc": np.log1p(200),
                    "directory_depth": 3,
                    "suffix": ".py",
                },
                {
                    "pair_id": "p4",
                    "repo": "repo-b",
                    "snapshot_tag": "v2.0",
                    "event_id": "CVE-2026-0004",
                    "event_observation_id": "evt-4",
                    "label": 0,
                    "kind": "control",
                    "file_path": "src/d.py",
                    "ground_truth_source": "reference_id",
                    "ground_truth_source_family": "reference_only",
                    "composite_score": 0.1,
                    "loc": 60,
                    "log_loc": np.log1p(60),
                    "directory_depth": 1,
                    "suffix": ".py",
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "pair_id": row["pair_id"],
                    "repo": row["repo"],
                    "snapshot_tag": row["snapshot_tag"],
                    "event_id": row["event_id"],
                    "label": row["label"],
                    "kind": row["kind"],
                    "file_path": row["file_path"],
                    "model": model,
                    "score": score,
                }
                for row, model, score in [
                    (dataset.iloc[0], "baseline_history", 0.55),
                    (dataset.iloc[0], "baseline_plus_composite", 0.85),
                    (dataset.iloc[1], "baseline_history", 0.35),
                    (dataset.iloc[1], "baseline_plus_composite", 0.15),
                    (dataset.iloc[2], "baseline_history", 0.60),
                    (dataset.iloc[2], "baseline_plus_composite", 0.80),
                    (dataset.iloc[3], "baseline_history", 0.40),
                    (dataset.iloc[3], "baseline_plus_composite", 0.20),
                ]
            ]
        )

        merged = merge_prediction_diagnostics(dataset, predictions)
        assert set(merged["opacity_stratum"]) == {"Q1", "Q2", "Q3", "Q4"}

        disagreement = build_model_disagreement_frame(merged)
        assert len(disagreement) == 4
        first_row = disagreement.loc[disagreement["pair_id"] == "p1"].iloc[0]
        assert pytest.approx(first_row["score_range"], rel=1e-6) == 0.30

        calibration = summarise_calibration_by_stratum(merged, n_bins=2)
        overall_row = calibration.loc[
            (calibration["model"] == "baseline_plus_composite")
            & (calibration["opacity_stratum"] == "overall")
        ].iloc[0]
        assert overall_row["absolute_error_mean"] < 0.3

    def test_all_file_calibration_and_disagreement_helpers(self):
        dataset = pd.DataFrame(
            [
                {
                    "repo": "repo-a",
                    "snapshot_tag": "v1.0",
                    "snapshot_key": "repo-a:v1.0",
                    "event_observation_id": "obs-1",
                    "label": 1,
                    "file_path": "src/a.c",
                    "composite_score": 0.9,
                    "loc": 120,
                    "log_loc": np.log1p(120),
                    "directory_depth": 1,
                    "suffix": ".c",
                    "advisory_ids": "CVE-2026-0001",
                    "source_families": "explicit_only",
                    "mapping_bases": "explicit_id",
                },
                {
                    "repo": "repo-a",
                    "snapshot_tag": "v1.0",
                    "snapshot_key": "repo-a:v1.0",
                    "event_observation_id": "obs-2",
                    "label": 0,
                    "file_path": "src/b.c",
                    "composite_score": 0.1,
                    "loc": 80,
                    "log_loc": np.log1p(80),
                    "directory_depth": 1,
                    "suffix": ".c",
                    "advisory_ids": "",
                    "source_families": "",
                    "mapping_bases": "",
                },
                {
                    "repo": "repo-b",
                    "snapshot_tag": "v2.0",
                    "snapshot_key": "repo-b:v2.0",
                    "event_observation_id": "obs-3",
                    "label": 1,
                    "file_path": "pkg/c.py",
                    "composite_score": 0.8,
                    "loc": 200,
                    "log_loc": np.log1p(200),
                    "directory_depth": 2,
                    "suffix": ".py",
                    "advisory_ids": "CVE-2026-0002",
                    "source_families": "reference_only",
                    "mapping_bases": "nvd_ref",
                },
                {
                    "repo": "repo-b",
                    "snapshot_tag": "v2.0",
                    "snapshot_key": "repo-b:v2.0",
                    "event_observation_id": "obs-4",
                    "label": 0,
                    "file_path": "pkg/d.py",
                    "composite_score": 0.2,
                    "loc": 60,
                    "log_loc": np.log1p(60),
                    "directory_depth": 2,
                    "suffix": ".py",
                    "advisory_ids": "",
                    "source_families": "",
                    "mapping_bases": "",
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "repo": row["repo"],
                    "snapshot_tag": row["snapshot_tag"],
                    "snapshot_key": row["snapshot_key"],
                    "event_observation_id": row["event_observation_id"],
                    "label": row["label"],
                    "file_path": row["file_path"],
                    "model": model,
                    "score": score,
                }
                for row, model, score in [
                    (dataset.iloc[0], "baseline_history_plus_structure", 0.65),
                    (dataset.iloc[0], "baseline_plus_composite", 0.85),
                    (dataset.iloc[1], "baseline_history_plus_structure", 0.35),
                    (dataset.iloc[1], "baseline_plus_composite", 0.15),
                    (dataset.iloc[2], "baseline_history_plus_structure", 0.60),
                    (dataset.iloc[2], "baseline_plus_composite", 0.80),
                    (dataset.iloc[3], "baseline_history_plus_structure", 0.40),
                    (dataset.iloc[3], "baseline_plus_composite", 0.20),
                ]
            ]
        )

        merged = merge_all_file_prediction_diagnostics(dataset, predictions, n_strata=4)
        disagreement = build_all_file_disagreement_frame(merged)
        calibration = summarise_all_file_calibration_by_stratum(
            merged,
            group_labels=["Q1", "Q2", "Q3", "Q4"],
            n_bins=2,
        )

        assert set(merged["opacity_stratum"]) == {"Q1", "Q2", "Q3", "Q4"}
        assert len(disagreement) == 4
        assert "brier_reliability" in calibration.columns
        assert brier_reliability([0, 1], [0.2, 0.8], n_bins=2) >= 0.0


class TestAuditedPanelHelpers:
    def test_history_pathspecs_cover_source_extensions(self):
        specs = _history_pathspecs([".py", "go", ".py", ""])

        assert specs == [":(glob)**/*.go", ":(glob)**/*.py"]

    def test_explode_and_load_security_audit_filters_range_only(self):
        catalog = pd.DataFrame(
            [
                {
                    "repo": "repo-a",
                    "event_id": "CVE-2026-0001",
                    "source": "explicit_id+nvd_ref",
                    "source_family": "explicit_plus_reference",
                    "fixed_commit": "abc123",
                    "snapshot_commit": "snap1",
                    "snapshot_tag": "v1.0",
                    "published": "2026-01-01T00:00:00+00:00",
                    "severity_label": "HIGH",
                    "changed_source_files": "src/a.c;src/b.c",
                },
                {
                    "repo": "repo-b",
                    "event_id": "OSV-1",
                    "source": "osv_range",
                    "source_family": "range_only",
                    "fixed_commit": "def456",
                    "snapshot_commit": "snap2",
                    "snapshot_tag": "v2.0",
                    "published": "2026-02-01T00:00:00+00:00",
                    "severity_label": "MEDIUM",
                    "changed_source_files": "pkg/c.py",
                },
            ]
        )
        audit = explode_event_catalog_to_audit_rows(catalog, reviewer="test")
        assert set(audit["review_decision"]) == {"accept", "ambiguous"}

        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.csv"
            audit.to_csv(audit_path, index=False)
            loaded = load_audited_security_table(audit_path)

        assert set(loaded["repo"]) == {"repo-a"}
        assert not loaded["source_family"].isin({"range_only"}).any()

    def test_snapshot_history_index_prefers_snapshot_paths_when_available(self, monkeypatch):
        captured = {}

        def fake_run_git(repo_path, args, **kwargs):
            captured["args"] = args
            return subprocess.CompletedProcess(
                ["git", "-C", str(repo_path), *args],
                0,
                stdout="",
                stderr="",
            )

        monkeypatch.setattr("cku_sim.analysis.audited_panel._run_git", fake_run_git)
        history = _snapshot_history_index(
            Path("/tmp/repo"),
            "deadbeef",
            0,
            [".py"],
            {},
            snapshot_paths=["src/a.py", "src/b.py"],
        )

        assert history == {}
        assert captured["args"][-2:] == ["src/a.py", "src/b.py"]

    def test_holdout_screen_and_split_keep_train_holdout_disjoint(self):
        audit = pd.DataFrame(
            [
                {"repo": "django-django", "snapshot_tag": "v1", "advisory_id": "a1", "file_path": "x"},
                {"repo": "django-django", "snapshot_tag": "v2", "advisory_id": "a2", "file_path": "y"},
                {"repo": "django-django", "snapshot_tag": "v3", "advisory_id": "a3", "file_path": "z"},
                {"repo": "psf-requests", "snapshot_tag": "v1", "advisory_id": "b1", "file_path": "x"},
                {"repo": "psf-requests", "snapshot_tag": "v2", "advisory_id": "b2", "file_path": "y"},
                {"repo": "psf-requests", "snapshot_tag": "v3", "advisory_id": "b3", "file_path": "z"},
            ]
        )
        screen = build_holdout_screen(audit, candidate_repos=["django-django", "psf-requests"], min_snapshots=3, min_events=3)
        assert screen["eligible_holdout"].all()

        train_corpus = [
            CorpusEntry(name="curl", git_url="https://example.com/curl.git"),
            CorpusEntry(name="django-django", git_url="https://example.com/django.git"),
        ]
        holdout_corpus = [
            CorpusEntry(name="django-django", git_url="https://example.com/django.git"),
            CorpusEntry(name="psf-requests", git_url="https://example.com/requests.git"),
        ]
        train_filtered, holdout_filtered = split_corpora_for_external_replication(
            train_corpus,
            holdout_corpus,
            candidate_repos=["django-django", "psf-requests"],
        )

        assert {entry.name for entry in train_filtered} == {"curl"}
        assert {entry.name for entry in holdout_filtered} == {"django-django", "psf-requests"}


class TestPromisorRetryHelpers:
    def test_run_git_retries_promisor_failure(self, monkeypatch):
        calls = []

        def fake_run(cmd, capture_output, text, encoding, errors, env):
            calls.append(env.copy())
            if len(calls) == 1:
                return subprocess.CompletedProcess(
                    cmd,
                    128,
                    stdout="",
                    stderr="fatal: could not fetch abc from promisor remote",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        proc = _run_git(Path("/tmp/repo"), ["show", "HEAD"])

        assert proc.returncode == 0
        assert proc.stdout == "ok\n"
        assert calls[0].get("GIT_NO_LAZY_FETCH") == "1"
        assert "GIT_NO_LAZY_FETCH" not in calls[1]


class TestInterventionMechanismHelpers:
    def test_intervention_audit_loader_filters_accept_only(self):
        audit = pd.DataFrame(
            [
                {
                    "repo": "django-django",
                    "anchor_tag": "5.0",
                    "anchor_commit": "a1",
                    "intervention_commit": "c1",
                    "intervention_parent": "a1",
                    "file_path": "django/db/models/base.py",
                    "commit_date": "2023-01-01T00:00:00+00:00",
                    "intervention_type": "refactor",
                    "keyword_basis": "refactor;cleanup",
                    "review_decision": "accept",
                    "reviewer": "test",
                    "notes": "accepted",
                },
                {
                    "repo": "django-django",
                    "anchor_tag": "5.1",
                    "anchor_commit": "a2",
                    "intervention_commit": "c2",
                    "intervention_parent": "a2",
                    "file_path": "django/forms/forms.py",
                    "commit_date": "2023-06-01T00:00:00+00:00",
                    "intervention_type": "cleanup",
                    "keyword_basis": "cleanup",
                    "review_decision": "reject",
                    "reviewer": "test",
                    "notes": "reject",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interventions.csv"
            audit.to_csv(path, index=False)
            loaded = load_intervention_audit_table(path)

        summary = summarise_intervention_audit_table(loaded)
        assert len(loaded) == 1
        assert loaded.iloc[0]["intervention_commit"] == "c1"
        assert summary["n_rows"] == 1
        assert summary["intervention_type_breakdown"] == {"refactor": 1}

    def test_future_security_lookup_uses_horizon(self):
        security_audit = pd.DataFrame(
            [
                {
                    "repo": "repo-a",
                    "file_path": "src/a.py",
                    "published_at": "2024-01-15T00:00:00+00:00",
                    "advisory_id": "GHSA-1",
                    "fixed_commit": "abc",
                },
                {
                    "repo": "repo-a",
                    "file_path": "src/a.py",
                    "published_at": "2027-01-15T00:00:00+00:00",
                    "advisory_id": "GHSA-2",
                    "fixed_commit": "def",
                },
            ]
        )
        lookup = build_security_label_lookup(security_audit, {})
        label, ids = lookup_future_security_events(
            lookup,
            repo="repo-a",
            file_path="src/a.py",
            anchor_date="2023-07-01T00:00:00+00:00",
            horizon_days=730,
        )

        assert label == 1
        assert ids == "GHSA-1"

    def test_select_best_control_candidate_prefers_nearest_structural_match(self):
        intervention = pd.Series(
            {
                "repo": "repo-a",
                "suffix": ".py",
                "directory_depth_bucket": 2,
                "pre_log_loc": math.log1p(100),
                "pre_log_prior_touches_total": math.log1p(10),
                "commit_epoch": 1000,
                "file_path": "pkg/core.py",
            }
        )
        maintenance = pd.DataFrame(
            [
                {
                    "repo": "repo-a",
                    "suffix": ".py",
                    "maintenance_depth_bucket": 2,
                    "pre_log_loc": math.log1p(95),
                    "pre_log_prior_touches_total": math.log1p(11),
                    "maintenance_epoch": 1000 + 10 * 86400,
                    "maintenance_commit": "c1",
                    "maintenance_id": "m1",
                    "file_path": "pkg/other.py",
                },
                {
                    "repo": "repo-a",
                    "suffix": ".py",
                    "maintenance_depth_bucket": 2,
                    "pre_log_loc": math.log1p(200),
                    "pre_log_prior_touches_total": math.log1p(80),
                    "maintenance_epoch": 1000 + 5 * 86400,
                    "maintenance_commit": "c2",
                    "maintenance_id": "m2",
                    "file_path": "pkg/far.py",
                },
            ]
        )

        match = select_best_control_candidate(intervention, maintenance)
        assert match is not None
        assert match["maintenance_id"] == "m1"
        assert match["match_window_days"] == 90

    def test_summarise_pairs_to_did_computes_expected_endpoint_deltas(self):
        scored = pd.DataFrame(
            [
                {
                    "intervention_id": "INT-0001",
                    "repo": "repo-a",
                    "role": "intervention",
                    "period": "pre",
                    "composite_score": 0.80,
                    "score_range": 0.20,
                    "absolute_error": 0.30,
                },
                {
                    "intervention_id": "INT-0001",
                    "repo": "repo-a",
                    "role": "intervention",
                    "period": "post",
                    "composite_score": 0.60,
                    "score_range": 0.10,
                    "absolute_error": 0.20,
                },
                {
                    "intervention_id": "INT-0001",
                    "repo": "repo-a",
                    "role": "control",
                    "period": "pre",
                    "composite_score": 0.50,
                    "score_range": 0.12,
                    "absolute_error": 0.10,
                },
                {
                    "intervention_id": "INT-0001",
                    "repo": "repo-a",
                    "role": "control",
                    "period": "post",
                    "composite_score": 0.55,
                    "score_range": 0.15,
                    "absolute_error": 0.12,
                },
            ]
        )

        did = summarise_pairs_to_did(scored)
        row = did.iloc[0]
        assert row["did_composite_score"] == pytest.approx(-0.25)
        assert row["did_score_range"] == pytest.approx(-0.13)
        assert row["did_absolute_error"] == pytest.approx(-0.12)


class TestProspectiveMappingHelpers:
    def test_snapshot_case_map_unions_candidate_commit_files(self, monkeypatch):
        def fake_list_changed_source_files(repo_path, commit, extensions):
            mapping = {
                "c1": ["pkg/a.py"],
                "c2": ["pkg/b.py"],
            }
            return mapping.get(commit, [])

        monkeypatch.setattr(
            "cku_sim.analysis.prospective_file_panel._list_changed_source_files",
            fake_list_changed_source_files,
        )
        future_events = pd.DataFrame(
            [
                {
                    "event_id": "CVE-1",
                    "fixed_commit": "c1",
                    "candidate_commits": "c1;c2",
                    "published_epoch": 1,
                    "source": "explicit_id",
                }
            ]
        )
        files_at_snapshot = {
            "pkg/a.py": {"path": "pkg/a.py"},
            "pkg/b.py": {"path": "pkg/b.py"},
            "pkg/c.py": {"path": "pkg/c.py"},
        }
        entry = CorpusEntry(
            name="repo",
            git_url="https://example.com/repo.git",
            source_extensions=[".py"],
        )

        case_map, event_rows = _build_snapshot_case_map(
            Path("/tmp/repo"),
            "snap1",
            files_at_snapshot,
            future_events,
            entry,
            changed_cache={},
        )

        assert set(case_map) == {"pkg/a.py", "pkg/b.py"}
        assert int(event_rows.loc[0, "changed_source_commit_count"]) == 2
        assert int(event_rows.loc[0, "changed_source_files_count"]) == 2

    def test_run_git_bytes_retries_promisor_failure(self, monkeypatch):
        calls = []

        def fake_run(cmd, input, capture_output, env):
            calls.append(env.copy())
            if len(calls) == 1:
                return subprocess.CompletedProcess(
                    cmd,
                    128,
                    stdout=b"",
                    stderr=b"fatal: missing blob object from promisor remote",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout=b"blob\n", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        proc = _run_git_bytes(Path("/tmp/repo"), ["cat-file", "--batch"], stdin=b"deadbeef\n")

        assert proc.returncode == 0
        assert proc.stdout == b"blob\n"
        assert calls[0].get("GIT_NO_LAZY_FETCH") == "1"
        assert "GIT_NO_LAZY_FETCH" not in calls[1]

    def test_summarise_external_replication_computes_primary_lifts(self):
        summary = {
            "models": {
                "baseline_history_plus_structure": {
                    "roc_auc": 0.55,
                    "average_precision": 0.40,
                    "brier_score": 0.22,
                },
                "baseline_plus_composite": {
                    "roc_auc": 0.61,
                    "average_precision": 0.45,
                    "brier_score": 0.20,
                },
            }
        }

        lifted = summarise_external_replication(summary)

        assert pytest.approx(lifted["roc_auc_lift"], rel=1e-6) == 0.06
        assert pytest.approx(lifted["average_precision_lift"], rel=1e-6) == 0.05
        assert pytest.approx(lifted["brier_lift"], rel=1e-6) == -0.02


class TestAuditedNegativeControlHelpers:
    def test_match_and_conditional_logit(self):
        security_pool = pd.DataFrame(
            [
                {
                    "repo": "repo-a",
                    "security_event_id": "evt-1",
                    "security_commit": "sec1",
                    "security_file": "src/a.c",
                    "security_suffix": ".c",
                    "security_subsystem_key": "src",
                    "security_directory_depth": 1.0,
                    "security_loc": 100.0,
                    "security_size_bytes": 1000.0,
                    "security_prior_touches_total": 10.0,
                    "security_prior_touches_365d": 3.0,
                    "security_total_churn": 50.0,
                    "security_churn_365d": 15.0,
                    "security_file_age_days": 400.0,
                    "security_composite": 0.70,
                    "security_ci_gzip": 0.50,
                    "security_entropy": 0.80,
                    "security_cc_density": 0.10,
                    "security_halstead": 0.30,
                },
                {
                    "repo": "repo-a",
                    "security_event_id": "evt-2",
                    "security_commit": "sec2",
                    "security_file": "src/b.c",
                    "security_suffix": ".c",
                    "security_subsystem_key": "src",
                    "security_directory_depth": 1.0,
                    "security_loc": 110.0,
                    "security_size_bytes": 1100.0,
                    "security_prior_touches_total": 11.0,
                    "security_prior_touches_365d": 4.0,
                    "security_total_churn": 55.0,
                    "security_churn_365d": 16.0,
                    "security_file_age_days": 410.0,
                    "security_composite": 0.72,
                    "security_ci_gzip": 0.52,
                    "security_entropy": 0.82,
                    "security_cc_density": 0.11,
                    "security_halstead": 0.31,
                },
                {
                    "repo": "repo-a",
                    "security_event_id": "evt-3",
                    "security_commit": "sec3",
                    "security_file": "src/c.c",
                    "security_suffix": ".c",
                    "security_subsystem_key": "src",
                    "security_directory_depth": 1.0,
                    "security_loc": 120.0,
                    "security_size_bytes": 1200.0,
                    "security_prior_touches_total": 12.0,
                    "security_prior_touches_365d": 5.0,
                    "security_total_churn": 60.0,
                    "security_churn_365d": 17.0,
                    "security_file_age_days": 420.0,
                    "security_composite": 0.74,
                    "security_ci_gzip": 0.54,
                    "security_entropy": 0.84,
                    "security_cc_density": 0.12,
                    "security_halstead": 0.32,
                },
            ]
        )
        bugfix_pool = pd.DataFrame(
            [
                {
                    "repo": "repo-a",
                    "bugfix_commit": "bug1",
                    "bugfix_subject": "ordinary fix",
                    "bugfix_file": "src/x.c",
                    "bugfix_suffix": ".c",
                    "bugfix_subsystem_key": "src",
                    "bugfix_directory_depth": 1.0,
                    "bugfix_loc": 100.0,
                    "bugfix_size_bytes": 1000.0,
                    "bugfix_prior_touches_total": 9.0,
                    "bugfix_prior_touches_365d": 3.0,
                    "bugfix_total_churn": 49.0,
                    "bugfix_churn_365d": 15.0,
                    "bugfix_file_age_days": 390.0,
                    "bugfix_composite": 0.40,
                    "bugfix_ci_gzip": 0.30,
                    "bugfix_entropy": 0.60,
                    "bugfix_cc_density": 0.05,
                    "bugfix_halstead": 0.20,
                },
                {
                    "repo": "repo-a",
                    "bugfix_commit": "bug2",
                    "bugfix_subject": "ordinary fix",
                    "bugfix_file": "src/y.c",
                    "bugfix_suffix": ".c",
                    "bugfix_subsystem_key": "src",
                    "bugfix_directory_depth": 1.0,
                    "bugfix_loc": 110.0,
                    "bugfix_size_bytes": 1100.0,
                    "bugfix_prior_touches_total": 10.0,
                    "bugfix_prior_touches_365d": 4.0,
                    "bugfix_total_churn": 54.0,
                    "bugfix_churn_365d": 16.0,
                    "bugfix_file_age_days": 405.0,
                    "bugfix_composite": 0.42,
                    "bugfix_ci_gzip": 0.31,
                    "bugfix_entropy": 0.61,
                    "bugfix_cc_density": 0.06,
                    "bugfix_halstead": 0.21,
                },
                {
                    "repo": "repo-a",
                    "bugfix_commit": "bug3",
                    "bugfix_subject": "ordinary fix",
                    "bugfix_file": "src/z.c",
                    "bugfix_suffix": ".c",
                    "bugfix_subsystem_key": "src",
                    "bugfix_directory_depth": 1.0,
                    "bugfix_loc": 120.0,
                    "bugfix_size_bytes": 1200.0,
                    "bugfix_prior_touches_total": 11.0,
                    "bugfix_prior_touches_365d": 5.0,
                    "bugfix_total_churn": 59.0,
                    "bugfix_churn_365d": 17.0,
                    "bugfix_file_age_days": 415.0,
                    "bugfix_composite": 0.44,
                    "bugfix_ci_gzip": 0.32,
                    "bugfix_entropy": 0.62,
                    "bugfix_cc_density": 0.07,
                    "bugfix_halstead": 0.22,
                },
            ]
        )

        pairs = match_audited_security_to_bugfix(security_pool, bugfix_pool)
        dataset = build_conditional_logit_dataset(pairs)
        fitted = fit_conditional_logit_models(dataset)

        assert len(pairs) == 3
        assert len(dataset) == 6
        assert dataset["pair_id"].nunique() == 3
        assert fitted == {} or "baseline_plus_composite" in fitted


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


class TestGitHubCorpusHelpers:
    def test_infer_source_extensions_maps_language(self):
        assert infer_source_extensions("Rust") == [".rs"]
        assert ".hpp" in infer_source_extensions("C++")

    def test_build_manifest_entries_emits_corpus_rows(self):
        candidates = [
            {
                "name": slug_to_local_name("curl/curl"),
                "full_name": "curl/curl",
                "git_url": "https://github.com/curl/curl.git",
                "primary_language": "C",
                "stars": 12345,
                "source_extensions": [".c", ".h"],
            }
        ]
        manifest = build_manifest_entries(candidates)

        assert manifest[0]["name"] == "curl-curl"
        assert manifest[0]["cpe_id"] is None
        assert manifest[0]["primary_language"] == "C"

    def test_is_research_corpus_candidate_rejects_guides(self):
        assert not is_research_corpus_candidate(
            {
                "full_name": "someone/awesome-python",
                "description": "awesome python list",
                "topics": [],
                "size_kb": 2000,
            }
        )
        assert is_research_corpus_candidate(
            {
                "full_name": "curl/curl",
                "description": "command line tool and library for transferring data with URLs",
                "topics": [],
                "size_kb": 5000,
            }
        )


class TestForwardPanelHelpers:
    def test_sample_release_snapshots_respects_date_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "repo"
            repo_path.mkdir()
            subprocess.run(["git", "init", str(repo_path)], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(repo_path), "config", "user.name", "Test User"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(repo_path), "config", "user.email", "test@example.com"],
                check=True,
                capture_output=True,
            )

            target = repo_path / "demo.txt"
            history = [
                ("v1.0.0", "2020-01-15T12:00:00+0000"),
                ("v2.0.0", "2022-06-15T12:00:00+0000"),
                ("v3.0.0", "2024-06-15T12:00:00+0000"),
            ]
            for idx, (tag, timestamp) in enumerate(history, start=1):
                target.write_text(f"{tag}\n")
                env = {
                    **dict(os.environ),
                    "GIT_AUTHOR_DATE": timestamp,
                    "GIT_COMMITTER_DATE": timestamp,
                }
                subprocess.run(
                    ["git", "-C", str(repo_path), "add", "demo.txt"],
                    check=True,
                    capture_output=True,
                    env=env,
                )
                subprocess.run(
                    ["git", "-C", str(repo_path), "commit", "-m", f"commit {idx}"],
                    check=True,
                    capture_output=True,
                    env=env,
                )
                subprocess.run(
                    ["git", "-C", str(repo_path), "tag", tag],
                    check=True,
                    capture_output=True,
                    env=env,
                )

            snapshots = sample_release_snapshots(
                repo_path,
                max_tags=10,
                min_gap_days=30,
                min_date=pd.Timestamp("2021-01-01T00:00:00+00:00"),
                max_date=pd.Timestamp("2023-12-31T00:00:00+00:00"),
            )

            assert [row["tag"] for row in snapshots] == ["v2.0.0"]

    def test_summarise_forward_panel_counts_snapshots(self):
        panel = pd.DataFrame(
            [
                {
                    "repo": "r1",
                    "snapshot_date": "2025-01-01T00:00:00+00:00",
                    "future_any_event": 0,
                    "future_event_count": 0,
                    "days_to_next_event": np.nan,
                    "composite_score": 0.2,
                    "log_total_loc": 3.0,
                    "log_total_bytes": 4.0,
                },
                {
                    "repo": "r1",
                    "snapshot_date": "2025-06-01T00:00:00+00:00",
                    "future_any_event": 1,
                    "future_event_count": 2,
                    "days_to_next_event": 14.0,
                    "composite_score": 0.6,
                    "log_total_loc": 3.2,
                    "log_total_bytes": 4.2,
                },
                {
                    "repo": "r2",
                    "snapshot_date": "2025-03-01T00:00:00+00:00",
                    "future_any_event": 0,
                    "future_event_count": 0,
                    "days_to_next_event": np.nan,
                    "composite_score": 0.3,
                    "log_total_loc": 3.1,
                    "log_total_bytes": 4.1,
                },
                {
                    "repo": "r2",
                    "snapshot_date": "2025-09-01T00:00:00+00:00",
                    "future_any_event": 1,
                    "future_event_count": 1,
                    "days_to_next_event": 30.0,
                    "composite_score": 0.7,
                    "log_total_loc": 3.3,
                    "log_total_bytes": 4.3,
                },
            ]
        )

        summary = summarise_forward_panel(panel)

        assert summary["n_snapshots"] == 4
        assert summary["n_repos"] == 2
        assert summary["future_event_rate"] == 0.5


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


class TestProspectivePanelHelpers:
    def test_cvss_v3_score_from_vector_parses_high(self):
        score = _cvss_v3_score_from_vector("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N")
        assert score == pytest.approx(7.5)

    def test_extract_osv_record_severity_uses_cvss_vector(self):
        severity = _extract_osv_record_severity(
            {
                "severity": [
                    {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N"}
                ]
            }
        )

        assert severity["label"] == "HIGH"
        assert severity["score"] == pytest.approx(7.5)

    def test_build_prospective_prediction_dataset_expands_pairs(self):
        pairs = pd.DataFrame(
            [
                {
                    "pair_id": 0,
                    "repo": "demo",
                    "snapshot_tag": "v1.0.0",
                    "snapshot_key": "demo:v1.0.0",
                    "event_id": "CVE-2026-0001",
                    "event_observation_id": "demo:v1.0.0:CVE-2026-0001",
                    "ground_truth_policy": "expanded_advisory_plus_explicit",
                    "ground_truth_source": "osv_ref+explicit_id",
                    "case_file_path": "src/a.c",
                    "case_suffix": ".c",
                    "case_loc": 100,
                    "case_size_bytes": 1000,
                    "case_ci_gzip": 0.6,
                    "case_shannon_entropy": 0.7,
                    "case_cyclomatic_density": 0.1,
                    "case_halstead_volume": 0.4,
                    "case_composite_score": 0.55,
                    "case_directory_depth": 1,
                    "case_prior_touches_total": 10,
                    "case_prior_touches_365d": 4,
                    "case_total_churn": 120,
                    "case_churn_365d": 30,
                    "case_author_count_total": 3,
                    "case_author_count_365d": 2,
                    "case_file_age_days": 200,
                    "case_latest_touch_days": 10,
                    "control_file_path": "src/b.c",
                    "control_suffix": ".c",
                    "control_loc": 90,
                    "control_size_bytes": 900,
                    "control_ci_gzip": 0.5,
                    "control_shannon_entropy": 0.6,
                    "control_cyclomatic_density": 0.05,
                    "control_halstead_volume": 0.3,
                    "control_composite_score": 0.45,
                    "control_directory_depth": 1,
                    "control_prior_touches_total": 7,
                    "control_prior_touches_365d": 3,
                    "control_total_churn": 90,
                    "control_churn_365d": 25,
                    "control_author_count_total": 2,
                    "control_author_count_365d": 2,
                    "control_file_age_days": 180,
                    "control_latest_touch_days": 14,
                }
            ]
        )

        dataset = build_prospective_prediction_dataset(pairs)

        assert len(dataset) == 2
        assert set(dataset["label"]) == {0, 1}
        assert set(dataset["kind"]) == {"case", "control"}
        assert set(dataset["suffix"]) == {".c"}
        assert set(dataset["event_observation_id"]) == {"demo:v1.0.0:CVE-2026-0001"}

    def test_summarise_prospective_pairs_reports_bootstrap(self):
        pairs = pd.DataFrame(
            [
                {
                    "pair_id": 0,
                    "repo": "r1",
                    "snapshot_key": "r1:v1",
                    "event_observation_id": "r1:v1:e1",
                    "ground_truth_policy": "expanded_advisory_plus_explicit",
                    "delta_composite": 0.05,
                    "case_loc": 100,
                    "control_loc": 80,
                },
                {
                    "pair_id": 1,
                    "repo": "r2",
                    "snapshot_key": "r2:v1",
                    "event_observation_id": "r2:v1:e2",
                    "ground_truth_policy": "expanded_advisory_plus_explicit",
                    "delta_composite": 0.01,
                    "case_loc": 120,
                    "control_loc": 100,
                },
                {
                    "pair_id": 2,
                    "repo": "r2",
                    "snapshot_key": "r2:v2",
                    "event_observation_id": "r2:v2:e3",
                    "ground_truth_policy": "expanded_advisory_plus_explicit",
                    "delta_composite": -0.01,
                    "case_loc": 95,
                    "control_loc": 92,
                },
            ]
        )

        summary = summarise_prospective_pairs(pairs)

        assert summary["n_pairs"] == 3
        assert summary["n_events"] == 3
        assert "bootstrap_primary_cluster" in summary

    def test_sample_audit_rows_deduplicates_event_observations(self):
        audit = pd.DataFrame(
            [
                {"repo": "r1", "snapshot_date": "2024-01-01", "event_id": "e1", "event_observation_id": "r1:v1:e1"},
                {"repo": "r1", "snapshot_date": "2024-01-01", "event_id": "e1", "event_observation_id": "r1:v1:e1"},
                {"repo": "r2", "snapshot_date": "2024-02-01", "event_id": "e2", "event_observation_id": "r2:v1:e2"},
            ]
        )

        sampled = sample_audit_rows(audit, sample_size=10, random_seed=7)

        assert len(sampled) == 2
        assert sampled["event_observation_id"].nunique() == 2

    def test_evaluate_external_holdout_scores_frozen_models(self):
        train = pd.DataFrame(
            [
                {
                    "pair_id": 0,
                    "repo": "train-a",
                    "snapshot_tag": "v1",
                    "event_id": "e1",
                    "label": 1,
                    "kind": "case",
                    "file_path": "src/a.c",
                    "suffix": ".c",
                    "loc": 100.0,
                    "size_bytes": 1000.0,
                    "ci_gzip": 0.6,
                    "shannon_entropy": 0.7,
                    "cyclomatic_density": 0.2,
                    "halstead_volume": 0.4,
                    "composite_score": 0.65,
                    "directory_depth": 1.0,
                    "prior_touches_total": 10.0,
                    "prior_touches_365d": 3.0,
                    "total_churn": 120.0,
                    "churn_365d": 25.0,
                    "author_count_total": 3.0,
                    "author_count_365d": 2.0,
                    "file_age_days": 300.0,
                    "latest_touch_days": 20.0,
                    "log_loc": np.log1p(100.0),
                    "log_size_bytes": np.log1p(1000.0),
                    "log_prior_touches_total": np.log1p(10.0),
                    "log_prior_touches_365d": np.log1p(3.0),
                    "log_total_churn": np.log1p(120.0),
                    "log_churn_365d": np.log1p(25.0),
                },
                {
                    "pair_id": 0,
                    "repo": "train-a",
                    "snapshot_tag": "v1",
                    "event_id": "e1",
                    "label": 0,
                    "kind": "control",
                    "file_path": "src/b.c",
                    "suffix": ".c",
                    "loc": 70.0,
                    "size_bytes": 800.0,
                    "ci_gzip": 0.4,
                    "shannon_entropy": 0.6,
                    "cyclomatic_density": 0.1,
                    "halstead_volume": 0.3,
                    "composite_score": 0.45,
                    "directory_depth": 1.0,
                    "prior_touches_total": 4.0,
                    "prior_touches_365d": 2.0,
                    "total_churn": 70.0,
                    "churn_365d": 15.0,
                    "author_count_total": 2.0,
                    "author_count_365d": 1.0,
                    "file_age_days": 250.0,
                    "latest_touch_days": 10.0,
                    "log_loc": np.log1p(70.0),
                    "log_size_bytes": np.log1p(800.0),
                    "log_prior_touches_total": np.log1p(4.0),
                    "log_prior_touches_365d": np.log1p(2.0),
                    "log_total_churn": np.log1p(70.0),
                    "log_churn_365d": np.log1p(15.0),
                },
                {
                    "pair_id": 1,
                    "repo": "train-b",
                    "snapshot_tag": "v2",
                    "event_id": "e2",
                    "label": 1,
                    "kind": "case",
                    "file_path": "lib/a.py",
                    "suffix": ".py",
                    "loc": 90.0,
                    "size_bytes": 1200.0,
                    "ci_gzip": 0.62,
                    "shannon_entropy": 0.72,
                    "cyclomatic_density": 0.18,
                    "halstead_volume": 0.42,
                    "composite_score": 0.66,
                    "directory_depth": 2.0,
                    "prior_touches_total": 12.0,
                    "prior_touches_365d": 5.0,
                    "total_churn": 150.0,
                    "churn_365d": 40.0,
                    "author_count_total": 4.0,
                    "author_count_365d": 2.0,
                    "file_age_days": 320.0,
                    "latest_touch_days": 18.0,
                    "log_loc": np.log1p(90.0),
                    "log_size_bytes": np.log1p(1200.0),
                    "log_prior_touches_total": np.log1p(12.0),
                    "log_prior_touches_365d": np.log1p(5.0),
                    "log_total_churn": np.log1p(150.0),
                    "log_churn_365d": np.log1p(40.0),
                },
                {
                    "pair_id": 1,
                    "repo": "train-b",
                    "snapshot_tag": "v2",
                    "event_id": "e2",
                    "label": 0,
                    "kind": "control",
                    "file_path": "lib/b.py",
                    "suffix": ".py",
                    "loc": 60.0,
                    "size_bytes": 700.0,
                    "ci_gzip": 0.38,
                    "shannon_entropy": 0.55,
                    "cyclomatic_density": 0.09,
                    "halstead_volume": 0.28,
                    "composite_score": 0.39,
                    "directory_depth": 2.0,
                    "prior_touches_total": 3.0,
                    "prior_touches_365d": 1.0,
                    "total_churn": 60.0,
                    "churn_365d": 10.0,
                    "author_count_total": 1.0,
                    "author_count_365d": 1.0,
                    "file_age_days": 210.0,
                    "latest_touch_days": 8.0,
                    "log_loc": np.log1p(60.0),
                    "log_size_bytes": np.log1p(700.0),
                    "log_prior_touches_total": np.log1p(3.0),
                    "log_prior_touches_365d": np.log1p(1.0),
                    "log_total_churn": np.log1p(60.0),
                    "log_churn_365d": np.log1p(10.0),
                },
            ]
        )
        holdout = train.copy()
        holdout["repo"] = ["holdout-a", "holdout-a", "holdout-b", "holdout-b"]

        predictions, repo_metrics, summary = evaluate_external_holdout(train, holdout)

        assert not predictions.empty
        assert set(repo_metrics["repo"]) == {"holdout-a", "holdout-b"}
        assert "baseline_plus_composite" in summary["models"]

    def test_fit_repo_fixed_effect_models_returns_composite_effect(self):
        dataset = pd.DataFrame(
            [
                {
                    "repo": "r1",
                    "event_observation_id": "r1:e1",
                    "label": 1,
                    "log_loc": 4.0,
                    "log_size_bytes": 6.0,
                    "directory_depth": 1.0,
                    "file_age_days": 400.0,
                    "log_prior_touches_total": 2.0,
                    "composite_score": 0.62,
                },
                {
                    "repo": "r1",
                    "event_observation_id": "r1:e1",
                    "label": 0,
                    "log_loc": 3.8,
                    "log_size_bytes": 5.8,
                    "directory_depth": 1.0,
                    "file_age_days": 380.0,
                    "log_prior_touches_total": 1.8,
                    "composite_score": 0.55,
                },
                {
                    "repo": "r1",
                    "event_observation_id": "r1:e2",
                    "label": 1,
                    "log_loc": 4.1,
                    "log_size_bytes": 6.2,
                    "directory_depth": 2.0,
                    "file_age_days": 420.0,
                    "log_prior_touches_total": 2.2,
                    "composite_score": 0.59,
                },
                {
                    "repo": "r1",
                    "event_observation_id": "r1:e2",
                    "label": 0,
                    "log_loc": 3.7,
                    "log_size_bytes": 5.7,
                    "directory_depth": 1.0,
                    "file_age_days": 300.0,
                    "log_prior_touches_total": 1.4,
                    "composite_score": 0.48,
                },
                {
                    "repo": "r2",
                    "event_observation_id": "r2:e3",
                    "label": 1,
                    "log_loc": 4.2,
                    "log_size_bytes": 6.3,
                    "directory_depth": 2.0,
                    "file_age_days": 410.0,
                    "log_prior_touches_total": 2.4,
                    "composite_score": 0.66,
                },
                {
                    "repo": "r2",
                    "event_observation_id": "r2:e3",
                    "label": 0,
                    "log_loc": 3.9,
                    "log_size_bytes": 5.9,
                    "directory_depth": 1.0,
                    "file_age_days": 360.0,
                    "log_prior_touches_total": 1.6,
                    "composite_score": 0.58,
                },
                {
                    "repo": "r2",
                    "event_observation_id": "r2:e4",
                    "label": 1,
                    "log_loc": 4.0,
                    "log_size_bytes": 6.1,
                    "directory_depth": 1.0,
                    "file_age_days": 390.0,
                    "log_prior_touches_total": 2.1,
                    "composite_score": 0.57,
                },
                {
                    "repo": "r2",
                    "event_observation_id": "r2:e4",
                    "label": 0,
                    "log_loc": 3.6,
                    "log_size_bytes": 5.6,
                    "directory_depth": 1.0,
                    "file_age_days": 320.0,
                    "log_prior_touches_total": 1.3,
                    "composite_score": 0.60,
                },
                {
                    "repo": "r1",
                    "event_observation_id": "r1:e5",
                    "label": 1,
                    "log_loc": 3.9,
                    "log_size_bytes": 5.9,
                    "directory_depth": 1.0,
                    "file_age_days": 340.0,
                    "log_prior_touches_total": 1.7,
                    "composite_score": 0.53,
                },
                {
                    "repo": "r1",
                    "event_observation_id": "r1:e5",
                    "label": 0,
                    "log_loc": 4.0,
                    "log_size_bytes": 6.0,
                    "directory_depth": 2.0,
                    "file_age_days": 430.0,
                    "log_prior_touches_total": 2.3,
                    "composite_score": 0.61,
                },
                {
                    "repo": "r2",
                    "event_observation_id": "r2:e6",
                    "label": 1,
                    "log_loc": 3.7,
                    "log_size_bytes": 5.8,
                    "directory_depth": 1.0,
                    "file_age_days": 310.0,
                    "log_prior_touches_total": 1.5,
                    "composite_score": 0.51,
                },
                {
                    "repo": "r2",
                    "event_observation_id": "r2:e6",
                    "label": 0,
                    "log_loc": 4.1,
                    "log_size_bytes": 6.2,
                    "directory_depth": 2.0,
                    "file_age_days": 440.0,
                    "log_prior_touches_total": 2.5,
                    "composite_score": 0.63,
                },
            ]
        )

        summary = fit_repo_fixed_effect_models(dataset)

        assert "baseline_plus_composite_repo_fixed_effects" in summary
        assert "composite_coef" in summary["baseline_plus_composite_repo_fixed_effects"]

    def test_summarise_reviewed_audit_counts_ambiguous_rows(self):
        reviewed = apply_review_decisions(
            pd.DataFrame(
                [
                    {
                        "repo": "curl",
                        "event_id": "CVE-2024-2379",
                        "fixed_commit": "aedbbdf18e689a5eee8dc39600914f5eda6c409c",
                        "ground_truth_source": "osv_range",
                        "case_file_in_changed_files": 1,
                        "changed_source_files_review_count": 1,
                        "commit_mentions_event_id": 0,
                    },
                    {
                        "repo": "git",
                        "event_id": "CVE-2022-24765",
                        "fixed_commit": "3b0bf2704980b1ed6018622bdf5377ec22289688",
                        "ground_truth_source": "explicit_id+osv_range",
                        "case_file_in_changed_files": 1,
                        "changed_source_files_review_count": 1,
                        "commit_mentions_event_id": 1,
                    },
                ]
            )
        )

        summary = summarise_reviewed_audit(reviewed)

        assert summary["n_reviewed"] == 2
        assert summary["overall_review"]["ambiguous"] == 1
        assert summary["overall_review"]["accept"] == 1

    def test_classify_ground_truth_source_collapses_range_only(self):
        assert classify_ground_truth_source("osv_range") == "range_only"
        assert classify_ground_truth_source("explicit_id+osv_range") == "explicit_plus_range"
        assert (
            classify_ground_truth_source("explicit_id+nvd_ref+osv_range+osv_ref")
            == "explicit_plus_reference_plus_range"
        )

    def test_sample_stratified_audit_rows_preserves_source_mix(self):
        audit = pd.DataFrame(
            [
                {
                    "repo": f"r{i}",
                    "snapshot_date": f"2024-01-{i:02d}",
                    "event_id": f"e{i}",
                    "event_observation_id": f"r{i}:v1:e{i}",
                    "ground_truth_source": "explicit_id" if i <= 8 else "osv_range",
                }
                for i in range(1, 13)
            ]
        )
        audit["source_family"] = audit["ground_truth_source"].map(classify_ground_truth_source)

        sampled = sample_stratified_audit_rows(
            audit,
            sample_size=6,
            stratify_by="ground_truth_source",
            random_seed=7,
        )

        assert len(sampled) == 6
        assert set(sampled["ground_truth_source"]) == {"explicit_id", "osv_range"}

    def test_filter_events_for_supported_policy_excludes_range_only(self):
        events = pd.DataFrame(
            [
                {"event_id": "e1", "source": "osv_range"},
                {"event_id": "e2", "source": "explicit_id+osv_range"},
                {"event_id": "e3", "source": "nvd_ref+osv_range+osv_ref"},
            ]
        )

        filtered = _filter_events_for_ground_truth_policy(
            events,
            ground_truth_policy="supported_advisory_plus_explicit",
        )

        assert list(filtered["event_id"]) == ["e2", "e3"]
        assert set(filtered["source_family"]) == {
            "explicit_plus_range",
            "reference_plus_range",
        }


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
        assert security.loc[0, "security_subsystem_key"] == "src"

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

    def test_summarise_bugfix_control_screen_counts_decisions(self):
        screened = pd.DataFrame(
            [
                {
                    "repo": "r1",
                    "review_decision": "accept",
                    "bugfix_file_in_changed_files": 1,
                    "has_security_keyword_signal": 0,
                    "has_security_adjacent_signal": 0,
                },
                {
                    "repo": "r1",
                    "review_decision": "ambiguous",
                    "bugfix_file_in_changed_files": 1,
                    "has_security_keyword_signal": 0,
                    "has_security_adjacent_signal": 1,
                },
                {
                    "repo": "r2",
                    "review_decision": "reject",
                    "bugfix_file_in_changed_files": 0,
                    "has_security_keyword_signal": 1,
                    "has_security_adjacent_signal": 1,
                },
            ]
        )

        summary = summarise_bugfix_control_screen(screened)

        assert summary["n_reviewed"] == 3
        assert summary["review_counts"]["accept"] == 1
        assert summary["review_counts"]["ambiguous"] == 1
        assert summary["review_counts"]["reject"] == 1
        assert summary["bugfix_file_in_changed_files_rate"] == pytest.approx(2 / 3)

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
