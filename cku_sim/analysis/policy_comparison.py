"""Policy-level robustness summaries for file-level vulnerability experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cku_sim.analysis.file_level_case_control import (
    GROUND_TRUTH_POLICIES,
    describe_ground_truth_policy,
    results_subdir_for_ground_truth_policy,
)
from cku_sim.analysis.bootstrap import flatten_bootstrap_interval

DEFAULT_POLICY_ORDER = [
    "nvd_commit_refs",
    "strict_nvd_event",
    "balanced_explicit_id_event",
    "expanded_advisory_event",
]


def predictive_results_subdir_for_e06(e06_subdir: str) -> str:
    """Return the default e07 output directory tied to an e06 results directory."""
    if e06_subdir == "e06_file_case_control":
        return "e07_predictive_validation"
    return f"e07_predictive_validation__{e06_subdir}"


def _load_json(path: Path) -> dict[str, object]:
    with open(path) as handle:
        return json.load(handle)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _first_present(mapping: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of the requested keys were present: {keys}")


def compile_policy_comparison(
    results_dir: Path,
    policies: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate effect and prediction summaries across ground-truth policies."""
    selected = policies or DEFAULT_POLICY_ORDER
    rows: list[dict[str, object]] = []

    for policy in selected:
        if policy not in GROUND_TRUTH_POLICIES:
            raise ValueError(f"Unknown ground-truth policy: {policy}")

        policy_meta = describe_ground_truth_policy(policy)
        e06_subdir = results_subdir_for_ground_truth_policy(policy)
        e07_subdir = predictive_results_subdir_for_e06(e06_subdir)
        e06_summary = _load_json(results_dir / e06_subdir / "summary.json")
        e07_summary = _load_json(results_dir / e07_subdir / "summary.json")

        pair_level = e06_summary["pair_level"]
        commit_level = e06_summary["commit_level"]
        baseline = e07_summary["models"]["baseline_size"]
        plus_composite = e07_summary["models"]["baseline_plus_composite"]

        rows.append(
            {
                "policy": policy,
                "policy_label": policy_meta["label"],
                "policy_description": policy_meta["description"],
                "e06_subdir": e06_subdir,
                "e07_subdir": e07_subdir,
                "pair_n_pairs": int(pair_level["n_pairs"]),
                "pair_n_commits": int(pair_level["n_commits"]),
                "pair_n_repos": int(pair_level["n_repos"]),
                "pair_n_events": _optional_int(pair_level.get("n_events")),
                "pair_mean_delta": float(pair_level["mean_delta_composite"]),
                "pair_median_delta": float(pair_level["median_delta_composite"]),
                "pair_positive_share": float(pair_level["positive_share"]),
                "pair_wilcoxon_pvalue": float(pair_level.get("wilcoxon_pvalue_greater", math.nan)),
                "commit_n_rows": int(_first_present(commit_level, "n_commit_events", "n_commits")),
                "commit_n_unique_commits": int(
                    _first_present(commit_level, "n_unique_commits", "n_commits")
                ),
                "commit_n_repos": int(commit_level["n_repos"]),
                "commit_n_events": _optional_int(commit_level.get("n_events")),
                "commit_mean_delta": float(commit_level["mean_delta_composite"]),
                "commit_median_delta": float(commit_level["median_delta_composite"]),
                "commit_positive_share": float(commit_level["positive_share"]),
                "commit_wilcoxon_pvalue": float(
                    commit_level.get("wilcoxon_pvalue_greater", math.nan)
                ),
                "predictive_n_files": int(e07_summary["n_files"]),
                "predictive_n_pairs": int(e07_summary["n_pairs"]),
                "predictive_n_commits": int(e07_summary["n_commits"]),
                "predictive_n_repos": int(e07_summary["n_repos"]),
                "pred_auc_baseline": float(baseline["roc_auc"]),
                "pred_auc_plus_composite": float(plus_composite["roc_auc"]),
                "pred_auc_lift": float(plus_composite["roc_auc"] - baseline["roc_auc"]),
                "pred_macro_auc_baseline": float(baseline["macro_roc_auc"]),
                "pred_macro_auc_plus_composite": float(plus_composite["macro_roc_auc"]),
                "pred_macro_auc_lift": float(
                    plus_composite["macro_roc_auc"] - baseline["macro_roc_auc"]
                ),
                "pred_pairacc_baseline": float(baseline["pairwise_accuracy"]),
                "pred_pairacc_plus_composite": float(plus_composite["pairwise_accuracy"]),
                "pred_pairacc_lift": float(
                    plus_composite["pairwise_accuracy"] - baseline["pairwise_accuracy"]
                ),
                "pred_macro_pairacc_baseline": float(baseline["macro_pairwise_accuracy"]),
                "pred_macro_pairacc_plus_composite": float(
                    plus_composite["macro_pairwise_accuracy"]
                ),
                "pred_macro_pairacc_lift": float(
                    plus_composite["macro_pairwise_accuracy"]
                    - baseline["macro_pairwise_accuracy"]
                ),
                **flatten_bootstrap_interval(
                    pair_level.get("bootstrap_primary_cluster"),
                    prefix="pair_primary_bootstrap",
                ),
                **flatten_bootstrap_interval(
                    pair_level.get("bootstrap_repo_cluster"),
                    prefix="pair_repo_bootstrap",
                ),
                **flatten_bootstrap_interval(
                    commit_level.get("bootstrap_primary_cluster"),
                    prefix="commit_primary_bootstrap",
                ),
                **flatten_bootstrap_interval(
                    commit_level.get("bootstrap_repo_cluster"),
                    prefix="commit_repo_bootstrap",
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_policy_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot the sample-size and robustness tradeoffs across policies."""
    if comparison.empty:
        return

    labels = comparison["policy_label"].tolist()
    x = list(range(len(comparison)))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].bar(
        [value - width / 2 for value in x],
        comparison["pair_n_pairs"],
        width=width,
        color="#9C755F",
        label="Matched pairs",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        comparison["commit_n_rows"],
        width=width,
        color="#D6B48A",
        alpha=0.9,
        label="Commit-event rows",
    )
    axes[0].set_title("Observation count")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(
        [value - width / 2 for value in x],
        comparison["pair_median_delta"],
        width=width,
        color="#4C78A8",
        label="Pair median",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        comparison["commit_median_delta"],
        width=width,
        color="#72B7B2",
        alpha=0.9,
        label="Commit-event median",
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Opacity effect size")
    axes[1].set_ylabel("Median composite delta")
    axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].bar(
        [value - width / 2 for value in x],
        comparison["pred_auc_lift"],
        width=width,
        color="#F58518",
        label="AUC lift",
    )
    axes[2].bar(
        [value + width / 2 for value in x],
        comparison["pred_pairacc_lift"],
        width=width,
        color="#54A24B",
        alpha=0.9,
        label="Pairwise lift",
    )
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Predictive lift over size baseline")
    axes[2].set_ylabel("Held-out improvement")
    axes[2].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[2].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
