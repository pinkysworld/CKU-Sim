"""Experiment 8: compare alternative ground-truth event-definition policies."""

from __future__ import annotations

import argparse
import json
import logging
import math

from cku_sim.analysis.file_level_case_control import GROUND_TRUTH_POLICIES
from cku_sim.analysis.policy_comparison import (
    compile_policy_comparison,
    plot_policy_comparison,
)
from cku_sim.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 8: ground-truth policy comparison")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="Comma-separated ground-truth policies to compare",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="e08_policy_comparison",
        help="Results subdirectory name under data/results/",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    results_dir = config.results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    policies = None
    if args.policies:
        policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
        unknown = [policy for policy in policies if policy not in GROUND_TRUTH_POLICIES]
        if unknown:
            raise ValueError(f"Unknown policies requested: {', '.join(sorted(unknown))}")

    comparison = compile_policy_comparison(config.results_dir, policies=policies)
    comparison.to_csv(results_dir / "policy_comparison.csv", index=False)
    with open(results_dir / "policy_comparison.json", "w") as handle:
        json.dump(comparison.to_dict(orient="records"), handle, indent=2)

    plot_policy_comparison(comparison, results_dir / "policy_comparison.pdf")
    plot_policy_comparison(comparison, results_dir / "policy_comparison.png")

    logger.info("=" * 60)
    logger.info("Experiment 8: Ground-truth policy comparison")
    logger.info("=" * 60)
    for row in comparison.to_dict(orient="records"):
        event_count = row["pair_n_events"]
        if isinstance(event_count, float) and math.isnan(event_count):
            event_count_text = "n/a"
        else:
            event_count_text = str(int(event_count))
        logger.info(
            (
                "%s: pairs=%d events=%s "
                "pair_median=%.4f pair_p=%.4g auc_lift=%.4f pairacc_lift=%.4f"
            ),
            row["policy_label"],
            row["pair_n_pairs"],
            event_count_text,
            row["pair_median_delta"],
            row["pair_wilcoxon_pvalue"],
            row["pred_auc_lift"],
            row["pred_pairacc_lift"],
        )


if __name__ == "__main__":
    main()
