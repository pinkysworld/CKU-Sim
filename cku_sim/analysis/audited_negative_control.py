"""Audited security-versus-bugfix negative-control utilities."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit

from cku_sim.analysis.bootstrap import clustered_delta_bootstrap
from cku_sim.analysis.label_audit import classify_ground_truth_source

logger = logging.getLogger(__name__)


def build_security_pool_from_e15(
    e15_pairs: pd.DataFrame,
    *,
    supported_only: bool = True,
) -> pd.DataFrame:
    """Project strict negative-control security rows into a reusable pool."""
    if e15_pairs.empty:
        return pd.DataFrame()
    rows = []
    for _, row in e15_pairs.iterrows():
        source = str(row.get("security_ground_truth_source", ""))
        source_family = classify_ground_truth_source(source)
        if supported_only and source_family not in {
            "explicit_only",
            "reference_only",
            "explicit_plus_range",
            "reference_plus_range",
            "explicit_plus_reference",
            "explicit_plus_reference_plus_range",
        }:
            continue
        rows.append(
            {
                "repo": str(row["repo"]),
                "security_event_id": str(row["security_event_id"]),
                "security_commit": str(row["security_commit"]),
                "security_parent": str(row["security_parent"]),
                "security_file": str(row["security_file"]),
                "security_suffix": str(row["security_suffix"]),
                "security_subsystem_key": str(row.get("security_subsystem_key", "")),
                "security_top_level_key": str(row.get("security_top_level_key", "")),
                "security_directory_depth": float(row.get("security_directory_depth", math.nan)),
                "security_loc": float(row.get("security_loc", math.nan)),
                "security_size_bytes": float(row.get("security_size_bytes", math.nan)),
                "security_prior_touches_total": float(row.get("security_prior_touches_total", math.nan)),
                "security_prior_touches_365d": float(row.get("security_prior_touches_365d", math.nan)),
                "security_total_churn": float(row.get("security_total_churn", math.nan)),
                "security_churn_365d": float(row.get("security_churn_365d", math.nan)),
                "security_file_age_days": float(row.get("security_file_age_days", math.nan)),
                "security_composite": float(row.get("security_composite", math.nan)),
                "security_ci_gzip": float(row.get("security_ci_gzip", math.nan)),
                "security_entropy": float(row.get("security_entropy", math.nan)),
                "security_cc_density": float(row.get("security_cc_density", math.nan)),
                "security_halstead": float(row.get("security_halstead", math.nan)),
                "security_source_family": source_family,
            }
        )
    security = pd.DataFrame(rows)
    if security.empty:
        return security
    return security.drop_duplicates(
        subset=["repo", "security_event_id", "security_commit", "security_file"]
    ).reset_index(drop=True)


def build_bugfix_pool_from_e15_and_audit(
    e15_pairs: pd.DataFrame,
    bugfix_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Build an accepted ordinary bug-fix feature pool."""
    if e15_pairs.empty or bugfix_audit.empty:
        return pd.DataFrame()
    accepted = bugfix_audit.loc[bugfix_audit["review_decision"] == "accept"].copy()
    keys = accepted[["repo", "fixed_commit", "file_path"]].drop_duplicates()

    bugfix = (
        e15_pairs[
            [
                "repo",
                "bugfix_commit",
                "bugfix_parent",
                "bugfix_subject",
                "bugfix_file",
                "bugfix_suffix",
                "bugfix_directory_depth",
                "bugfix_subsystem_key",
                "bugfix_top_level_key",
                "bugfix_size_bytes",
                "bugfix_loc",
                "bugfix_prior_touches_total",
                "bugfix_prior_touches_365d",
                "bugfix_total_churn",
                "bugfix_churn_365d",
                "bugfix_file_age_days",
                "bugfix_composite",
                "bugfix_ci_gzip",
                "bugfix_entropy",
                "bugfix_cc_density",
                "bugfix_halstead",
            ]
        ]
        .drop_duplicates(subset=["repo", "bugfix_commit", "bugfix_file"])
        .rename(
            columns={
                "bugfix_commit": "fixed_commit",
                "bugfix_file": "file_path",
            }
        )
    )
    merged = keys.merge(
        bugfix,
        on=["repo", "fixed_commit", "file_path"],
        how="inner",
        validate="one_to_one",
    )
    return merged.rename(
        columns={
            "fixed_commit": "bugfix_commit",
            "file_path": "bugfix_file",
        }
    ).reset_index(drop=True)


def match_audited_security_to_bugfix(
    security_pool: pd.DataFrame,
    bugfix_pool: pd.DataFrame,
) -> pd.DataFrame:
    """Match accepted security files to accepted ordinary bug-fix controls."""
    if security_pool.empty or bugfix_pool.empty:
        return pd.DataFrame()

    pairs = []
    used_bugfix: set[tuple[str, str, str]] = set()
    pair_id = 0
    security_pool = security_pool.sort_values(
        ["repo", "security_event_id", "security_file"]
    ).reset_index(drop=True)
    for _, security in security_pool.iterrows():
        repo = str(security["repo"])
        candidates = bugfix_pool.loc[
            (bugfix_pool["repo"] == repo)
            & (bugfix_pool["bugfix_suffix"] == security["security_suffix"])
            & (bugfix_pool["bugfix_subsystem_key"] == security["security_subsystem_key"])
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.loc[
            [
                (row["repo"], row["bugfix_commit"], row["bugfix_file"]) not in used_bugfix
                for _, row in candidates.iterrows()
            ]
        ].copy()
        if candidates.empty:
            continue
        security_log_loc = math.log1p(float(security["security_loc"]))
        security_log_touch = math.log1p(float(security["security_prior_touches_total"]))
        candidates["distance"] = (
            (np.log1p(candidates["bugfix_loc"].astype(float)) - security_log_loc).abs()
            + (
                np.log1p(candidates["bugfix_prior_touches_total"].astype(float))
                - security_log_touch
            ).abs()
        )
        candidates = candidates.sort_values(
            ["distance", "bugfix_directory_depth", "bugfix_file"]
        )
        match = candidates.iloc[0]
        used_bugfix.add((repo, str(match["bugfix_commit"]), str(match["bugfix_file"])))
        pairs.append(
            {
                "pair_id": pair_id,
                "repo": repo,
                "security_event_id": str(security["security_event_id"]),
                "security_commit": str(security["security_commit"]),
                "security_file": str(security["security_file"]),
                "security_suffix": str(security["security_suffix"]),
                "security_subsystem_key": str(security["security_subsystem_key"]),
                "security_loc": float(security["security_loc"]),
                "security_size_bytes": float(security["security_size_bytes"]),
                "security_directory_depth": float(security["security_directory_depth"]),
                "security_prior_touches_total": float(security["security_prior_touches_total"]),
                "security_prior_touches_365d": float(security["security_prior_touches_365d"]),
                "security_total_churn": float(security["security_total_churn"]),
                "security_churn_365d": float(security["security_churn_365d"]),
                "security_file_age_days": float(security["security_file_age_days"]),
                "security_composite": float(security["security_composite"]),
                "security_ci_gzip": float(security["security_ci_gzip"]),
                "security_entropy": float(security["security_entropy"]),
                "security_cc_density": float(security["security_cc_density"]),
                "security_halstead": float(security["security_halstead"]),
                "bugfix_commit": str(match["bugfix_commit"]),
                "bugfix_subject": str(match.get("bugfix_subject", "")),
                "bugfix_file": str(match["bugfix_file"]),
                "bugfix_suffix": str(match["bugfix_suffix"]),
                "bugfix_subsystem_key": str(match["bugfix_subsystem_key"]),
                "bugfix_loc": float(match["bugfix_loc"]),
                "bugfix_size_bytes": float(match["bugfix_size_bytes"]),
                "bugfix_directory_depth": float(match["bugfix_directory_depth"]),
                "bugfix_prior_touches_total": float(match["bugfix_prior_touches_total"]),
                "bugfix_prior_touches_365d": float(match["bugfix_prior_touches_365d"]),
                "bugfix_total_churn": float(match["bugfix_total_churn"]),
                "bugfix_churn_365d": float(match["bugfix_churn_365d"]),
                "bugfix_file_age_days": float(match["bugfix_file_age_days"]),
                "bugfix_composite": float(match["bugfix_composite"]),
                "bugfix_ci_gzip": float(match["bugfix_ci_gzip"]),
                "bugfix_entropy": float(match["bugfix_entropy"]),
                "bugfix_cc_density": float(match["bugfix_cc_density"]),
                "bugfix_halstead": float(match["bugfix_halstead"]),
                "distance": float(match["distance"]),
                "delta_composite": float(security["security_composite"] - match["bugfix_composite"]),
                "delta_ci_gzip": float(security["security_ci_gzip"] - match["bugfix_ci_gzip"]),
                "delta_entropy": float(security["security_entropy"] - match["bugfix_entropy"]),
                "delta_cc_density": float(security["security_cc_density"] - match["bugfix_cc_density"]),
                "delta_halstead": float(security["security_halstead"] - match["bugfix_halstead"]),
            }
        )
        pair_id += 1
    return pd.DataFrame(pairs)


def summarise_audited_negative_control(pairs: pd.DataFrame) -> dict[str, object]:
    """Summarise the audited negative-control matched pairs."""
    if pairs.empty:
        return {"n_pairs": 0, "n_security_events": 0, "n_repos": 0}
    delta = pairs["delta_composite"].dropna()
    summary = {
        "n_pairs": int(len(pairs)),
        "n_security_events": int(pairs["security_event_id"].nunique()),
        "n_repos": int(pairs["repo"].nunique()),
        "mean_delta_composite": float(delta.mean()),
        "median_delta_composite": float(delta.median()),
        "positive_share": float((delta > 0).mean()),
        "mean_match_distance": float(pairs["distance"].mean()),
    }
    if len(delta) >= 3:
        wilcoxon = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
        summary["wilcoxon_pvalue_greater"] = float(wilcoxon.pvalue)
        summary["wilcoxon_statistic"] = float(wilcoxon.statistic)
    bootstrap_security = clustered_delta_bootstrap(
        pairs.rename(columns={"security_event_id": "event_observation_id"}),
        cluster_col="event_observation_id",
        delta_col="delta_composite",
    )
    if bootstrap_security:
        summary["bootstrap_security_cluster"] = bootstrap_security
    return summary


def build_conditional_logit_dataset(pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert matched security-vs-bugfix pairs into long-form conditional-logit data."""
    if pairs.empty:
        return pd.DataFrame()
    rows = []
    for _, row in pairs.iterrows():
        common = {
            "pair_id": int(row["pair_id"]),
            "repo": str(row["repo"]),
            "security_event_id": str(row["security_event_id"]),
        }
        for kind, label in (("security", 1), ("bugfix", 0)):
            rows.append(
                {
                    **common,
                    "label": int(label),
                    "kind": kind,
                    "log_loc": math.log1p(float(row[f"{kind}_loc"])),
                    "log_size_bytes": math.log1p(float(row[f"{kind}_size_bytes"])),
                    "directory_depth": float(row[f"{kind}_directory_depth"]),
                    "log_prior_touches_total": math.log1p(float(row[f"{kind}_prior_touches_total"])),
                    "log_prior_touches_365d": math.log1p(float(row[f"{kind}_prior_touches_365d"])),
                    "log_total_churn": math.log1p(float(row[f"{kind}_total_churn"])),
                    "log_churn_365d": math.log1p(float(row[f"{kind}_churn_365d"])),
                    "file_age_days": float(row[f"{kind}_file_age_days"]),
                    "cyclomatic_density": float(row[f"{kind}_cc_density"]),
                    "halstead_volume": float(row[f"{kind}_halstead"]),
                    "composite_score": float(row[f"{kind}_composite"]),
                }
            )
    return pd.DataFrame(rows)


def fit_conditional_logit_models(dataset: pd.DataFrame) -> dict[str, object]:
    """Fit frozen baseline and baseline-plus-composite conditional-logit models."""
    if dataset.empty or dataset["pair_id"].nunique() < 3:
        return {}

    work = dataset.copy()
    feature_cols = [
        "log_loc",
        "log_size_bytes",
        "directory_depth",
        "log_prior_touches_total",
        "log_prior_touches_365d",
        "log_total_churn",
        "log_churn_365d",
        "file_age_days",
        "cyclomatic_density",
        "halstead_volume",
    ]
    for col in feature_cols + ["composite_score"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        if work[col].dropna().empty:
            work[col] = 0.0
        else:
            work[col] = work[col].fillna(work[col].median())

    y = work["label"].astype(int)
    groups = work["pair_id"].astype(int)
    x_base = work[feature_cols]
    x_plus = work[feature_cols + ["composite_score"]]
    varying_base = x_base.groupby(groups).nunique(dropna=False).gt(1).any(axis=0)
    x_base = x_base.loc[:, varying_base]
    x_plus = x_plus.loc[:, list(x_base.columns) + ["composite_score"]]

    def _fit_with_fallback(endog, exog):
        model = ConditionalLogit(endog, exog, groups=groups)
        try:
            return model.fit(disp=False, skip_hessian=True), "standard"
        except Exception as exc:
            logger.warning(
                "Conditional logit standard fit failed (%s); retrying with elastic-net regularization.",
                exc,
            )
            try:
                return model.fit_regularized(alpha=1e-6, refit=False), "regularized"
            except Exception as exc2:
                logger.warning("Conditional logit failed: %s", exc2)
                return None, "failed"

    base_model, base_fit_method = _fit_with_fallback(y, x_base)
    plus_model, plus_fit_method = _fit_with_fallback(y, x_plus)
    if base_model is None or plus_model is None:
        return {}

    base_llf = getattr(base_model, "llf", float("nan"))
    plus_llf = getattr(plus_model, "llf", float("nan"))
    base_aic = getattr(base_model, "aic", float("nan"))
    plus_aic = getattr(plus_model, "aic", float("nan"))
    if np.isfinite(base_llf) and np.isfinite(plus_llf):
        lr_stat = max(0.0, 2.0 * (plus_llf - base_llf))
        lr_pvalue = float(stats.chi2.sf(lr_stat, df=1))
    else:
        lr_stat = float("nan")
        lr_pvalue = float("nan")
    try:
        conf = plus_model.conf_int().loc["composite_score"]
        conf_lo = float(conf[0])
        conf_hi = float(conf[1])
    except Exception:
        conf_lo = float("nan")
        conf_hi = float("nan")
    try:
        composite_pvalue = float(plus_model.pvalues["composite_score"])
    except Exception:
        composite_pvalue = float("nan")
    return {
        "baseline_history_plus_structure": {
            "log_likelihood": float(base_llf),
            "aic": float(base_aic),
            "fit_method": base_fit_method,
        },
        "baseline_plus_composite": {
            "log_likelihood": float(plus_llf),
            "aic": float(plus_aic),
            "fit_method": plus_fit_method,
            "composite_coef": float(plus_model.params["composite_score"]),
            "composite_pvalue": composite_pvalue,
            "composite_ci_lo": conf_lo,
            "composite_ci_hi": conf_hi,
            "lr_statistic_vs_baseline": float(lr_stat),
            "lr_pvalue_vs_baseline": lr_pvalue,
        },
    }
