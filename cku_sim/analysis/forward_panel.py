"""Forward-looking release-level panel analysis."""

from __future__ import annotations

import logging
import math
from datetime import timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from git import Repo
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cku_sim.analysis.file_level_case_control import (
    _get_commit_epoch,
    _run_git,
    _select_primary_event_commit,
)
from cku_sim.collectors.osv_collector import (
    canonicalise_osv_event_id,
    extract_osv_event_candidates,
    fetch_osv_records_for_repo,
    repo_url_variants,
)
from cku_sim.collectors.repo_metrics import compute_opacity
from cku_sim.core.config import Config, CorpusEntry

logger = logging.getLogger(__name__)

FORWARD_MODEL_SPECS = {
    "baseline_size": {
        "numeric": ["log_total_loc", "log_total_bytes", "log_num_files"],
        "categorical": ["category"],
    },
    "baseline_plus_composite": {
        "numeric": ["log_total_loc", "log_total_bytes", "log_num_files", "composite_score"],
        "categorical": ["category"],
    },
}


def sample_release_snapshots(
    repo_path: Path,
    *,
    max_tags: int = 20,
    min_gap_days: int = 30,
    min_date: pd.Timestamp | None = None,
    max_date: pd.Timestamp | None = None,
) -> list[dict[str, object]]:
    """Sample release-like tag snapshots from oldest to newest."""
    repo = Repo(str(repo_path))
    rows: list[dict[str, object]] = []
    seen_commits: set[str] = set()

    for tag in repo.tags:
        try:
            commit = tag.commit
            commit_sha = commit.hexsha
            if commit_sha in seen_commits:
                continue
            seen_commits.add(commit_sha)
            rows.append(
                {
                    "tag": str(tag),
                    "commit": commit_sha,
                    "date": commit.committed_datetime.astimezone(timezone.utc),
                }
            )
        except Exception:
            continue

    rows.sort(key=lambda item: (item["date"], item["commit"]))
    if not rows:
        return []

    if min_date is not None or max_date is not None:
        filtered: list[dict[str, object]] = []
        for row in rows:
            current = _to_utc_timestamp(row["date"])
            if min_date is not None and current < min_date:
                continue
            if max_date is not None and current > max_date:
                continue
            filtered.append(row)
        rows = filtered
        if not rows:
            return []

    if min_gap_days > 0:
        filtered: list[dict[str, object]] = []
        min_gap = pd.Timedelta(days=min_gap_days)
        last_date: pd.Timestamp | None = None
        for row in rows:
            current = pd.Timestamp(row["date"])
            if last_date is None or current - last_date >= min_gap:
                filtered.append(row)
                last_date = current
        rows = filtered or rows

    if len(rows) > max_tags:
        indices = np.linspace(0, len(rows) - 1, max_tags, dtype=int)
        rows = [rows[int(index)] for index in indices]

    return rows


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def collect_repo_forward_events(
    repo_path: Path,
    entry: CorpusEntry,
    cache_dir: Path,
    *,
    rate_limit: float = 0.1,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Collect published security events with a primary fixing commit."""
    records = fetch_osv_records_for_repo(
        repo_path,
        entry.git_url,
        cache_dir,
        rate_limit=rate_limit,
        batch_size=batch_size,
    )
    repo_urls = repo_url_variants(entry.git_url)
    event_candidates = extract_osv_event_candidates(records, repo_urls)
    records_by_event: dict[str, list[dict]] = {}
    for record in records:
        event_id = canonicalise_osv_event_id(record)
        records_by_event.setdefault(event_id, []).append(record)

    metadata_cache: dict[str, tuple[str, tuple[str, ...], int] | None] = {}
    rows: list[dict[str, object]] = []
    for event_id, repos_records in sorted(records_by_event.items()):
        candidate_entry = event_candidates.get(event_id)
        if candidate_entry is None or not candidate_entry["commits"]:
            continue
        selected_commit = _select_primary_event_commit(
            repo_path,
            entry,
            set(candidate_entry["commits"]),
            metadata_cache,
        )
        if selected_commit is None:
            continue

        published_values = [
            _to_utc_timestamp(record["published"])
            for record in repos_records
            if record.get("published")
        ]
        if not published_values:
            continue
        published = min(published_values)
        fixed_epoch = _get_commit_epoch(repo_path, selected_commit)
        aliases = sorted(
            {
                str(token).upper()
                for record in repos_records
                for token in [record.get("id"), *record.get("aliases", [])]
                if token
            }
        )
        rows.append(
            {
                "repo": entry.name,
                "event_id": event_id,
                "published": published.isoformat(),
                "published_epoch": int(published.timestamp()),
                "fixed_commit": selected_commit,
                "fixed_epoch": fixed_epoch,
                "aliases": ";".join(aliases),
                "source": "+".join(sorted(candidate_entry["sources"])),
            }
        )

    return pd.DataFrame(rows)


def build_forward_panel(
    repo_paths: dict[str, Path],
    corpus: list[CorpusEntry],
    config: Config,
    *,
    max_tags: int = 20,
    min_tag_gap_days: int = 30,
    horizon_days: int = 365,
    analysis_date: pd.Timestamp | None = None,
    lookback_years: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a release-level panel with future event outcomes."""
    osv_cache_dir = config.processed_dir / "osv_cache"
    ancestor_cache: dict[tuple[str, str, str], bool] = {}
    panel_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    analysis_date = analysis_date or pd.Timestamp.now(tz="UTC")
    max_snapshot_date = analysis_date - pd.Timedelta(days=horizon_days)
    min_snapshot_date = None
    if lookback_years is not None and lookback_years > 0:
        min_snapshot_date = max_snapshot_date - pd.Timedelta(days=365 * lookback_years)

    for entry in corpus:
        repo_path = repo_paths.get(entry.name)
        if repo_path is None or not repo_path.exists():
            logger.info("Skipping %s: repository checkout not available", entry.name)
            continue

        snapshots = sample_release_snapshots(
            repo_path,
            max_tags=max_tags,
            min_gap_days=min_tag_gap_days,
            min_date=min_snapshot_date,
            max_date=max_snapshot_date,
        )
        if len(snapshots) < 2:
            logger.info("Skipping %s: fewer than two usable release tags", entry.name)
            continue

        events = collect_repo_forward_events(
            repo_path,
            entry,
            osv_cache_dir,
            rate_limit=config.osv_rate_limit,
            batch_size=config.osv_query_batch_size,
        )
        if events.empty:
            logger.info("Skipping %s: no usable forward events", entry.name)
            continue

        event_frames.append(events)
        opacity_rows = compute_snapshot_opacity_rows(repo_path, entry, snapshots, config)
        if not opacity_rows:
            continue

        for order, row in enumerate(opacity_rows):
            snapshot_epoch = int(_to_utc_timestamp(row["snapshot_date"]).timestamp())
            future = _future_events_for_snapshot(
                repo_path,
                row["snapshot_commit"],
                snapshot_epoch,
                events,
                horizon_days=horizon_days,
                ancestor_cache=ancestor_cache,
            )
            row.update(
                {
                    "snapshot_order": order,
                    "analysis_date": analysis_date.isoformat(),
                    "fully_observed_horizon": 1,
                    "future_event_count": int(len(future)),
                    "future_any_event": int(not future.empty),
                    "days_to_next_event": (
                        float((future["published_epoch"].min() - snapshot_epoch) / 86400.0)
                        if not future.empty
                        else math.nan
                    ),
                    "future_event_ids": ";".join(future["event_id"].astype(str).tolist()),
                }
            )
            panel_rows.append(row)

    panel_df = pd.DataFrame(panel_rows)
    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    if not panel_df.empty:
        panel_df["log_total_loc"] = np.log1p(panel_df["total_loc"])
        panel_df["log_total_bytes"] = np.log1p(panel_df["total_bytes"])
        panel_df["log_num_files"] = np.log1p(panel_df["num_files"])
        panel_df["category"] = panel_df["category"].fillna("expanded")

    return panel_df, events_df


def compute_snapshot_opacity_rows(
    repo_path: Path,
    entry: CorpusEntry,
    snapshots: list[dict[str, object]],
    config: Config,
) -> list[dict[str, object]]:
    """Compute opacity metrics for sampled tag snapshots via temporary checkouts."""
    repo = Repo(str(repo_path))
    original_ref = repo.head.commit.hexsha
    try:
        if not repo.head.is_detached:
            original_ref = repo.active_branch.name
    except Exception:
        pass

    rows: list[dict[str, object]] = []
    try:
        for snapshot in snapshots:
            commit = str(snapshot["commit"])
            repo.git.checkout(commit, force=True)
            opacity = compute_opacity(repo_path, entry, snapshot_id=str(snapshot["tag"]), config=config)
            if opacity is None:
                continue
            row = opacity.to_dict()
            row.update(
                {
                    "repo": entry.name,
                    "full_name": entry.full_name or entry.name,
                    "category": entry.category,
                    "primary_language": entry.primary_language,
                    "stars": entry.stars,
                    "snapshot_tag": snapshot["tag"],
                    "snapshot_commit": commit,
                    "snapshot_date": pd.Timestamp(snapshot["date"]).isoformat(),
                }
            )
            rows.append(row)
    finally:
        try:
            repo.git.checkout(original_ref, force=True)
        except Exception as exc:
            logger.warning("Could not restore %s to %s: %s", entry.name, original_ref, exc)

    return rows


def _future_events_for_snapshot(
    repo_path: Path,
    snapshot_commit: str,
    snapshot_epoch: int,
    events: pd.DataFrame,
    *,
    horizon_days: int,
    ancestor_cache: dict[tuple[str, str, str], bool],
) -> pd.DataFrame:
    upper_epoch = snapshot_epoch + horizon_days * 86400
    subset = events.loc[
        (events["published_epoch"] > snapshot_epoch)
        & (events["published_epoch"] <= upper_epoch)
        & (events["fixed_epoch"] > snapshot_epoch)
    ].copy()
    if subset.empty:
        return subset

    mask = []
    for _, row in subset.iterrows():
        key = (str(repo_path), snapshot_commit, str(row["fixed_commit"]))
        if key not in ancestor_cache:
            proc = _run_git(
                repo_path,
                ["merge-base", "--is-ancestor", snapshot_commit, str(row["fixed_commit"])],
            )
            ancestor_cache[key] = proc.returncode == 0
        mask.append(ancestor_cache[key])

    return subset.loc[mask].copy()


def summarise_forward_panel(panel_df: pd.DataFrame) -> dict[str, object]:
    """Summarise the forward-looking panel."""
    if panel_df.empty:
        return {}

    any_event = panel_df.loc[panel_df["future_any_event"] == 1, "composite_score"]
    no_event = panel_df.loc[panel_df["future_any_event"] == 0, "composite_score"]
    mann_whitney = math.nan
    if len(any_event) > 0 and len(no_event) > 0:
        mann_whitney = float(
            stats.mannwhitneyu(
                any_event.to_numpy(),
                no_event.to_numpy(),
                alternative="greater",
            ).pvalue
        )

    quartiles = pd.qcut(
        panel_df["composite_score"],
        q=min(4, panel_df["composite_score"].nunique()),
        duplicates="drop",
    )
    quartile_summary = (
        panel_df.assign(opacity_quartile=quartiles.astype(str))
        .groupby("opacity_quartile", as_index=False)
        .agg(
            n_snapshots=("future_any_event", "size"),
            future_event_rate=("future_any_event", "mean"),
            mean_future_event_count=("future_event_count", "mean"),
            median_composite=("composite_score", "median"),
        )
        .to_dict(orient="records")
    )

    summary = {
        "n_snapshots": int(len(panel_df)),
        "n_repos": int(panel_df["repo"].nunique()),
        "snapshot_start": str(panel_df["snapshot_date"].min()),
        "snapshot_end": str(panel_df["snapshot_date"].max()),
        "future_event_rate": float(panel_df["future_any_event"].mean()),
        "mean_future_event_count": float(panel_df["future_event_count"].mean()),
        "median_days_to_next_event": float(panel_df["days_to_next_event"].median(skipna=True)),
        "median_composite_future_event": float(any_event.median()) if len(any_event) else math.nan,
        "median_composite_no_event": float(no_event.median()) if len(no_event) else math.nan,
        "mann_whitney_pvalue": mann_whitney,
        "opacity_quartiles": quartile_summary,
    }
    glm = fit_forward_association_models(panel_df)
    if glm:
        summary["association_models"] = glm
    return summary


def fit_forward_association_models(panel_df: pd.DataFrame) -> dict[str, object]:
    """Fit simple clustered GLMs for future-event outcomes."""
    data = panel_df.copy()
    if (
        data.empty
        or len(data) < 20
        or data["repo"].nunique() < 2
        or data["future_any_event"].nunique() < 2
    ):
        return {}

    X = data[["log_total_loc", "log_total_bytes", "composite_score"]].copy()
    X = sm.add_constant(X, has_constant="add")
    groups = data["repo"]
    summary: dict[str, object] = {}

    try:
        logit_model = sm.GLM(
            data["future_any_event"],
            X,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": groups})
        conf = logit_model.conf_int().loc["composite_score"]
        summary["logit_future_any_event"] = {
            "coef": float(logit_model.params["composite_score"]),
            "pvalue": float(logit_model.pvalues["composite_score"]),
            "ci_lo": float(conf[0]),
            "ci_hi": float(conf[1]),
        }
    except Exception as exc:
        logger.warning("Forward logit model failed: %s", exc)

    try:
        poisson_model = sm.GLM(
            data["future_event_count"],
            X,
            family=sm.families.Poisson(),
        ).fit(cov_type="cluster", cov_kwds={"groups": groups})
        conf = poisson_model.conf_int().loc["composite_score"]
        summary["poisson_future_event_count"] = {
            "coef": float(poisson_model.params["composite_score"]),
            "pvalue": float(poisson_model.pvalues["composite_score"]),
            "ci_lo": float(conf[0]),
            "ci_hi": float(conf[1]),
        }
    except Exception as exc:
        logger.warning("Forward Poisson model failed: %s", exc)

    return summary


def evaluate_forward_prediction(
    panel_df: pd.DataFrame,
    model_specs: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run leave-one-repository-out prediction for future event occurrence."""
    model_specs = model_specs or FORWARD_MODEL_SPECS
    if panel_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    logo = LeaveOneGroupOut()
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for train_idx, test_idx in logo.split(panel_df, panel_df["future_any_event"], groups=panel_df["repo"]):
        train = panel_df.iloc[train_idx].copy()
        test = panel_df.iloc[test_idx].copy()
        held_out_repo = str(test["repo"].iloc[0])

        if train["future_any_event"].nunique() < 2:
            continue

        for model_name, spec in model_specs.items():
            feature_cols = spec["numeric"] + spec["categorical"]
            pipeline = _build_forward_pipeline(spec["numeric"], spec["categorical"])
            pipeline.fit(train[feature_cols], train["future_any_event"])
            y_score = pipeline.predict_proba(test[feature_cols])[:, 1]

            fold_pred = test[
                [
                    "repo",
                    "snapshot_tag",
                    "snapshot_date",
                    "future_any_event",
                    "future_event_count",
                ]
            ].copy()
            fold_pred["model"] = model_name
            fold_pred["score"] = y_score
            prediction_frames.append(fold_pred)

            metrics = _score_binary_predictions(test["future_any_event"], y_score)
            metrics["model"] = model_name
            metrics["held_out_repo"] = held_out_repo
            metrics["n_snapshots"] = int(len(test))
            fold_rows.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    fold_metrics_df = pd.DataFrame(fold_rows)
    summary: dict[str, object] = {
        "n_snapshots": int(len(panel_df)),
        "n_repos": int(panel_df["repo"].nunique()),
        "models": {},
    }
    if predictions.empty:
        return predictions, fold_metrics_df, summary

    for model_name in model_specs:
        model_preds = predictions.loc[predictions["model"] == model_name].copy()
        model_folds = fold_metrics_df.loc[fold_metrics_df["model"] == model_name].copy()
        overall = _score_binary_predictions(model_preds["future_any_event"], model_preds["score"].to_numpy())
        if not model_folds.empty:
            overall["macro_roc_auc"] = float(model_folds["roc_auc"].mean(skipna=True))
            overall["macro_average_precision"] = float(
                model_folds["average_precision"].mean(skipna=True)
            )
            overall["macro_brier_score"] = float(model_folds["brier_score"].mean(skipna=True))
        summary["models"][model_name] = overall

    return predictions, fold_metrics_df, summary


def _build_forward_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def _score_binary_predictions(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    result = {
        "roc_auc": math.nan,
        "average_precision": math.nan,
        "brier_score": float(brier_score_loss(y_true_array, y_score)),
        "log_loss": float(log_loss(y_true_array, y_score, labels=[0, 1])),
    }
    if len(np.unique(y_true_array)) >= 2:
        result["roc_auc"] = float(roc_auc_score(y_true_array, y_score))
        result["average_precision"] = float(average_precision_score(y_true_array, y_score))
    return result


def plot_forward_event_rate_by_quartile(panel_df: pd.DataFrame, output_path: Path) -> None:
    """Plot future event rates by composite-opacity quartile."""
    if panel_df.empty:
        return
    quartiles = pd.qcut(
        panel_df["composite_score"],
        q=min(4, panel_df["composite_score"].nunique()),
        duplicates="drop",
    )
    summary = (
        panel_df.assign(opacity_quartile=quartiles.astype(str))
        .groupby("opacity_quartile", as_index=False)
        .agg(
            future_event_rate=("future_any_event", "mean"),
            mean_future_event_count=("future_event_count", "mean"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(summary["opacity_quartile"], summary["future_event_rate"], color="#4C78A8")
    axes[0].set_ylabel("Future-event rate")
    axes[0].set_title("Future event probability by opacity quartile")
    axes[1].bar(summary["opacity_quartile"], summary["mean_future_event_count"], color="#F58518")
    axes[1].set_ylabel("Mean future-event count")
    axes[1].set_title("Future event count by opacity quartile")
    for ax in axes:
        ax.set_xlabel("Composite opacity quartile")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_forward_model_comparison(summary: dict[str, object], output_path: Path) -> None:
    """Plot pooled model metrics for the forward panel."""
    if not summary or not summary.get("models"):
        return
    models = list(summary["models"].keys())
    roc_aucs = [summary["models"][name]["roc_auc"] for name in models]
    aps = [summary["models"][name]["average_precision"] for name in models]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].bar(models, roc_aucs, color=["#7A9E9F", "#4C78A8"])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Pooled ROC AUC")
    axes[0].set_title("Held-out discrimination")
    axes[1].bar(models, aps, color=["#C6B38E", "#F58518"])
    axes[1].set_ylabel("Pooled average precision")
    axes[1].set_title("Held-out precision-recall")
    for ax in axes:
        ax.set_xticks(range(len(models)), models, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
