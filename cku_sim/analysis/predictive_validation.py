"""Out-of-sample predictive validation for file-level vulnerability signals."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_SPECS = {
    "baseline_size": {
        "numeric": ["log_loc", "log_size_bytes"],
        "categorical": ["suffix"],
    },
    "baseline_plus_composite": {
        "numeric": ["log_loc", "log_size_bytes", "composite_score"],
        "categorical": ["suffix"],
    },
    "baseline_plus_components": {
        "numeric": [
            "log_loc",
            "log_size_bytes",
            "ci_gzip",
            "shannon_entropy",
            "cyclomatic_density",
            "halstead_volume",
        ],
        "categorical": ["suffix"],
    },
}


def build_prediction_dataset(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Convert matched case/control pairs into a file-level prediction dataset."""
    rows: list[dict[str, object]] = []

    for pair_id, row in pairs_df.reset_index(drop=True).iterrows():
        common = {
            "pair_id": int(pair_id),
            "repo": row["repo"],
            "commit": row["commit"],
            "cve_ids": row["cve_ids"],
            "event_id": row.get("event_id", row["cve_ids"]),
            "ground_truth_policy": row.get("ground_truth_policy", "nvd_commit_refs"),
            "ground_truth_source": row.get("ground_truth_source", "nvd_ref"),
        }

        case_file = str(row["case_file"])
        rows.append(
            {
                **common,
                "label": 1,
                "kind": "case",
                "file_path": case_file,
                "suffix": Path(case_file).suffix.lower(),
                "loc": float(row["case_loc"]),
                "size_bytes": float(row["case_size_bytes"]),
                "ci_gzip": float(row["case_ci_gzip"]),
                "shannon_entropy": float(row["case_entropy"]),
                "cyclomatic_density": float(row["case_cc_density"]),
                "halstead_volume": float(row["case_halstead"]),
                "composite_score": float(row["case_composite"]),
            }
        )

        control_file = str(row["control_file"])
        rows.append(
            {
                **common,
                "label": 0,
                "kind": "control",
                "file_path": control_file,
                "suffix": Path(control_file).suffix.lower(),
                "loc": float(row["control_loc"]),
                "size_bytes": float(row["control_size_bytes"]),
                "ci_gzip": float(row["control_ci_gzip"]),
                "shannon_entropy": float(row["control_entropy"]),
                "cyclomatic_density": float(row["control_cc_density"]),
                "halstead_volume": float(row["control_halstead"]),
                "composite_score": float(row["control_composite"]),
            }
        )

    dataset = pd.DataFrame(rows)
    dataset["log_loc"] = np.log1p(dataset["loc"])
    dataset["log_size_bytes"] = np.log1p(dataset["size_bytes"])
    dataset["suffix"] = dataset["suffix"].replace("", "<none>")
    return dataset


def pairwise_accuracy(predictions: pd.DataFrame, score_col: str) -> float:
    """Measure whether each case outranks its matched control."""
    if predictions.empty:
        return math.nan

    wins = []
    for _, group in predictions.groupby("pair_id"):
        if len(group) != 2 or set(group["label"]) != {0, 1}:
            continue
        case_score = float(group.loc[group["label"] == 1, score_col].iloc[0])
        control_score = float(group.loc[group["label"] == 0, score_col].iloc[0])
        if case_score > control_score:
            wins.append(1.0)
        elif case_score < control_score:
            wins.append(0.0)
        else:
            wins.append(0.5)

    return float(np.mean(wins)) if wins else math.nan


def _build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
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


def _score_predictions(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
    }


def evaluate_leave_one_repo_out(
    dataset: pd.DataFrame,
    model_specs: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run leave-one-repository-out prediction and collect held-out metrics."""
    model_specs = model_specs or MODEL_SPECS
    logo = LeaveOneGroupOut()

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for train_idx, test_idx in logo.split(dataset, dataset["label"], groups=dataset["repo"]):
        train = dataset.iloc[train_idx].copy()
        test = dataset.iloc[test_idx].copy()
        held_out_repo = str(test["repo"].iloc[0])

        for model_name, spec in model_specs.items():
            feature_cols = spec["numeric"] + spec["categorical"]
            pipeline = _build_pipeline(spec["numeric"], spec["categorical"])
            pipeline.fit(train[feature_cols], train["label"])
            y_score = pipeline.predict_proba(test[feature_cols])[:, 1]

            fold_pred = test[
                ["pair_id", "repo", "commit", "label", "kind", "file_path"]
            ].copy()
            fold_pred["model"] = model_name
            fold_pred["score"] = y_score
            prediction_frames.append(fold_pred)

            fold_metrics = _score_predictions(test["label"], y_score)
            fold_metrics["pairwise_accuracy"] = pairwise_accuracy(fold_pred, "score")
            fold_metrics["model"] = model_name
            fold_metrics["held_out_repo"] = held_out_repo
            fold_metrics["n_files"] = int(len(test))
            fold_metrics["n_pairs"] = int(test["pair_id"].nunique())
            fold_rows.append(fold_metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_rows)

    summary: dict[str, object] = {
        "n_files": int(len(dataset)),
        "n_pairs": int(dataset["pair_id"].nunique()),
        "n_commits": int(dataset["commit"].nunique()),
        "n_repos": int(dataset["repo"].nunique()),
        "models": {},
    }

    for model_name in model_specs:
        model_preds = predictions.loc[predictions["model"] == model_name].copy()
        model_folds = fold_metrics_df.loc[fold_metrics_df["model"] == model_name].copy()

        overall = _score_predictions(model_preds["label"], model_preds["score"].to_numpy())
        overall["pairwise_accuracy"] = pairwise_accuracy(model_preds, "score")
        overall["macro_roc_auc"] = float(model_folds["roc_auc"].mean())
        overall["macro_average_precision"] = float(model_folds["average_precision"].mean())
        overall["macro_brier_score"] = float(model_folds["brier_score"].mean())
        overall["macro_pairwise_accuracy"] = float(model_folds["pairwise_accuracy"].mean())
        summary["models"][model_name] = overall

    return predictions, fold_metrics_df, summary


def plot_model_comparison(summary: dict[str, object], output_path: Path) -> None:
    """Plot pooled out-of-sample model metrics for quick comparison."""
    models = list(summary["models"].keys())
    roc_aucs = [summary["models"][name]["roc_auc"] for name in models]
    pair_accs = [
        summary["models"][name].get("pairwise_accuracy", float("nan"))
        for name in models
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(models, roc_aucs, color=["#7A9E9F", "#4C78A8", "#F58518"])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.45, max(0.75, max(roc_aucs) + 0.05))
    axes[0].set_ylabel("Pooled ROC AUC")
    axes[0].set_title("Held-out discrimination")

    if any(pd.notna(pair_accs)):
        valid_pair_accs = [value for value in pair_accs if pd.notna(value)]
        axes[1].bar(models, pair_accs, color=["#7A9E9F", "#4C78A8", "#F58518"])
        axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylim(0.45, max(0.75, max(valid_pair_accs) + 0.05))
        axes[1].set_ylabel("Pairwise ranking accuracy")
        axes[1].set_title("Held-out pair ranking")
    else:
        axes[1].bar(models, [summary["models"][name]["average_precision"] for name in models], color=["#7A9E9F", "#4C78A8", "#F58518"])
        axes[1].set_ylabel("Average precision")
        axes[1].set_title("Held-out precision-recall")

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pooled_roc_curves(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot pooled ROC curves from all held-out predictions."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = {
        "baseline_size": "#7A9E9F",
        "baseline_plus_composite": "#4C78A8",
        "baseline_plus_components": "#F58518",
    }

    for model_name, group in predictions.groupby("model"):
        fpr, tpr, _ = roc_curve(group["label"], group["score"])
        auc = roc_auc_score(group["label"], group["score"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=palette.get(model_name))

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Leave-one-repository-out ROC curves")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
