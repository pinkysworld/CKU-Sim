"""Publication-quality visualisations for CKU-Sim results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

# Use non-interactive backend for server/CI environments
matplotlib.use("Agg")

# Consistent style
STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}
plt.rcParams.update(STYLE)

# Colour palette
PALETTE = {
    "high_opacity": "#d62728",
    "low_opacity": "#2ca02c",
    "mixed": "#1f77b4",
    "aware": "#1f77b4",
    "blind": "#ff7f0e",
}


def plot_synthetic_separation(
    df: pd.DataFrame, output_path: Path | str
) -> None:
    """Plot CI vs regularity for synthetic codebases (Simulation 1).

    Args:
        df: DataFrame from run_synthetic_separation with regularity, ci_mean columns.
        output_path: Path to save figure (PDF or PNG).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.scatter(
        df["regularity"], df["ci_mean"],
        c=df["regularity"], cmap="RdYlGn", s=40, edgecolors="black", linewidths=0.5,
        zorder=3,
    )

    # Trend line
    z = np.polyfit(df["regularity"], df["ci_mean"], 2)
    x_smooth = np.linspace(0, 1, 100)
    ax.plot(x_smooth, np.polyval(z, x_smooth), "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Regularity score (0 = irregular, 1 = regular)")
    ax.set_ylabel("Compressibility index (higher = more opaque)")
    ax.set_title("Synthetic Codebase Opacity Separation")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)


def plot_corpus_opacity(
    df: pd.DataFrame, output_path: Path | str
) -> None:
    """Bar chart of composite opacity scores across the corpus, coloured by category."""
    df_sorted = df.sort_values("composite_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.35)))

    colors = [PALETTE.get(cat, "#999999") for cat in df_sorted["category"]]
    ax.barh(df_sorted["name"], df_sorted["composite_score"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Composite opacity score")
    ax.set_title("Structural Opacity Across Corpus")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PALETTE[k], label=k.replace("_", " ").title())
               for k in ("high_opacity", "low_opacity", "mixed") if k in PALETTE]
    ax.legend(handles=handles, loc="lower right")

    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)


def plot_opacity_vs_cve(
    df: pd.DataFrame, output_path: Path | str
) -> None:
    """Scatter plot: composite opacity vs CVE density with regression line."""
    valid = df.dropna(subset=["composite_score", "cve_density"])
    if len(valid) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [PALETTE.get(cat, "#999999") for cat in valid["category"]]
    ax.scatter(
        valid["composite_score"], valid["cve_density"],
        c=colors, s=60, edgecolors="black", linewidths=0.5, zorder=3,
    )

    # Label points
    for _, row in valid.iterrows():
        ax.annotate(
            row["name"], (row["composite_score"], row["cve_density"]),
            fontsize=7, alpha=0.7, xytext=(5, 5), textcoords="offset points",
        )

    # Regression line
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(
        valid["composite_score"], valid["cve_density"]
    )
    x_line = np.linspace(valid["composite_score"].min(), valid["composite_score"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.5,
            label=f"r={r:.3f}, p={p:.3f}")

    ax.set_xlabel("Composite opacity score")
    ax.set_ylabel("CVE density (CVEs per KLOC)")
    ax.set_title("Structural Opacity vs Vulnerability Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)


def plot_insurance_comparison(
    result: dict, output_path: Path | str
) -> None:
    """Side-by-side comparison of opacity-aware vs opacity-blind loss distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss distributions
    ax = axes[0]
    for model, color, label in [
        ("aware_model", PALETTE["aware"], "Opacity-aware"),
        ("blind_model", PALETTE["blind"], "Opacity-blind"),
    ]:
        var99 = result[model]["VaR_99"]
        ax.axvline(var99, color=color, linestyle="--", alpha=0.8,
                   label=f"{label} VaR99: ${var99:,.0f}")

    ax.set_xlabel("Portfolio loss ($)")
    ax.set_ylabel("Density")
    ax.set_title("VaR Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Gap chart
    ax = axes[1]
    gaps = result["gaps"]
    gap_labels = [k for k in gaps if k.startswith("VaR")]
    gap_values = [gaps[k] for k in gap_labels]
    clean_labels = [k.replace("_gap_pct", "").replace("_", " ") for k in gap_labels]

    bars = ax.bar(clean_labels, gap_values, color=PALETTE["aware"],
                  edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Gap (%)")
    ax.set_title("Opacity-Aware vs Blind: Risk Underestimation")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, gap_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_algorithm_robustness(
    df: pd.DataFrame, output_path: Path | str
) -> None:
    """Pairwise scatter plots of CI across compression algorithms."""
    pairs = [("ci_gzip", "ci_lzma"), ("ci_gzip", "ci_zstd"), ("ci_lzma", "ci_zstd")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (a, b) in zip(axes, pairs):
        if a not in df.columns or b not in df.columns:
            continue
        colors = [PALETTE.get(cat, "#999999") for cat in df["category"]]
        ax.scatter(df[a], df[b], c=colors, s=40, edgecolors="black", linewidths=0.5)

        # Identity line
        lims = [min(df[a].min(), df[b].min()), max(df[a].max(), df[b].max())]
        ax.plot(lims, lims, "k--", alpha=0.3)

        from scipy import stats
        rho, _ = stats.spearmanr(df[a], df[b])
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"ρ = {rho:.4f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Compression Algorithm Robustness", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_weight_sensitivity(
    sensitivity_df: pd.DataFrame, output_path: Path | str
) -> None:
    """Distribution of Spearman ρ under random weight perturbation."""
    if sensitivity_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(sensitivity_df["spearman_rho"], bins=50, color=PALETTE["aware"],
            edgecolor="black", linewidth=0.5, alpha=0.8)

    mean_rho = sensitivity_df["spearman_rho"].mean()
    ax.axvline(mean_rho, color="red", linestyle="--",
               label=f"Mean ρ = {mean_rho:.4f}")

    ax.set_xlabel("Spearman ρ (composite opacity vs CVE density)")
    ax.set_ylabel("Count")
    ax.set_title("Weight Sensitivity Analysis (1000 random weight vectors)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
