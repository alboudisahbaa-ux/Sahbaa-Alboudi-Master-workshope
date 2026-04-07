"""Exploratory Data Analysis — advanced, explanatory visualizations.

Generates a comprehensive set of publication-quality figures that cover:
  1. Dataset overview dashboard (shape, types, quality indicators)
  2. Class distribution (annotated bar + pie)
  3. Data-quality matrix (missing & infinite per column)
  4. Correlation analysis (clustered heatmap + top correlated pairs)
  5. Per-feature statistics (skewness, kurtosis, outlier %)
  6. Feature distributions (violin plots split by class)
  7. KDE histograms for top features per class
  8. Outlier z-score landscape
  9. Feature variance ranking
 10. PCA projection (with explained-variance curve + class centroids)
 11. t-SNE 2-D embedding (non-linear structure)
 12. Pairplot of top discriminative features
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Plotting defaults ────────────────────────────────────────────
_STYLE = "whitegrid"
_PALETTE = "Set2"
_DPI = 180
_TITLE_SIZE = 14
_LABEL_SIZE = 11


def _apply_style() -> None:
    sns.set_style(_STYLE)
    plt.rcParams.update({
        "axes.titlesize": _TITLE_SIZE,
        "axes.labelsize": _LABEL_SIZE,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": _DPI,
    })


class DataEDA:
    """Generate advanced EDA summaries and explanatory visualizations."""

    def __init__(self, cfg: dict):
        data_cfg = cfg.get("data", {})
        out_cfg = cfg.get("output", {})
        eda_cfg = cfg.get("eda", {})

        self.target_col = data_cfg.get("target_column", "label")
        self.seed = cfg.get("project", {}).get("seed", 42)
        self.reports_dir = Path(out_cfg.get("reports_dir", "reports"))
        self.eda_dir = self.reports_dir / "eda"

        self.sample_size = int(eda_cfg.get("sample_size", 50000))
        self.top_n_features = int(eda_cfg.get("top_n_features", 12))
        self.corr_top_features = int(eda_cfg.get("corr_top_features", 20))
        self.max_classes_plot = int(eda_cfg.get("max_classes_plot", 20))
        self.tsne_perplexity = int(eda_cfg.get("tsne_perplexity", 30))
        self.tsne_sample = int(eda_cfg.get("tsne_sample", 8000))
        self.pairplot_features = int(eda_cfg.get("pairplot_features", 4))

    # ================================================================
    # Public entry
    # ================================================================
    def run(self, df: pd.DataFrame) -> dict:
        """Execute the full EDA pipeline. Returns a summary dict."""
        if self.target_col not in df.columns:
            raise KeyError(
                f"Target column '{self.target_col}' not found for EDA. "
                f"Available columns include: {list(df.columns)[:10]}…"
            )

        _apply_style()
        self.eda_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running advanced EDA → %s", self.eda_dir)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        inf_count = int(np.isinf(df[numeric_cols].values).sum()) if numeric_cols else 0
        missing_count = int(df.isnull().sum().sum())
        dup_count = int(df.duplicated().sum())

        summary = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_columns": len(numeric_cols),
            "non_numeric_columns": int(df.shape[1]) - len(numeric_cols),
            "missing_values": missing_count,
            "infinite_values": inf_count,
            "duplicate_rows": dup_count,
            "class_distribution": {
                str(k): int(v)
                for k, v in df[self.target_col].value_counts().to_dict().items()
            },
        }

        self._save_summary(summary, df)

        # --- Plots (order matters for narrative flow) ---
        self._plot_dataset_overview_dashboard(summary, df)
        self._plot_class_distribution(df)
        self._plot_data_quality_matrix(df)
        self._plot_feature_statistics_table(df)

        sampled = self._sample(df)
        self._plot_correlation_analysis(sampled)
        self._plot_feature_distributions_violin(sampled)
        self._plot_kde_histograms(sampled)
        self._plot_outlier_landscape(sampled)
        self._plot_feature_variance_ranking(sampled)
        self._plot_pca_projection(sampled)
        self._plot_tsne_projection(sampled)
        self._plot_pairplot(sampled)

        logger.info("Advanced EDA complete — %d figures saved to %s",
                     len(list(self.eda_dir.glob("*.png"))), self.eda_dir)
        return summary

    # ================================================================
    # Helpers
    # ================================================================
    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.sample_size:
            return df
        return df.sample(n=self.sample_size, random_state=self.seed)

    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include=[np.number]).copy()

    def _clean_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        num = self._numeric(df)
        num = num.replace([np.inf, -np.inf], np.nan)
        return num.fillna(num.median())

    def _top_var_cols(self, df: pd.DataFrame, n: int) -> List[str]:
        num = self._numeric(df)
        return num.var().sort_values(ascending=False).head(n).index.tolist()

    def _savefig(self, name: str) -> None:
        plt.savefig(self.eda_dir / name, dpi=_DPI, bbox_inches="tight",
                     facecolor="white", edgecolor="none")
        plt.close()

    # ================================================================
    # 0. Persist summary artefacts
    # ================================================================
    def _save_summary(self, summary: dict, df: pd.DataFrame) -> None:
        with (self.eda_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Class distribution CSV
        pd.DataFrame({
            "class": list(summary["class_distribution"].keys()),
            "count": list(summary["class_distribution"].values()),
        }).to_csv(self.eda_dir / "class_distribution.csv", index=False)

        # Full descriptive statistics CSV
        df.describe(include="all").T.to_csv(self.eda_dir / "descriptive_stats.csv")

    # ================================================================
    # 1. Dataset overview dashboard
    # ================================================================
    def _plot_dataset_overview_dashboard(self, summary: dict, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

        # -- KPI cards (top row, first 4 slots) --
        kpis = [
            ("Rows", f"{summary['rows']:,}"),
            ("Columns", f"{summary['columns']:,}"),
            ("Missing Values", f"{summary['missing_values']:,}"),
            ("Infinite Values", f"{summary['infinite_values']:,}"),
        ]
        for idx, (label, value) in enumerate(kpis):
            ax = fig.add_subplot(gs[0, idx])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor="#e8f4fd", edgecolor="#4a90d9",
                                  linewidth=2)
            ax.add_patch(box)
            ax.text(0.5, 0.62, value, ha="center", va="center",
                    fontsize=22, fontweight="bold", color="#2c3e50")
            ax.text(0.5, 0.28, label, ha="center", va="center",
                    fontsize=11, color="#7f8c8d")

        # -- Dtype breakdown pie (bottom-left) --
        ax_dtype = fig.add_subplot(gs[1, 0:2])
        dtype_counts = df.dtypes.astype(str).value_counts()
        colors = sns.color_palette("pastel", len(dtype_counts))
        wedges, texts, autotexts = ax_dtype.pie(
            dtype_counts.values, labels=dtype_counts.index,
            autopct="%1.0f%%", colors=colors, startangle=140,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax_dtype.set_title("Column Data Types", fontsize=12, fontweight="bold")

        # -- Duplicates + quality bar (bottom-right) --
        ax_q = fig.add_subplot(gs[1, 2:4])
        total = summary["rows"]
        clean = total - summary["duplicate_rows"]
        bars = ax_q.barh(
            ["Unique rows", "Duplicate rows"],
            [clean, summary["duplicate_rows"]],
            color=["#27ae60", "#e74c3c"],
            edgecolor="white",
        )
        for bar, val in zip(bars, [clean, summary["duplicate_rows"]]):
            ax_q.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                      f"{val:,} ({val / total * 100:.1f}%)",
                      va="center", fontsize=10)
        ax_q.set_xlim(0, total * 1.25)
        ax_q.set_title("Data Uniqueness", fontsize=12, fontweight="bold")
        ax_q.set_xlabel("Number of Rows")

        fig.suptitle("Dataset Overview Dashboard", fontsize=16, fontweight="bold", y=1.01)
        self._savefig("01_dataset_overview_dashboard.png")

    # ================================================================
    # 2. Class distribution — bar + pie
    # ================================================================
    def _plot_class_distribution(self, df: pd.DataFrame) -> None:
        counts = df[self.target_col].value_counts().head(self.max_classes_plot)
        total = counts.sum()
        pct = (counts / total * 100).round(1)

        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(18, 6),
                                              gridspec_kw={"width_ratios": [3, 2]})

        # Bar chart with annotations
        palette = sns.color_palette("crest", len(counts))
        bars = ax_bar.bar(range(len(counts)), counts.values, color=palette,
                          edgecolor="white", linewidth=0.8)
        for bar, p in zip(bars, pct.values):
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + total * 0.005,
                        f"{int(h):,}\n({p}%)", ha="center", va="bottom", fontsize=8)
        ax_bar.set_xticks(range(len(counts)))
        ax_bar.set_xticklabels([textwrap.shorten(str(c), 20) for c in counts.index],
                                rotation=40, ha="right", fontsize=9)
        ax_bar.set_ylabel("Count")
        ax_bar.set_title("Class Frequency (Bar Chart)", fontweight="bold")
        ax_bar.spines[["top", "right"]].set_visible(False)

        # Pie / donut chart
        wedges, texts, autotexts = ax_pie.pie(
            counts.values,
            labels=[textwrap.shorten(str(c), 15) for c in counts.index],
            autopct="%1.1f%%",
            colors=palette,
            startangle=90,
            pctdistance=0.78,
            wedgeprops={"width": 0.45, "edgecolor": "white"},
            textprops={"fontsize": 8},
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_fontweight("bold")
        ax_pie.set_title("Class Proportion (Donut Chart)", fontweight="bold")

        # Majority / minority annotation
        majority = counts.idxmax()
        minority = counts.idxmin()
        ratio = counts.max() / max(counts.min(), 1)
        fig.text(0.5, -0.02,
                 f"Majority class: {majority} ({counts.max():,})  |  "
                 f"Minority class: {minority} ({counts.min():,})  |  "
                 f"Imbalance ratio: {ratio:.1f}:1",
                 ha="center", fontsize=10, style="italic", color="#555")

        fig.suptitle("Class Distribution Analysis", fontsize=15, fontweight="bold")
        self._savefig("02_class_distribution.png")

    # ================================================================
    # 3. Data quality matrix (missing + infinite heatmap)
    # ================================================================
    def _plot_data_quality_matrix(self, df: pd.DataFrame) -> None:
        numeric_df = self._numeric(df)
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        inf_pct = pd.Series(0.0, index=df.columns)
        if not numeric_df.empty:
            inf_per_col = np.isinf(numeric_df).sum()
            inf_pct[inf_per_col.index] = (inf_per_col / len(df) * 100)

        # Keep only columns that have any issue, limit to top 40
        problem_cols = missing_pct.index[
            (missing_pct > 0) | (inf_pct[missing_pct.index] > 0)
        ][:40]

        if problem_cols.empty:
            logger.info("No data-quality issues detected — skipping quality matrix.")
            # Still produce a small "all clear" figure
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "All columns clean — no missing or infinite values detected",
                    ha="center", va="center", fontsize=13, color="#27ae60",
                    fontweight="bold", transform=ax.transAxes)
            ax.axis("off")
            fig.suptitle("Data Quality Matrix", fontsize=14, fontweight="bold")
            self._savefig("03_data_quality_matrix.png")
            return

        quality_df = pd.DataFrame({
            "Missing %": missing_pct[problem_cols].values,
            "Infinite %": inf_pct[problem_cols].values,
        }, index=[textwrap.shorten(str(c), 25) for c in problem_cols])

        fig, ax = plt.subplots(figsize=(10, max(4, len(problem_cols) * 0.35)))
        sns.heatmap(quality_df, annot=True, fmt=".2f", cmap="YlOrRd",
                     linewidths=0.5, cbar_kws={"label": "Percentage (%)"}, ax=ax)
        ax.set_title("Data Quality Matrix — Missing & Infinite Values per Column",
                      fontweight="bold")
        ax.set_ylabel("")
        self._savefig("03_data_quality_matrix.png")

    # ================================================================
    # 4. Feature statistics table (skew, kurtosis, outlier %)
    # ================================================================
    def _plot_feature_statistics_table(self, df: pd.DataFrame) -> None:
        numeric_df = self._numeric(df)
        if numeric_df.shape[1] < 1:
            return

        clean = numeric_df.replace([np.inf, -np.inf], np.nan)
        stats_rows = []
        for col in clean.columns:
            s = clean[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_pct = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100
            stats_rows.append({
                "Feature": textwrap.shorten(str(col), 28),
                "Mean": f"{s.mean():.2f}",
                "Std": f"{s.std():.2f}",
                "Skewness": f"{s.skew():.2f}",
                "Kurtosis": f"{s.kurtosis():.2f}",
                "Outlier %": f"{outlier_pct:.1f}",
            })

        # Show top features sorted by absolute skewness
        stats_df = pd.DataFrame(stats_rows)
        stats_df["_abs_skew"] = stats_df["Skewness"].astype(float).abs()
        stats_df = stats_df.sort_values("_abs_skew", ascending=False).head(25)
        stats_df = stats_df.drop(columns="_abs_skew").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, max(4, len(stats_df) * 0.38)))
        ax.axis("off")
        table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        # Header styling
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4a90d9")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f0f4f8")

        fig.suptitle("Feature Statistics — Skewness, Kurtosis & Outlier Profile (Top 25)",
                      fontsize=13, fontweight="bold", y=0.98)
        self._savefig("04_feature_statistics_table.png")

        stats_df.to_csv(self.eda_dir / "feature_statistics.csv", index=False)

    # ================================================================
    # 5. Correlation analysis (clustered heatmap + top pairs)
    # ================================================================
    def _plot_correlation_analysis(self, df: pd.DataFrame) -> None:
        numeric_df = self._numeric(df)
        if numeric_df.shape[1] < 2:
            logger.info("Not enough numeric columns for correlation analysis.")
            return

        top_cols = self._top_var_cols(df, self.corr_top_features)
        corr = numeric_df[top_cols].corr()

        # Clustered heatmap
        g = sns.clustermap(
            corr, cmap="RdBu_r", center=0, figsize=(14, 12),
            linewidths=0.3, annot=False,
            dendrogram_ratio=(0.12, 0.12),
            cbar_pos=(0.02, 0.82, 0.03, 0.15),
            xticklabels=[textwrap.shorten(str(c), 18) for c in top_cols],
            yticklabels=[textwrap.shorten(str(c), 18) for c in top_cols],
        )
        g.fig.suptitle(
            f"Clustered Correlation Heatmap (Top {len(top_cols)} Variable Features)",
            fontsize=14, fontweight="bold", y=1.02,
        )
        g.savefig(self.eda_dir / "05a_correlation_clustered_heatmap.png",
                   dpi=_DPI, bbox_inches="tight")
        plt.close()

        # Top correlated pairs bar chart
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        pairs_df = pd.DataFrame(pairs, columns=["Feature A", "Feature B", "r"])
        pairs_df["abs_r"] = pairs_df["r"].abs()
        top_pairs = pairs_df.sort_values("abs_r", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 7))
        pair_labels = [
            f"{textwrap.shorten(str(a), 14)} ↔ {textwrap.shorten(str(b), 14)}"
            for a, b in zip(top_pairs["Feature A"], top_pairs["Feature B"])
        ]
        colors = ["#e74c3c" if r < 0 else "#2980b9" for r in top_pairs["r"]]
        ax.barh(range(len(top_pairs)), top_pairs["r"].values, color=colors,
                edgecolor="white")
        ax.set_yticks(range(len(top_pairs)))
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.set_xlabel("Pearson Correlation (r)")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title("Top 20 Most Correlated Feature Pairs", fontweight="bold")
        ax.invert_yaxis()
        for i, r in enumerate(top_pairs["r"].values):
            ax.text(r + 0.01 * np.sign(r), i, f"{r:.3f}", va="center", fontsize=8)
        self._savefig("05b_top_correlated_pairs.png")

        top_pairs.to_csv(self.eda_dir / "top_correlated_pairs.csv", index=False)

    # ================================================================
    # 6. Feature distributions — violin plots split by class
    # ================================================================
    def _plot_feature_distributions_violin(self, df: pd.DataFrame) -> None:
        numeric_df = self._numeric(df)
        if numeric_df.empty:
            return

        top_cols = self._top_var_cols(df, min(self.top_n_features, 8))
        n = len(top_cols)
        ncols = 2
        nrows = (n + 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
        axes = np.atleast_2d(axes).flatten()

        for idx, col in enumerate(top_cols):
            ax = axes[idx]
            plot_df = df[[col, self.target_col]].copy()
            plot_df[self.target_col] = plot_df[self.target_col].astype(str)
            # Clip extreme values for visual clarity
            q_lo, q_hi = plot_df[col].quantile(0.01), plot_df[col].quantile(0.99)
            plot_df[col] = plot_df[col].clip(q_lo, q_hi)
            sns.violinplot(data=plot_df, x=self.target_col, y=col, ax=ax,
                           hue=self.target_col, palette=_PALETTE,
                           inner="quartile", linewidth=0.7,
                           cut=0, density_norm="width", legend=False)
            ax.set_title(textwrap.shorten(str(col), 35), fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30, labelsize=7)

        for idx in range(len(top_cols), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Feature Distributions by Class (Violin Plots)",
                      fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._savefig("06_feature_distributions_violin.png")

    # ================================================================
    # 7. KDE histograms for top features per class
    # ================================================================
    def _plot_kde_histograms(self, df: pd.DataFrame) -> None:
        numeric_df = self._numeric(df)
        if numeric_df.empty:
            return

        top_cols = self._top_var_cols(df, min(self.top_n_features, 6))
        classes = df[self.target_col].unique()
        palette = sns.color_palette("tab10", len(classes))
        n = len(top_cols)
        ncols = 2
        nrows = (n + 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
        axes = np.atleast_2d(axes).flatten()

        for idx, col in enumerate(top_cols):
            ax = axes[idx]
            for ci, cls in enumerate(classes):
                subset = df.loc[df[self.target_col] == cls, col].dropna()
                q_lo, q_hi = subset.quantile(0.01), subset.quantile(0.99)
                subset = subset.clip(q_lo, q_hi)
                sns.kdeplot(subset, ax=ax, label=str(cls), color=palette[ci % len(palette)],
                            fill=True, alpha=0.25, linewidth=1.2)
            ax.set_title(textwrap.shorten(str(col), 35), fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            if idx == 0:
                ax.legend(fontsize=6, loc="upper right", ncol=2)
            else:
                ax.legend().set_visible(False)

        for idx in range(len(top_cols), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("KDE Density Histograms by Class (Top Features)",
                      fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._savefig("07_kde_histograms.png")

    # ================================================================
    # 8. Outlier z-score landscape
    # ================================================================
    def _plot_outlier_landscape(self, df: pd.DataFrame) -> None:
        clean = self._clean_numeric(df)
        if clean.shape[1] < 2:
            return

        top_cols = self._top_var_cols(df, min(self.top_n_features, 15))
        subset = clean[top_cols]

        z = np.abs(sp_stats.zscore(subset, nan_policy="omit"))
        outlier_pct = np.asarray((z > 3).mean(axis=0) * 100).ravel()
        mean_z = np.asarray(z.mean(axis=0)).ravel()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6),
                                        gridspec_kw={"width_ratios": [2, 1]})

        # Heatmap of z-score medians per feature
        z_summary = pd.DataFrame({
            "Outlier % (|z|>3)": outlier_pct,
            "Mean |z|": mean_z,
        }, index=[textwrap.shorten(str(c), 22) for c in top_cols])

        sns.heatmap(z_summary, annot=True, fmt=".2f", cmap="YlOrRd",
                     linewidths=0.5, ax=ax1)
        ax1.set_title("Outlier Profile per Feature", fontweight="bold")
        ax1.set_ylabel("")

        # Box plot of z-scores
        z_df = pd.DataFrame(
            np.abs(sp_stats.zscore(subset, nan_policy="omit")),
            columns=[textwrap.shorten(str(c), 18) for c in top_cols],
        )
        z_melted = z_df.melt(var_name="Feature", value_name="|z-score|")
        sns.boxplot(data=z_melted, x="|z-score|", y="Feature", ax=ax2,
                    hue="Feature", showfliers=False, palette="coolwarm",
                    orient="h", legend=False)
        ax2.axvline(3, color="red", linestyle="--", linewidth=1.2, label="|z|=3 threshold")
        ax2.legend(fontsize=8)
        ax2.set_title("Z-Score Distribution", fontweight="bold")

        fig.suptitle("Outlier Landscape — Z-Score Analysis",
                      fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._savefig("08_outlier_landscape.png")

    # ================================================================
    # 9. Feature variance ranking
    # ================================================================
    def _plot_feature_variance_ranking(self, df: pd.DataFrame) -> None:
        numeric_df = self._numeric(df)
        if numeric_df.shape[1] < 2:
            return

        variance = numeric_df.var().sort_values(ascending=True)
        top = variance.tail(min(30, len(variance)))

        fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.32)))
        colors = sns.color_palette("viridis", len(top))
        ax.barh(range(len(top)), top.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([textwrap.shorten(str(c), 25) for c in top.index], fontsize=8)
        ax.set_xlabel("Variance")
        ax.set_title("Feature Variance Ranking (Top 30)", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate values
        for i, v in enumerate(top.values):
            ax.text(v + top.max() * 0.01, i, f"{v:.1f}", va="center", fontsize=7)

        self._savefig("09_feature_variance_ranking.png")

    # ================================================================
    # 10. PCA projection (explained variance + 2-D scatter with centroids)
    # ================================================================
    def _plot_pca_projection(self, df: pd.DataFrame) -> None:
        clean = self._clean_numeric(df)
        if clean.shape[1] < 2:
            return

        scaled = StandardScaler().fit_transform(clean.values)
        n_comp = min(clean.shape[1], 10)
        pca_full = PCA(n_components=n_comp, random_state=self.seed)
        pca_full.fit(scaled)

        pca2 = PCA(n_components=2, random_state=self.seed)
        proj = pca2.fit_transform(scaled)

        labels = df.loc[clean.index, self.target_col].astype(str).values
        pca_df = pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "Class": labels})

        fig = plt.figure(figsize=(18, 7))
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.5, 2.5, 1.5], wspace=0.3)

        # -- Explained variance curve --
        ax_var = fig.add_subplot(gs[0])
        cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
        ax_var.bar(range(1, n_comp + 1), pca_full.explained_variance_ratio_ * 100,
                   color="#3498db", alpha=0.7, label="Individual")
        ax_var.plot(range(1, n_comp + 1), cum_var, "o-", color="#e74c3c",
                    linewidth=2, label="Cumulative")
        ax_var.axhline(95, color="grey", linestyle="--", linewidth=0.8)
        ax_var.set_xlabel("Principal Component")
        ax_var.set_ylabel("Explained Variance (%)")
        ax_var.set_title("Scree Plot", fontweight="bold")
        ax_var.legend(fontsize=8)
        for i, v in enumerate(cum_var):
            ax_var.annotate(f"{v:.1f}%", (i + 1, v), textcoords="offset points",
                            xytext=(0, 8), fontsize=7, ha="center")

        # -- 2-D scatter --
        ax_scatter = fig.add_subplot(gs[1])
        classes = pca_df["Class"].unique()
        palette = sns.color_palette("tab10", len(classes))
        for ci, cls in enumerate(classes):
            mask = pca_df["Class"] == cls
            ax_scatter.scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
                               s=12, alpha=0.5, color=palette[ci % len(palette)],
                               label=textwrap.shorten(str(cls), 15), edgecolors="none")
            # Class centroid
            cx = pca_df.loc[mask, "PC1"].mean()
            cy = pca_df.loc[mask, "PC2"].mean()
            ax_scatter.scatter(cx, cy, s=180, color=palette[ci % len(palette)],
                               edgecolors="black", linewidth=1.5, marker="X", zorder=5)

        ev1 = pca2.explained_variance_ratio_[0] * 100
        ev2 = pca2.explained_variance_ratio_[1] * 100
        ax_scatter.set_xlabel(f"PC 1 ({ev1:.1f}% var)")
        ax_scatter.set_ylabel(f"PC 2 ({ev2:.1f}% var)")
        ax_scatter.set_title("PCA 2-D Projection with Class Centroids", fontweight="bold")
        ax_scatter.legend(fontsize=7, loc="best", ncol=2, markerscale=1.5)

        # -- Top loadings --
        ax_load = fig.add_subplot(gs[2])
        loadings = pd.DataFrame(pca2.components_.T,
                                 columns=["PC1", "PC2"],
                                 index=clean.columns)
        top_load = loadings["PC1"].abs().sort_values(ascending=False).head(10)
        load_vals = loadings.loc[top_load.index, "PC1"]
        colors = ["#e74c3c" if v < 0 else "#2980b9" for v in load_vals]
        ax_load.barh(range(len(load_vals)), load_vals.values, color=colors)
        ax_load.set_yticks(range(len(load_vals)))
        ax_load.set_yticklabels([textwrap.shorten(str(c), 18) for c in load_vals.index],
                                 fontsize=8)
        ax_load.invert_yaxis()
        ax_load.set_xlabel("PC1 Loading")
        ax_load.set_title("Top 10 PC1 Loadings", fontweight="bold")

        fig.suptitle("Principal Component Analysis", fontsize=15, fontweight="bold")
        self._savefig("10_pca_projection.png")

    # ================================================================
    # 11. t-SNE projection
    # ================================================================
    def _plot_tsne_projection(self, df: pd.DataFrame) -> None:
        clean = self._clean_numeric(df)
        if clean.shape[1] < 2:
            return

        # Subsample for t-SNE performance
        n = min(self.tsne_sample, len(clean))
        if n < len(clean):
            idx = clean.sample(n=n, random_state=self.seed).index
        else:
            idx = clean.index

        scaled = StandardScaler().fit_transform(clean.loc[idx].values)
        perp = min(self.tsne_perplexity, n - 1)

        logger.info("Running t-SNE on %d samples (perplexity=%d)…", n, perp)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=self.seed,
                     max_iter=800, init="pca", learning_rate="auto")
        proj = tsne.fit_transform(scaled)

        labels = df.loc[idx, self.target_col].astype(str).values
        tsne_df = pd.DataFrame({"t-SNE 1": proj[:, 0], "t-SNE 2": proj[:, 1],
                                 "Class": labels})

        fig, ax = plt.subplots(figsize=(12, 9))
        classes = tsne_df["Class"].unique()
        palette = sns.color_palette("tab10", len(classes))
        for ci, cls in enumerate(classes):
            mask = tsne_df["Class"] == cls
            ax.scatter(tsne_df.loc[mask, "t-SNE 1"], tsne_df.loc[mask, "t-SNE 2"],
                       s=15, alpha=0.6, color=palette[ci % len(palette)],
                       label=textwrap.shorten(str(cls), 15), edgecolors="none")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title("t-SNE 2-D Embedding of Network Flows", fontweight="bold")
        ax.legend(fontsize=8, loc="best", ncol=2, markerscale=2)

        fig.text(0.5, -0.01,
                 f"Perplexity={perp} | Samples={n:,} | "
                 f"KL divergence={tsne.kl_divergence_:.4f}",
                 ha="center", fontsize=9, style="italic", color="#666")

        self._savefig("11_tsne_projection.png")

    # ================================================================
    # 12. Pairplot of top discriminative features
    # ================================================================
    def _plot_pairplot(self, df: pd.DataFrame) -> None:
        numeric_df = self._numeric(df)
        if numeric_df.shape[1] < 2:
            return

        top_cols = self._top_var_cols(df, self.pairplot_features)
        plot_df = df[top_cols + [self.target_col]].copy()
        plot_df[self.target_col] = plot_df[self.target_col].astype(str)

        # Clip extremes for visual clarity
        for col in top_cols:
            lo, hi = plot_df[col].quantile(0.02), plot_df[col].quantile(0.98)
            plot_df[col] = plot_df[col].clip(lo, hi)

        # Subsample for speed
        if len(plot_df) > 5000:
            plot_df = plot_df.sample(n=5000, random_state=self.seed)

        g = sns.pairplot(plot_df, hue=self.target_col, palette="tab10",
                          diag_kind="kde",
                          plot_kws={"s": 12, "alpha": 0.45, "edgecolor": "none"},
                          diag_kws={"fill": True, "alpha": 0.4, "linewidth": 1})
        g.figure.suptitle("Pairplot — Top Discriminative Features",
                           fontsize=14, fontweight="bold", y=1.02)
        g.savefig(self.eda_dir / "12_pairplot.png", dpi=_DPI, bbox_inches="tight")
        plt.close()
