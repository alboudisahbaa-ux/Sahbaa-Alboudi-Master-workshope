"""Error analysis — misclassification patterns and insights (Phase 4)."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyze misclassification patterns across models."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def analyse(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X_test: np.ndarray,
        feature_names: list,
        class_names: list | None = None,
        save_dir: str | None = None,
    ) -> Dict[str, Any]:
        """Run full error analysis for a single model.

        Returns a dict with misclassification stats and a DataFrame of
        misclassified samples.
        """
        mask_wrong = y_true != y_pred
        n_wrong = mask_wrong.sum()
        n_total = len(y_true)

        logger.info(
            "[%s] Misclassified: %d / %d (%.2f%%)",
            model_name, n_wrong, n_total, 100 * n_wrong / n_total,
        )

        if n_wrong == 0:
            logger.info("[%s] Perfect predictions — no errors to analyze", model_name)
            return {"n_wrong": 0, "error_rate": 0.0}

        # ── Misclassification per class ───────────────────────────
        classes = np.unique(y_true)
        per_class = {}
        for c in classes:
            mask_c = y_true == c
            wrong_c = (mask_wrong & mask_c).sum()
            total_c = mask_c.sum()
            label = class_names[c] if class_names and c < len(class_names) else str(c)
            per_class[label] = {
                "total": int(total_c),
                "misclassified": int(wrong_c),
                "error_rate": wrong_c / total_c if total_c > 0 else 0.0,
            }

        per_class_df = pd.DataFrame(per_class).T
        per_class_df.index.name = "Class"
        logger.info("[%s] Per-class errors:\n%s", model_name, per_class_df.to_string())

        # ── Feature distribution comparison ───────────────────────
        df_test = pd.DataFrame(X_test, columns=feature_names)
        df_test["_correct"] = ~mask_wrong

        # Mean feature values: correct vs incorrect
        correct_means = df_test[df_test["_correct"]].drop(columns=["_correct"]).mean()
        wrong_means = df_test[~df_test["_correct"]].drop(columns=["_correct"]).mean()

        diff = (wrong_means - correct_means).abs().sort_values(ascending=False)
        top_diff_features = diff.head(10)

        logger.info(
            "[%s] Top features differentiating correct/incorrect:\n%s",
            model_name, top_diff_features.to_string(),
        )

        # ── Plot: per-class error rates ───────────────────────────
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of per-class error rates
        ax = axes[0]
        per_class_df["error_rate"].plot(kind="bar", ax=ax, color="salmon")
        ax.set_title(f"{model_name} — Per-Class Error Rate")
        ax.set_ylabel("Error Rate")
        ax.set_ylim(0, max(per_class_df["error_rate"].max() * 1.3, 0.1))
        ax.tick_params(axis="x", rotation=0)

        # Bar chart of top feature differences
        ax = axes[1]
        top_diff_features.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"{model_name} — Feature Diff (incorrect vs correct)")
        ax.set_xlabel("Absolute Mean Difference")

        plt.tight_layout()
        if save_dir:
            path = f"{save_dir}/error_analysis_{model_name}.png"
            plt.savefig(path, dpi=150)
            logger.info("Error analysis plot saved to %s", path)
        plt.show()

        # ── Confidence analysis (if probabilities available) ──────
        result: Dict[str, Any] = {
            "n_wrong": int(n_wrong),
            "error_rate": n_wrong / n_total,
            "per_class": per_class,
            "top_diff_features": top_diff_features.to_dict(),
        }
        return result

    def compare_errors(
        self,
        evaluator_results: Dict[str, Dict[str, Any]],
        save_dir: str | None = None,
    ) -> pd.DataFrame:
        """Compare misclassification rates across all evaluated models."""
        rows = []
        for name, m in evaluator_results.items():
            y_true = m.get("y_true")
            y_pred = m.get("y_pred")
            if y_true is None or y_pred is None:
                continue
            wrong = (y_true != y_pred).sum()
            total = len(y_true)
            rows.append({
                "Model": name,
                "Total": total,
                "Errors": wrong,
                "Error Rate": wrong / total,
                "Accuracy": 1 - wrong / total,
            })

        df = pd.DataFrame(rows).sort_values("Error Rate")

        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            df.set_index("Model")["Error Rate"].plot(kind="bar", ax=ax, color="coral")
            ax.set_title("Error Rate Comparison Across Models")
            ax.set_ylabel("Error Rate")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                path = f"{save_dir}/error_rate_comparison.png"
                plt.savefig(path, dpi=150)
                logger.info("Error comparison saved to %s", path)
            plt.show()

        return df
