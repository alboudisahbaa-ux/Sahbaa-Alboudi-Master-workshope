"""Model evaluation — advanced metrics, ROC/PR curves, reports (Phase 4)."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute and store evaluation metrics for one or more models."""

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("evaluation", {})
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        best_params: dict | None = None,
    ) -> Dict[str, Any]:
        """Compute metrics and store them internally.

        Parameters
        ----------
        y_proba : optional probability array (n_samples,) for binary
                  or (n_samples, n_classes) for multiclass.
        """
        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
            "y_true": y_true,
            "y_pred": y_pred,
        }

        # ROC-AUC / PR-AUC when probabilities are available
        if y_proba is not None:
            metrics["y_proba"] = y_proba
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary: use positive-class probability
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = roc_auc_score(y_true, proba_pos)
                metrics["pr_auc"] = average_precision_score(y_true, proba_pos)
            else:
                # Multiclass: one-vs-rest
                try:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                except ValueError:
                    metrics["roc_auc"] = None

        if best_params:
            metrics["best_params"] = best_params

        self.results[model_name] = metrics

        roc_str = f"  roc_auc={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else ""
        logger.info(
            "[%s] accuracy=%.4f  f1_weighted=%.4f  mcc=%.4f  kappa=%.4f%s",
            model_name,
            metrics["accuracy"],
            metrics["f1_weighted"],
            metrics["matthews_corrcoef"],
            metrics["cohen_kappa"],
            roc_str,
        )
        return metrics

    # ── Summary ───────────────────────────────────────────────────
    def summary_table(self) -> pd.DataFrame:
        """Return a tidy DataFrame comparing all evaluated models."""
        rows = []
        for name, m in self.results.items():
            row = {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Bal. Acc": m["balanced_accuracy"],
                "Precision": m["precision_weighted"],
                "Recall": m["recall_weighted"],
                "F1": m["f1_weighted"],
                "F1 (macro)": m["f1_macro"],
                "MCC": m["matthews_corrcoef"],
                "Kappa": m["cohen_kappa"],
            }
            if "roc_auc" in m and m["roc_auc"] is not None:
                row["ROC-AUC"] = m["roc_auc"]
            if "pr_auc" in m:
                row["PR-AUC"] = m["pr_auc"]
            rows.append(row)
        return pd.DataFrame(rows).sort_values("F1", ascending=False)

    def print_reports(self) -> None:
        """Print classification reports for every evaluated model."""
        for name, m in self.results.items():
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            print(m["classification_report"])

    # ── Confusion matrices ────────────────────────────────────────
    def plot_confusion_matrices(self, save_dir: str | None = None) -> None:
        """Plot confusion matrix heatmaps for all evaluated models."""
        n = len(self.results)
        if n == 0:
            return
        cols = min(n, 4)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 5 * rows_n))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for ax, (name, m) in zip(axes, self.results.items()):
            cm = m["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        # Hide any empty subplots
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        if save_dir:
            path = f"{save_dir}/confusion_matrices.png"
            plt.savefig(path, dpi=150)
            logger.info("Confusion matrices saved to %s", path)
        plt.show()

    # ── ROC Curves ────────────────────────────────────────────────
    def plot_roc_curves(self, save_dir: str | None = None) -> None:
        """Plot ROC curves for all models that have probability predictions."""
        models_with_proba = {
            n: m for n, m in self.results.items() if "y_proba" in m
        }
        if not models_with_proba:
            logger.info("No probability predictions — skipping ROC curves")
            return

        fig, ax = plt.subplots(figsize=(10, 7))

        for name, m in models_with_proba.items():
            y_true = m["y_true"]
            y_proba = m["y_proba"]
            n_classes = len(np.unique(y_true))

            if n_classes == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                fpr, tpr, _ = roc_curve(y_true, proba_pos)
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2,
                        label=f"{name} (AUC = {roc_auc_val:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            path = f"{save_dir}/roc_curves.png"
            plt.savefig(path, dpi=150)
            logger.info("ROC curves saved to %s", path)
        plt.show()

    # ── Precision-Recall Curves ───────────────────────────────────
    def plot_pr_curves(self, save_dir: str | None = None) -> None:
        """Plot Precision-Recall curves for models with probability predictions."""
        models_with_proba = {
            n: m for n, m in self.results.items() if "y_proba" in m
        }
        if not models_with_proba:
            logger.info("No probability predictions — skipping PR curves")
            return

        fig, ax = plt.subplots(figsize=(10, 7))

        for name, m in models_with_proba.items():
            y_true = m["y_true"]
            y_proba = m["y_proba"]
            n_classes = len(np.unique(y_true))

            if n_classes == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, proba_pos)
                ap = average_precision_score(y_true, proba_pos)
                ax.plot(recall_vals, precision_vals, lw=2,
                        label=f"{name} (AP = {ap:.4f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            path = f"{save_dir}/pr_curves.png"
            plt.savefig(path, dpi=150)
            logger.info("PR curves saved to %s", path)
        plt.show()

    # ── Comparison bar chart ──────────────────────────────────────
    def plot_comparison_bar(self, save_dir: str | None = None) -> None:
        """Bar chart comparing key metrics across models."""
        df = self.summary_table()
        if df.empty:
            return

        plot_cols = ["Accuracy", "Precision", "Recall", "F1", "MCC", "Kappa"]
        available = [c for c in plot_cols if c in df.columns]

        df.set_index("Model")[available].plot(kind="bar", figsize=(12, 5))
        plt.title("Model Performance Comparison")
        plt.ylim(-0.1, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_dir:
            path = f"{save_dir}/model_comparison.png"
            plt.savefig(path, dpi=150)
            logger.info("Comparison chart saved to %s", path)
        plt.show()
