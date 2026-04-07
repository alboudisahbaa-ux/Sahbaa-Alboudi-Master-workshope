"""Model explainability — SHAP and LIME analysis (Phase 4)."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Generate SHAP and LIME explanations for trained models."""

    def __init__(self, cfg: dict):
        exp_cfg = cfg.get("explainability", {})
        self.shap_enabled = exp_cfg.get("shap_enabled", True)
        self.lime_enabled = exp_cfg.get("lime_enabled", True)
        self.shap_samples = exp_cfg.get("shap_samples", 500)
        self.lime_samples = exp_cfg.get("lime_samples", 5)
        self.feature_names: list = []
        self.shap_values_store: Dict[str, Any] = {}

    # ── SHAP ──────────────────────────────────────────────────────
    def shap_explain(
        self,
        model_name: str,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: list,
        save_dir: str | None = None,
    ) -> None:
        """Compute SHAP values and generate summary + dependence plots."""
        if not self.shap_enabled:
            return

        import shap

        self.feature_names = feature_names
        n_samples = min(self.shap_samples, X_test.shape[0])
        X_sample = X_test[:n_samples]

        logger.info("[%s] Computing SHAP values on %d samples …",
                     model_name, n_samples)

        # Choose explainer based on model type
        try:
            # Tree-based models (RF, XGB, LGBM, CatBoost, GB)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            # Fallback: KernelExplainer for any model
            bg = shap.sample(X_train, min(100, X_train.shape[0]))
            explainer = shap.KernelExplainer(model.predict, bg)
            shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values may be a list of 2 arrays
        if isinstance(shap_values, list):
            sv = shap_values[1]  # take positive-class SHAP values
        else:
            sv = shap_values

        self.shap_values_store[model_name] = {
            "shap_values": sv,
            "X_sample": X_sample,
        }

        # ── SHAP Summary plot (beeswarm) ─────────────────────────
        plt.figure()
        shap.summary_plot(
            sv,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            path = f"{save_dir}/shap_summary_{model_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("SHAP summary saved to %s", path)
        plt.show()

        # ── SHAP Bar plot (global feature importance) ─────────────
        plt.figure()
        shap.summary_plot(
            sv,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        if save_dir:
            path = f"{save_dir}/shap_bar_{model_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("SHAP bar plot saved to %s", path)
        plt.show()

        logger.info("[%s] SHAP analysis complete", model_name)

    def shap_dependence(
        self,
        model_name: str,
        feature_idx: int | str,
        save_dir: str | None = None,
    ) -> None:
        """Generate a SHAP dependence plot for a specific feature."""
        import shap

        if model_name not in self.shap_values_store:
            logger.warning("No SHAP values for %s — run shap_explain first", model_name)
            return

        store = self.shap_values_store[model_name]
        plt.figure()
        shap.dependence_plot(
            feature_idx,
            store["shap_values"],
            store["X_sample"],
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        if save_dir:
            feat_label = feature_idx if isinstance(feature_idx, str) else self.feature_names[feature_idx]
            path = f"{save_dir}/shap_dep_{model_name}_{feat_label}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── LIME ──────────────────────────────────────────────────────
    def lime_explain(
        self,
        model_name: str,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
        class_names: list | None = None,
        save_dir: str | None = None,
    ) -> None:
        """Generate LIME explanations for a few test samples."""
        if not self.lime_enabled:
            return

        import lime
        import lime.lime_tabular

        logger.info("[%s] Generating LIME explanations for %d samples …",
                     model_name, self.lime_samples)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names or ["BENIGN", "DDoS"],
            mode="classification",
        )

        # Determine predict function
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            # Wrap models without predict_proba
            def predict_fn(X):
                preds = model.predict(X)
                n_cls = len(class_names) if class_names else 2
                proba = np.zeros((len(preds), n_cls))
                for i, p in enumerate(preds):
                    proba[i, int(p)] = 1.0
                return proba

        for i in range(min(self.lime_samples, X_test.shape[0])):
            exp = explainer.explain_instance(
                X_test[i],
                predict_fn,
                num_features=10,
            )

            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME — {model_name} | Sample {i} (true={y_test[i]})")
            fig.tight_layout()

            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                path = f"{save_dir}/lime_{model_name}_sample{i}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                logger.info("LIME explanation saved to %s", path)
            plt.show()

        logger.info("[%s] LIME analysis complete", model_name)

    # ── Feature importance ranking from SHAP ──────────────────────
    def shap_importance_table(self, model_name: str) -> pd.DataFrame | None:
        """Return a DataFrame of mean |SHAP| values per feature."""
        if model_name not in self.shap_values_store:
            return None

        sv = self.shap_values_store[model_name]["shap_values"]
        mean_abs = np.abs(sv).mean(axis=0)

        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean |SHAP|": mean_abs,
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        return df
