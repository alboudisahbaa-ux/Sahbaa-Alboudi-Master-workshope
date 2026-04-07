"""Feature engineering — selection, importance, and analysis.

Fixes the original bug where SelectKBest features were incorrectly
paired with unrelated Random Forest importance values.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from .preprocessor import SplitData

logger = logging.getLogger(__name__)

SCORE_FUNCS = {
    "f_classif": f_classif,
    "mutual_info_classif": mutual_info_classif,
}


class FeatureEngineer:
    """Analyse and rank features using statistical tests and tree importance."""

    def __init__(self, cfg: dict):
        feat = cfg.get("features", {})
        self.k = feat.get("select_k_best", 10)
        self.score_func_name = feat.get("score_func", "f_classif")
        self.use_rf = feat.get("use_rf_importance", True)
        self.rf_n_est = feat.get("rf_n_estimators", 100)
        self.seed = cfg["project"].get("seed", 42)

    def analyse(self, split: SplitData) -> pd.DataFrame:
        """Return a DataFrame of feature importances.

        Columns: feature, kbest_score, rf_importance (if enabled).
        """
        feature_names = np.array(split.feature_names)

        # --- SelectKBest --------------------------------------------------
        k = min(self.k, split.X_train.shape[1])
        score_fn = SCORE_FUNCS.get(self.score_func_name, f_classif)
        selector = SelectKBest(score_func=score_fn, k=k)
        selector.fit(split.X_train, split.y_train)

        mask = selector.get_support()
        selected = feature_names[mask]
        scores = selector.scores_[mask]
        logger.info("SelectKBest top-%d features: %s", k, list(selected))

        result = pd.DataFrame({"feature": selected, "kbest_score": scores})

        # --- Random Forest importance (full feature set) ------------------
        if self.use_rf:
            rf = RandomForestClassifier(
                n_estimators=self.rf_n_est, random_state=self.seed, n_jobs=-1
            )
            rf.fit(split.X_train, split.y_train)

            # Map importance to the CORRECT feature names
            all_imp = pd.DataFrame({
                "feature": feature_names,
                "rf_importance": rf.feature_importances_,
            })

            # Merge on feature name so scores are correctly aligned
            result = result.merge(all_imp, on="feature", how="left")
            result = result.sort_values("rf_importance", ascending=False)
            logger.info("RF importance computed for %d features", len(feature_names))

        return result

    def plot(self, importance_df: pd.DataFrame, save_path: str | None = None):
        """Bar chart of feature importances."""
        col = "rf_importance" if "rf_importance" in importance_df.columns else "kbest_score"
        plt.figure(figsize=(10, 6))
        sns.barplot(x=col, y="feature", data=importance_df.head(self.k))
        plt.title(f"Top-{self.k} Feature Importances ({col})")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info("Feature importance plot saved to %s", save_path)
        plt.show()
