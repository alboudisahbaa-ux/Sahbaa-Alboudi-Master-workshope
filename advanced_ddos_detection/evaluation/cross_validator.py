"""K-Fold Cross-Validation evaluator (Phase 2)."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate

from ..models.base import BaseModel

logger = logging.getLogger(__name__)


class CrossValidator:
    """Run stratified K-fold CV for any BaseModel-compatible estimator."""

    def __init__(self, cfg: dict):
        cv_cfg = cfg.get("cross_validation", {})
        self.n_splits = cv_cfg.get("n_splits", 5)
        self.strategy = cv_cfg.get("strategy", "stratified")
        self.seed = cfg["project"].get("seed", 42)

        tuning = cfg.get("tuning", {})
        self.scoring = tuning.get("scoring", "f1_weighted")
        self.n_jobs = tuning.get("n_jobs", -1)

        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Run K-fold CV and store per-fold scores."""
        if self.strategy == "stratified":
            kf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )
        else:
            kf = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )

        # Build a fresh estimator for CV
        model.build()
        estimator = model.model

        logger.info(
            "[%s] Running %d-fold CV (strategy=%s, scoring=%s)",
            model.name, self.n_splits, self.strategy, self.scoring,
        )

        scoring_map = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
        }

        cv_results = cross_validate(
            estimator, X, y,
            cv=kf,
            scoring=scoring_map,
            n_jobs=self.n_jobs,
            return_train_score=False,
        )

        entry = {}
        for metric_key, scorer_name in scoring_map.items():
            scores = cv_results[f"test_{scorer_name}"]
            entry[f"{metric_key}_mean"] = scores.mean()
            entry[f"{metric_key}_std"] = scores.std()
            entry[f"{metric_key}_folds"] = scores.tolist()

        self.results[model.name] = entry
        logger.info(
            "[%s] CV accuracy=%.4f±%.4f  f1=%.4f±%.4f",
            model.name,
            entry["accuracy_mean"], entry["accuracy_std"],
            entry["f1_weighted_mean"], entry["f1_weighted_std"],
        )
        return entry

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame with mean ± std for each model."""
        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}",
                "Precision": f"{m['precision_weighted_mean']:.4f}±{m['precision_weighted_std']:.4f}",
                "Recall": f"{m['recall_weighted_mean']:.4f}±{m['recall_weighted_std']:.4f}",
                "F1": f"{m['f1_weighted_mean']:.4f}±{m['f1_weighted_std']:.4f}",
            })
        return pd.DataFrame(rows)

    def plot_cv_scores(self, save_dir: str | None = None) -> None:
        """Box plot of per-fold F1 scores across models."""
        if not self.results:
            return

        data = {}
        for name, m in self.results.items():
            data[name] = m["f1_weighted_folds"]

        df = pd.DataFrame(data)
        df.plot(kind="box", figsize=(10, 5))
        plt.title(f"{self.n_splits}-Fold Cross-Validation F1 Scores")
        plt.ylabel("F1 (weighted)")
        plt.tight_layout()
        if save_dir:
            path = f"{save_dir}/cv_boxplot.png"
            plt.savefig(path, dpi=150)
            logger.info("CV boxplot saved to %s", path)
        plt.show()
