"""Ensemble methods — Voting and Stacking classifiers (Phase 3)."""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from .base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Builds a Voting or Stacking ensemble from trained base models."""

    def __init__(
        self,
        ensemble_type: str,
        base_models: List[BaseModel],
        cfg: dict,
    ):
        """
        Parameters
        ----------
        ensemble_type : "voting" or "stacking"
        base_models   : already-trained BaseModel instances
        cfg           : full pipeline config dict
        """
        self.ensemble_type = ensemble_type
        self.base_models = base_models
        self.name = f"Ensemble_{ensemble_type}"

        ens_cfg = cfg.get("ensemble", {})
        self.voting_type = ens_cfg.get("voting_type", "hard")
        self.stack_final = ens_cfg.get("stacking_final_estimator", "logistic_regression")
        self.cv = ens_cfg.get("cv", 5)
        self.n_jobs = cfg.get("tuning", {}).get("n_jobs", -1)

        self.model = None
        self.best_params: Dict[str, Any] = {}

    def _make_estimators(self):
        """Build a list of (name, estimator) tuples from base models."""
        estimators = []
        for m in self.base_models:
            if m.model is not None:
                estimators.append((m.name, m.model))
        return estimators

    def build(self, **kwargs) -> None:
        estimators = self._make_estimators()
        if len(estimators) < 2:
            raise ValueError(
                f"Need ≥2 base estimators for ensemble, got {len(estimators)}"
            )

        if self.ensemble_type == "stacking":
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=self.cv,
                n_jobs=self.n_jobs,
            )
        else:  # voting (default)
            self.model = VotingClassifier(
                estimators=estimators,
                voting=self.voting_type,
                n_jobs=self.n_jobs,
            )

        logger.info(
            "[%s] Built with %d base models: %s",
            self.name, len(estimators), [e[0] for e in estimators],
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if self.model is None:
            self.build()
        logger.info("[%s] Training ensemble …", self.name)
        self.model.fit(X_train, y_train)
        logger.info("[%s] Training complete", self.name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "ensemble_type": self.ensemble_type,
            "n_base_models": len(self.base_models),
            "base_models": [m.name for m in self.base_models],
        }
