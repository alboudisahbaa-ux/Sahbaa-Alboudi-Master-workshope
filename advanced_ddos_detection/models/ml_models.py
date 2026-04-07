"""Scikit-learn / XGBoost / LightGBM / CatBoost classifiers (Phase 3).

Supports GridSearchCV, RandomizedSearchCV, and Optuna-based tuning.
"""

import logging
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .base import BaseModel

logger = logging.getLogger(__name__)

# Registry of supported sklearn-style classifiers
_SKLEARN_MODELS = {
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "catboost": CatBoostClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
}

# Default constructor kwargs for models that need them
_MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "xgboost": {"eval_metric": "logloss", "verbosity": 0},
    "lightgbm": {"verbose": -1},
    "catboost": {"verbose": 0},
    "logistic_regression": {"max_iter": 1000},
}


class SklearnModel(BaseModel):
    """Wraps any sklearn-API classifier with GridSearch / RandomSearch / Optuna tuning."""

    def __init__(self, model_key: str, param_grid: dict, tuning_cfg: dict):
        self.name = model_key
        self.model_key = model_key
        self.param_grid = param_grid
        self.cv = tuning_cfg.get("cv_folds", 5)
        self.scoring = tuning_cfg.get("scoring", "f1_weighted")
        self.n_jobs = tuning_cfg.get("n_jobs", -1)
        self.search_method = tuning_cfg.get("search_method", "grid")
        self.n_iter = tuning_cfg.get("n_iter", 50)
        self.seed = tuning_cfg.get("seed", 42)
        self.model = None
        self.best_params: Dict[str, Any] = {}

    def build(self, **kwargs) -> None:
        cls = _SKLEARN_MODELS.get(self.model_key)
        if cls is None:
            raise ValueError(
                f"Unknown model key '{self.model_key}'. "
                f"Available: {list(_SKLEARN_MODELS)}"
            )
        defaults = _MODEL_DEFAULTS.get(self.model_key, {})
        self.model = cls(**defaults)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if self.model is None:
            self.build()

        if self.search_method == "optuna":
            self._fit_optuna(X_train, y_train)
        elif self.search_method == "random":
            self._fit_random_search(X_train, y_train)
        else:
            self._fit_grid_search(X_train, y_train)

    def _fit_grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info("[%s] Starting GridSearchCV (cv=%d, scoring=%s)",
                     self.name, self.cv, self.scoring)
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        logger.info("[%s] Best params: %s", self.name, self.best_params)

    def _fit_random_search(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info("[%s] Starting RandomizedSearchCV (n_iter=%d, cv=%d)",
                     self.name, self.n_iter, self.cv)
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        logger.info("[%s] Best params: %s", self.name, self.best_params)

    def _fit_optuna(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Bayesian hyperparameter optimisation via Optuna."""
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cls = _SKLEARN_MODELS[self.model_key]
        defaults = _MODEL_DEFAULTS.get(self.model_key, {})

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for key, vals in self.param_grid.items():
                if isinstance(vals, list):
                    if all(isinstance(v, int) for v in vals):
                        params[key] = trial.suggest_int(key, min(vals), max(vals))
                    elif all(isinstance(v, float) for v in vals):
                        params[key] = trial.suggest_float(key, min(vals), max(vals))
                    else:
                        params[key] = trial.suggest_categorical(key, vals)
                else:
                    params[key] = vals

            merged = {**defaults, **params}
            estimator = cls(**merged)
            scores = cross_val_score(
                estimator, X_train, y_train,
                cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_iter, show_progress_bar=False)

        best = {**defaults, **study.best_params}
        self.model = cls(**best)
        self.model.fit(X_train, y_train)
        self.best_params = study.best_params
        logger.info("[%s] Optuna best params: %s (score=%.4f)",
                     self.name, study.best_params, study.best_value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self.best_params
