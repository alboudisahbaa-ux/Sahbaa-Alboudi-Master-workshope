"""Model factory — builds model instances from the YAML config (Phase 3)."""

import logging
from typing import List

from .base import BaseModel
from .ml_models import SklearnModel
from .dl_model import DeepLearningModel, LSTMModel, CNN1DModel
from .ensemble import EnsembleModel

logger = logging.getLogger(__name__)

_DL_REGISTRY = {
    "mlp": DeepLearningModel,
    "lstm": LSTMModel,
    "cnn1d": CNN1DModel,
}


class ModelFactory:
    """Create all enabled models declared in the configuration."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def create_ml_models(self) -> List[BaseModel]:
        """Instantiate and return all enabled sklearn/xgb models."""
        models: List[BaseModel] = []
        tuning = self.cfg.get("tuning", {})

        for key, spec in self.cfg.get("ml_models", {}).items():
            if not spec.get("enabled", True):
                logger.info("Skipping disabled model: %s", key)
                continue
            m = SklearnModel(
                model_key=key,
                param_grid=spec.get("params", {}),
                tuning_cfg=tuning,
            )
            models.append(m)
            logger.info("Registered ML model: %s", key)

        return models

    def create_dl_models(
        self, n_features: int, n_classes: int
    ) -> List[BaseModel]:
        """Instantiate and return all enabled DL models (MLP, LSTM, CNN1D)."""
        models: List[BaseModel] = []

        # Original MLP
        dl_cfg = self.cfg.get("deep_learning", {})
        if dl_cfg.get("enabled", True):
            m = DeepLearningModel(self.cfg, n_features=n_features, n_classes=n_classes)
            m.build()
            models.append(m)
            logger.info("Registered DL model: MLP")

        # LSTM
        lstm_cfg = self.cfg.get("deep_learning_lstm", {})
        if lstm_cfg.get("enabled", False):
            m = LSTMModel(self.cfg, n_features=n_features, n_classes=n_classes)
            m.build()
            models.append(m)
            logger.info("Registered DL model: LSTM")

        # CNN1D
        cnn_cfg = self.cfg.get("deep_learning_cnn", {})
        if cnn_cfg.get("enabled", False):
            m = CNN1DModel(self.cfg, n_features=n_features, n_classes=n_classes)
            m.build()
            models.append(m)
            logger.info("Registered DL model: CNN1D")

        return models

    # Backward compat — returns first DL model or None
    def create_dl_model(self, n_features: int, n_classes: int) -> BaseModel | None:
        """Instantiate the primary DL model (MLP) if enabled."""
        dl_cfg = self.cfg.get("deep_learning", {})
        if not dl_cfg.get("enabled", True):
            logger.info("Deep learning model disabled — skipping")
            return None
        model = DeepLearningModel(self.cfg, n_features=n_features, n_classes=n_classes)
        model.build()
        logger.info("Registered DL model")
        return model

    def create_ensemble(
        self, trained_models: List[BaseModel]
    ) -> List[EnsembleModel]:
        """Create ensemble models from already-trained base models."""
        ens_cfg = self.cfg.get("ensemble", {})
        if not ens_cfg.get("enabled", False):
            return []

        ensembles: List[EnsembleModel] = []
        for etype in ens_cfg.get("types", ["voting"]):
            e = EnsembleModel(
                ensemble_type=etype,
                base_models=trained_models,
                cfg=self.cfg,
            )
            ensembles.append(e)
            logger.info("Registered ensemble: %s", e.name)

        return ensembles
