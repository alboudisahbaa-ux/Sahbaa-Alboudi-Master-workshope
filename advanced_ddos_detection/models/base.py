"""Abstract base class for all models."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseModel(ABC):
    """Every model (ML or DL) inherits this interface."""

    name: str = "BaseModel"

    @abstractmethod
    def build(self, **kwargs) -> None:
        """Construct / initialise the model."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""

    def get_params(self) -> Dict[str, Any]:
        """Return current hyper-parameters (for logging)."""
        return {}
