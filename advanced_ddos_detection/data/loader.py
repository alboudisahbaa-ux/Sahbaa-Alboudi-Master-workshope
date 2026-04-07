"""Data loading with validation and basic profiling."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and perform initial validation on the CIC-DDoS2019 CSV."""

    def __init__(self, cfg: dict):
        self.file_path = cfg["data"]["file_path"]
        self.target_col = cfg["data"]["target_column"]

    def load(self) -> pd.DataFrame:
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. "
                "Download from Kaggle and place it in the data/ folder."
            )

        logger.info("Loading dataset from %s", path)
        df = pd.read_csv(path)
        logger.info("Loaded %d rows × %d columns", *df.shape)

        # Quick validation
        if self.target_col not in df.columns:
            raise KeyError(
                f"Target column '{self.target_col}' not in dataset. "
                f"Available: {list(df.columns)}"
            )

        # Profile summary
        n_missing = int(df.isnull().sum().sum())
        n_inf = int(np.isinf(df.select_dtypes(include=[np.number]).values).sum())
        label_counts = df[self.target_col].value_counts()
        logger.info("Missing values: %d | Infinite values: %d", n_missing, n_inf)
        logger.info("Label distribution:\n%s", label_counts.to_string())

        return df
