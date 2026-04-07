"""Data preprocessing — encoding, cleaning, splitting, scaling, balancing.

Key fix vs. original: all imputation / scaling is fit ONLY on the
training set, then applied to test, eliminating data leakage.

Phase 2 additions:
  - KNN imputation option
  - IQR-based outlier handling
  - SMOTE oversampling (applied to training set only)
"""

import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


@dataclass
class SplitData:
    """Container for train / test arrays and metadata."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list = field(default_factory=list)
    label_encoder: LabelEncoder | None = None
    scaler: object = None


class DataPreprocessor:
    """Pipeline: encode → drop → clean → split → impute → outlier → scale → balance."""

    def __init__(self, cfg: dict):
        self.target_col = cfg["data"]["target_column"]
        self.drop_cols = cfg["data"].get("drop_columns", [])
        self.test_size = cfg["data"].get("test_size", 0.2)
        self.stratify = cfg["data"].get("stratify", True)
        self.seed = cfg["project"].get("seed", 42)

        prep = cfg.get("preprocessing", {})
        self.handle_inf = prep.get("handle_inf", True)
        self.handle_nan = prep.get("handle_nan", True)
        self.nan_strategy = prep.get("nan_strategy", "mean")
        self.knn_neighbors = prep.get("knn_neighbors", 5)
        self.clip_lower = float(prep.get("clip_lower", -1e6))
        self.clip_upper = float(prep.get("clip_upper", 1e6))
        self.scaler_name = prep.get("scaler", "standard")
        self.outlier_method = prep.get("outlier_method", "none")
        self.iqr_factor = float(prep.get("iqr_factor", 1.5))

        bal = cfg.get("balancing", {})
        self.balance_enabled = bal.get("enabled", False)
        self.balance_method = bal.get("method", "none")
        self.smote_k = bal.get("smote_k_neighbors", 5)

    # ------------------------------------------------------------------
    def _handle_outliers_iqr(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cap outliers using IQR bounds computed on the training set."""
        Q1 = X_train.quantile(0.25)
        Q3 = X_train.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - self.iqr_factor * IQR
        upper = Q3 + self.iqr_factor * IQR

        X_train = X_train.clip(lower=lower, upper=upper, axis=1)
        X_test = X_test.clip(lower=lower, upper=upper, axis=1)
        logger.info("IQR outlier capping applied (factor=%.1f)", self.iqr_factor)
        return X_train, X_test

    def _apply_smote(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to the training set only."""
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(k_neighbors=self.smote_k, random_state=self.seed)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE: %d → %d samples (k=%d)",
            len(y_train), len(y_res), self.smote_k,
        )
        return X_res, y_res

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> SplitData:
        """Execute full preprocessing pipeline and return SplitData."""
        df = df.copy()

        # 1. Drop explicitly configured columns
        df = df.drop(columns=self.drop_cols, errors="ignore")
        logger.info("Dropped configured columns: %s", self.drop_cols)

        # 2. Encode target
        label_enc = LabelEncoder()
        df[self.target_col] = label_enc.fit_transform(df[self.target_col].astype(str))
        logger.info("Encoded target classes: %s", list(label_enc.classes_))

        # 3. Drop remaining string/object columns (identifiers, IPs, etc.)
        remaining_str = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if remaining_str:
            df = df.drop(columns=remaining_str)
            logger.info("Dropped remaining string columns: %s", remaining_str)

        # 4. Drop zero-variance (constant) columns
        nunique = df.drop(columns=[self.target_col]).nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            df = df.drop(columns=constant_cols)
            logger.info("Dropped %d zero-variance columns: %s", len(constant_cols), constant_cols)

        # 5. Separate X / y
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].values
        feature_names = list(X.columns)

        # 6. Split BEFORE any imputation / scaling (prevents leakage)
        stratify_arr = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=stratify_arr,
        )
        logger.info(
            "Split → train %d | test %d (stratify=%s)",
            len(X_train), len(X_test), self.stratify,
        )

        # 7. Handle inf → NaN
        if self.handle_inf:
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)

        # 8. Handle NaN (fit on train, transform both)
        if self.handle_nan:
            if self.nan_strategy == "knn":
                imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                X_train = pd.DataFrame(
                    imputer.fit_transform(X_train),
                    columns=feature_names, index=X_train.index,
                )
                X_test = pd.DataFrame(
                    imputer.transform(X_test),
                    columns=feature_names, index=X_test.index,
                )
                logger.info("KNN imputation (k=%d)", self.knn_neighbors)
            else:
                if self.nan_strategy == "median":
                    fill_values = X_train.median()
                else:  # mean (default)
                    fill_values = X_train.mean()
                X_train = X_train.fillna(fill_values)
                X_test = X_test.fillna(fill_values)

        # 9. Outlier handling
        if self.outlier_method == "iqr":
            X_train, X_test = self._handle_outliers_iqr(X_train, X_test)
        else:
            # Simple clip as fallback
            X_train = X_train.clip(lower=self.clip_lower, upper=self.clip_upper)
            X_test = X_test.clip(lower=self.clip_lower, upper=self.clip_upper)

        # 10. Scale (fit on train only)
        scaler_cls = SCALERS.get(self.scaler_name, StandardScaler)
        scaler = scaler_cls()
        X_train_arr = scaler.fit_transform(X_train)
        X_test_arr = scaler.transform(X_test)
        logger.info("Scaling: %s", self.scaler_name)

        # 11. SMOTE balancing (on training set only, after scaling)
        if self.balance_enabled and self.balance_method == "smote":
            X_train_arr, y_train = self._apply_smote(X_train_arr, y_train)

        return SplitData(
            X_train=X_train_arr,
            X_test=X_test_arr,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            label_encoder=label_enc,
            scaler=scaler,
        )
