"""Keras deep-learning models wrapped in the BaseModel interface (Phase 3).

Architectures: MLP, LSTM, CNN1D — all configurable via YAML.
"""

import logging
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    LSTM, Reshape, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)

from .base import BaseModel

logger = logging.getLogger(__name__)

_OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
}


class DeepLearningModel(BaseModel):
    """Configurable MLP for binary/multi-class DDoS classification."""

    name = "DL_MLP"

    def __init__(self, cfg: dict, n_features: int, n_classes: int):
        arch = cfg["deep_learning"]["architecture"]
        train = cfg["deep_learning"]["training"]

        self.units = arch.get("units", [64, 32])
        self.activation = arch.get("activation", "relu")
        self.dropout = arch.get("dropout_rate", 0.3)
        self.use_bn = arch.get("use_batch_norm", True)

        self.epochs = train.get("epochs", 100)
        self.batch_size = train.get("batch_size", 32)
        self.lr = train.get("learning_rate", 0.001)
        self.optimizer_name = train.get("optimizer", "adam")
        self.es_patience = train.get("early_stopping_patience", 10)
        self.rlr_patience = train.get("reduce_lr_patience", 5)
        self.rlr_factor = train.get("reduce_lr_factor", 0.2)
        self.min_lr = train.get("min_lr", 1e-5)

        self.n_features = n_features
        self.n_classes = n_classes
        self.model: Sequential | None = None
        self.history = None

    def build(self, **kwargs) -> None:
        layers = []
        for i, u in enumerate(self.units):
            if i == 0:
                layers.append(Dense(u, activation=self.activation,
                                    input_shape=(self.n_features,)))
            else:
                layers.append(Dense(u, activation=self.activation))
            if self.use_bn:
                layers.append(BatchNormalization())
            layers.append(Dropout(self.dropout))

        layers.append(Dense(self.n_classes, activation="softmax"))

        self.model = Sequential(layers)
        opt_cls = _OPTIMIZERS.get(self.optimizer_name, Adam)
        self.model.compile(
            optimizer=opt_cls(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("MLP model built: %s", [l.name for l in self.model.layers])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if self.model is None:
            self.build()

        y_cat = tf.keras.utils.to_categorical(y_train, num_classes=self.n_classes)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.es_patience,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=self.rlr_factor,
                              patience=self.rlr_patience, min_lr=self.min_lr),
        ]

        validation_split = kwargs.get("validation_split", 0.2)

        logger.info("Training MLP for up to %d epochs …", self.epochs)
        self.history = self.model.fit(
            X_train, y_cat,
            validation_split=validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        stopped = len(self.history.history["loss"])
        logger.info("MLP training stopped at epoch %d", stopped)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

    def get_params(self) -> Dict[str, Any]:
        return {
            "arch": "MLP",
            "units": self.units,
            "dropout": self.dropout,
            "lr": self.lr,
            "optimizer": self.optimizer_name,
            "epochs_ran": len(self.history.history["loss"]) if self.history else 0,
        }


class LSTMModel(BaseModel):
    """LSTM-based model treating each feature as a time step (Phase 3)."""

    name = "DL_LSTM"

    def __init__(self, cfg: dict, n_features: int, n_classes: int):
        dl = cfg.get("deep_learning_lstm", cfg.get("deep_learning", {}))
        arch = dl.get("architecture", {})
        train = dl.get("training", cfg["deep_learning"]["training"])

        self.lstm_units = arch.get("lstm_units", [64, 32])
        self.dense_units = arch.get("dense_units", [32])
        self.dropout = arch.get("dropout_rate", 0.3)
        self.activation = arch.get("activation", "relu")

        self.epochs = train.get("epochs", 100)
        self.batch_size = train.get("batch_size", 32)
        self.lr = train.get("learning_rate", 0.001)
        self.optimizer_name = train.get("optimizer", "adam")
        self.es_patience = train.get("early_stopping_patience", 10)
        self.rlr_patience = train.get("reduce_lr_patience", 5)
        self.rlr_factor = train.get("reduce_lr_factor", 0.2)
        self.min_lr = train.get("min_lr", 1e-5)

        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None

    def build(self, **kwargs) -> None:
        inp = Input(shape=(self.n_features, 1))
        x = inp

        for i, units in enumerate(self.lstm_units):
            return_seq = i < len(self.lstm_units) - 1
            x = LSTM(units, return_sequences=return_seq)(x)
            x = Dropout(self.dropout)(x)

        for units in self.dense_units:
            x = Dense(units, activation=self.activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout)(x)

        out = Dense(self.n_classes, activation="softmax")(x)

        self.model = Model(inputs=inp, outputs=out)
        opt_cls = _OPTIMIZERS.get(self.optimizer_name, Adam)
        self.model.compile(
            optimizer=opt_cls(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("LSTM model built: %d LSTM layers, %d dense layers",
                     len(self.lstm_units), len(self.dense_units))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if self.model is None:
            self.build()

        # Reshape for LSTM: (samples, features, 1)
        X_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_cat = tf.keras.utils.to_categorical(y_train, num_classes=self.n_classes)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.es_patience,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=self.rlr_factor,
                              patience=self.rlr_patience, min_lr=self.min_lr),
        ]

        validation_split = kwargs.get("validation_split", 0.2)

        logger.info("Training LSTM for up to %d epochs …", self.epochs)
        self.history = self.model.fit(
            X_3d, y_cat,
            validation_split=validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        stopped = len(self.history.history["loss"])
        logger.info("LSTM training stopped at epoch %d", stopped)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_3d = X.reshape(X.shape[0], X.shape[1], 1)
        probs = self.model.predict(X_3d, verbose=0)
        return np.argmax(probs, axis=1)

    def get_params(self) -> Dict[str, Any]:
        return {
            "arch": "LSTM",
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs_ran": len(self.history.history["loss"]) if self.history else 0,
        }


class CNN1DModel(BaseModel):
    """1D-CNN model treating features as a 1D signal (Phase 3)."""

    name = "DL_CNN1D"

    def __init__(self, cfg: dict, n_features: int, n_classes: int):
        dl = cfg.get("deep_learning_cnn", cfg.get("deep_learning", {}))
        arch = dl.get("architecture", {})
        train = dl.get("training", cfg["deep_learning"]["training"])

        self.filters = arch.get("filters", [64, 128])
        self.kernel_size = arch.get("kernel_size", 3)
        self.pool_size = arch.get("pool_size", 2)
        self.dense_units = arch.get("dense_units", [64])
        self.dropout = arch.get("dropout_rate", 0.3)
        self.activation = arch.get("activation", "relu")

        self.epochs = train.get("epochs", 100)
        self.batch_size = train.get("batch_size", 32)
        self.lr = train.get("learning_rate", 0.001)
        self.optimizer_name = train.get("optimizer", "adam")
        self.es_patience = train.get("early_stopping_patience", 10)
        self.rlr_patience = train.get("reduce_lr_patience", 5)
        self.rlr_factor = train.get("reduce_lr_factor", 0.2)
        self.min_lr = train.get("min_lr", 1e-5)

        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None

    def build(self, **kwargs) -> None:
        inp = Input(shape=(self.n_features, 1))
        x = inp

        for n_filters in self.filters:
            x = Conv1D(n_filters, self.kernel_size, activation=self.activation,
                       padding="same")(x)
            x = BatchNormalization()(x)
            # Only pool if there are enough time steps
            if x.shape[1] is not None and x.shape[1] >= self.pool_size * 2:
                x = MaxPooling1D(pool_size=self.pool_size)(x)
            x = Dropout(self.dropout)(x)

        x = GlobalAveragePooling1D()(x)

        for units in self.dense_units:
            x = Dense(units, activation=self.activation)(x)
            x = Dropout(self.dropout)(x)

        out = Dense(self.n_classes, activation="softmax")(x)

        self.model = Model(inputs=inp, outputs=out)
        opt_cls = _OPTIMIZERS.get(self.optimizer_name, Adam)
        self.model.compile(
            optimizer=opt_cls(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("CNN1D model built: filters=%s, kernel=%d",
                     self.filters, self.kernel_size)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        if self.model is None:
            self.build()

        # Reshape for CNN: (samples, features, 1)
        X_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_cat = tf.keras.utils.to_categorical(y_train, num_classes=self.n_classes)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.es_patience,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=self.rlr_factor,
                              patience=self.rlr_patience, min_lr=self.min_lr),
        ]

        validation_split = kwargs.get("validation_split", 0.2)

        logger.info("Training CNN1D for up to %d epochs …", self.epochs)
        self.history = self.model.fit(
            X_3d, y_cat,
            validation_split=validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        stopped = len(self.history.history["loss"])
        logger.info("CNN1D training stopped at epoch %d", stopped)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_3d = X.reshape(X.shape[0], X.shape[1], 1)
        probs = self.model.predict(X_3d, verbose=0)
        return np.argmax(probs, axis=1)

    def get_params(self) -> Dict[str, Any]:
        return {
            "arch": "CNN1D",
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs_ran": len(self.history.history["loss"]) if self.history else 0,
        }
