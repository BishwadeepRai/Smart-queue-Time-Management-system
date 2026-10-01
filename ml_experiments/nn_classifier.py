"""Reusable feed-forward neural network for call-centre service-standard classification."""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_classifier(
    n_features: int,
    hidden_layers: list[int],
    learning_rate: float,
    dropout_rate: float = 0.0,
    seed: int = SEED,
) -> keras.Model:
    """Binary classifier: ReLU hidden layers, sigmoid output, binary cross-entropy."""
    keras.backend.clear_session()
    set_seeds(seed)

    model = keras.Sequential(name="CallCentreNN")
    model.add(keras.Input(shape=(n_features,), name="features"))
    for i, units in enumerate(hidden_layers, start=1):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i}"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, seed=seed, name=f"dropout_{i}"))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def predict_labels(model: keras.Model, X, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict(X, verbose=0).ravel()
    pred = (proba >= threshold).astype(int)
    return pred, proba


def classification_metrics(y_true, y_pred) -> dict:
    """Metrics for both classes plus macro/weighted averages."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=["False (not met)", "True (met)"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_false": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_false": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_false": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_true": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_true": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_true": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
    }
