"""
Model definitions and training orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_model
from .evaluation import timed_fit_predict
from .evaluation import summarise_results
from .features import FeatureData

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """
    Declarative model configuration.
    """

    name: str
    feature_set: str  # "simple" or "extended"
    build_model: Callable[[], object]


def default_model_specs() -> List[ModelSpec]:
    """
    Provide a small suite of baseline models.
    """
    return [
        ModelSpec(
            name="Linear (simple)",
            feature_set="simple",
            build_model=lambda: LinearRegression(n_jobs=-1),
        ),
        ModelSpec(
            name="Linear (extended)",
            feature_set="extended",
            build_model=lambda: LinearRegression(n_jobs=-1),
        ),
        ModelSpec(
            name="Random Forest",
            feature_set="extended",
            build_model=lambda: RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            name="Neural Network",
            feature_set="extended",
            build_model=lambda: Pipeline(
                [
                    ("scale", StandardScaler(with_mean=False)),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            random_state=42,
                            max_iter=50,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _pick_features(feature_data: FeatureData, feature_set: str):
    if feature_set == "simple":
        return feature_data.X_simple
    if feature_set == "extended":
        return feature_data.X_extended
    raise ValueError(f"Unknown feature_set: {feature_set}")


def train_val_split(feature_data: FeatureData, test_size: float = 0.2, random_state: int = 42):
    """
    Create matching train/test splits for both simple and extended feature sets.
    """
    X_train_ext, X_test_ext, y_train, y_test = train_test_split(
        feature_data.X_extended,
        feature_data.y,
        test_size=test_size,
        random_state=random_state,
    )
    # Align simple features with the same row indices to ensure consistency.
    X_train_simple = feature_data.X_simple.loc[X_train_ext.index]
    X_test_simple = feature_data.X_simple.loc[X_test_ext.index]
    return {
        "simple": (X_train_simple, X_test_simple, y_train, y_test),
        "extended": (X_train_ext, X_test_ext, y_train, y_test),
    }


def train_and_evaluate(specs: Iterable[ModelSpec], feature_data: FeatureData) -> pd.DataFrame:
    """
    Fit each model specification and summarise key metrics.
    """
    splits = train_val_split(feature_data)
    rows: List[Dict[str, float]] = []

    for spec in specs:
        LOGGER.info("Training model: %s", spec.name)
        X_train, X_test, y_train, y_test = splits[spec.feature_set]
        model = spec.build_model()

        timing = timed_fit_predict(model, X_train, y_train, X_test)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        rows.append(
            {
                "Model": spec.name,
                "Feature set": spec.feature_set,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Fit_s": timing.fit_s,
                "Predict_s": timing.predict_s,
            }
        )

    return summarise_results(rows)

