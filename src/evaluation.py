"""
Evaluation helpers shared across model runs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class TimingResult:
    fit_s: float
    predict_s: float


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute core regression metrics.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def timed_fit_predict(model, X_train, y_train, X_test) -> TimingResult:
    """
    Fit a model and time training + prediction separately.
    """
    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit

    start_pred = time.perf_counter()
    _ = model.predict(X_test)
    pred_time = time.perf_counter() - start_pred
    return TimingResult(fit_s=fit_time, predict_s=pred_time)


def summarise_results(rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """
    Convert collected metric rows into a clean dataframe.
    """
    df = pd.DataFrame(rows)
    metric_cols = ["MAE", "RMSE", "R2", "Fit_s", "Predict_s"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    total_col = "Total_s"
    df[total_col] = df["Fit_s"] + df["Predict_s"]
    return df.sort_values("R2", ascending=False).reset_index(drop=True)

