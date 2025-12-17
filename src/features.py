"""
Feature construction helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureData:
    """Container for prepared modelling data."""

    df_model: pd.DataFrame
    X_simple: pd.DataFrame
    X_extended: pd.DataFrame
    y: pd.Series
    feature_cols_simple: Sequence[str]
    feature_cols_ext: Sequence[str]


def trim_outliers(df: pd.DataFrame, target: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Clip extreme target outliers using quantiles to stabilise model training.
    """
    lower = df[target].quantile(lower_q)
    upper = df[target].quantile(upper_q)
    trimmed = df[(df[target] >= lower) & (df[target] <= upper)].copy()
    LOGGER.info("Trimmed %s outliers outside [%s, %s]; %s rows remain", target, lower, upper, len(trimmed))
    return trimmed


def build_modelling_frame(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Select modelling columns and drop rows with missing values.
    """
    model_cols = [
        "TRANS_VALUE",
        "PROCEDURE_AREA",
        "ACTUAL_AREA",
        "BEDROOMS",
        "PROCEDURE_CODE",
        "IS_FREE_HOLD_CODE",
        "IS_OFFPLAN_CODE",
        "GROUP_CODE",
        "AREA_EN",
    ]
    model_cols = [col for col in model_cols if col in df_clean.columns]
    df_model = df_clean[model_cols].dropna().copy()
    LOGGER.info("Model frame shape before outlier trim: %s", df_model.shape)
    return trim_outliers(df_model, target="TRANS_VALUE")


def build_feature_data(df_model: pd.DataFrame) -> FeatureData:
    """
    Create train-ready feature matrices with both simple and extended sets.
    """
    df_model_dummies = pd.get_dummies(df_model, columns=["AREA_EN"], drop_first=True)

    feature_cols_simple = [
        "PROCEDURE_AREA",
        "ACTUAL_AREA",
        "BEDROOMS",
        "PROCEDURE_CODE",
        "IS_FREE_HOLD_CODE",
    ]
    feature_cols_simple = [c for c in feature_cols_simple if c in df_model_dummies.columns]

    extra_cols = [c for c in ["IS_OFFPLAN_CODE", "GROUP_CODE"] if c in df_model_dummies.columns]
    area_dummy_cols = [c for c in df_model_dummies.columns if c.startswith("AREA_EN_")]
    feature_cols_ext = feature_cols_simple + extra_cols + area_dummy_cols

    X_simple = df_model_dummies[feature_cols_simple]
    X_extended = df_model_dummies[feature_cols_ext]
    y = df_model_dummies["TRANS_VALUE"]

    LOGGER.info("Simple features: %d; extended features: %d", X_simple.shape[1], X_extended.shape[1])
    return FeatureData(
        df_model=df_model_dummies,
        X_simple=X_simple,
        X_extended=X_extended,
        y=y,
        feature_cols_simple=feature_cols_simple,
        feature_cols_ext=feature_cols_ext,
    )

