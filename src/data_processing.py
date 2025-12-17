"""
Data loading and cleaning utilities for the Dubai real-estate dataset.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Columns that are removed entirely during cleaning
DEFAULT_DROP_COLS = [
    "TRANSACTION_NUMBER",
    "INSTANCE_DATE",
    "TOTAL_BUYER",
    "TOTAL_SELLER",
    "MASTER_PROJECT_EN",
    "PROJECT_EN",
]


def load_raw_data(csv_path: Path | str) -> pd.DataFrame:
    """
    Load the raw transactions CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    LOGGER.info("Loading raw data from %s", path)
    return pd.read_csv(path)


def extract_bedrooms(value: Any) -> float | int | np.nan:
    """
    Convert ROOM text like "3 B/R", "STUDIO", or "2 BED" into a numeric bedroom count.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).upper().strip()
    if "STUDIO" in text:
        return 0
    match = re.search(r"\d+", text)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return np.nan
    return np.nan


def map_procedure_category(text: Any) -> str:
    """
    Map raw procedure descriptions into a small, stable set of categories.
    """
    if pd.isna(text):
        return "Other"
    lowered = str(text).lower()
    if "mortgage" in lowered:
        return "Mortgage Registration"
    if "sale" in lowered or "transfer" in lowered:
        return "Sale Registration"
    if "initial" in lowered:
        return "Initial Registration"
    if "lease" in lowered:
        return "Lease Registration"
    if "gift" in lowered:
        return "Gift Transfer"
    return "Other"


def _encode_binary_flag(df: pd.DataFrame, column: str, mapping: Dict[str, int], *, default: int = 0) -> pd.Series:
    """
    Normalise string flags into 0/1 codes using a simple mapping.
    """
    return (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(default)
        .astype(int)
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight cleaning and feature normalisation.

    This keeps the logic deliberately minimal and deterministic to make it easy
    to reuse outside of the notebook.
    """
    LOGGER.info("Cleaning dataframe with %d rows", len(df))
    df_clean = df.drop(columns=DEFAULT_DROP_COLS, errors="ignore").copy()

    df_clean["BEDROOMS"] = df_clean["ROOMS_EN"].apply(extract_bedrooms)

    if "GROUP_EN" in df_clean.columns:
        df_clean["GROUP_CODE"] = df_clean["GROUP_EN"].astype("category").cat.codes
    else:
        df_clean["GROUP_CODE"] = 0

    df_clean["PROCEDURE_CAT"] = df_clean["PROCEDURE_EN"].apply(map_procedure_category)
    procedure_map = {
        "Mortgage Registration": 1,
        "Sale Registration": 2,
        "Initial Registration": 3,
        "Lease Registration": 4,
        "Gift Transfer": 5,
        "Other": 99,
    }
    df_clean["PROCEDURE_CODE"] = df_clean["PROCEDURE_CAT"].map(procedure_map).astype(int)

    df_clean["IS_OFFPLAN_CODE"] = _encode_binary_flag(
        df_clean,
        "IS_OFFPLAN_EN",
        mapping={"off-plan": 1, "off plan": 1},
    )
    df_clean["IS_FREE_HOLD_CODE"] = _encode_binary_flag(
        df_clean,
        "IS_FREE_HOLD_EN",
        mapping={"free hold": 1, "freehold": 1},
    )

    LOGGER.info("Finished cleaning. Columns now: %s", sorted(df_clean.columns))
    return df_clean

