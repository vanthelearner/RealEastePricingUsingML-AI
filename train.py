"""
Command-line entrypoint to run the modelling pipeline outside the notebook.

Example
-------
python train.py --data data/transactions-2025-11-20.csv --output outputs --sample-frac 0.1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_processing import clean_dataframe, load_raw_data
from src.features import build_feature_data, build_modelling_frame
from src.models import default_model_specs, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dubai real-estate pricing models.")
    parser.add_argument("--data", required=True, help="Path to the raw transactions CSV.")
    parser.add_argument("--output", default="outputs", help="Directory to write results.")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fraction of data to sample for quicker experimentation (0 < f <= 1).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling and splits.")
    return parser.parse_args()


def maybe_sample(df: pd.DataFrame, sample_frac: float, random_state: int) -> pd.DataFrame:
    if sample_frac >= 1.0:
        return df
    if sample_frac <= 0:
        raise ValueError("sample_frac must be > 0.")
    LOGGER.info("Sampling %.0f%% of rows for a quicker run", sample_frac * 100)
    return df.sample(frac=sample_frac, random_state=random_state)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(args.data)
    sampled_df = maybe_sample(raw_df, sample_frac=args.sample_frac, random_state=args.random_state)

    df_clean = clean_dataframe(sampled_df)
    df_model = build_modelling_frame(df_clean)
    feature_data = build_feature_data(df_model)

    specs = default_model_specs()
    results = train_and_evaluate(specs, feature_data)

    metrics_path = output_dir / "metrics.csv"
    results.to_csv(metrics_path, index=False)

    LOGGER.info("Saved metrics to %s", metrics_path.resolve())
    LOGGER.info("Top results:\n%s", results.head())


if __name__ == "__main__":
    main()

