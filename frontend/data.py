# ===============================================
# Data Loading and Preprocessing Module
# ===============================================

import numpy as np
import pandas as pd
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

from .config import DATA_PATH, TARGET_COLUMN


def load_historical_data() -> pd.DataFrame:
    """
    Load and preprocess historical solar radiation data.

    Returns:
        Preprocessed DataFrame with all features
    """
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime"],
        dayfirst=True,
        index_col="datetime",
    )

    df = df.sort_index().apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index, dayfirst=True)

    # Prepare data with all features
    df_day = df.copy()
    df_day.dropna(subset=[TARGET_COLUMN], inplace=True)

    # Add cyclical encoding for time features
    df_day["hour"] = df_day.index.hour
    df_day["month"] = df_day.index.month
    df_day["day_of_year"] = df_day.index.dayofyear
    df_day["hour_sin"] = np.sin(2 * np.pi * df_day["hour"] / 24)
    df_day["hour_cos"] = np.cos(2 * np.pi * df_day["hour"] / 24)
    df_day["month_sin"] = np.sin(2 * np.pi * df_day["month"] / 12)
    df_day["month_cos"] = np.cos(2 * np.pi * df_day["month"] / 12)
    df_day["doy_sin"] = np.sin(2 * np.pi * df_day["day_of_year"] / 365)
    df_day["doy_cos"] = np.cos(2 * np.pi * df_day["day_of_year"] / 365)

    return df_day


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics about the data.

    Args:
        df: Loaded DataFrame

    Returns:
        Dictionary with summary statistics
    """
    return {
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "total_records": len(df),
        "avg_radiation": df[TARGET_COLUMN].mean(),
        "max_radiation": df[TARGET_COLUMN].max(),
        "columns": list(df.columns),
    }
