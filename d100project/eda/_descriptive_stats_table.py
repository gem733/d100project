import numpy as np
import pandas as pd


def descriptive_stats_table(df):
    """
    Create a descriptive statistics table for all columns in a DataFrame. Zeros are to be treated as missing values.

    Includes:
    - data type
    - count of non-null values
    - number of missing values
    - number of unique values
    - most frequent value (mode)
    - frequency of the mode
    - numeric stats where applicable

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Copy for stats only
    df_stats = df.copy()

    # Replace zeros with NaN for numeric columns
    num_cols = df_stats.select_dtypes(include="number").columns
    df_stats[num_cols] = df_stats[num_cols].replace(0, np.nan)

    # Base descriptive stats
    desc = df_stats.describe(include="all").T

    # Add extra columns (no overlaps)
    desc["dtype"] = df.dtypes
    desc["missing"] = df_stats.isna().sum()
    desc["mode"] = df_stats.mode().iloc[0]
    desc["mode_freq"] = df_stats.apply(
        lambda x: x.value_counts(dropna=True).iloc[0]
        if x.notna().any() else np.nan
    )

    return desc
