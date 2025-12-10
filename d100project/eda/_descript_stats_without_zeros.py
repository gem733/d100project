import pandas as pd
import numpy as np

def descript_stats_without_zeros(df):
    """
    Prints descriptive statistics for each column,
    ignoring zeros (treated as missing).
    
    Args:
        df (pd.DataFrame): The dataset to describe
    """
    print("===== Descriptive Statistics (Zero Treated as Missing) =====")
    
    cleaned_df = df.copy()

    # Replace zeros with NaN only for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include='number').columns
    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].replace(0, np.nan)

    # Drop rows where values are missing *per column* during describe()
    stats = cleaned_df.describe(include='all')
    
    print(stats)