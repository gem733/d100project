import pandas as pd
import numpy as np

def extract_dates(df, column):

    """
    Extract date components (year, month) from a specified date column in the DataFrame, if a row doesn't have a valid date, replace month with NaN, and year with average year.

    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The name of the column containing date information.

    Returns:
    DataFrame: DataFrame with new columns for year, month, and day extracted from the date column.
    """

    # Convert the specified column to datetime, coerce errors to NaT
    df[column] = pd.to_datetime(df[column], errors='coerce')

    # Extract year, month, and day
    df['year'] = df[column].dt.year
    df['month'] = df[column].dt.month

    # Calculate average year excluding NaNs
    avg_year = int(df['year'].mean())

    # Replace NaN months with NaN and NaN years with average year
    df['month'] = df['month'].where(df[column].notna(), np.nan)
    df['year'] = df['year'].where(df[column].notna(), avg_year)

    return df