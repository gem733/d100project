import numpy as np

def replace_missing_values(df, column):
    """
    Replace missing values, NaNs, and zero values in the dataset with mean, and create a new column indicating which values were replaced.

    Parameters:
    data (DataFrame): A list of numerical values where some values may be None (missing).
    column (str): The name of the column to process.

    Returns:
    Dataframe: a dataframe with missing values replaced and an indicator column.
    """
    mean_value = df[column].replace(0, np.nan).mean()
    indicator_column = f"{column}_was_missing"
    
    df[indicator_column] = df[column].isna() | (df[column] == 0)
    df[column] = df[column].replace(0, np.nan).fillna(mean_value)
    
    return df