import numpy as np

def replace_missing_values(df, column):
    """
    Replace missing values, NaNs, and zero values in the dataset with mean.

    Parameters:
    data (list): A list of numerical values where some values may be None (missing).
    replacement_value (int, float): The value to replace missing values with. Default is 0.

    Returns:
    list: A new list with missing values replaced.
    """
    # Calculate the mean of the column excluding NaNs and zeros
    mean_value = df[column][(df[column] != 0) & (df[column].notna())].mean()

    # Replace NaNs and zeros with the calculated mean
    df[column] = df[column].replace(0, np.nan)
    df[column] = df[column].fillna(mean_value)

    return df