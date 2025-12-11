def remove_missing_values(df, columns):
    """
    Remove rows with missing data, NaNs, and rows with zero in specified columns.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list): List of column names to check for missing data.

    Returns:
    DataFrame: DataFrame with rows containing missing data in specified columns removed.
    """
    for column in columns:
        df = df[df[column].notna() & (df[column] != 0)]
    return df