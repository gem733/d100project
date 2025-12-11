def remove_columns(df, columns):
    """
    Remove specified columns from the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list): List of column names to be removed.

    Returns:
    DataFrame: DataFrame with specified columns removed.
    """
    return df.drop(columns=columns)