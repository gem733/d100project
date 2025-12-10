def count_missing_values(df, column):
   
    """
    Count the number of missing values and 0s in a specified column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    column (str): The name of the column to check for missing values.

    Returns:
    int: The count of missing values in the specified column.
    """

    return df[column].isnull().sum() + (df[column] == 0).sum()