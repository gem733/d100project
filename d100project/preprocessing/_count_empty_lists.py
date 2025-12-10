def count_empty_lists(dataframe, column):
    """
    Count the number of empty lists in a specified column of a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to check for empty lists.

    Returns:
    int: The count of empty lists in the specified column.
    """
    empty_list_count = dataframe[column + "_list"].apply(lambda x: isinstance(x, list) and len(x) == 0).sum()
    print("Number of missing values:", empty_list_count)