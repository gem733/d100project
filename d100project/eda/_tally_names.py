def tally_names(df, column):
    """
    Count how many times each UNIQUE STRING appears in a column.

    Args:
        df (pd.DataFrame)
        column (str)

    Returns:
        pd.DataFrame: counts of each unique string
    """
    # Explode the lists to get individual names
    exploded = df.explode(column)

    # Drop NaN values
    exploded = exploded[exploded[column].notna()]

    # Use value_counts() to count unique names
    tally = exploded[column].value_counts().reset_index()

    tally.columns = ["Value", "Count"]
    return tally

