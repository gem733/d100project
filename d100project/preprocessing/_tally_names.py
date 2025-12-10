def tally_strings(df, column):
    """
    Count how many times each FULL STRING appears in a column.

    Args:
        df (pd.DataFrame)
        column (str)

    Returns:
        pd.DataFrame: counts of each unique string
    """
    # Ensure everything is treated as a whole string
    strings = df[column].dropna().astype(str)

    # Use value_counts(), which counts whole strings correctly
    tally = strings.value_counts().reset_index()

    tally.columns = ["Value", "Count"]
    return tally
