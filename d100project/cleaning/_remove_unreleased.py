def _remove_unreleased(df):
    """
    Remove rows where the 'status' column is not 'Released'.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: DataFrame with only released entries.
    """
    return df[df['status'] == 'Released']