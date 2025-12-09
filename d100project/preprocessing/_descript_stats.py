def descript_stats(df):
    """
    Prints descriptive statistics of the DataFrame

    Args:
        df (pd.DataFrame): The dataset to describe
    """
    print("===== Descriptive Statistics =====")
    print(df.describe(include='all'))
