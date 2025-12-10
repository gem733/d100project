def dtype(df):
    """
    Prints the type of each column in the DataFrame

    Args:
        df (pd.DataFrame): The dataset to describe
    """
    print("===== Data Types =====")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")