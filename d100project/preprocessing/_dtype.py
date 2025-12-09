def dtype():
    def describe_data(df):
    """
    Prints summary of the dataframe: dtypes, missing values, descriptive statistics
    """
    print("===== Data Types =====")
    print(df.dtypes)
    print("\n===== Missing Values =====")
    print(df.isnull().sum())
    print("\n===== Descriptive Statistics =====")
    display(df.describe()