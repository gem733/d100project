import pandas as pd
import ast

def extract_name(df, column):
    """
    Converts a stringified JSON column into a list of names for any column
    containing a list of dictionaries with a 'name' key.
    
    Args:
        df (pd.DataFrame): The dataset
        column (str): Name of the column to process
    
    Returns:
        pd.DataFrame: Original DataFrame with a new column '{column}_list'
    """
    def parse_items(x):
        if pd.isna(x):
            return []
        try:
            items = ast.literal_eval(x)  # safely convert string to list/dict
            return [d["name"] for d in items if "name" in d]
        except Exception:
            return []

    df[column + "_list"] = df[column].apply(parse_items)
    return df
