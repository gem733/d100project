import pandas as pd
from pathlib import Path

def load_cleaned_parquet():
    """
    Load the cleaned parquet file into a pandas DataFrame.

    Args:
        parquet_path (str or Path): The file path to the parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the parquet file.
    """
    parquet_path = Path(__file__).parent / "cleaned_data.parquet"
    df = pd.read_parquet(parquet_path)
    return df