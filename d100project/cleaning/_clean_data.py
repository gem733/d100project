from pathlib import Path

# Import your preprocessing functions
from d100project.data._load_data import load_data
from d100project.cleaning._remove_missing_values import remove_missing_values
from d100project.cleaning._extract_dates import extract_dates
from d100project.cleaning._replace_missing_values import replace_missing_values

def cleaned_data():
    """
    Loads raw data, applies cleaning functions, and saves cleaned dataframe as a parquet file in the data folder.

    Args:
    df (DataFrame): The input DataFrame to be cleaned.

    Returns:
    DataFrame: Cleaned DataFrame after removing and replacing missing values.
    """
    # Load raw data
    df = load_data()

    # Remove rows with missing values in specified columns
    columns_to_check = ['revenue']  
    df = remove_missing_values(df, columns_to_check)

    # Extract date features from a date column
    df = extract_dates(df, 'release_date')

    # Replace missing values in specified columns
    columns_to_replace = ['budget', 'runtime']
    for column in columns_to_replace:
        df = replace_missing_values(df, column)

    # Save cleaned data to parquet file
    output_path = Path(__file__).parent/"data"/"cleaned_data.parquet"
    df.to_parquet(output_path, index=False)

    return df