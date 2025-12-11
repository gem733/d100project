import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_root)) 

# Import your preprocessing functions
from d100project.data._load_data import load_data
from d100project.cleaning._remove_missing_values import remove_missing_values
from d100project.cleaning._extract_dates import extract_dates
from d100project.cleaning._replace_missing_values import replace_missing_values
from d100project.eda._extract_name import extract_name
from d100project.cleaning._remove_columns import remove_columns
from d100project.cleaning._remove_unreleased import remove_unreleased

def cleaned_data():
    """
    Loads raw data, applies cleaning functions, and creates a parquet file in the data folder, which it saves the dataframe into.

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

    # Replace missing values in specified columns
    columns_to_replace = ['budget', 'runtime']  
    for column in columns_to_replace:
        df = replace_missing_values(df, column)

    # Extract date components from a date column
    df = extract_dates(df, 'release_date')  

        # Extract names components from list
    df = extract_name(df, 'genres')
    df = extract_name(df, 'production_companies')  
    df = extract_name(df, 'production_countries')   
    df = extract_name(df, 'spoken_languages')

    # Remove unreleased movies
    df = remove_unreleased(df)

    # Remove unnecessary columns
    columns_to_remove = ['genres', 'homepage', 'tagline', 'overview', 'status', 'release_date', 'production_companies', 'production_countries', 'spoken_languages', 'keywords', 'original_title', 'popularity', 'title', 'vote_count', 'vote_average']
    df = remove_columns(df, columns_to_remove)

    # Build the output path relative to this script's location
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir.parent / 'data' / 'cleaned_data.parquet'
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    # Save to Parquet
    df.to_parquet(output_path, index=False, engine='fastparquet')

    return df

if __name__ == "__main__":
    cleaned_data()