import pandas as pd
from pathlib import Path

def load_data():

    """
    Load the dataset from movieds.csv file.
    Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download&select=tmdb_5000_movies.csv
    
    I had to download the dataset manually as Kaggle does not allow direct downloading via scripts without authentication.
    
    Returns: pd.DataFrame: DataFrame containing data on 5000 movies.
    """

    
    data_path = Path(__file__).parent/"data"/"movieds.csv"  # felxible path that is correct when run from anywhere
    df = pd.read_csv(data_path)
    return df