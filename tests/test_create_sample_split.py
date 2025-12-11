from d100project.data._create_sample_split import create_sample_split
import pandas as pd

def test_create_sample_split():

    """
    Test that create_sample_split() creates a sample split correctly.
    """
    df_movies = pd.read_parquet("d100project/data/cleaned_data.parquet")

    df_split = create_sample_split(df_movies, id_column="id", training_frac=0.8)
    
    # Check that it returns a DataFrame

    assert df_split is not None
    assert not df_split.empty

    # Check that 'sample' column exists in df_split

    assert "sample" in df_split.columns

    # Check that 'sample' column contains only 'train' and 'test'

    unique_samples = df_split["sample"].unique()
    assert set(unique_samples).issubset({"train", "test"})

    n_rows = len(df_split)
    n_train = (df_split["sample"] == "train").sum()
    train_frac = n_train / n_rows

    # Allow a tolerence (±5%)
    assert 0.75 <= train_frac <= 0.85, f"Training fraction {train_frac:.2f} not ~0.8"