from d100project.data._load_cleaned_parquet import load_cleaned_parquet

def test_load_cleaned_parquet():

    """
    Test that load_cleaned_parquet() loads the  dataset correctly.
    """

    df = load_cleaned_parquet()
    
    # Check that it returns a DataFrame

    assert df is not None
    assert not df.empty

    # Check that all columns exist in df

    expected_columns = ["budget", "genres_list", "id"]
    for col in expected_columns:
        assert col in df.columns
