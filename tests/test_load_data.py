from d100project.data._load_data import load_data

def test_load_data():

    """
    Test that load_data() loads the movies dataset correctly.
    """

    df = load_data()
    
    # Check that it returns a DataFrame

    assert df is not None
    assert not df.empty

    # Check that all columns exist in df

    expected_columns = ["budget", "genres", "id", "keywords", "original_language", "original_title", "overview", "popularity", "production_companies", "production_countries", "release_date", "revenue", "runtime", "spoken_languages", "status", "tagline", "title", "vote_average", "vote_count"]
    for col in expected_columns:
        assert col in df.columns
