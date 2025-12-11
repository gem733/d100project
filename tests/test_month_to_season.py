import pandas as pd
from d100project.preprocessing._month_to_season import MonthToSeasonTransformer


def test_month_to_season_transformer():
    # Sample input data
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "month": [1, 4, 7, 10]  # Winter, Spring, Summer, Autumn
    })

    transformer = MonthToSeasonTransformer(month_column="month")

    df_transformed = transformer.fit_transform(df)

    # 1. Check original month column is removed
    assert "month" not in df_transformed.columns

    # 2. Check expected season dummy columns exist
    expected_cols = [
        "month__Winter",
        "month__Spring",
        "month__Summer",
        "month__Autumn"
    ]

    for col in expected_cols:
        assert col in df_transformed.columns

    # 3. Check values are correct
    expected_values = pd.DataFrame({
        "month__Winter": [1, 0, 0, 0],
        "month__Spring": [0, 1, 0, 0],
        "month__Summer": [0, 0, 1, 0],
        "month__Autumn": [0, 0, 0, 1],
    })

    pd.testing.assert_frame_equal(
        df_transformed[expected_values.columns].reset_index(drop=True),
        expected_values
    )