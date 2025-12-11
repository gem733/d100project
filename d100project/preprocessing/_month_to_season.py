# i'm going to use the use the season of release as a feature, so I need to get a dummy variable for each season from the month
from sklearn.base import BaseEstimator, TransformerMixin

class MonthToSeasonTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts month numbers to season dummy variables.
    Seasons are defined as:
        - Winter: December (12), January (1), February (2)
        - Spring: March (3), April (4), May (5)
        - Summer: June (6), July (7), August (8)
        - Autumn: September (9), October (10), November (11)
    """
    def __init__(self, month_column):
        self.month_column = month_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        season_mapping = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11]
        }

        # Create dummy variables
        for season, months in season_mapping.items():
            X[f"{self.month_column}__{season}"] = \
                X[self.month_column].apply(
                    lambda m: 1 if m in months else 0
                )

        # Drop original month column
        X.drop(columns=[self.month_column], inplace=True)

        return X