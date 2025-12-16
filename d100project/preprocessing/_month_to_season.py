# i'm going to use the use the season of release as a feature, so I need to get a dummy variable for each season from the month
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MonthToSeasonTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts month numbers to season dummy variables.
    Seasons are defined as:
        - Winter: December (12), January (1), February (2)
        - Spring: March (3), April (4), May (5)
        - Summer: June (6), July (7), August (8)
        - Autumn: September (9), October (10), November (11)
    """
    def __init__(self, month_column='month'):
        self.month_column = month_column
        self.season_columns = ["season_spring", "season_summer", "season_fall", "season_winter"]

    def fit(self, X, y=None):
        # Nothing to learn, just return self
        return self

    def transform(self, X):
        # Make a copy to avoid modifying original
        X_ = X.copy()

        # Map months to season names
        def month_to_season(month):
            if month in [12, 1, 2]:
                return "season_winter"
            elif month in [3, 4, 5]:
                return "season_spring"
            elif month in [6, 7, 8]:
                return "season_summer"
            else:
                return "season_fall"

        # Convert month column to seasons
        seasons = X_[self.month_column].apply(month_to_season)

        # One-hot encode seasons
        X_season = pd.get_dummies(seasons)
        # Ensure all four columns exist
        for col in self.season_columns:
            if col not in X_season:
                X_season[col] = 0

        # Return as numpy array
        return X_season[self.season_columns].values

    def get_feature_names_out(self, input_features=None):
        # Return proper feature names for ColumnTransformer
        return self.season_columns