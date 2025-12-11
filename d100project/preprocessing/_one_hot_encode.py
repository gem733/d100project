from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ListOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode columns that contain lists.
    For each unique item in any list in the columns, create a new dummy column.
    """
    def __init__(self, columns):
        self.columns = columns
        self.unique_items_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            unique_items = set()
            X[col].dropna().apply(lambda lst: unique_items.update(lst))
            self.unique_items_[col] = sorted(unique_items)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            for item in self.unique_items_[col]:
                dummy_col_name = f"{col}__{item}"
                X_transformed[dummy_col_name] = X_transformed[col].apply(
                    lambda lst: 1 if isinstance(lst, list) and item in lst else 0
                )
            X_transformed.drop(columns=[col], inplace=True)
        return X_transformed
