from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ListOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode columns that contain lists or single strings.
    For each unique item in any list or string in the columns, create a new dummy column.
    """

    def __init__(self, columns):
        self.columns = columns
        self.unique_items_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by collecting all unique items in the specified columns.
        """
        for col in self.columns:
            unique_items = set()
            
            def add_items(val):
                if isinstance(val, list):
                    unique_items.update(val)
                elif isinstance(val, str):
                    unique_items.add(val)
                # ignore NaN or other types

            X[col].dropna().apply(add_items)
            self.unique_items_[col] = sorted(unique_items)

        return self

    def transform(self, X):
        """
        Transform the DataFrame by adding one-hot encoded columns for each item.
        """
        X = X.copy()
        all_dummies = []

        for col in self.columns:
            for item in self.unique_items_[col]:
                dummy_col_name = f"{col}__{item}"
                dummies = X[col].apply(
                    lambda val: 1 if (item in val if isinstance(val, list) else val == item) else 0
                )
                all_dummies.append(dummies.rename(dummy_col_name))

        if all_dummies:
            X = pd.concat([X] + all_dummies, axis=1)
        
        X.drop(columns=self.columns, inplace=True)
        return X