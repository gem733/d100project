from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # no columns, ColumnTransformer passes selected columns

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X):
        return np.log1p(X)  # X is an array
