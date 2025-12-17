from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts transforms a number using the natural log
    """
        
    def __init__(self):
        pass  # no columns, ColumnTransformer passes selected columns

    def fit(self, X, y=None):
        return self  # nothing to fit
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided for LogTransformer.")
        return [f"{f}_log" for f in input_features]

    def transform(self, X):
        return np.log1p(X)  # X is an array
