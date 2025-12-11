from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies the natural logarithm to specified numerical variables.
    It handles non-positive values by shifting the data before applying the log transformation.

    Parameters
    ----------
    variables : list
        List of numerical variable names to be transformed.
    """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list of variable names.")
        self.variables = variables
        self.shift_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer by calculating the necessary shift for each variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series, optional
            The target variable (not used).

        Returns
        -------
        self : LogTransformer
            Fitted transformer.
        """
        X = X.copy()
        for var in self.variables:
            min_value = X[var].min()
            if min_value <= 0:
                self.shift_[var] = abs(min_value) + 1e-6  # Small constant to avoid log(0)
            else:
                self.shift_[var] = 0
        return self

    def log_transform(self, X):
        """
        Apply the log transformation to the specified variables.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed data with log applied to specified variables.
        """
        X = X.copy()
        for var in self.variables:
            shift_value = self.shift_.get(var, 0)
            X[var] = np.log(X[var] + shift_value)
        return X