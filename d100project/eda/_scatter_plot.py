import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def scatter_plot(df, y_column, x_column):
    """
    Plot a scatter plot for the specified x and y columns in the DataFrame with estimated line of best fit.
    Args:
        df (pd.DataFrame): The dataset
        y_column (str): The name of the column to be plotted on the y-axis
        x_column (str): The name of the column to be plotted on the x-axis
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.5)
    plt.title(f'Scatter Plot of {y_column} vs {x_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)

    # Clean numeric data and drop NaNs
    clean = df[[x_column, y_column]].apply(pd.to_numeric, errors='coerce').dropna()

    # Fit line only if enough data exists
    if len(clean) > 1:
        x = clean[x_column]
        y = clean[y_column]
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, color='red')

    plt.tight_layout()
    plt.show()