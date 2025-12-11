import matplotlib.pyplot as plt
import numpy as np

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

    # Fit and plot line of best fit
    if df[x_column].dtype in ['int64', 'float64'] and df[y_column].dtype in ['int64', 'float64']:
        m, b = np.polyfit(df[x_column].dropna(), df[y_column].dropna(), 1)
        plt.plot(df[x_column], m*df[x_column] + b, color='red')

    plt.tight_layout()
    plt.show()