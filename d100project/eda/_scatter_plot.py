import matplotlib.pyplot as plt


def scatter_plot(df, y_column, x_column):
    """
    Plot a scatter plot for the specified x and y columns in the DataFrame.

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
    plt.grid(True)
    plt.tight_layout()
    plt.show()