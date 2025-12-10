import pandas as pd
import matplotlib.pyplot as plt

def plot_hist_range(df, column, bins='auto', min_value=None, max_value=None):
    """
    Plot the distribution of a numeric column from a DataFrame for values between given limits.

    Args:
        df (pd.DataFrame): The dataset
        column (str): Column name to plot
        bins (int or str): Number of histogram bins or 'auto'. Defaults to 'auto'.
        min_value (float): Minimum value to include. Defaults to None.
        max_value (float): Maximum value to include. Defaults to None.
    """
    # Convert column to numeric (invalid values become NaN)
    data = pd.to_numeric(df[column], errors='coerce')

    # Filter by min and max values
    if min_value is not None:
        data = data[data >= min_value]
    if max_value is not None:
        data = data[data <= max_value]

    # Drop missing values
    cleaned = data.dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(cleaned, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    title = f'Distribution of {column}'
    if min_value is not None and max_value is not None:
        title += f' (between {min_value} and {max_value})'
    elif min_value is not None:
        title += f' (>= {min_value})'
    elif max_value is not None:
        title += f' (<= {max_value})'
    plt.title(title)

    plt.tight_layout()
    plt.show()
