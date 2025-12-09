# This is basically the same as _plot_hist.py but I noticed if the data had missing values it would use a zero. So this histrogram replaces zeros with NaNs and counts how many were replaces them with Nans.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_hist(df, column, bins='auto'):
    """
    Plot the distribution of a column from a DataFrame.
    
    Args:
        df (pd.DataFrame): The dataset
        column (str): Column name to plot
        bins (int or 'auto', optional): Number of bins for numeric data. Defaults to 'auto'.
    """

    col_data = df[column]

    # Convert column to numeric (invalid values become NaN)
    data = pd.to_numeric(col_data, errors='coerce')

    # Replace zero values with NaN
    data = data.replace(0, np.nan)

    # Drop all missing values (zeros + invalid values)
    cleaned = data.dropna()

    # Count number of removed entries
    dropped = len(col_data) - len(cleaned)
    
    plt.figure(figsize=(8, 5))
    
    if np.issubdtype(data.dtype, np.number):
        # Numeric column → histogram
        sns.histplot(data, bins=bins, kde=True, color='skyblue')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column} (Numeric)')
    else:
        # Categorical/object column → bar plot
        counts = data.value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette='pastel')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'Distribution of {column} (Categorical)')
        plt.xticks(rotation=45, ha='right')
    
    # Add number of dropped values below the plot
    plt.text(
        0.5, -0.15,
        f"Dropped {dropped} missing value(s).",
        fontsize=10,
        ha='center',
        va='center',
        transform=plt.gca().transAxes
    )

    plt.tight_layout()
    plt.show()