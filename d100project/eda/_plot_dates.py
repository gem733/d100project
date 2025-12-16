import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def plot_dates(df, column):
    """
    Plot a histogram of the distribution of dates (by year) in a DataFrame column
    with a smooth distribution line.

    Args:
        df (pd.DataFrame): The dataset
        column (str): Column name containing date values
    """

    # Convert to datetime safely
    dates = pd.to_datetime(df[column], errors='coerce').dropna()
    years = dates.dt.year

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2), color='blue',
            edgecolor='black', alpha=0.6, density=True)

    # Smooth line (KDE)
    kde = gaussian_kde(years)
    x_vals = np.linspace(years.min(), years.max(), 500)
    ax.plot(x_vals, kde(x_vals), color='darkblue', linewidth=2, label='Density')

    ax.set_title(f"Distribution of {column} by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()
