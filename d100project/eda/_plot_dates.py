import matplotlib.pyplot as plt
import pandas as pd

def plot_dates(df, column):
    """
    Plot a histogram of the distribution of dates (by year) in a DataFrame column.
    """

    # Convert to datetime safely
    dates = pd.to_datetime(df[column], errors='coerce').dropna()

    # Extract min/max year as Python ints
    min_year = int(dates.dt.year.min())
    max_year = int(dates.dt.year.max())

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        dates.dt.year,
        bins=range(min_year, max_year + 2),
        edgecolor='black'
    )
    ax.set_title(f"Distribution of {column} by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency")

    plt.show()