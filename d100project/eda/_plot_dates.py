import matplotlib.pyplot as plt#
import pandas as pd

def plot_dates(column):
    """Plot dates in a column.

    Parameters
    ----------
    column : pd.Series
        A pandas Series containing date values.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object containing the date plot.
    """

    # Convert to datetime if not already
    dates = pd.to_datetime(column.dropna())

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dates.dt.year, bins=range(dates.dt.year.min(), dates.dt.year.max() + 1), edgecolor='black')
    ax.set_title('Distribution of Dates by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Frequency')

    return fig