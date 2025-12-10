import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix(df, columns):
    """
    Plot a correlation matrix for the specified numeric columns in the DataFrame, inlcuding a column that contains dates, where we extract the year from the date column.

    Args:
        df (pd.DataFrame): The dataset
        columns (list of str): List of column names to include in the correlation matrix
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = df[columns].copy()

    # Convert date columns to year if any
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.year
        # Optional: convert string-like dates to datetime first
        elif pd.api.types.is_object_dtype(data[col]):
            try:
                data[col] = pd.to_datetime(data[col], errors='coerce').dt.year
            except:
                pass

    # Compute the correlation matrix
    corr = data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()

