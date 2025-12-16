import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def plot_category(df, list_column, target_column, top_n=5):
    """
    Plot the top N most common items in a list column as a bar plot,
    with a secondary y-axis showing average target value.
    
    Args:
        df (pd.DataFrame): Dataset
        list_column (str): Column containing lists
        target_column (str): Numeric column (e.g., revenue)
        top_n (int): Number of top items to show
    """

    # Flatten the list column into individual items
    all_items = df[list_column].dropna().explode()
    
    # Count occurrences
    counts = all_items.value_counts().head(top_n)
    
    # Filter df to rows containing the top items
    top_items_set = set(counts.index)
    
    # Explode and filter to top items only
    exploded = df[[list_column, target_column]].dropna(subset=[list_column])
    exploded = exploded.explode(list_column)
    exploded = exploded[exploded[list_column].isin(top_items_set)]
    
    # Exclude zero revenue rows
    exploded = exploded[exploded[target_column] > 0]
    
    # Compute average revenue per item
    avg_revenue = exploded.groupby(list_column)[target_column].mean().loc[counts.index]
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10,6))

    # Left axis: counts
    ax1.bar(counts.index, counts.values, color='skyblue', edgecolor='black')
    ax1.set_ylabel('Count', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Right axis: average revenue
    ax2 = ax1.twinx()
    ax2.plot(avg_revenue.index, avg_revenue.values, color='darkblue', marker='o', linewidth=2, label='Average Revenue')
    ax2.set_ylabel('Average Revenue', color='darkblue')
    ax2.tick_params(axis='y', labelcolor='darkblue')
    
    # Titles and labels
    plt.title(f"Top {top_n} items in '{list_column}' and their Average {target_column}")
    ax1.set_xlabel(list_column)
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.show()
