import pandas as pd 
from collections import Counter

def tally_items(df, column_list):
    """
    Counts how often each item appears in a list column.
    
    Args:
        df (pd.DataFrame)
        column_list (str): name of the list column
    
    Returns:
        pd.DataFrame: table with item and count
    """
    all_items = []
    for items in df[column_list]:
        all_items.extend(items)
    
    counter = Counter(all_items)
    
    # Convert to a DataFrame and sort by count
    tally_df = pd.DataFrame(counter.items(), columns=["Item", "Count"])
    tally_df = tally_df.sort_values("Count", ascending=False).reset_index(drop=True)
    
    return tally_df
