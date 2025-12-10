import pandas as pd
from collections import Counter

def tally_strings(df, column):
    """
    Counts how often each string appears in a list column.
    
    Args:
        df (pd.DataFrame)
        column_list (str): name of the list column

    Returns:
        pd.DataFrame: table with string and count
    """

    all_strings = []
    for items in df[column]:
        all_strings.extend(items)
    
    counter = Counter(all_strings)
    
    # Convert to a DataFrame and sort by count
    tally_df = pd.DataFrame(counter.items(), columns=["String", "Count"])
    tally_df = tally_df.sort_values("Count", ascending=False).reset_index(drop=True)
    
    return tally_df