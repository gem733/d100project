def conditional_means(df, condition_col_list, target_col, top_n=20):
    """
    Use the column containing lists, to work out the conditional means of another column, given each value in the list column, as well as the undconitioned mean, and the mean for columns where the list is empty.

    Args:
        df (pd.DataFrame): The dataset
        condition_col_list (str): Column name to condition on
        target_col (str): Column name to calculate means for

    Returns:
        dict: A dictionary with condition values as keys and corresponding means as values
    """
    means = {}
    
    # Calculate the overall mean
    overall_mean = df[target_col].mean()
    means['overall'] = overall_mean

    # Calculate the mean for rows where the list is empty
    empty_mean = df[df[condition_col_list].apply(lambda x: len(x) == 0)][target_col].mean()
    means['empty'] = empty_mean

    # Explode the list column to get individual condition values
    exploded = df.explode(condition_col_list)

    # Drop rows where the exploded value is NaN
    exploded = exploded[exploded[condition_col_list].notna()]

    # Count frequency of each item
    freq = exploded[condition_col_list].value_counts()

    # Limit to top N most common items
    top_items = freq.head(top_n).index.tolist()

    print(f"Conditional means of '{target_col}' for the top {top_n} most common items in '{condition_col_list}':")

    means_list = []
    for item in top_items:
        mean_val = exploded.loc[exploded[condition_col_list] == item, target_col].mean()
        means_list.append((item, mean_val))

    # Sort by mean descending
    means_list.sort(key=lambda x: x[1], reverse=True)

    # Print with commas and 0 decimal places
    for item, mean_val in means_list:
        print(f"{item}: {mean_val:,.0f}")