def conditional_means(df, condition_col_list, target_col):
    """
    Use the column containing lists, to work out the conditional means of another column, given each value in the list column, as well as the undoncitioned mean, and the mean for columns where the list is empty.

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

    # Calculate means for each unique condition value
    for condition in exploded[condition_col_list].dropna().unique():
        condition_mean = exploded[exploded[condition_col_list] == condition][target_col].mean()
        means[condition] = condition_mean

    return means