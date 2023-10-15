import numpy as np
import pandas as pd
from tqdm import tqdm
from common_utils import downcast_to_32bit

# ========================================================================================
# 1. EDA Function for this competition
# ========================================================================================
def filter_df(df, stock_id=None, date_id=None, seconds=None, reset_index=False):
    """
    Filter a DataFrame based on specified conditions.

    Args:
        df (pandas.DataFrame): The input DataFrame to be filtered.
        stock_id (int, optional): Filter rows where the 'stock_id' column matches this value.
        date_id (int or str, optional): Filter rows where the 'date_id' column matches this value.
        seconds (int, tuple, or list, optional): Filter rows based on 'seconds' column conditions.
        reset_index (bool, optional): Whether to reset the index of the filtered DataFrame.

    Returns:
        pandas.DataFrame: The filtered DataFrame based on the specified conditions.

    The function filters the input DataFrame 'df' based on the specified conditions. It allows you to filter rows
    based on values in the 'stock_id' and 'date_id' columns and apply various conditions to the 'seconds' column.

    - 'stock_id': If specified, only rows with a matching 'stock_id' are retained.
    - 'date_id': If specified, only rows with a matching 'date_id' are retained. 'date_id' can be provided as an int or str.
    - 'seconds': If specified as an int, only rows with 'seconds' matching the value are retained.
                If specified as a tuple, only rows with 'seconds' within the specified range are retained.
                If specified as a list, only rows with 'seconds' matching values in the list are retained.

    The filtered DataFrame is returned, and you can choose to reset its index using the 'reset_index' parameter.
    """
    # For each argument, get the conditons series based on the input data format
    conds_dict = {}
    for input_, input_arg_name in zip([stock_id, date_id, seconds], ["stock_id", "date_id", "seconds"]):
        if input_ is None:
            conds_dict[input_arg_name] = ~df[input_arg_name].isnull()
        elif isinstance(input_, int):
            conds_dict[input_arg_name] = (df[input_arg_name] == input_)
        elif isinstance(input_, str):
            conds_dict[input_arg_name] = (df[input_arg_name] == int(input_))
        elif isinstance(input_, tuple):
            if len(input_) == 2:
                conds_dict[input_arg_name] = df[input_arg_name].between(input_[0], input_[1])
            else:
                print(f"{input_arg_name} tuple shouldn't have more than 2 dimension")
        elif isinstance(input_, list):
            conds_dict[input_arg_name] = df[input_arg_name].isin(input_)
    
    # Filter the dataframe using AND (&) Operator
    conditions = [conds for conds in conds_dict.values()]
    df_subset = df.loc[pd.concat(conditions, axis=1).all(axis=1)]
    
    if reset_index:
        df_subset = df_subset.reset_index(drop=True)
    return df_subset

# ========================================================================================
# 2. Preprocessing Function for this competition
# ========================================================================================
def clean_df(df, columns_to_drop=['row_id', 'time_id']):
    """
    Clean and prepare a DataFrame for analysis.

    Args:
        df (pandas.DataFrame): The input DataFrame to be cleaned.
        columns_to_drop (list, optional): A list of column names to be dropped from the DataFrame.

    Returns:
        pandas.DataFrame: The cleaned DataFrame with specified transformations.

    The function cleans and prepares the input DataFrame 'df' for analysis by performing the following operations:

    1. Drops specified columns from the DataFrame based on the 'columns_to_drop' parameter.
    2. Downcasts numeric columns to 32-bit data types for memory optimization.
    3. Renames selected columns for improved readability.
    4. Calculates a new column 'real_imb_size' by multiplying 'imb_size' with 'imb_flag' if 'imb_size' exists in the DataFrame.

    Parameters:
        - df: The input DataFrame to be cleaned.
        - columns_to_drop: A list of column names to be dropped. Default columns include 'row_id' and 'time_id'.

    Returns:
        A cleaned and transformed DataFrame ready for further analysis.
    """
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df = downcast_to_32bit(df)
    df = df.rename(
        columns={
            "seconds_in_bucket": "seconds",
            "imbalance_size": "imb_size",
            "imbalance_buy_sell_flag": "imb_flag",
            "reference_price": "ref_price",
            "wap": "wa_price", 
        }
    )
    if "imb_size" in df.columns and "real_imb_size" not in df.columns:
        position = df.columns.get_loc("imb_size")
        df.insert(position + 1, "real_imb_size", df["imb_size"] * df["imb_flag"])
    return df

# ========================================================================================
# 3. Postprocessing Function for this competition
# ========================================================================================
def goto_conversion(listOfOdds, total=1, eps=1e-6, isAmericanOdds=False):

    # Convert American Odds to Decimal Odds
    if isAmericanOdds:
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i]
            isNegativeAmericanOdds = currOdds < 0
            if isNegativeAmericanOdds:
                currDecimalOdds = 1 + (100/(currOdds*-1))
            else: #Is non-negative American Odds
                currDecimalOdds = 1 + (currOdds/100)
            listOfOdds[i] = currDecimalOdds

    # Error Catchers
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2')
    if any(x < 1 for x in listOfOdds):
        raise ValueError('All odds must be >= 1, set isAmericanOdds parameter to True if using American Odds')

    # Computation
    listOfProbabilities = [1/x for x in listOfOdds] #initialize probabilities using inverse odds
    listOfSe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities] #compute the standard error (SE) for each probability
    step = (sum(listOfProbabilities) - total)/sum(listOfSe) #compute how many steps of SE the probabilities should step back by
    outputListOfProbabilities = [min(max(x - (y*step),eps),1) for x,y in zip(listOfProbabilities, listOfSe)]
    return outputListOfProbabilities

def zero_sum(listOfPrices, listOfVolumes):
    # Compute standard errors assuming standard deviation is same for all stocks
    listOfSe = [x**0.5 for x in listOfVolumes]
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices

