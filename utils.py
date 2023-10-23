# ========================================================================================
# Import
# ========================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import psutil
import warnings
from colorama import Fore, Back, Style
from datetime import datetime, timedelta
from itertools import repeat
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

# ========================================================================================
# Master Config
# ========================================================================================
# 1. Clipping Config
# ========================================================================================
PRICE_CLIP_PERCENTILE = 0.01 # 1 - (2 * PRICE_CLIPPER_TAIL / 100) will remained unclip
VOLUME_CLIPPER_UPPER_TAIL = 0.01 # only the top VOLUME_CLIPPER_UPPER_TAIL proportion will be clipped

MIN_TARGET = -100 # Target lower than -100 will be clip to -100
MAX_TARGET = 100 # Target higher than 100 will be clip to 100

MILD_TARGET_LOWER_BOUND = -4.5
MILD_TARGET_UPPER_BOUND = 4.5

# ========================================================================================
# Settings functions
# ========================================================================================
# Suppress UserWarning
warnings.filterwarnings('ignore', category=UserWarning, message='No warning is required here')

# # Auto reload magic word
# %load_ext autoreload
# %autoreload

# ========================================================================================
# System functions
# ========================================================================================    
# Process.memory_info is expressed in bytes, so convert to megabytes
def check_memory_usage(unit="gb", color="green"):
    '''
    Check and print the RAM (memory) usage of the current process in the specified unit.

    Parameters:
        unit (str, optional): The unit in which to display the memory usage. Options are "kb" (kilobytes), "mb" (megabytes),
                              and "gb" (gigabytes). Default is "gb".
    '''
    unit_multipliers = {
        "kb": 2 ** 10,
        "mb": 2 ** 20,
        "gb": 2 ** 30,
    }
    unit = unit.strip().lower()
    if unit not in unit_multipliers:
        print(f"The unit provided is not available, available list of units: {list(unit_multipliers.keys())}, default to GB")
        unit = "gb"
    ram_used = psutil.Process().memory_info().rss / unit_multipliers[unit]
    cprint(f"RAM used: {ram_used:.1f} {unit.upper()}", color=color)
    return ram_used

def check_memory_by_global_variable(excludes="_", size_threshold=1):
    """
    Check the memory usage of global variables and return variables exceeding a specified size threshold.

    Parameters:
    - excludes (str, optional): Prefix to exclude variable names (default is "_").
    - size_threshold (float, optional): Minimum memory size threshold in megabytes (default is 1).

    Returns:
    - pandas DataFrame: A DataFrame containing variable names and their memory sizes (in MB)
      for variables exceeding the size threshold.

    This function checks the memory usage of global variables in the current Python session. It excludes variables whose
    names start with the specified 'excludes' prefix. The 'size_threshold' parameter defines the minimum memory size
    (in MB) for variables to be included in the result.
    """
    name_list = [name for name, var in globals().items() if not name.startswith("_")]
    size_list = [sys.getsizeof(globals()[name]) / (1024 ** 2) for name in name_list]
    var_size_df = pd.DataFrame(
        dict(variable_name=name_list, size_in_mb=size_list)
    ).sort_values(by="size_in_mb", ascending=False).reset_index(drop=True)
    var_size_df = var_size_df.loc[var_size_df["size_in_mb"] > size_threshold]
    return var_size_df    
    
def get_time_now():
    '''
    Get the current time in string format, for logging use
    '''
    return datetime.now().strftime("%H:%M:%S")

# ========================================================================================
# Common Functions
# ========================================================================================    
def list_diff(list1, list2, sort=False):
    """
    Compute the difference between two lists and optionally sort the result.

    Parameters:
    - list1 (list): The first list.
    - list2 (list): The second list.
    - sort (bool, optional): Whether to sort the result (default is False).

    Returns:
    - list: A list containing elements that are in 'list1' but not in 'list2'.
    """
    result = list(set(list1) - set(list2))
    if sort:
        result = sorted(result)
    return result

# ========================================================================================
# EDA check functions
# ======================================================================================== 
def check_auc(df, col, target_col="is_positive_target", verbose=0):
    """
    Calculate and check the Area Under the Receiver Operating Characteristic Curve (AUC) for a given column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the column for which to calculate the AUC.
    - target_col (str, optional): The name of the binary target column (default is "is_positive_target").
    - verbose (int, optional): Verbosity level for printing messages (0 for no messages, 1 for basic info, 2 for detailed info).

    Returns:
    - float: The AUC (Area Under the ROC Curve) for the specified column and target.

    This function calculates the AUC for a specified column and binary target column in a DataFrame. It handles cases
    where there are missing values in the specified column by dropping rows with missing values. If the AUC is less
    than 0.5, it calculates the effective AUC as 1 minus the AUC.

    If `verbose` is set to 1, the function prints basic information about the AUC value. If set to 2, it provides
    detailed information, including messages about data loss due to missing values and highlights low AUC values in red.
    """
    temp = df[[col, target_col]].dropna()
    proportion_drop = 1 - (temp.shape[0] / df.shape[0])
    if proportion_drop > 1e-5 and verbose:
        cprint(f"{proportion_drop:.3%} of the data dropped due to NA in {col}")
    auc = roc_auc_score(temp[target_col], temp[col].clip(-1e6, 1e6))
    if verbose:
        if auc < 0.5:
            cprint(f"The AUC of {col} on {target_col} is {auc:.4f} (effective AUC - {1-auc:.4f})", color="red")
        else:
            cprint(f"The AUC of {col} on {target_col} is {auc:.4f}", color="blue")
            
    if auc < 0.5:
        auc = (1 - auc)
    return auc, proportion_drop

# ========================================================================================
# Pandas processing functions
# ======================================================================================== 
# Get specific columns from dataframe
def get_cols(df, contains="", startswith="", endswith="", excludes="ColumnNamesThatWillNeverBeUsed", debug=False):
    """
    Get a list of column names from a DataFrame based on various filtering criteria.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to extract column names.
    contains (str or list of str, optional): Substrings that column names must contain (default is an empty string).
    startswith (str or list of str, optional): Prefixes that column names must start with (default is an empty string).
    endswith (str or list of str, optional): Suffixes that column names must end with (default is an empty string).
    excludes (str or list of str, optional): Column names to exclude from the selection (default is "ColumnNamesThatWillNeverBeUsed").
    debug (bool, optional): If True, print debugging information (default is False).

    Returns:
    list: A list of selected column names that meet the specified criteria.

    The function extracts a list of column names from the given DataFrame (df) based on the specified criteria. 
    You can filter columns by requiring them to contain specific substrings, start with certain prefixes, end with particular suffixes, 
    or exclude specific column names. The function returns the selected column names as a list.
    """
    if isinstance(contains, str):
        contains = [contains]
    if isinstance(startswith, str):
        startswith = [startswith]
    if isinstance(endswith, str):
        endswith = [endswith]
    if isinstance(excludes, str):
        excludes = [excludes]
    
    if debug:
        cprint(f"Contains {contains}, Startswith {startswith}, Endswith {endswith}, Excludes {excludes}", color="blue")
    if len(startswith) > 1:
        startswith = tuple(startswith)
    else:
        startswith = startswith[0]
    if len(endswith) > 1:
        endswith = tuple(endswith)
    else:
        endswith = endswith[0]
    dfc = df.columns
    selected_columns = dfc[
        dfc.str.startswith(startswith)
        & dfc.str.endswith(endswith)
        & dfc.str.contains("|".join(contains))
        & ~dfc.str.contains("|".join(excludes))
    ].tolist()
    return selected_columns

def downcast_to_32bit(df, excludes=[], verbose=1):
    """
    Downcast numeric columns in the DataFrame to 32-bit data types.

    Args:
        df (pandas.DataFrame): The input DataFrame to be modified.
        excludes (list, optional): A list of column names to exclude from downcasting.
        verbose (int, optional): Controls the verbosity of the process. 
            Set to 1 for progress bar, 0 to disable it.

    Returns:
        pandas.DataFrame: The modified DataFrame with downcasted columns.

    This function identifies numeric columns in the input DataFrame (float64 and int64) and
    downcasts their data types to 32-bit (float32 and int32) to save memory. Columns
    specified in the 'excludes' list are skipped.
    """
    if verbose:
        cprint("Before downcast: ", end="\t", color="green")
        check_memory_usage(color="green")
        
    float64_columns = df.dtypes[df.dtypes == "float64"].index.tolist()
    float64_columns = list_diff(float64_columns, excludes)
    
    int64_columns = df.dtypes[df.dtypes == "int64"].index.tolist()
    int64_columns = list_diff(int64_columns, excludes)
    
    for col in tqdm(float64_columns + int64_columns, disable=not verbose):
        if col in float64_columns:
            df[col] = df[col].astype(np.float32)
        elif col in int64_columns:
            df[col] = df[col].astype(np.int32)
            
    if verbose:
        cprint("After downcast: ", end="\t", color="blue")
        check_memory_usage(color="blue")
    
    return df

def my_power(series, factor):
    """
    Compute the element-wise power of a pandas Series with support for negative values (avoid domain error).

    Parameters:
    - series (pandas Series): The input pandas Series containing numeric values.
    - factor (float): The exponent to raise each element to.

    Returns:
    - pandas Series: A new Series with elements raised to the power of 'factor', considering negative values.

    This function calculates the element-wise power of a pandas Series, 'series', with an exponent 'factor'. It handles
    negative values by first taking the absolute value and then raising it to the power, ensuring that the sign is 
    retained in the result.
    """
    return np.where(
        series >= 0, 
        series ** factor, 
        -(series.abs() ** factor)
    )

def my_log(series):
    """
    Compute the element-wise natural logarithm of a pandas Series with support for negative values (avoid log negative error).

    Parameters:
    - series (pandas Series): The input pandas Series containing numeric values.

    Returns:
    - pandas Series: A new Series with elements transformed using the natural logarithm, considering negative values.

    This function calculates the element-wise natural logarithm (logarithm with base e) of a pandas Series, 'series'.
    It handles negative values by first taking the absolute value and then applying the natural logarithm, ensuring that
    the sign is retained in the result.
    """
    return np.where(
        series >= 0, 
        np.log1p(series), 
        -(np.log1p(series.abs()))
    )

def my_concat(df_list, n=4, reset_index=False):
    """
    Concatenate a list of pandas DataFrames into a single DataFrame while controlling the number of concatenations per operation.

    Parameters:
        df_list (list): A list of pandas DataFrames to concatenate.
        n (int, optional): The maximum number of DataFrames to concatenate in each operation. Default is 4.
        reset_index (bool, optional): If True, reset the index of the resulting DataFrame. Default is False.

    Returns:
        pandas.DataFrame: The concatenated DataFrame.

    Note:
        - You can adjust the value of 'n' based on memory constraints by specifying a lower value to reduce memory usage.
        - Lowering 'n' may lead to more concatenation operations but can help conserve memory when concatenating large DataFrames.
    """
    list_len = len(df_list)
    if list_len <= n:
        if reset_index:
            return pd.concat(df_list, ignore_index=reset_index)
        else:
            return pd.concat(df_list)
    else:
        chunk_size = (list_len // n) + 1
        mylist = []
        max_, i = 0, 0
        while max_ < list_len:
            min_ = i * chunk_size
            max_ = min((i+1) * chunk_size, list_len)
            i += 1
            mylist.append(my_concat(df_list[min_:max_]))
        return my_concat(mylist, reset_index=reset_index)
    
# ========================================================================================
# Logging functions
# ======================================================================================== 
fore_color_dict = {
    "green": Fore.GREEN,
    "blue": Fore.BLUE, 
    "red": Fore.RED,
    "yellow": Fore.YELLOW,
    "cyan": Fore.CYAN,
    "magenta": Fore.MAGENTA,
    "black": Fore.BLACK,
    "white": Fore.WHITE
}

def cprint(string, color=None, **kwargs):
    """
    Custom (similar to) print function but allows printing text in various colors using ANSI color codes.

    Parameters:
        string (str): The text to be printed.
        color (str, optional): The color in which to print the text. Default is None (no color).
        **kwargs: Additional keyword arguments that will be passed to the built-in print function.

    Usage:
        - To print text in a specific color, provide the 'color' parameter with one of the following color names:
          'green', 'blue', 'red', 'yellow', 'cyan', 'magenta', 'black', 'white'.
        - If an invalid color name is provided, a warning message will be printed in yellow, and the text will be printed in the default color.
    """
    if isinstance(string, list):
        string = ", ".join(string)
    if color is None:
        print(string, **kwargs)
    else:
        color_lower = color.strip().lower()
        try:
            color_in_fore = fore_color_dict[color_lower]
            print(f"{color_in_fore}{Style.BRIGHT}" + string + f"{Style.RESET_ALL}", **kwargs)
        except BaseException:
            cprint(f"Your color is not found, the available color is {list(fore_color_dict.keys())}", color="yellow")
            print(string, **kwargs)

# ========================================================================================
# Cheking Functions
# ======================================================================================== 
def calculate_psi(expected, actual, buckettype='bins', buckets=1000, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

# ========================================================================================
# Gradient Boosting Functions
# ======================================================================================== 
# Plot feature importances
def plot_feature_importance(features=None, importances=None, imp_df=None, title=None, limit=50, 
                            figsize=(16, 9), ascending=False, plot_chart=True, return_df=False):
    """
    Plot feature importances in a bar chart and optionally return the sorted DataFrame.

    Parameters:
    - features (list, optional): List of feature names (default is None).
    - importances (list, optional): List of feature importances (default is None).
    - imp_df (DataFrame, optional): DataFrame containing feature names and importances (default is None).
    - title (str, optional): Title for the bar chart (default is None).
    - limit (int, optional): Number of top features to display in the chart (default is 50).
    - figsize (tuple, optional): Size of the figure (width, height) in inches (default is (16, 9)).
    - ascending (bool, optional): Whether to sort feature importances in ascending order (default is False).
    - plot_chart (bool, optional): Whether to plot the feature importances as a bar chart (default is True).
    - return_df (bool, optional): Whether to return the sorted DataFrame (default is False).

    Returns:
    - None or DataFrame: If 'return_df' is True, returns the sorted DataFrame of feature importances.
    """
    if imp_df is None:
        imp_df = pd.DataFrame(dict(feature=features, feature_importance=importances))
        imp_df = imp_df.sort_values(by="feature_importance", ascending=ascending).reset_index(drop=True)
        imp_df["feature_importance_rank"] = imp_df["feature_importance"].rank(ascending=False, method="dense").astype(int)

    if limit:
        imp_plot = imp_df.iloc[:limit]
    else:
        imp_plot = imp_df
    
    if plot_chart:
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            x=imp_plot["feature_importance"], 
            y=imp_plot["feature"], 
            ax=ax
        )
        if title is not None:
            plt.title(title)
        plt.show()
    if return_df:
        return imp_df

def get_feature_summary(lgbm_model, df, clustering_threshold=1):
    """
    Get a summary of feature importance and feature clustering for a given LightGBM model and dataset.

    Parameters:
    - lgbm_model (lightgbm.Booster): A trained LightGBM model.
    - df (pd.DataFrame): The dataset containing the features.
    - clustering_threshold (float, optional): The threshold for clustering features based on their correlation (default is 1).

    Returns:
    - Tuple (pd.DataFrame, pd.DataFrame, np.ndarray): A tuple containing:
      - pd.DataFrame: A DataFrame summarizing feature importance, including features and importance scores.
      - pd.DataFrame: A DataFrame summarizing feature clustering, including features, cluster assignments, and correlation matrix.
      - np.ndarray: A linkage matrix resulting from hierarchical clustering.

    This function calculates the feature importance using the provided LightGBM model, and it clusters features based
    on their pairwise correlation using hierarchical clustering. The resulting summary DataFrames include feature names,
    importance scores, cluster assignments, and correlation information. It provides insights into both feature importance
    and inter-feature relationships.
    """
    feature_imp_df = plot_feature_importance(
        features=lgbm_model.feature_name(), importances=lgbm_model.feature_importance(), return_df=True, plot_chart=False
    )
    cprint(f"{get_time_now()} Calculating Feature Correlation...", color="green")
    feature_list = feature_imp_df["feature"].tolist()
    corr_df = df[feature_list].corr()
    
    # Assuming you have a correlation DataFrame named 'feature_correlations'
    # The linkage method and distance metric can be adjusted to suit your data
    cprint(f"{get_time_now()} Forming Feature Clusters...", color="green")
    linkage_matrix = sch.linkage(corr_df.values, method='ward', metric='euclidean')
    
    # Set a threshold or specify the number of clusters
    clusters = fcluster(linkage_matrix, clustering_threshold, criterion='distance')
    feature_cluster_df = pd.DataFrame(dict(feature=feature_list, clusters=clusters))
    
    feature_summary_df = feature_imp_df.merge(feature_cluster_df, on="feature", how="left")
    return feature_summary_df, corr_df, linkage_matrix  

# Inference on whole dataset, by batch
def lgbm_inference_by_batch(model, data, batch_size=8192, verbose=1):
    """
    Perform inference using a LightGBM model on a dataset, processing it in batches.

    Parameters:
    - model: A trained LightGBM model used for inference.
    - data: The dataset on which inference will be performed.
    - batch_size (int, optional): The size of each batch for processing the data (default is 8192).
    - verbose (int, optional): Verbosity level, where 1 displays a progress bar and 0 disables it (default is 1).

    Returns:
    - score_list (numpy.ndarray): An array of inference scores generated by the model.
    """
    score_list = []
    for i in tqdm(range(int(data.shape[0] / batch_size) + 1), disable=not verbose):
        try:
            score_list.extend(model.predict(
                data[model.feature_name()].iloc[int(i * batch_size): int((i+1) * batch_size)], 
            ))
        except:
            print("Too Many Batch")
    return np.array(score_list)

# ========================================================================================
# General Plotting Functions
# ======================================================================================== 
def plot_scatterplot(df, x_col, y_col, hue_col=None, figsize=(18, 10), ticksize=7, annotate=False, 
                     x_lower_thr=-np.inf, y_lower_thr=-np.inf, x_upper_thr=np.inf, y_upper_thr=np.inf, **kwargs):
    """
    Create a scatterplot to visualize the relationship between two variables from a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to be plotted.
    - column (str): The name of the column to be plotted on the x-axis.
    - column2 (str): The name of the column to be plotted on the y-axis.
    - hue_column (str, optional): The column to differentiate data points using colors (default is None).
    - figsize (tuple, optional): The size of the figure (width, height) in inches (default is (18, 10)).
    - ticksize (int, optional): The size of data points in the plot (default is 7).
    - annotate (bool, optional): Whether to annotate data points (default is False).
    - column_thr (float, optional): A threshold for annotating data points in the x-axis (default is positive infinity).
    - column2_thr (float, optional): A threshold for annotating data points in the y-axis (default is positive infinity).
    - **kwargs: Additional keyword arguments to customize the scatterplot.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, style=hue_col, s=ticksize, legend="full", **kwargs) # palette="deep", 
    ax.set_title(f"Scatterplot of {y_col} (y) against {x_col} (x)")
    
    if annotate:
        temp_df = df[[x_col, y_col]]
        temp_df = temp_df.loc[~(temp_df[x_col].between(x_lower_thr, x_upper_thr)) | ~(temp_df[y_col].between(y_lower_thr, y_upper_thr))]
        for idx, row in temp_df.iterrows():
            ax.annotate(idx, xy=(row[x_col], row[y_col]), xytext=(row[x_col], row[y_col]))
    if hue_col is not None:
        ax.legend()
    plt.show()

# ========================================================================================
# COMPETITION FUNCTIONS
# ========================================================================================
# 1. EDA Function for this competition
# ========================================================================================
def filter_df(df, stock_id=None, date_id=None, seconds=None, reset_index=False, meta_columns=["stock_id", "date_id", "seconds"]):
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
    for input_, input_arg_name in zip([stock_id, date_id, seconds], meta_columns):
        if input_arg_name not in df.columns:
            conds_dict[input_arg_name] = pd.Series(repeat(True, df.shape[0]))
        elif input_ is None:
            conds_dict[input_arg_name] = ~df[input_arg_name].isnull()
        elif isinstance(input_, (int, float, np.number)):
            conds_dict[input_arg_name] = (df[input_arg_name].between(int(input_) - 1e-6, int(input_) + 1e-6))
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
def clean_df(df, columns_to_drop=['row_id', 'time_id'], verbose=0):
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
    4. Update the column 'imb_size' by multiplying 'imb_size' with 'imb_flag' if 'imb_size' exists in the DataFrame.

    Parameters:
        - df: The input DataFrame to be cleaned.
        - columns_to_drop: A list of column names to be dropped. Default columns include 'row_id' and 'time_id'.

    Returns:
        A cleaned and transformed DataFrame ready for further analysis.
    """
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df = downcast_to_32bit(df, verbose=verbose)
    df = df.rename(
        columns={
            "seconds_in_bucket": "seconds",
            "imbalance_size": "imb_size",
            "imbalance_buy_sell_flag": "imb_flag",
            "reference_price": "ref_price",
            "wap": "wa_price", 
        }
    )
    
    # if "imb_size" in df.columns and "real_imb_size" not in df.columns:
    #     position = df.columns.get_loc("imb_size")
    #     df.insert(position + 1, "real_imb_size", df["imb_size"] * df["imb_flag"])
    
    ## I don't think the absolute magnitude is useful, we can replace it first
    ## if we want to get the magnitude of raw imb volume, I can always take the abs() later
    if "imb_size" in df.columns:
        df["imb_size"] = df["imb_size"] * df["imb_flag"]
        
    return df

def get_price_clippers(df, price_cols, price_clip_percentile=PRICE_CLIP_PERCENTILE):
    """
    Calculate price clipping bounds for specified price columns in a DataFrame.

    This function computes the lower and upper clipping bounds for each of the specified price columns
    in the given DataFrame. Price clipping is a statistical technique used to remove outliers from
    the data, ensuring that extreme values do not unduly influence the analysis.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - price_cols (list): A list of column names for which clipping bounds are calculated.
    - price_clip_percentile (float, optional): The percentile range for clipping (default is PRICE_CLIP_PERCENTILE).

    Returns:
    - price_clippers (dict): A dictionary where keys are price column names, and values are tuples
      representing the lower and upper clipping bounds for each price column.
    """
    price_clippers = {}
    for price_col in price_cols:
        upper_bound = np.percentile(df[price_col].dropna(), 100 - price_clip_percentile)
        lower_bound = np.percentile(df[price_col].dropna(), price_clip_percentile)
        price_clippers[price_col] = (lower_bound, upper_bound)
        cprint(f"For {price_col}, the global clip bound is", end=" ", color="blue")
        cprint(f"({lower_bound:.4f}, {upper_bound:.4f})", color="green")
    return price_clippers

def get_volume_clippers(df, volume_cols, volume_clip_upper_percentile=VOLUME_CLIPPER_UPPER_TAIL):
    """
    Generate volume clippers for specified columns in a DataFrame.

    This function calculates upper clip bounds for each specified column based on a given
    percentile of the data distribution. It returns a dictionary where keys are column names,
    and values are tuples representing the clip bounds. The lower bound is always negative
    infinity, and the upper bound is determined by the specified percentile of the column data.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to be clipped.
        volume_cols (list): A list of column names for which volume clippers should be generated.
        volume_clip_upper_percentile (float, optional): The percentile for the upper clip bound.
            It should be a value between 0 and 100. Default is VOLUME_CLIPPER_UPPER_TAIL.

    Returns:
        dict: A dictionary where keys are column names, and values are tuples representing
        the clip bounds in the format (-inf, upper_bound).
    """
    volume_clippers = {}
    for volume_col in volume_cols:
        upper_bound = int(round(np.percentile(df[volume_col].dropna(), 100 - volume_clip_upper_percentile), -3))
        volume_clippers[volume_col] = (-np.inf, upper_bound)
        cprint(f"For {volume_col}, the global clip bound is", end=" ", color="blue")
        cprint(f"(-inf, {upper_bound:,.0f})", color="green")
    return volume_clippers

def clip_df(df, price_clippers=None, volume_clippers=None):
    """
    Clip specified columns in a DataFrame to specified bounds.

    This function clips the columns in a given DataFrame based on provided clipper bounds.
    It can be used to clip price and volume columns and, if applicable, target columns.
    If no clipper bounds are provided, it calculates default bounds using helper functions.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be processed.
        price_clippers (dict, optional): A dictionary containing clipper bounds for price columns.
            If not provided, default bounds are calculated using get_price_clippers.
        volume_clippers (dict, optional): A dictionary containing clipper bounds for volume columns.
            If not provided, default bounds are calculated using get_volume_clippers.

    Returns:
        pandas.DataFrame: A DataFrame with specified columns clipped to the specified bounds.

    Notes:
        - Columns to be clipped are determined by their names: price columns (ending with "price"),
          volume columns (ending with "size"), and flag columns (ending with "flag").
        - Target columns are clipped if a column named "target" exists in the DataFrame.
          Additional transformed binary target columns are created for feature analysis and
          potential model support.
    """
    # Find the column list for each group
    price_cols = get_cols(df, endswith="price")
    volume_cols = get_cols(df, endswith="size")
    flag_cols = get_cols(df, endswith="flag")
    base_cols = price_cols + volume_cols + flag_cols
    
    if price_clippers is None:
        price_clippers = get_price_clippers(df, price_cols)
    
    if volume_clippers is None:
        volume_clippers = get_volume_clippers(df, volume_cols)
    
    # Clip price columns
    for price_col in price_cols:
        df[price_col] = df[price_col].clip(*price_clippers[price_col])
        
    # Clip volume columns
    for volume_col in volume_cols:
        df[volume_col] = df[volume_col].clip(*volume_clippers[volume_col])
    
    # Clip target columns (if applicable)
    if "target" in df.columns:
        df["clipped_target"] = df["target"].clip(MIN_TARGET, MAX_TARGET)
        # This 2 transformed binary targets are for the feature analysis and potentially for supporting model(s)
        df["is_positive_target"] = (df["target"] > 0).astype(int)
        df["is_mild_target"] = df["target"].between(MILD_TARGET_LOWER_BOUND, MILD_TARGET_UPPER_BOUND).astype(int)
    return df

def calc_robust_scale(master_df, df, base_col, groupby, log=False):
    """
    Calculate and add robust scaled values to a master DataFrame.

    Parameters:
    - master_df (DataFrame): The master DataFrame to which the robust scaled values will be added.
    - df (DataFrame): The DataFrame containing data used for scaling.
    - base_col (str): The name of the column to be scaled.
    - groupby (str): The name of the column used for grouping data.
    - log (bool, optional): If True, apply logarithmic transformation to the data before scaling (default is False).

    Returns:
    - master_df (DataFrame): The master DataFrame with the robust scaled values added.
    """
    if log:
        temp = my_log(df[base_col])
    else:
        temp = df[base_col]
    # rbt sc == Robust Scaled
    master_df[f"{base_col}_{groupby}_rbtsc"] = (temp - df[f"{base_col}_{groupby}_median"]) / (df[f"{base_col}_{groupby}_pct75"] - df[f"{base_col}_{groupby}_pct25"])
    return master_df

def calc_std_scale(master_df, df, base_col, groupby, log=False):
    """
    Calculate and add standard scaled values to a master DataFrame.

    Parameters:
    - master_df (DataFrame): The master DataFrame to which the standard scaled values will be added.
    - df (DataFrame): The DataFrame containing data used for scaling.
    - base_col (str): The name of the column to be scaled.
    - groupby (str): The name of the column used for grouping data.
    - log (bool, optional): If True, apply logarithmic transformation to the data before scaling (default is False).

    Returns:
    - master_df (DataFrame): The master DataFrame with the standard scaled values added.
    """
    if log:
        temp = my_log(df[base_col])
    else:
        temp = df[base_col]
    # std sc == Standard Scaled
    master_df[f"{base_col}_{groupby}_stdsc"] = (temp - df[f"{base_col}_{groupby}_mean"]) / df[f"{base_col}_{groupby}_std"]
    return master_df

def scale_base_columns(df, _level_stats_df, base_columns, level_col="stock", join_col="stock_id", verbose=0):
    """
    Scale and transform specified base columns in a DataFrame using statistical data from another DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame that contains the base columns to be scaled.
    - _level_stats_df (DataFrame): A DataFrame containing statistical data used for scaling.
    - base_columns (list): A list of column names in the input DataFrame to be scaled.
    - level_col (str, optional): The name of the column used for grouping data (default is "stock").
    - join_col (str, optional): The name of the column used for merging the input and statistics DataFrames (default is "stock_id").
    - verbose (int, optional): Verbosity level, where 0 means no progress bar (default is 0).

    Returns:
    - df (DataFrame): The input DataFrame with scaled and transformed columns.
    """
    for col in tqdm(base_columns, disable=not verbose):
        if col.endswith("_size"):
            log = True
        else:
            log = False
        df_subset = df.loc[:, ["stock_id", "date_id", "seconds"] + [col]]
        
        # if isinstance(join_col, list):
        #     for col in join_col:
        #         if col not in _level_stats_df.columns:
        #             _level_stats_df = _level_stats_df.reset_index()
        #             break
        # else:
        #     if join_col not in _level_stats_df.columns:
        #         _level_stats_df = _level_stats_df.reset_index()
                
        df_subset = df_subset.merge(
            _level_stats_df[get_cols(_level_stats_df, startswith=col)].reset_index(), on=join_col, how="left"
        )
        df = calc_robust_scale(df, df_subset, base_col=col, groupby=level_col, log=log)
        df = calc_std_scale(df, df_subset, base_col=col, groupby=level_col, log=log)
    return df

# ========================================================================================
# 3. Postprocessing Function for this competition
# ========================================================================================
def goto_conversion(listOfOdds, total=1, eps=1e-6, isAmericanOdds=False):
    """
    Convert a list of odds to probabilities, taking into account the total probability and whether the odds are in American format.

    Parameters:
    - listOfOdds (list): A list of odds to be converted to probabilities.
    - total (float, optional): The desired total probability (default is 1).
    - eps (float, optional): A small epsilon value to prevent division by zero (default is 1e-6).
    - isAmericanOdds (bool, optional): Set to True if the odds are in American format; otherwise, False (default is False).

    Returns:
    - outputListOfProbabilities (list): A list of probabilities corresponding to the input odds.

    Raises:
    - ValueError: If the length of listOfOdds is less than 2 or if any odds are less than 1 (when using non-American odds).
    """
    # Convert American Odds to Decimal Odds
    if isAmericanOdds:
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i]
            isNegativeAmericanOdds = currOdds < 0
            if isNegativeAmericanOdds:
                currDecimalOdds = 1 + (100/(currOdds*-1))
            # Is non-negative American Odds
            else: 
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
    """
    Adjust a list of prices to have a zero-sum while considering associated volumes.

    Parameters:
    - listOfPrices (list): A list of prices for a set of stocks.
    - listOfVolumes (list): A list of corresponding volumes for the same set of stocks.

    Returns:
    - outputListOfPrices (list): A list of adjusted prices with a zero-sum.
    """
    # Compute standard errors assuming standard deviation is same for all stocks
    listOfSe = [x**0.5 for x in listOfVolumes]
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices