import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import psutil
import warnings
from colorama import Fore, Back, Style
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

# ========================================================================================
# Common Setup
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
def check_memory_usage(unit="gb", color=""):
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

# ========================================================================================
# General Plotting Functions
# ======================================================================================== 
def plot_scatterplot(df, column, column2, hue_column=None, figsize=(18, 10), ticksize=7, **kwargs):
    """
    Create a scatterplot to visualize the relationship between two variables from a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to be plotted.
    - column (str): The name of the column to be plotted on the x-axis.
    - column2 (str): The name of the column to be plotted on the y-axis.
    - hue_column (str, optional): The column to differentiate data points using colors (default is None).
    - figsize (tuple, optional): The size of the figure (width, height) in inches (default is (18, 10)).
    - ticksize (int, optional): The size of data points in the plot (default is 7).
    - **kwargs: Additional keyword arguments to customize the scatterplot.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=column, y=column2, hue=hue_column, style=hue_column, 
                    s=ticksize, legend="full", **kwargs) # palette="deep", 
    ax.set_title(f"Scatterplot of {column2} (y) against {column} (x)")
    if hue_column is not None:
        ax.legend()
    plt.show()
