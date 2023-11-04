# ========================================================================================
# Import
# ========================================================================================
import gc
import joblib
import lightgbm as lgb
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
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=False, nb_workers=12)

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.stats import pearsonr, chi2_contingency

import lightgbm as lgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, log_evaluation, early_stopping

# ========================================================================================
# Master Config
# ========================================================================================
# 0. Constants
# ========================================================================================
META_COLUMNS = ["stock_id", "date_id", "seconds"]

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
def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'pct{:02.0f}'.format(n*100)
    return percentile_

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

def read_data(path='', columns=None, **kwargs):
    """
    Read data from various file formats and return it as a DataFrame.

    Parameters:
        path (str): The path to the data file to be read.
        columns (list, optional): A list of column names to select from the data (default is None).
        **kwargs: Additional keyword arguments specific to the data reading method.

    Returns:
        pd.DataFrame or str: A Pandas DataFrame containing the data, or "Nothing" if the file format is unknown.
    """
    if path.endswith(".parquet"):
        if columns is not None:
            data = pd.read_parquet(path, columns=columns, **kwargs)
        else:
            data = pd.read_parquet(path, **kwargs)
    elif path.endswith(".ftr"):
        data = pd.read_feather(path, **kwargs)
    elif path.endswith(".csv"):
        data = pd.read_csv(path, **kwargs)
    elif path.endswith(".pkl"):
        data = joblib.load(path, **kwargs)
    else:
        print("Unknown file format")
        data = "Nothing"
    return data

def read_model(filepath=''):
    filename = filepath.split("/")[-1]
    if filename.startswith("lgbm") and filename.endswith(".txt"):
        return lgb.Booster(model_file=filepath)
    elif filename.endswith(".cbm"):
        # Because this competition using Regressor, this is hard coded
        return CatBoostRegressor().load_model(filepath)
    else:
        return "X tau apa model format ini"

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

def check_target_mean(df, feature_col, target_col="target", feature_class=20, target_class=5, plot_chart=True):
    feature_class = min(df[feature_col].nunique(), feature_class)
    df[f"{feature_col}_bins"] = pd.qcut(df[feature_col], q=feature_class, duplicates="drop").cat.codes.replace(-1, np.nan)
    
    df.groupby(f"{feature_col}_bins")[target_col].mean().plot()
    plt.show()

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
        -(np.log1p(np.abs(series)))
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
# Validate Feature Functions
# ======================================================================================== 
def check_target_dependency(df, feature_col, target_col="target", feature_class=50, target_class=10, 
                            lower_bound=-1e10, upper_bound=1e10, 
                            plot_chart=True, return_table=True, conduct_chi_square_test=True, verbose=0):
    """
    Analyze the dependency between a feature and a target variable by creating a contingency table.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        feature_col (str): The name of the column containing the feature variable.
        target_col (str, optional): The name of the column containing the target variable (default is "target").
        feature_class (int, optional): The number of bins for the feature variable (default is 20).
        target_class (int, optional): The number of bins for the target variable (default is 5).
        plot_chart (bool, optional): Whether to plot a heatmap of the contingency table (default is True).
        return_table (bool, optional): Whether to return the contingency table (default is True).
        conduct_chi_square_test (bool, optional): Whether to conduct a chi-square test on the contingency table (default is True).

    Returns:
        pd.DataFrame or tuple: Depending on the parameters, returns a contingency table DataFrame
        (if 'return_table' is True), or a tuple containing chi-squared test statistics
        (chi2, p_value, dof) if 'return_table' is False and 'conduct_chi_square_test' is True.

    The function calculates the contingency table between 'feature_col' and 'target_col', optionally plots a heatmap of the table,
    and can conduct a chi-square test to assess the statistical significance of the relationship.
    """
    if df[feature_col].nunique() < 300:
        feature_class = df[feature_col].nunique()
        
    # Clip to avoid infinity, then bin the feature
    feature_series = df[feature_col].clip(lower_bound, upper_bound)
    feature_group = pd.qcut(feature_series, q=feature_class, duplicates="drop").cat.codes.replace(-1, np.nan)
    
    # Bin the target
    target_group = pd.qcut(df[target_col], q=target_class, duplicates="drop").cat.codes.replace(-1, np.nan)
    
    table = pd.crosstab(feature_group, target_group)
    table.index.name = f"{feature_col}_by_{feature_class}_bins"
    table.columns.name = f"target_by_{target_class}_bins"
    
    if plot_chart:
        sns.heatmap(table, cmap="coolwarm", fmt=".0f", annot=True)
        plt.show()

    if conduct_chi_square_test:
        chi2, p_value, dof, expected = chi2_contingency(table)
        if verbose:
            print(chi2, p_value, dof)
        
    if return_table:
        return table, chi2, p_value, dof, expected
    else:
        return chi2, p_value, dof, expected

def run_chi_square_tests(df, feature_columns=None, target_col="target", feature_class=50, target_class=10, min_log_p=-745, plot_chart=False, return_table=True):
    """
    Perform chi-square tests for feature-target independence and return log-transformed p-values.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing data to analyze.
        feature_columns (list, optional): A list of feature column names to analyze. If None, it's inferred from the DataFrame (default is None).
        target_col (str, optional): The name of the target column (default is "target").
        feature_class (int, optional): The number of classes (bins) for the feature variable (default is 50).
        target_class (int, optional): The number of classes (bins) for the target variable (default is 10).
        min_log_p (float, optional): The minimum log-transformed p-value to clip values below (default is -400).
        plot_chart (bool, optional): Whether to plot a bar chart of log-transformed p-values (default is False).
        return_table (bool, optional): Whether to return log-transformed p-values in a DataFrame (default is True).

    Returns:
        pd.DataFrame or tuple: If return_table is True, returns a DataFrame containing log-transformed p-values for each feature.
        If plot_chart is True, also plots a bar chart of log-transformed p-values.
    """
    # -745 is the magic number because np.exp(-745) is betul betul 0 in my Python
    if feature_columns is None:
        feature_columns = list_diff(df.columns, ["stock_id", "date_id", "clipped_target", "target", "is_positive_target", "is_mild_target"])
    
    log_p_values_dict = {}
    for column_name in tqdm(feature_columns):
        log_p_values = []
        for stock_id in range(200):
            chi2, p_value, dof, expected = check_target_dependency(
                filter_df(df, stock_id=stock_id), feature_col=column_name, feature_class=feature_class, target_class=target_class, 
                plot_chart=False, conduct_chi_square_test=True, return_table=False
            )
            log_p_values.append(np.log(p_value))

        log_p_values_dict[column_name] = log_p_values

    log_chi_square_p_df = pd.DataFrame(log_p_values_dict)
    log_chi_square_p_df = log_chi_square_p_df.replace(-np.inf, min_log_p)
    log_chi_square_p_df["stock_id"] = range(200)
    log_chi_square_p_df = log_chi_square_p_df.set_index("stock_id")
    log_chi_square_p_median = log_chi_square_p_df.median().sort_values()[::-1]
    
    if plot_chart:
        plt.figure(figsize=(17, 6))
        log_chi_square_p_median.plot.barh()
        plt.show()
        
    if return_table:
        return log_chi_square_p_df, log_chi_square_p_median
    
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
def get_lgbm_dataset(df, start_date, end_date, feature_list, target="target", free_raw_data=True):
    """
    Create a LightGBM dataset for training or testing with the specified data.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        start_date (int): The start date for filtering the data.
        end_date (int): The end date for filtering the data.
        feature_list (list): A list of feature column names.
        target (str, optional): The name of the target column (default is "target").
        free_raw_data (bool, optional): Whether to keep the raw data free after creating the dataset (default is True).

    Returns:
        lgb.Dataset: A LightGBM dataset ready for training or testing.
    """
    df_subset = filter_df(df, date_id=(start_date, end_date))
    data = lgb.Dataset(
        df_subset.loc[:, feature_list], 
        df_subset[target],
        free_raw_data=free_raw_data
    )
    return data

# Define the custom evaluation metric for MAE
def lgbm_feval_mae(preds, train_data):
    """
    Custom evaluation metric for Mean Absolute Error (MAE) in LightGBM.

    Parameters:
        preds (array-like): Predicted values from the model.
        train_data (lgb.Dataset): Training data containing labels.

    Returns:
        tuple: A tuple containing the metric name ('mae'), the calculated MAE value, and a flag (False).
    """
    labels = train_data.get_label()
    mae = np.abs(preds - labels).mean()
    return 'mae', mae, False

def train_lgbm(data, feature_list, lgbm_params, train_target="clipped_target", train_start_date=0, train_end_date=420, val_start_date=421, val_end_date=480, 
               es=True, eval_freq=100, get_val_pred=True):
    """
    Train a LightGBM model with the specified data and parameters.

    Parameters:
        data (pd.DataFrame): The input data containing features and target variables.
        feature_list (list): A list of feature column names.
        lgbm_params (dict): LightGBM model hyperparameters.
        train_target (str, optional): The name of the target column for training (default is "clipped_target").
        train_start_date (int, optional): The start date for training data (default is 0).
        train_end_date (int, optional): The end date for training data (default is 420).
        val_end_date (int, optional): The end date for validation data (default is 480).
        eval_freq (int, optional): Evaluation frequency during training (default is 100).
        get_val_pred (bool, optional): Whether to obtain validation predictions (default is True).

    Returns:
        tuple: A tuple containing the trained LightGBM model, a DataFrame with validation predictions (if requested), and the best MAE score achieved during training.
    """
    cprint(f"{get_time_now()} Preparing Dataset...", color="green")
    train_data = get_lgbm_dataset(data, start_date=train_start_date, end_date=train_end_date, feature_list=feature_list, target=train_target) 
    valid_data = get_lgbm_dataset(data, start_date=val_start_date, end_date=val_end_date, feature_list=feature_list, target="target", free_raw_data=False)
    
    cprint(f"{get_time_now()} Training...", color="green")
    callbacks = [log_evaluation(eval_freq)]
    
    if es:
        callbacks += [early_stopping(eval_freq, first_metric_only=True, verbose=True, min_delta=5e-5)]
        
    model = lgb.train(
        params=lgbm_params,
        train_set=train_data, 
        valid_sets=[valid_data, train_data], 
        feval=lgbm_feval_mae, 
        # categorical_feature=["stock_id"],
        callbacks=callbacks
    )
    best_score = model.best_score["valid_0"]["l1"]
    del train_data
    gc.collect()
    
    if get_val_pred:
        cprint(f"{get_time_now()} Getting Validation Prediction...", color="green")
        val_df = filter_df(data, date_id=(train_end_date + 1, val_end_date))[META_COLUMNS].reset_index(drop=True)
        val_df["val_pred"] = model.predict(valid_data.get_data())
    else:
        val_df = pd.DataFrame()
        
    del valid_data
    
    return model, val_df, best_score

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

def plot_heatmap(df, figsize=(15, 8), annot=False, fmt='.3g'):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data=df, annot=annot, cmap="coolwarm", fmt=fmt)
    plt.show()    
    
# ========================================================================================
# COMPETITION FUNCTIONS
# ========================================================================================
# 1. Preprocessing Functions for this competition
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

def sort_df(df, sort_by=META_COLUMNS, reset_index=True):
    """
    Sort a DataFrame based on specified columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to be sorted.
        sort_by (list, optional): A list of column names to sort the DataFrame by (default is META_COLUMNS).
        reset_index (bool, optional): Whether to reset the index after sorting (default is True).

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    df = df.sort_values(by=sort_by)
    gc.collect()
    if reset_index:
        return df.reset_index(drop=True)
    else:
        return df

def clean_df(df, missing_stock_dates=None, columns_to_drop=['row_id', 'time_id'], drop_null=False, verbose=0):
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
    if "wap" in df.columns:
        df = df.rename(
            columns={
                "seconds_in_bucket": "seconds",
                "imbalance_size": "imb_size",
                "imbalance_buy_sell_flag": "imb_flag",
                "reference_price": "ref_price",
                "wap": "wa_price", 
            }
        )
    # Remove rows with missing imb_size (it can be any price / volume columns)
    # The assumption here is one missing => whole stock-date missing
    if missing_stock_dates is not None:
        null_indices = []
        for stock_id, date_id in missing_stock_dates:
            date_id = date_id[0]
            null_imb_size_index = df.loc[(df["stock_id"] == stock_id) & (df["date_id"] == date_id)].index.tolist()
            null_indices.extend(null_imb_size_index)
            
        df = df.drop(null_indices, axis=0, errors="ignore").reset_index(drop=True)
    
    # I don't think the absolute magnitude is useful, we can replace it first
    # if we want to get the magnitude of raw imb volume, we can always take the abs() later
    if "imb_size" in df.columns and df["imb_size"].min() >= 0:
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
        upper_bound = int(round(np.percentile(df[volume_col].dropna(), 100 - volume_clip_upper_percentile), -1))
        volume_clippers[volume_col] = (-upper_bound, upper_bound)
        cprint(f"For {volume_col}, the global clip bound is", end=" ", color="blue")
        cprint(f"(-{upper_bound:,.0f}, {upper_bound:,.0f})", color="green")
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
    
    # if price_clippers is None:
    #     price_clippers = get_price_clippers(df, price_cols)
    
    # if volume_clippers is None:
    #     volume_clippers = get_volume_clippers(df, volume_cols)
    
    # # Clip price columns
    # for price_col in price_cols:
    #     df[price_col] = df[price_col].clip(*price_clippers[price_col])
        
    # # Clip volume columns
    # for volume_col in volume_cols:
    #     df[volume_col] = df[volume_col].clip(*volume_clippers[volume_col])
    
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
# 2. Cross Sectional Features Functions
# ========================================================================================

# ========================================================================================
# 3. Temporal Features Functions
# ========================================================================================
def increment_date_id(df, reset_index=True):
    if "date_id" not in df.columns:
        index = list(df.index.names)
        if "date_id" in index:
            df = df.reset_index()
            df["date_id"] += 1
            df = df.set_index(index)
        return df
    else:
        df["date_id"] += 1
        return df

def calc_intraday_gradient(intraday_array):
    """
    Calculate the intraday gradient (slope) of a time series.

    Parameters:
        intraday_array (array-like): An array containing the values of a time series for a single intraday period.

    Returns:
        float: The calculated gradient (slope) of the time series, indicating the rate of change.

    The function computes the gradient of the given time series, which represents the rate of change or slope of the values
    over time. It uses linear regression to fit a line to the data and returns the gradient of that line. A positive gradient
    indicates an upward trend, while a negative gradient indicates a downward trend in the time series.
    """
    return np.polyfit(range(len(intraday_array)), intraday_array, 1)[0]

def calc_ma_features(df, columns, groupby=["stock_id"], window_sizes=[2]):
    """
    Calculate moving average features for specified columns within a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list of str): The columns for which moving average features will be calculated.
        groupby (list of str, optional): The columns to group by before calculating moving averages (default is ["stock_id"]).
        window_sizes (list of int, optional): The window sizes for the moving averages (default is [2]).

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns containing moving average features.

    This function calculates moving averages for the specified columns within the DataFrame, grouped by the specified
    columns (e.g., 'stock_id'). It computes moving averages with different window sizes and adds new columns to the
    DataFrame to store the results. The column names for moving averages are constructed using the original column names
    and the window size (e.g., 'column_name_ma2' for a 2-day moving average).
    """
    # For all col in columns, calculate the moving average of window size k
    for k in window_sizes:
        df[[f"{col}_ma{k}" for col in columns]] = (
            df.groupby(groupby)[columns].rolling(k).mean()
        ).values
    return df

def calc_diff_features(df, columns, groupby=["stock_id"], lag_distances=[1]):
    """
    Calculate lag difference features for specified columns within a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list of str): The columns for which lag difference features will be calculated.
        groupby (list of str, optional): The columns to group by before calculating lag differences (default is ["stock_id"]).
        lag_distances (list of int, optional): The lag distances for calculating differences (default is [1]).

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns containing lag difference features.

    This function calculates lag difference features for the specified columns within the DataFrame, grouped by the
    specified columns (e.g., 'stock_id'). It computes differences between values at lag distances and adds new columns
    to the DataFrame to store the results. The column names for lag differences are constructed using the original column
    names and the lag distance (e.g., 'column_name_ld1' for a lag difference of 1).
    """
    # For all col in columns, calculate the lag difference (Lag 0 - Lag d) values
    # When d = 1, this is same as first order difference
    for d in lag_distances:
        df[[f"{col}_ld{d}" for col in columns]] = (
            df.groupby(groupby)[columns].diff(d)
        ).values
    return df

def calc_pct_chg_features(df, columns, groupby=["stock_id"], lag_distances=[1]):
    """
    Calculate percentage change features for specified columns within a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list of str): The columns for which percentage change features will be calculated.
        groupby (list of str, optional): The columns to group by before calculating percentage changes (default is ["stock_id"]).
        lag_distances (list of int, optional): The lag distances for calculating percentage changes (default is [1]).

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns containing percentage change features.
    """
    # For all col in columns, calculate the lag ratio difference (Lag 0 / Lag d) values
    for d in lag_distances:
        df[[f"{col}_pc{d}" for col in columns]] = (
            df.groupby(groupby)[columns].pct_change(d)
        ).values
    return df

def calc_interday_gradient_features(df, columns, groupby=["stock_id"], lookback_periods=[5], verbose=0):
    """
    Calculate interday gradient features for specified columns within a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list of str): The columns for which interday gradient features will be calculated.
        groupby (list of str, optional): The columns to group by before calculating interday gradients (default is ["stock_id"]).
        lookback_periods (list of int, optional): The lookback periods for calculating gradients (default is [3, 6]).

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns containing interday gradient features.

    This function calculates interday gradient features for the specified columns within the DataFrame, grouped by the
    specified columns (e.g., 'stock_id'). It computes gradients over specified lookback periods and adds new columns
    to the DataFrame to store the results. The column names for interday gradients are constructed using the original
    column names and the lookback period (e.g., 'column_name_3days_gradient' for a 3-day interday gradient).
    """
    gradients_dict = dict()
    for column in tqdm(columns, disable=not verbose):
        for lookback_period in lookback_periods:
            gradient_list = []
            for i, w in enumerate(df.groupby(groupby)[column].rolling(lookback_period)):
                if len(w) < lookback_period:
                    gradient_list.append(np.nan)
                else:
                    gradient, intercept = np.polyfit(range(len(w)), w, 1)
                    gradient_list.append(gradient)
            gradients_dict[f"{column}_{lookback_period}d_grad"] = gradient_list
            
    gradients_df = pd.DataFrame(gradients_dict)
    df[gradients_df.columns] = gradients_df.values
    return df

# ========================================================================================
# 4. Master feature engineering functions for this competition
# ========================================================================================
def get_master_daily_target_data(master_df, missing_stock_dates, groupby=["stock_id", "date_id"], verbose=0):
    # Sort the dataframe first
    master_df = master_df.sort_values(by=META_COLUMNS).reset_index(drop=True)
    
    master_daily_target_data = []
    for round_, seconds_period in tqdm(enumerate([(0, 540), (0, 290), (300, 540)]), disable=not verbose):
        # Filter dataframe
        master_subset = filter_df(master_df, seconds=seconds_period)
        
        # Calc the gradients for the intraday array
        daily_target_grads = master_subset.groupby(groupby)["target"].apply(calc_intraday_gradient).apply(pd.Series)
        daily_target_grads.columns = [f"target_r{round_}_grad"]
        daily_target_grads.drop(columns=f"target_r{round_}_intercept", errors="ignore", inplace=True)
        
        # Compute the agg statistics of target variable for every day
        daily_target_stats = master_subset.groupby(groupby)["target"].agg(["mean", "std", "min", "max"]).add_prefix(f"daily_target_r{round_}_")

        # Compute the first order difference then the agg statistics (only mean is good I think) of target variable for every day
        master_subset["target_fod"] = master_subset.groupby(groupby)["target"].diff(1)
        daily_target_fod_stats = master_subset.groupby(groupby)["target_fod"].agg(["mean"]).add_prefix(f"target_fod_r{round_}_")

        master_subset["target_sod"] = master_subset.groupby(groupby)["target_fod"].diff(1)
        daily_target_sod_stats = master_subset.groupby(groupby)["target_sod"].agg(["mean"]).add_prefix(f"target_sod_r{round_}_")

        # Horizontally stack all the dataframes above
        daily_target_data = pd.concat([daily_target_grads, daily_target_stats, daily_target_fod_stats, daily_target_sod_stats], axis=1)
        
        # Append this dataframe for this master df subset into list
        master_daily_target_data.append(daily_target_data)
    
    master_daily_target_data = pd.concat(master_daily_target_data, axis=1)
    master_daily_target_data = master_daily_target_data.reset_index()
    
    # Fill in the missing entries
    for stock_id, date_id in missing_stock_dates:
        master_daily_target_data.loc[master_daily_target_data.shape[0]] = (
            filter_df(master_daily_target_data, stock_id=stock_id, date_id=(date_id - 1, date_id + 1)).mean(axis=0)
        )
    
    # Sort the imputed rows to above, and cast back meta columns to integer and set them as index
    master_daily_target_data = master_daily_target_data.sort_values(by=groupby).reset_index(drop=True)
    master_daily_target_data[groupby] = master_daily_target_data[groupby].astype(np.int32)
    master_daily_target_data = master_daily_target_data.set_index(groupby)
        
    return master_daily_target_data

def generate_interday_target_features(daily_target_data, merge_gt=False, shift=True, verbose=0):
    # if shift:
    #     # Shift the target agg value to avoid future data leakage
    #     lag_daily_target_data = daily_target_data.shift(1)
    # else:
    lag_daily_data = daily_target_data
    
    # Generate features for intraday gradient columns
    gradient_columns = get_cols(lag_daily_data, endswith="grad")
    lag_daily_data = calc_diff_features(lag_daily_data, columns=gradient_columns, groupby=["stock_id"], lag_distances=[1])
    lag_daily_data = calc_ma_features(lag_daily_data, columns=gradient_columns, groupby=["stock_id"], window_sizes=[2])
    
    # # Generate features for intraday standard deviation columns
    # std_columns = get_cols(lag_daily_target_data, endswith="std")
    # lag_daily_target_data = calc_pct_chg_features(lag_daily_target_data, columns=std_columns, groupby=["stock_id"], lag_distances=[1])
    
    # Generate interday gradient features for intraday agg stat columns
    if verbose:
        cprint(f"Generating Interday gradient features for each column...", color="blue")
    agg_stat_columns = get_cols(lag_daily_data, endswith=["mean", "min", "max", "std"], excludes=["ground_truth", "fod", "sod"])
    lag_daily_data = calc_interday_gradient_features(lag_daily_data, agg_stat_columns, groupby=["stock_id"], lookback_periods=[3, 6], verbose=verbose)
    
    # We are using data from N days ago to today for building features, but we can only use it for tmr, so we increment date_id by 1
    lag_daily_data = lag_daily_data.reset_index()
    lag_daily_data["date_id"] += 1
    lag_daily_data = lag_daily_data.set_index(["stock_id", "date_id"])
    
    # Merge the target mean per day for feature analysis
    if merge_gt:
        lag_daily_data = lag_daily_data.merge(
            daily_target_data["daily_target_r0_mean"].rename("daily_ground_truth_mean"), 
            left_on=["stock_id", "date_id"], right_index=True, how="left"
        )
    return lag_daily_data

def get_master_daily_price_data(master_df, verbose=0):
    # Sort the dataframe first
    master_df = master_df.sort_values(by=META_COLUMNS).reset_index(drop=True)
    
    master_daily_price_data = []
    for round_, seconds_period in tqdm(enumerate([(0, 540), (0, 290), (300, 540)]), disable=not verbose):
        # Filter dataframe
        master_subset = filter_df(master_df, seconds=seconds_period)
        
        # No matter how, also exclude far price and near price
        if seconds_period[0] == 300:
            price_columns = get_cols(master_df, contains="price", excludes=["far", "near"])
        else:
            price_columns = get_cols(master_df, contains="price", excludes=["far", "near"])
        
        # Calc the gradients for the intraday array (multiply by 100 to avoid underflow?)
        daily_price_grads = master_subset.groupby(["stock_id", "date_id"])[price_columns].apply(calc_intraday_gradient).apply(pd.Series) * 100
        daily_price_grads.columns = [f"{price_col}_r{round_}_grad" for price_col in price_columns]
        
        # Compute the agg statistics of target variable for every day
        daily_price_stats = master_subset.groupby(["stock_id", "date_id"])[price_columns].agg(["mean", "std", "min", "max", "last"])
        daily_price_stats.columns = daily_price_stats.columns.map(f'_r{round_}_'.join)

        # Compute the first order difference then the agg statistics (only mean is good I think) of target variable for every day
        master_subset = calc_diff_features(master_subset, columns=price_columns, groupby=["stock_id", "date_id"], lag_distances=[1])
        fod_columns = master_subset.iloc[:, -len(price_columns):].columns
        
        daily_price_fod_stats = master_subset.groupby(["stock_id", "date_id"])[fod_columns].agg(["mean"]) * 100
        daily_price_fod_stats.columns = daily_price_fod_stats.columns.map(f'_r{round_}_'.join)
        
        # Horizontally stack all the dataframes above
        daily_price_data = pd.concat([daily_price_grads, daily_price_stats, daily_price_fod_stats], axis=1)
        
        # Append this dataframe for this master df subset into list
        master_daily_price_data.append(daily_price_data)
    
    master_daily_price_data = pd.concat(master_daily_price_data, axis=1)
    return master_daily_price_data

def generate_interday_price_features(daily_price_data, merge_gt=False, shift=True, verbose=0):
    # if shift:
    #     # Shift the target agg value to avoid future data leakage
    #     lag_daily_price_data = daily_price_data.shift(1)
    # else:
    lag_daily_data = daily_price_data
    
    # Generate features for intraday gradient columns
    gradient_columns = get_cols(lag_daily_data, endswith="grad")
    lag_daily_data = calc_diff_features(lag_daily_data, columns=gradient_columns, groupby=["stock_id"], lag_distances=[1])
    lag_daily_data = calc_ma_features(lag_daily_data, columns=gradient_columns, groupby=["stock_id"], window_sizes=[2])
    
    # # Generate features for intraday standard deviation columns
    # std_columns = get_cols(lag_daily_target_data, endswith="std")
    # lag_daily_target_data = calc_pct_chg_features(lag_daily_target_data, columns=std_columns, groupby=["stock_id"], lag_distances=[1])
    
    # Generate interday gradient features for intraday agg stat columns
    if verbose:
        cprint(f"Generating Interday gradient features for each column...", color="blue")
        
    agg_stat_columns = get_cols(lag_daily_data, endswith=["mean", "std", "last"], excludes=["ground_truth", "fod", "sod"])
    lag_daily_data = calc_interday_gradient_features(lag_daily_data, agg_stat_columns, groupby=["stock_id"], lookback_periods=[3, 6], verbose=verbose)
    
    # We are using data from N days ago to today for building features, but we can only use it for tmr, so we increment date_id by 1
    lag_daily_data = lag_daily_data.reset_index()
    lag_daily_data["date_id"] += 1
    lag_daily_data = lag_daily_data.set_index(["stock_id", "date_id"])
    
    # Merge the target mean per day for feature analysis
    if merge_gt:
        lag_daily_data = lag_daily_data.merge(
            daily_target_data["daily_target_r0_mean"].rename("daily_ground_truth_mean"), 
            left_on=["stock_id", "date_id"], right_index=True, how="left"
        )
    return lag_daily_data

def get_master_daily_volume_data(master_df, verbose=0):
    # Sort the dataframe first
    master_df = master_df.sort_values(by=META_COLUMNS).reset_index(drop=True)
    
    master_daily_data = []
    for round_, seconds_period in tqdm(enumerate([(0, 540), (0, 290), (300, 540)]), disable=not verbose):
        # Filter dataframe
        master_subset = filter_df(master_df, seconds=seconds_period)
        master_subset["trade_size"] = master_subset["bid_size"] + master_subset["ask_size"]
        
        # Get volume columns
        volume_columns = get_cols(master_subset, contains="size", excludes=["bid", "ask"])
        
        # Calc the gradients for the intraday array (multiply by 100 to avoid underflow?)
        daily_grads = master_subset.groupby(["stock_id", "date_id"])[volume_columns].apply(calc_intraday_gradient).apply(pd.Series)
        daily_grads.columns = [f"{col}_r{round_}_grad" for col in volume_columns]
        
        # Compute the agg statistics of target variable for every day
        daily_stats = master_subset.groupby(["stock_id", "date_id"])[volume_columns].agg(["mean", "std", "min", "max"])
        daily_stats.columns = daily_stats.columns.map(f'_r{round_}_'.join)

        # Compute the first order difference then the agg statistics (only mean is good I think) of target variable for every day
        master_subset = calc_diff_features(master_subset, columns=volume_columns, groupby=["stock_id", "date_id"], lag_distances=[1])
        fod_columns = master_subset.iloc[:, -len(volume_columns):].columns
        
        daily_fod_stats = master_subset.groupby(["stock_id", "date_id"])[fod_columns].agg(["mean"])
        daily_fod_stats.columns = daily_fod_stats.columns.map(f'_r{round_}_'.join)
        
        # Horizontally stack all the dataframes above
        daily_data = pd.concat([daily_grads, daily_stats, daily_fod_stats], axis=1)
        
        # Append this dataframe for this master df subset into list
        master_daily_data.append(daily_data)
    
    master_daily_data = pd.concat(master_daily_data, axis=1)
    return master_daily_data

def generate_interday_volume_features(daily_volume_data, merge_gt=False, shift=True, verbose=0):
    # if shift:
    #     # Shift the interday value to avoid future data leakage
    #     lag_daily_data = daily_volume_data.shift(1)
    # else:
    lag_daily_data = daily_volume_data
    
    # Generate features for intraday gradient columns
    gradient_columns = get_cols(lag_daily_data, endswith="grad")
    lag_daily_data = calc_diff_features(lag_daily_data, columns=gradient_columns, groupby=["stock_id"], lag_distances=[1])
    lag_daily_data = calc_ma_features(lag_daily_data, columns=gradient_columns, groupby=["stock_id"], window_sizes=[2])
    
    # Generate interday gradient features for intraday agg stat columns
    if verbose:
        cprint(f"Generating Interday gradient features for each column...", color="blue")
        
    agg_stat_columns = get_cols(lag_daily_data, endswith=["mean", "min", "max", "std"], excludes=["ground_truth", "fod", "sod"])
    lag_daily_data = calc_interday_gradient_features(lag_daily_data, agg_stat_columns, groupby=["stock_id"], lookback_periods=[3, 6], verbose=verbose)
    
    # We are using data from N days ago to today for building features, but we can only use it for tmr, so we increment date_id by 1
    lag_daily_data = lag_daily_data.reset_index()
    lag_daily_data["date_id"] += 1
    lag_daily_data = lag_daily_data.set_index(["stock_id", "date_id"])
    
    # Merge the target mean per day for feature analysis
    if merge_gt:
        lag_daily_data = lag_daily_data.merge(
            daily_target_data["daily_target_r0_mean"].rename("daily_ground_truth_mean"), 
            left_on=["stock_id", "date_id"], right_index=True, how="left"
        )
    return lag_daily_data

# ========================================================================================
# 5. Postprocessing Function for this competition
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

# ========================================================================================
# 6. Preparation Function for inference simulation
# ========================================================================================
def setup_validation_zip(data_dir, val_start_date=421, val_end_date=480):
    """
    Prepare data for validation using a zip iterator.

    Parameters:
        data_dir (str): The directory containing the data files.
        val_start_date (int, optional): The starting date for the validation data (default is 421).
        val_end_date (int, optional): The ending date for the validation data (default is 480).

    Returns:
        zip: A zip iterator containing three components: (dataframes, revealed_targets, submissions).
    """
    master_df = pd.read_csv(f"{data_dir}/optiver-trading-at-the-close/train.csv")
    val = master_df.loc[master_df["date_id"] >= val_start_date].reset_index(drop=True)
    
    iter_test = joblib.load(f'{data_dir}/optiver-test-data/iter_test_copy.pkl')
    for count, (sample_chunk, sample_revealed_target, sample_sub) in enumerate(iter_test):
        if count == 1:
            break
    
    df_list, revealed_target_list, submission_list = [], [], []
    for date_id in tqdm(range(val_start_date, val_end_date + 1)):
        for seconds in np.arange(0, 550, 10):
            # Append test dataframe chunk
            df = filter_df(val, date_id=date_id, seconds=seconds, reset_index=True, meta_columns=["stock_id", "date_id", "seconds_in_bucket"])
            df = df.drop(columns=["time_id", "target"])
            df_list.append(df)

            # Append revealed targets dataframe chunk
            if seconds > 0:
                temp = sample_revealed_target.copy()
                temp["date_id"] = date_id
                temp["seconds_in_bucket"] = seconds
            else:
                # Get yesterday revealed targets
                temp = filter_df(master_df, date_id=int(date_id-1), reset_index=True, meta_columns=["stock_id", "date_id", "seconds_in_bucket"])
                temp = temp[["stock_id", "date_id", "seconds_in_bucket", "target"]].rename(
                    columns={"target": "revealed_target"}
                )
                temp["revealed_date_id"] = temp["date_id"]
                temp["date_id"] += 1
                temp["revealed_time_id"] = (temp["revealed_date_id"] * 55 + temp["seconds_in_bucket"]).astype(int)

            revealed_target_list.append(temp)

            # Append submission dataframe chunk
            sub = df[["row_id"]].copy()
            sub["target"] = 0
            submission_list.append(sub)
    
    return zip(df_list, revealed_target_list, submission_list)