"""
Helper utility functions for the 10-K Analysis Toolkit.
"""

import os
import re
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(dir_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters:
    -----------
    dir_path : str
        Directory path to ensure
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")

def setup_matplotlib_style():
    """
    Set up consistent matplotlib styling for all visualizations.
    """
    plt.style.use('fivethirtyeight')
    sns.set_palette('Set2')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

def normalize_ticker(ticker):
    """
    Normalize ticker symbol (uppercase, remove special characters).
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to normalize
        
    Returns:
    --------
    str
        Normalized ticker symbol
    """
    if not ticker:
        return ""
    
    # Convert to uppercase
    ticker = ticker.upper()
    
    # Remove special characters
    ticker = re.sub(r'[^A-Z0-9]', '', ticker)
    
    return ticker

def format_number(value, precision=2, prefix='', suffix='', with_commas=True, 
                  abbreviate=False, is_percentage=False):
    """
    Format a number for display with various options.
    
    Parameters:
    -----------
    value : float or int
        Number to format
    precision : int
        Number of decimal places
    prefix : str
        Prefix to add (e.g., '$')
    suffix : str
        Suffix to add (e.g., '%')
    with_commas : bool
        Whether to include thousand separators
    abbreviate : bool
        Whether to abbreviate large numbers (K, M, B)
    is_percentage : bool
        Whether to multiply by 100 and add % suffix
        
    Returns:
    --------
    str
        Formatted number
    """
    if value is None:
        return "N/A"
    
    try:
        # Convert to float
        value = float(value)
        
        # Apply percentage conversion if specified
        if is_percentage:
            value = value * 100
            suffix = '%' + suffix
        
        # Abbreviate if specified
        if abbreviate:
            if abs(value) >= 1e9:
                value = value / 1e9
                suffix = 'B' + suffix
            elif abs(value) >= 1e6:
                value = value / 1e6
                suffix = 'M' + suffix
            elif abs(value) >= 1e3:
                value = value / 1e3
                suffix = 'K' + suffix
        
        # Format the number
        if with_commas:
            formatted = f"{value:,.{precision}f}"
        else:
            formatted = f"{value:.{precision}f}"
        
        # Add prefix and suffix
        return f"{prefix}{formatted}{suffix}"
        
    except (ValueError, TypeError):
        return "N/A"

def filter_dataframe(df, filters):
    """
    Apply filters to a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to filter
    filters : dict
        Dictionary of column-value pairs to filter on
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if column in filtered_df.columns:
            if isinstance(value, (list, tuple, set)):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df

def get_date_range_filter(df, date_column, start_date=None, end_date=None):
    """
    Create a date range filter mask for a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to filter
    date_column : str
        Name of the date column
    start_date : str or datetime
        Start date for filter
    end_date : str or datetime
        End date for filter
        
    Returns:
    --------
    pandas.Series
        Boolean mask for filtering
    """
    # Ensure date column is datetime type
    if date_column in df.columns:
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Initialize mask
        mask = pd.Series(True, index=df.index)
        
        # Apply start date filter
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            mask = mask & (df[date_column] >= start_date)
        
        # Apply end date filter
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            mask = mask & (df[date_column] <= end_date)
        
        return mask
    
    return pd.Series(True, index=df.index)

def create_report_header(title, subtitle=None, date=None):
    """
    Create a formatted header for reports.
    
    Parameters:
    -----------
    title : str
        Report title
    subtitle : str
        Report subtitle
    date : str or datetime
        Report date
        
    Returns:
    --------
    str
        Formatted report header
    """
    header = f"{title}\n"
    header += "=" * len(title) + "\n\n"
    
    if subtitle:
        header += f"{subtitle}\n"
        header += "-" * len(subtitle) + "\n\n"
    
    if date:
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str = date
        
        header += f"Generated: {date_str}\n\n"
    else:
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    return header

def merge_dataframes(df1, df2, on=None, how='inner'):
    """
    Merge two DataFrames with error handling.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First DataFrame
    df2 : pandas.DataFrame
        Second DataFrame
    on : list or str
        Columns to merge on
    how : str
        Type of merge (inner, outer, left, right)
        
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame
    """
    try:
        # If 'on' is not specified, look for common columns
        if on is None:
            common_cols = set(df1.columns).intersection(set(df2.columns))
            if not common_cols:
                logger.error("No common columns found for merge")
                return None
            
            on = list(common_cols)
        
        # Perform merge
        merged_df = pd.merge(df1, df2, on=on, how=how)
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging DataFrames: {str(e)}")
        return None

def extract_year_from_date(df, date_column, year_column=None):
    """
    Extract year from a date column in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to process
    date_column : str
        Name of the date column
    year_column : str
        Name for the new year column (if None, use 'year')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added year column
    """
    result_df = df.copy()
    
    if date_column in result_df.columns:
        if not pd.api.types.is_datetime64_dtype(result_df[date_column]):
            result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
        
        year_col = year_column if year_column else 'year'
        result_df[year_col] = result_df[date_column].dt.year
    
    return result_df

def calculate_growth_rates(df, group_column, date_column, value_column, growth_column=None):
    """
    Calculate year-over-year growth rates for a value column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to process
    group_column : str
        Column to group by (e.g., 'ticker')
    date_column : str
        Date column for sorting
    value_column : str
        Column to calculate growth rates for
    growth_column : str
        Name for the growth rate column (if None, use value_column + '_growth')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added growth rate column
    """
    result_df = df.copy()
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_dtype(result_df[date_column]):
        result_df[date_column] = pd.to_datetime(result_df[date_column])
    
    # Set growth column name
    growth_col = growth_column if growth_column else f"{value_column}_growth"
    
    # Initialize growth column
    result_df[growth_col] = np.nan
    
    # Calculate growth rates for each group
    for group, group_data in result_df.groupby(group_column):
        # Sort by date
        sorted_data = group_data.sort_values(date_column)
        
        # Calculate growth rates
        sorted_data[growth_col] = sorted_data[value_column].pct_change() * 100
        
        # Update result_df with calculated growth rates
        result_df.loc[sorted_data.index, growth_col] = sorted_data[growth_col]
    
    return result_df

def calculate_summary_statistics(df, group_columns, value_columns, include_count=True):
    """
    Calculate summary statistics for value columns grouped by specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to process
    group_columns : list
        Columns to group by
    value_columns : list
        Columns to calculate statistics for
    include_count : bool
        Whether to include count in the statistics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with summary statistics
    """
    if not isinstance(group_columns, (list, tuple)):
        group_columns = [group_columns]
    
    if not isinstance(value_columns, (list, tuple)):
        value_columns = [value_columns]
    
    # Define aggregation functions
    agg_funcs = ['mean', 'median', 'std', 'min', 'max']
    if include_count:
        agg_funcs.append('count')
    
    # Group and aggregate
    summary = df.groupby(group_columns)[value_columns].agg(agg_funcs)
    
    return summary

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a DataFrame column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to process
    column : str
        Column to check for outliers
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    pandas.Series
        Boolean mask indicating outliers
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame")
        return pd.Series(False, index=df.index)
    
    # Get values
    values = df[column].dropna()
    
    if method == 'iqr':
        # IQR method
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return (df[column] < lower_bound) | (df[column] > upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return pd.Series(False, index=df.index)
        
        z_scores = (df[column] - mean) / std
        
        return z_scores.abs() > threshold
        
    else:
        logger.error(f"Unknown outlier detection method: {method}")
        return pd.Series(False, index=df.index)

# Example usage
if __name__ == "__main__":
    # Test setup_matplotlib_style
    setup_matplotlib_style()
    
    # Test format_number
    print(format_number(1234567.89, prefix='$', abbreviate=True))  # $1.23M
    print(format_number(0.1234, is_percentage=True))  # 12.34%
    
    # Test normalize_ticker
    print(normalize_ticker('AAPL.US'))  # AAPL