"""
Basic visualization functions for 10-K filing analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default style
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')

def configure_plot_style(ax, title, xlabel, ylabel, legend=True, grid=True, fontsize=12):
    """Configure common plot style elements"""
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize-2)
    if legend:
        ax.legend(fontsize=fontsize-2)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return ax

def plot_time_series(df, date_column, value_column, company_column=None, title=None, figsize=(12, 6)):
    """
    Plot time series data from 10-K filings
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    date_column : str
        Name of the column with dates
    value_column : str
        Name of the column with values to plot
    company_column : str, optional
        Name of the column with company identifiers for multi-company plots
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if company_column is not None:
        # Plot time series for each company
        for company, group in df.groupby(company_column):
            group = group.sort_values(by=date_column)
            ax.plot(group[date_column], group[value_column], marker='o', linewidth=2, label=company)
        
        default_title = f'{value_column} Over Time by Company'
    else:
        # Plot single time series
        df_sorted = df.sort_values(by=date_column)
        ax.plot(df_sorted[date_column], df_sorted[value_column], marker='o', linewidth=2, color='#1f77b4')
        default_title = f'{value_column} Over Time'
    
    # Configure style
    configure_plot_style(
        ax, 
        title=title if title else default_title,
        xlabel='Date',
        ylabel=value_column
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    return fig, ax

def plot_metric_distribution(df, metric_column, group_column=None, title=None, figsize=(12, 6), kind='hist'):
    """
    Plot distribution of a metric from 10-K filings
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    metric_column : str
        Name of the column with metric to analyze
    group_column : str, optional
        Name of the column to group by (e.g., company, industry, year)
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    kind : str
        Type of plot ('hist', 'box', 'violin')
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if kind == 'hist':
        if group_column:
            for group, data in df.groupby(group_column):
                sns.histplot(data[metric_column], label=group, kde=True, alpha=0.6, ax=ax)
        else:
            sns.histplot(df[metric_column], kde=True, ax=ax)
            
        default_title = f'Distribution of {metric_column}'
            
    elif kind == 'box':
        if group_column:
            sns.boxplot(x=group_column, y=metric_column, data=df, ax=ax)
        else:
            sns.boxplot(y=metric_column, data=df, ax=ax)
            
        default_title = f'Box Plot of {metric_column}'
            
    elif kind == 'violin':
        if group_column:
            sns.violinplot(x=group_column, y=metric_column, data=df, ax=ax)
        else:
            sns.violinplot(y=metric_column, data=df, ax=ax)
            
        default_title = f'Violin Plot of {metric_column}'
    
    # Configure style
    configure_plot_style(
        ax, 
        title=title if title else default_title,
        xlabel=group_column if group_column else '',
        ylabel=metric_column
    )
    
    # Rotate x-axis labels if needed
    if group_column:
        plt.xticks(rotation=45)
    
    return fig, ax

def create_wordcloud(text_data, title='Word Cloud of 10-K Filings', figsize=(12, 8), 
                     background_color='white', colormap='viridis', max_words=200):
    """
    Create a word cloud visualization from 10-K filing text
    
    Parameters:
    -----------
    text_data : str or list
        Text data to visualize. If list, it will be joined with spaces.
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    background_color : str, optional
        Background color for the word cloud
    colormap : str, optional
        Colormap for the word cloud
    max_words : int, optional
        Maximum number of words to include
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # If text_data is a list, join with spaces
    if isinstance(text_data, list):
        text_data = ' '.join(text_data)
    
    # Generate word cloud
    wordcloud = WordCloud(
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        width=800,
        height=500,
        contour_width=1,
        contour_color='steelblue',
        random_state=42
    ).generate(text_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax

def plot_sentiment_analysis(df, date_column, sentiment_column, company_column=None, 
                           title=None, figsize=(14, 7)):
    """
    Plot sentiment analysis results from 10-K filings over time
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    date_column : str
        Name of the column with dates
    sentiment_column : str
        Name of the column with sentiment scores
    company_column : str, optional
        Name of the column with company identifiers
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add horizontal line at y=0 (neutral sentiment)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    if company_column:
        # Plot sentiment for each company
        for company, group in df.groupby(company_column):
            group = group.sort_values(by=date_column)
            ax.plot(group[date_column], group[sentiment_column], marker='o', linewidth=2, label=company)
            
        default_title = 'Sentiment Analysis Over Time by Company'
    else:
        # Plot overall sentiment
        df_sorted = df.sort_values(by=date_column)
        
        # Create colormap based on sentiment values
        colors = ['#d7191c' if x < 0 else '#1a9641' for x in df_sorted[sentiment_column]]
        
        ax.scatter(df_sorted[date_column], df_sorted[sentiment_column], c=colors, s=80, alpha=0.7)
        ax.plot(df_sorted[date_column], df_sorted[sentiment_column], color='#2c7fb8', alpha=0.5)
        
        default_title = 'Sentiment Analysis Over Time'
    
    # Add sentiment regions
    ax.axhspan(-1, -0.1, alpha=0.2, color='red', label='Negative')
    ax.axhspan(-0.1, 0.1, alpha=0.2, color='gray', label='Neutral')
    ax.axhspan(0.1, 1, alpha=0.2, color='green', label='Positive')
    
    # Configure style
    configure_plot_style(
        ax, 
        title=title if title else default_title,
        xlabel='Date',
        ylabel='Sentiment Score'
    )
    
    # Set y-axis limits
    ax.set_ylim(-1.1, 1.1)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    return fig, ax

def plot_comparative_metrics(df, metrics, company_column, title=None, figsize=(14, 8)):
    """
    Create a comparative bar chart for multiple metrics across companies
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    metrics : list
        List of metric column names to compare
    company_column : str
        Name of the column with company identifiers
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # Number of companies and metrics
    companies = df[company_column].unique()
    n_companies = len(companies)
    n_metrics = len(metrics)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set width of bars
    bar_width = 0.8 / n_metrics
    
    # Set positions for the bars
    positions = np.arange(n_companies)
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        # Get average value for each company
        values = [df[df[company_column] == company][metric].mean() for company in companies]
        
        # Plot bars
        ax.bar(positions + i * bar_width - (n_metrics - 1) * bar_width / 2, 
               values, bar_width, label=metric, alpha=0.8)
    
    # Configure style
    configure_plot_style(
        ax, 
        title=title if title else 'Comparative Metrics by Company',
        xlabel='Company',
        ylabel='Value'
    )
    
    # Set x-ticks at the center of the company groups
    ax.set_xticks(positions)
    ax.set_xticklabels(companies, rotation=45, ha='right')
    
    return fig, ax

def plot_correlation_heatmap(df, columns=None, title='Correlation Heatmap of 10-K Metrics', 
                            figsize=(12, 10), cmap='coolwarm', annot=True):
    """
    Create a correlation heatmap for 10-K filing metrics
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    columns : list, optional
        List of columns to include in correlation analysis
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap for the heatmap
    annot : bool, optional
        Whether to annotate the heatmap with correlation values
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # Select columns if specified, otherwise use all numeric columns
    if columns:
        data = df[columns].select_dtypes(include=[np.number])
    else:
        data = df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annot, 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
                fmt='.2f', ax=ax)
    
    # Set title
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, ax

def plot_stacked_bar(data, analysis_type, selected_year=None, selected_therapy_area=None, 
                     selected_geography=None, title=None, figsize=(14, 8), 
                     color_palette='viridis', sort_by=None):
    """
    Create a stacked bar chart for 10-K filings analysis.
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the data for visualization
    analysis_type : str
        Type of analysis being performed (e.g., 'Revenue', 'R&D Spending')
    selected_year : int or str, optional
        Year filter for the data
    selected_therapy_area : str, optional
        Therapy area filter for the data
    selected_geography : str, optional
        Geographic region filter for the data
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    color_palette : str, optional
        Color palette for the bars
    sort_by : str, optional
        Column to sort the data by
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # Filter data based on selections
    filtered_data = data.copy()
    
    if selected_year is not None:
        if 'Year' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Year'] == selected_year]
        elif 'Filing_Year' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Filing_Year'] == selected_year]
    
    if selected_therapy_area is not None and 'Therapy_Area' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Therapy_Area'] == selected_therapy_area]
    
    if selected_geography is not None and 'Geography' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Geography'] == selected_geography]
    
    # Check if we have data
    if filtered_data.empty:
        logger.warning("No data available with the selected filters")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available with the selected filters", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig, ax
    
    # Determine grouping columns based on what's available
    if 'Company' in filtered_data.columns:
        primary_group = 'Company'
    elif 'Ticker' in filtered_data.columns:
        primary_group = 'Ticker'
    else:
        logger.warning("No company identifier column found")
        primary_group = filtered_data.columns[0]
    
    # Determine secondary grouping if available
    if 'Therapy_Area' in filtered_data.columns and not selected_therapy_area:
        secondary_group = 'Therapy_Area'
    elif 'Product_Line' in filtered_data.columns:
        secondary_group = 'Product_Line'
    elif 'Geography' in filtered_data.columns and not selected_geography:
        secondary_group = 'Geography'
    else:
        secondary_group = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for stacked bar chart
    if secondary_group:
        # Pivot data for stacked bars
        pivot_data = filtered_data.pivot_table(
            index=primary_group, 
            columns=secondary_group, 
            values=analysis_type, 
            aggfunc='sum'
        ).fillna(0)
        
        # Sort if requested
        if sort_by:
            if sort_by in pivot_data.columns:
                pivot_data = pivot_data.sort_values(by=sort_by, ascending=False)
            elif sort_by == 'total':
                pivot_data = pivot_data.sort_values(by=pivot_data.sum(axis=1), ascending=False)
        
        # Plot stacked bar
        pivot_data.plot(kind='bar', stacked=True, ax=ax, colormap=color_palette)
        
    else:
        # Simple bar chart grouped by primary_group
        grouped_data = filtered_data.groupby(primary_group)[analysis_type].sum().reset_index()
        
        # Sort if requested
        if sort_by:
            grouped_data = grouped_data.sort_values(by=analysis_type, ascending=False)
        
        # Plot bar chart
        sns.barplot(x=primary_group, y=analysis_type, data=grouped_data, ax=ax, palette=color_palette)
    
    # Format y-axis as percentage if appropriate
    if 'percentage' in analysis_type.lower() or 'ratio' in analysis_type.lower() or 'margin' in analysis_type.lower():
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Set title and labels
    if title:
        chart_title = title
    else:
        filters = []
        if selected_year:
            filters.append(f"Year: {selected_year}")
        if selected_therapy_area:
            filters.append(f"Therapy Area: {selected_therapy_area}")
        if selected_geography:
            filters.append(f"Geography: {selected_geography}")
        
        filter_text = " | ".join(filters)
        chart_title = f"{analysis_type} by {primary_group}"
        if secondary_group:
            chart_title += f" and {secondary_group}"
        if filter_text:
            chart_title += f" ({filter_text})"
    
    ax.set_title(chart_title, fontsize=16, fontweight='bold')
    ax.set_xlabel(primary_group, fontsize=14)
    ax.set_ylabel(analysis_type, fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend with better placement
    if secondary_group:
        ax.legend(title=secondary_group, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig, ax

def plot_year_over_year_comparison(df, date_column, value_column, group_column=None,
                                  title='Year-over-Year Comparison', normalize=False):
    """
    Create bar chart comparing metrics year over year.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    date_column : str
        Name of the column with dates
    value_column : str
        Name of the column with values to compare
    group_column : str, optional
        Name of the column to group by (e.g., company)
    title : str, optional
        Plot title
    normalize : bool, optional
        Whether to normalize values as percentage change from first year
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract year from date
    df['year'] = df[date_column].dt.year
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if group_column:
        # Group data by year and group_column
        groups = df.groupby(group_column)
        
        # Number of years and groups
        years = sorted(df['year'].unique())
        groups_list = sorted(df[group_column].unique())
        n_years = len(years)
        n_groups = len(groups_list)
        
        # Set width of bars
        bar_width = 0.8 / n_groups
        
        # Set positions for the bars
        positions = np.arange(n_years)
        
        # Plot bars for each group
        for i, (group_name, group_data) in enumerate(groups):
            # Calculate yearly averages
            yearly_data = group_data.groupby('year')[value_column].mean()
            
            # Get values for each year
            values = [yearly_data.get(year, np.nan) for year in years]
            
            # Normalize values if requested
            if normalize and not np.isnan(values[0]) and values[0] != 0:
                base_value = values[0]
                values = [(v / base_value - 1) * 100 if not np.isnan(v) else np.nan for v in values]
            
            # Plot bars
            ax.bar(positions + i * bar_width - (n_groups - 1) * bar_width / 2, 
                  values, bar_width, label=group_name, alpha=0.8)
        
        # Set x-ticks at the center of the year groups
        ax.set_xticks(positions)
        ax.set_xticklabels([int(year) for year in years])
        
    else:
        # Group data by year only
        yearly_data = df.groupby('year')[value_column].mean()
        years = sorted(yearly_data.index)
        
        # Get values for each year
        values = yearly_data.values
        
        # Normalize values if requested
        if normalize and values[0] != 0:
            values = (values / values[0] - 1) * 100
        
        # Set bar colors based on year-over-year change
        colors = ['#d7191c' if values[i] < values[i-1] else '#1a9641' for i in range(1, len(values))]
        colors.insert(0, '#1a9641')  # First year is green by default
        
        # Plot bars
        ax.bar([int(year) for year in years], values, color=colors, alpha=0.8)
        
        # Add value annotations above bars
        for i, (year, value) in enumerate(zip(years, values)):
            if normalize:
                ax.annotate(f'{value:.1f}%', xy=(int(year), value), 
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom')
            else:
                ax.annotate(f'{value:.2f}', xy=(int(year), value), 
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom')
    
    # Configure style
    if normalize:
        ylabel = 'Percentage Change from Base Year (%)'
    else:
        ylabel = value_column
        
    configure_plot_style(
        ax, 
        title=title,
        xlabel='Year',
        ylabel=ylabel
    )
    
    # Convert year labels to integers
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x)))
    
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2018-01-01', end='2022-01-01', freq='YS')
    companies = ['Company A', 'Company B', 'Company C']
    
    data = []
    for company in companies:
        for date in dates:
            value = np.random.normal(100, 20)
            sentiment = np.random.uniform(-0.5, 0.5)
            data.append({
                'date': date,
                'company': company,
                'value': value,
                'sentiment': sentiment
            })
    
    df = pd.DataFrame(data)
    
    # Create and show a time series plot
    fig, ax = plot_time_series(df, 'date', 'value', 'company')
    plt.show()