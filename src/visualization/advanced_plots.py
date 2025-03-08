"""
Advanced visualization functions for 10-K filing analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

def create_interactive_time_series(df, date_column, value_columns, company_column=None, 
                                  title='Interactive Time Series of 10-K Metrics'):
    """
    Create an interactive time series plot using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    date_column : str
        Name of the column with dates
    value_columns : list
        List of columns with values to plot
    company_column : str, optional
        Name of the column with company identifiers
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    if company_column:
        # Create a figure with a subplot for each company
        companies = df[company_column].unique()
        
        fig = make_subplots(
            rows=len(companies), 
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f'Company: {company}' for company in companies],
            vertical_spacing=0.1
        )
        
        # Add traces for each company and value
        for i, company in enumerate(companies):
            company_data = df[df[company_column] == company].sort_values(by=date_column)
            
            for value_col in value_columns:
                fig.add_trace(
                    go.Scatter(
                        x=company_data[date_column],
                        y=company_data[value_col],
                        mode='lines+markers',
                        name=f'{company} - {value_col}',
                        hovertemplate='%{x}<br>%{y}<extra></extra>'
                    ),
                    row=i+1, 
                    col=1
                )
    else:
        # Create a single plot with multiple traces for each value
        fig = go.Figure()
        
        # Sort data by date
        df_sorted = df.sort_values(by=date_column)
        
        # Add traces for each value
        for value_col in value_columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[date_column],
                    y=df_sorted[value_col],
                    mode='lines+markers',
                    name=value_col,
                    hovertemplate='%{x}<br>%{y}<extra></extra>'
                )
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        legend_title='Metrics',
        hovermode='closest',
        template='plotly_white',
        height=250 * (len(companies) if company_column else 1) + 100
    )
    
    return fig

def create_heatmap_over_time(df, date_column, company_column, value_column, 
                            title='Heatmap of Metrics Over Time'):
    """
    Create a heatmap showing the evolution of a metric over time for different companies
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    date_column : str
        Name of the column with dates
    company_column : str
        Name of the column with company identifiers
    value_column : str
        Name of the column with values to plot
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract year from date
    df = df.copy()
    df['year'] = df[date_column].dt.year
    
    # Pivot data
    pivot_df = df.pivot_table(
        index=company_column,
        columns='year',
        values=value_column,
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x='Year', y=company_column, color=value_column),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=company_column,
        coloraxis_colorbar=dict(title=value_column),
        height=500,
        width=900
    )
    
    # Display year values as integers
    fig.update_xaxes(type='category')
    
    return fig

def create_interactive_scatter(df, x_column, y_column, color_column=None, size_column=None,
                              hover_data=None, title='Interactive Scatter Plot'):
    """
    Create an interactive scatter plot using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    x_column : str
        Name of the column for x-axis
    y_column : str
        Name of the column for y-axis
    color_column : str, optional
        Name of the column for color encoding
    size_column : str, optional
        Name of the column for size encoding
    hover_data : list, optional
        List of columns to include in hover information
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        size=size_column,
        hover_name=color_column if color_column else None,
        hover_data=hover_data,
        title=title,
        height=600,
        width=900
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else '',
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

def create_interactive_bar_chart(df, x_column, y_column, color_column=None, 
                               title='Interactive Bar Chart'):
    """
    Create an interactive bar chart using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    x_column : str
        Name of the column for x-axis (categories)
    y_column : str
        Name of the column for y-axis (values)
    color_column : str, optional
        Name of the column for color encoding
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create bar chart
    fig = px.bar(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        title=title,
        height=500,
        width=900,
        barmode='group' if color_column else None
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else '',
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

def create_interactive_pie_chart(df, names_column, values_column, title='Interactive Pie Chart'):
    """
    Create an interactive pie chart using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    names_column : str
        Name of the column for slice names
    values_column : str
        Name of the column for slice values
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create pie chart
    fig = px.pie(
        df,
        names=names_column,
        values=values_column,
        title=title,
        height=500,
        width=700
    )
    
    # Update layout
    fig.update_layout(
        legend_title=names_column,
        template='plotly_white'
    )
    
    # Update traces
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value'
    )
    
    return fig

def create_interactive_heatmap(df, x_column, y_column, z_column, 
                              title='Interactive Heatmap'):
    """
    Create an interactive heatmap using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    x_column : str
        Name of the column for x-axis
    y_column : str
        Name of the column for y-axis
    z_column : str
        Name of the column for values (color intensity)
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create pivot table
    pivot_table = df.pivot_table(index=y_column, columns=x_column, values=z_column, aggfunc='mean')
    
    # Create heatmap
    fig = px.imshow(
        pivot_table,
        labels=dict(x=x_column, y=y_column, color=z_column),
        color_continuous_scale='RdBu_r',
        title=title,
        height=600,
        width=900
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        coloraxis_colorbar=dict(title=z_column),
        template='plotly_white'
    )
    
    return fig

def create_radar_chart(df, categories, value_column, group_column, 
                      title='Radar Chart Comparison'):
    """
    Create a radar chart (spider plot) for comparing metrics across companies
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    categories : list
        List of category columns to include in radar
    value_column : str
        Name of the column for values
    group_column : str
        Name of the column for grouping (e.g., company)
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Get unique groups
    groups = df[group_column].unique()
    
    # Add a trace for each group
    for group in groups:
        group_data = df[df[group_column] == group]
        
        # Extract values for each category
        values = []
        for category in categories:
            category_data = group_data[group_data['category'] == category]
            if not category_data.empty:
                values.append(category_data[value_column].values[0])
            else:
                values.append(0)
                
        # Add values back to the first position to close the loop
        values.append(values[0])
        categories_closed = categories + [categories[0]]
        
        # Add trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name=group
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        showlegend=True,
        title=title,
        height=600,
        width=700
    )
    
    return fig

def plot_embeddings_2d(embeddings, labels=None, method='pca', title=None, figsize=(12, 10)):
    """
    Plot 2D visualization of document embeddings from 10-K filings
    
    Parameters:
    -----------
    embeddings : numpy array
        Document embeddings matrix
    labels : list or array, optional
        Labels for coloring the points
    method : str, optional
        Dimensionality reduction method ('pca' or 'tsne')
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects
    """
    # Reduce dimensionality to 2D
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        method_name = 'PCA'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]//2))
        embedding_2d = reducer.fit_transform(embeddings)
        method_name = 't-SNE'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    if labels is not None:
        # Get unique labels
        unique_labels = np.unique(labels)
        
        # Create colormap
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        
        # Plot each label group
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                      c=[cmap(i)], label=label, alpha=0.7, s=80)
        
        ax.legend()
    else:
        # Plot all points with a single color
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=80)
    
    # Set title and labels
    default_title = f'2D {method_name} Projection of Document Embeddings'
    ax.set_title(title if title else default_title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method_name} Dimension 1', fontsize=14)
    ax.set_ylabel(f'{method_name} Dimension 2', fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig, ax

def create_waterfall_chart(df, category_column, value_column, title='Waterfall Chart'):
    """
    Create a waterfall chart showing contributions to a total
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    category_column : str
        Name of the column for categories
    value_column : str
        Name of the column for values
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Sort data by value
    df_sorted = df.sort_values(by=value_column, ascending=False)
    
    # Get categories and values
    categories = df_sorted[category_column].tolist()
    values = df_sorted[value_column].tolist()
    
    # Calculate cumulative values
    cumulative = np.cumsum(values).tolist()
    total = sum(values)
    
    # Add 'Total' category
    categories.append('Total')
    values.append(total)
    
    # Create measure and text arrays
    measure = ['relative'] * len(df_sorted) + ['total']
    text = [f'{value:.1f}' for value in values]
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name='Waterfall',
        orientation='v',
        measure=measure,
        x=categories,
        textposition='outside',
        text=text,
        y=values,
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        height=500,
        width=900,
        template='plotly_white'
    )
    
    return fig

def create_sunburst_chart(df, path_columns, value_column, title='Sunburst Chart'):
    """
    Create a sunburst chart for hierarchical data
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    path_columns : list
        List of columns defining the hierarchy (from root to leaves)
    value_column : str
        Name of the column for values
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create sunburst chart
    fig = px.sunburst(
        df,
        path=path_columns,
        values=value_column,
        title=title,
        height=700,
        width=700
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white'
    )
    
    return fig

def create_treemap_chart(df, path_columns, value_column, color_column=None, 
                        title='Treemap Chart'):
    """
    Create a treemap chart for hierarchical data
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    path_columns : list
        List of columns defining the hierarchy (from root to leaves)
    value_column : str
        Name of the column for values (size of boxes)
    color_column : str, optional
        Name of the column for color encoding
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create treemap chart
    fig = px.treemap(
        df,
        path=path_columns,
        values=value_column,
        color=color_column,
        title=title,
        height=700,
        width=900
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white'
    )
    
    return fig

def create_bubble_chart(df, x_column, y_column, size_column, color_column, 
                      text_column=None, title='Bubble Chart'):
    """
    Create a bubble chart with multiple dimensions
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    x_column : str
        Name of the column for x-axis
    y_column : str
        Name of the column for y-axis
    size_column : str
        Name of the column for bubble size
    color_column : str
        Name of the column for color encoding
    text_column : str, optional
        Name of the column for hover text
    title : str, optional
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create bubble chart
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        size=size_column,
        color=color_column,
        text=text_column,
        title=title,
        height=600,
        width=900,
        size_max=60,
        hover_name=text_column or color_column
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column,
        template='plotly_white'
    )
    
    # Update traces
    fig.update_traces(
        textposition='top center',
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )
    
    return fig

def create_interactive_dashboard(financial_df, sentiment_df=None, topic_df=None):
    """
    Create an interactive dashboard with multiple visualizations
    
    Parameters:
    -----------
    financial_df : pandas DataFrame
        DataFrame with financial metrics
    sentiment_df : pandas DataFrame, optional
        DataFrame with sentiment metrics
    topic_df : pandas DataFrame, optional
        DataFrame with topic modeling results
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Financial Metrics Over Time',
            'Sentiment Analysis',
            'Company Comparison',
            'Topic Distribution'
        ],
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add financial metrics time series
    if 'filing_date' in financial_df.columns and 'revenue' in financial_df.columns:
        financial_df['filing_date'] = pd.to_datetime(financial_df['filing_date'])
        
        for ticker, company_data in financial_df.groupby('ticker'):
            company_data = company_data.sort_values('filing_date')
            
            fig.add_trace(
                go.Scatter(
                    x=company_data['filing_date'],
                    y=company_data['revenue'],
                    mode='lines+markers',
                    name=f'{ticker} Revenue',
                    legendgroup='1'
                ),
                row=1, col=1
            )
    
    # Add sentiment analysis
    if sentiment_df is not None and 'filing_date' in sentiment_df.columns:
        sentiment_df['filing_date'] = pd.to_datetime(sentiment_df['filing_date'])
        
        for ticker, company_data in sentiment_df.groupby('ticker'):
            company_data = company_data.sort_values('filing_date')
            
            fig.add_trace(
                go.Scatter(
                    x=company_data['filing_date'],
                    y=company_data['lexicon_net_score'],
                    mode='lines+markers',
                    name=f'{ticker} Sentiment',
                    legendgroup='2'
                ),
                row=1, col=2
            )
    
    # Add company comparison
    if 'profit_margin' in financial_df.columns:
        # Group by company and get the most recent data
        latest_financial = financial_df.sort_values('filing_date').groupby('ticker').last()
        
        fig.add_trace(
            go.Bar(
                x=latest_financial.index,
                y=latest_financial['profit_margin'],
                name='Profit Margin',
                legendgroup='3'
            ),
            row=2, col=1
        )
    
    # Add topic distribution
    if topic_df is not None and 'topic' in topic_df.columns and 'probability' in topic_df.columns:
        # Group by topic and get average probability
        topic_dist = topic_df.groupby('topic')['probability'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(
                x=topic_dist['topic'],
                y=topic_dist['probability'],
                name='Topic Distribution',
                legendgroup='4'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='10-K Filing Analysis Dashboard',
        height=800,
        width=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=2)
    fig.update_xaxes(title_text='Company', row=2, col=1)
    fig.update_xaxes(title_text='Topic', row=2, col=2)
    
    fig.update_yaxes(title_text='Revenue', row=1, col=1)
    fig.update_yaxes(title_text='Sentiment Score', row=1, col=2)
    fig.update_yaxes(title_text='Profit Margin', row=2, col=1)
    fig.update_yaxes(title_text='Probability', row=2, col=2)
    
    return fig

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    dates = pd.date_range(start='2020-01-01', end='2022-01-01', freq='QS')
    companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    # Financial data
    financial_data = []
    for company in companies:
        for date in dates:
            financial_data.append({
                'ticker': company,
                'filing_date': date,
                'revenue': np.random.uniform(10, 100) * (1 + companies.index(company) * 0.2),
                'profit_margin': np.random.uniform(0.05, 0.25)
            })
    
    financial_df = pd.DataFrame(financial_data)
    
    # Create and show an interactive time series
    fig = create_interactive_time_series(
        financial_df,
        date_column='filing_date',
        value_columns=['revenue', 'profit_margin'],
        company_column='ticker',
        title='Financial Metrics Over Time'
    )
    
    print("Interactive visualization created successfully.")