"""
Tests for the visualization modules (basic_plots, advanced_plots).
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.visualization.basic_plots import (
    plot_time_series, plot_metric_distribution, create_wordcloud,
    plot_sentiment_analysis, plot_comparative_metrics, plot_correlation_heatmap
)
from src.visualization.advanced_plots import (
    create_interactive_time_series, create_heatmap_over_time,
    create_interactive_scatter, create_interactive_bar_chart
)

class TestBasicPlots(unittest.TestCase):
    """Test suite for basic plotting functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample data for testing
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='YS')
        companies = ['AAPL', 'MSFT', 'GOOGL']
        
        data = []
        for company in companies:
            for date in dates:
                data.append({
                    'ticker': company,
                    'filing_date': date,
                    'revenue': np.random.uniform(50, 200) * 1e9,
                    'profit': np.random.uniform(5, 30) * 1e9,
                    'sentiment': np.random.uniform(-0.5, 0.5)
                })
        
        self.sample_df = pd.DataFrame(data)
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        # Call the function
        fig, ax = plot_time_series(
            self.sample_df,
            date_column='filing_date',
            value_column='revenue',
            company_column='ticker',
            title='Test Time Series'
        )
        
        # Check that we got a figure and axes
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check the title
        self.assertEqual(ax.get_title(), 'Test Time Series')
        
        # Close the figure to avoid warnings
        plt.close(fig)
    
    def test_plot_metric_distribution(self):
        """Test metric distribution plotting."""
        # Test with histogram
        fig, ax = plot_metric_distribution(
            self.sample_df,
            metric_column='revenue',
            group_column='ticker',
            title='Test Distribution',
            kind='hist'
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        plt.close(fig)
        
        # Test with box plot
        fig, ax = plot_metric_distribution(
            self.sample_df,
            metric_column='revenue',
            group_column='ticker',
            title='Test Distribution',
            kind='box'
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        plt.close(fig)
    
    def test_create_wordcloud(self):
        """Test word cloud creation."""
        # Create sample text data
        text_data = "This is a sample text for creating a word cloud. The word cloud should contain the most frequent words."
        
        # Call the function
        fig, ax = create_wordcloud(
            text_data,
            title='Test Word Cloud',
            figsize=(8, 6)
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        plt.close(fig)
    
    def test_plot_sentiment_analysis(self):
        """Test sentiment analysis plotting."""
        fig, ax = plot_sentiment_analysis(
            self.sample_df,
            date_column='filing_date',
            sentiment_column='sentiment',
            company_column='ticker',
            title='Test Sentiment'
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), 'Test Sentiment')
        plt.close(fig)
    
    def test_plot_comparative_metrics(self):
        """Test comparative metrics plotting."""
        fig, ax = plot_comparative_metrics(
            self.sample_df,
            metrics=['revenue', 'profit'],
            company_column='ticker',
            title='Test Comparison'
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), 'Test Comparison')
        plt.close(fig)
    
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting."""
        # Add some correlated columns to the sample data
        self.sample_df['metric1'] = self.sample_df['revenue'] * 0.1 + np.random.normal(0, 1e9, len(self.sample_df))
        self.sample_df['metric2'] = self.sample_df['revenue'] * 0.2 + np.random.normal(0, 1e9, len(self.sample_df))
        self.sample_df['metric3'] = self.sample_df['profit'] * 0.3 + np.random.normal(0, 1e9, len(self.sample_df))
        
        fig, ax = plot_correlation_heatmap(
            self.sample_df,
            columns=['revenue', 'profit', 'metric1', 'metric2', 'metric3'],
            title='Test Correlation'
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), 'Test Correlation')
        plt.close(fig)


class TestAdvancedPlots(unittest.TestCase):
    """Test suite for advanced plotting functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample data for testing
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='YS')
        companies = ['AAPL', 'MSFT', 'GOOGL']
        
        data = []
        for company in companies:
            for date in dates:
                data.append({
                    'ticker': company,
                    'filing_date': date,
                    'revenue': np.random.uniform(50, 200) * 1e9,
                    'profit': np.random.uniform(5, 30) * 1e9,
                    'margin': np.random.uniform(0.1, 0.3),
                    'sentiment': np.random.uniform(-0.5, 0.5)
                })
        
        self.sample_df = pd.DataFrame(data)
    
    def test_create_interactive_time_series(self):
        """Test interactive time series creation."""
        fig = create_interactive_time_series(
            self.sample_df,
            date_column='filing_date',
            value_columns=['revenue', 'profit'],
            company_column='ticker',
            title='Test Interactive Time Series'
        )
        
        # Check that we got a figure
        self.assertIsInstance(fig, go.Figure)
        
        # Check the title
        self.assertEqual(fig.layout.title.text, 'Test Interactive Time Series')
    
    def test_create_heatmap_over_time(self):
        """Test heatmap over time creation."""
        fig = create_heatmap_over_time(
            self.sample_df,
            date_column='filing_date',
            company_column='ticker',
            value_column='revenue',
            title='Test Heatmap'
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, 'Test Heatmap')
    
    def test_create_interactive_scatter(self):
        """Test interactive scatter plot creation."""
        fig = create_interactive_scatter(
            self.sample_df,
            x_column='revenue',
            y_column='profit',
            color_column='ticker',
            title='Test Scatter'
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, 'Test Scatter')
    
    def test_create_interactive_bar_chart(self):
        """Test interactive bar chart creation."""
        # Calculate average margin by company
        margin_by_company = self.sample_df.groupby('ticker')['margin'].mean().reset_index()
        
        fig = create_interactive_bar_chart(
            margin_by_company,
            x_column='ticker',
            y_column='margin',
            title='Test Bar Chart'
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, 'Test Bar Chart')


if __name__ == '__main__':
    unittest.main()