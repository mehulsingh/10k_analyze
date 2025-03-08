#!/usr/bin/env python
"""
Command-line interface for the 10-K Analysis Toolkit.
"""

import argparse
import os
import sys
import pandas as pd
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data.data_loader import SECDataLoader
from src.data.data_preprocessor import FilingPreprocessor
from src.analysis.text_analysis import TextAnalyzer
from src.analysis.financial_analysis import FinancialAnalyzer
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.visualization.basic_plots import (
    plot_time_series, create_wordcloud, plot_sentiment_analysis,
    plot_comparative_metrics, plot_correlation_heatmap
)
from src.visualization.advanced_plots import (
    create_interactive_time_series, create_interactive_scatter,
    create_interactive_bar_chart, create_bubble_chart
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='10-K Analysis Toolkit CLI')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download 10-K filings')
    download_parser.add_argument('--tickers', nargs='+', required=True, help='List of ticker symbols')
    download_parser.add_argument('--years', nargs='+', type=int, help='List of years')
    download_parser.add_argument('--output', default='data/raw/filings.pkl', help='Output file path')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess 10-K filings')
    preprocess_parser.add_argument('--input', default='data/raw/filings.pkl', help='Input file path')
    preprocess_parser.add_argument('--output', default='data/processed/processed_filings.pkl', help='Output file path')
    preprocess_parser.add_argument('--extract-sections', action='store_true', help='Extract sections from filings')
    preprocess_parser.add_argument('--clean-text', action='store_true', help='Clean text content')
    
    # Text analysis command
    text_parser = subparsers.add_parser('text-analysis', help='Perform text analysis on 10-K filings')
    text_parser.add_argument('--input', default='data/processed/processed_filings.pkl', help='Input file path')
    text_parser.add_argument('--output', default='data/results/text_analysis.pkl', help='Output file path')
    text_parser.add_argument('--sections', nargs='+', default=['item_1a', 'item_7'], help='Sections to analyze')
    
    # Financial analysis command
    financial_parser = subparsers.add_parser('financial-analysis', help='Perform financial analysis on 10-K filings')
    financial_parser.add_argument('--input', default='data/processed/processed_filings.pkl', help='Input file path')
    financial_parser.add_argument('--output', default='data/results/financial_analysis.pkl', help='Output file path')
    
    # Sentiment analysis command
    sentiment_parser = subparsers.add_parser('sentiment-analysis', help='Perform sentiment analysis on 10-K filings')
    sentiment_parser.add_argument('--input', default='data/processed/processed_filings.pkl', help='Input file path')
    sentiment_parser.add_argument('--output', default='data/results/sentiment_analysis.pkl', help='Output file path')
    sentiment_parser.add_argument('--sections', nargs='+', default=['item_7'], help='Sections to analyze')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations from analysis results')
    viz_parser.add_argument('--financial', default='data/results/financial_analysis.pkl', help='Financial analysis file')
    viz_parser.add_argument('--sentiment', default='data/results/sentiment_analysis.pkl', help='Sentiment analysis file')
    viz_parser.add_argument('--text', default='data/results/text_analysis.pkl', help='Text analysis file')
    viz_parser.add_argument('--output-dir', default='output/visualizations', help='Output directory for visualizations')
    viz_parser.add_argument('--format', choices=['png', 'pdf', 'html', 'all'], default='all', help='Output format')
    
    # Pipeline command (run all steps)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full analysis pipeline')
    pipeline_parser.add_argument('--tickers', nargs='+', required=True, help='List of ticker symbols')
    pipeline_parser.add_argument('--years', nargs='+', type=int, help='List of years')
    pipeline_parser.add_argument('--output-dir', default='output', help='Output directory')
    
    return parser.parse_args()

def download_filings(args):
    """Download 10-K filings."""
    logger.info(f"Downloading 10-K filings for tickers: {args.tickers}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize loader
    loader = SECDataLoader()
    
    # Download filings
    filings_df = loader.load_filings(args.tickers, years=args.years)
    
    # Save to file
    filings_df.to_pickle(args.output)
    
    logger.info(f"Downloaded {len(filings_df)} filings to {args.output}")
    
    return filings_df

def preprocess_filings(args):
    """Preprocess 10-K filings."""
    logger.info(f"Preprocessing 10-K filings from {args.input}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load filings
    filings_df = pd.read_pickle(args.input)
    
    # Initialize preprocessor
    preprocessor = FilingPreprocessor()
    
    # Process filings
    processed_df = preprocessor.process_filings(
        filings_df,
        extract_sections=args.extract_sections,
        clean_text=args.clean_text
    )
    
    # Save to file
    processed_df.to_pickle(args.output)
    
    logger.info(f"Preprocessed {len(processed_df)} filings to {args.output}")
    
    return processed_df

def text_analysis(args):
    """Perform text analysis on 10-K filings."""
    logger.info(f"Performing text analysis on 10-K filings from {args.input}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load filings
    filings_df = pd.read_pickle(args.input)
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    # Analyze filings
    analysis_df = analyzer.analyze_filings(filings_df, sections=args.sections)
    
    # Save to file
    analysis_df.to_pickle(args.output)
    
    logger.info(f"Text analysis results saved to {args.output}")
    
    return analysis_df

def financial_analysis(args):
    """Perform financial analysis on 10-K filings."""
    logger.info(f"Performing financial analysis on 10-K filings from {args.input}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load filings
    filings_df = pd.read_pickle(args.input)
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer()
    
    # Analyze filings
    analysis_df = analyzer.analyze_filings(filings_df)
    
    # Save to file
    analysis_df.to_pickle(args.output)
    
    logger.info(f"Financial analysis results saved to {args.output}")
    
    return analysis_df

def sentiment_analysis(args):
    """Perform sentiment analysis on 10-K filings."""
    logger.info(f"Performing sentiment analysis on 10-K filings from {args.input}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load filings
    filings_df = pd.read_pickle(args.input)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze filings
    analysis_df = analyzer.analyze_filings(filings_df, sections=args.sections)
    
    # Save to file
    analysis_df.to_pickle(args.output)
    
    logger.info(f"Sentiment analysis results saved to {args.output}")
    
    return analysis_df

def create_visualizations(args):
    """Create visualizations from analysis results."""
    logger.info(f"Creating visualizations from analysis results")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load analysis results
    financial_df = pd.read_pickle(args.financial) if os.path.exists(args.financial) else None
    sentiment_df = pd.read_pickle(args.sentiment) if os.path.exists(args.sentiment) else None
    text_df = pd.read_pickle(args.text) if os.path.exists(args.text) else None
    
    # List to track created visualizations
    visualizations = []
    
    # Create visualizations based on available data
    if financial_df is not None:
        logger.info("Creating financial visualizations")
        
        # Time series visualization
        if 'filing_date' in financial_df.columns and 'revenue' in financial_df.columns:
            # Matplotlib version (static)
            fig, ax = plot_time_series(
                financial_df,
                date_column='filing_date',
                value_column='revenue',
                company_column='ticker',
                title='Revenue Over Time by Company'
            )
            
            static_path = os.path.join(args.output_dir, 'revenue_time_series')
            if args.format in ['png', 'all']:
                fig.savefig(f"{static_path}.png", dpi=300, bbox_inches='tight')
                visualizations.append(f"{static_path}.png")
            if args.format in ['pdf', 'all']:
                fig.savefig(f"{static_path}.pdf", bbox_inches='tight')
                visualizations.append(f"{static_path}.pdf")
            plt.close(fig)
            
            # Plotly version (interactive)
            if args.format in ['html', 'all']:
                fig = create_interactive_time_series(
                    financial_df,
                    date_column='filing_date',
                    value_columns=['revenue', 'net_income'] if 'net_income' in financial_df.columns else ['revenue'],
                    company_column='ticker',
                    title='Financial Metrics Over Time'
                )
                
                interactive_path = os.path.join(args.output_dir, 'revenue_time_series_interactive.html')
                fig.write_html(interactive_path)
                visualizations.append(interactive_path)
        
        # Comparative metrics visualization
        if 'ticker' in financial_df.columns:
            # Get the most recent data for each company
            if 'filing_date' in financial_df.columns:
                financial_df['filing_date'] = pd.to_datetime(financial_df['filing_date'])
                latest_financial = financial_df.sort_values('filing_date').groupby('ticker').last().reset_index()
            else:
                latest_financial = financial_df
            
            # Select metrics to compare
            potential_metrics = ['revenue', 'net_income', 'operating_income', 'profit_margin', 'roa']
            metrics_to_compare = [m for m in potential_metrics if m in latest_financial.columns]
            
            if metrics_to_compare:
                # Matplotlib version (static)
                fig, ax = plot_comparative_metrics(
                    latest_financial,
                    metrics=metrics_to_compare,
                    company_column='ticker',
                    title='Financial Metrics Comparison by Company'
                )
                
                static_path = os.path.join(args.output_dir, 'financial_metrics_comparison')
                if args.format in ['png', 'all']:
                    fig.savefig(f"{static_path}.png", dpi=300, bbox_inches='tight')
                    visualizations.append(f"{static_path}.png")
                if args.format in ['pdf', 'all']:
                    fig.savefig(f"{static_path}.pdf", bbox_inches='tight')
                    visualizations.append(f"{static_path}.pdf")
                plt.close(fig)
                
                # Plotly version (interactive)
                if args.format in ['html', 'all'] and len(metrics_to_compare) > 0:
                    # Create separate charts for different metrics
                    for metric in metrics_to_compare:
                        fig = create_interactive_bar_chart(
                            latest_financial,
                            x_column='ticker',
                            y_column=metric,
                            title=f'{metric.replace("_", " ").title()} by Company'
                        )
                        
                        interactive_path = os.path.join(args.output_dir, f'{metric}_comparison_interactive.html')
                        fig.write_html(interactive_path)
                        visualizations.append(interactive_path)
    
    if sentiment_df is not None:
        logger.info("Creating sentiment visualizations")
        
        # Filter to a specific section (e.g., MD&A)
        if 'section' in sentiment_df.columns:
            mda_sentiment = sentiment_df[sentiment_df['section'] == 'item_7'].copy()
            
            if not mda_sentiment.empty and 'filing_date' in mda_sentiment.columns:
                # Ensure date column is datetime
                mda_sentiment['filing_date'] = pd.to_datetime(mda_sentiment['filing_date'])
                
                # Sentiment time series
                if 'lexicon_net_score' in mda_sentiment.columns:
                    fig, ax = plot_sentiment_analysis(
                        mda_sentiment,
                        date_column='filing_date',
                        sentiment_column='lexicon_net_score',
                        company_column='ticker',
                        title='MD&A Sentiment Over Time by Company'
                    )
                    
                    static_path = os.path.join(args.output_dir, 'mda_sentiment_time_series')
                    if args.format in ['png', 'all']:
                        fig.savefig(f"{static_path}.png", dpi=300, bbox_inches='tight')
                        visualizations.append(f"{static_path}.png")
                    if args.format in ['pdf', 'all']:
                        fig.savefig(f"{static_path}.pdf", bbox_inches='tight')
                        visualizations.append(f"{static_path}.pdf")
                    plt.close(fig)
                
                # Interactive sentiment visualization
                if args.format in ['html', 'all']:
                    sentiment_metrics = ['lexicon_net_score', 'textblob_polarity', 'lexicon_uncertainty_score', 'lexicon_litigious_score']
                    available_metrics = [m for m in sentiment_metrics if m in mda_sentiment.columns]
                    
                    if available_metrics:
                        fig = create_interactive_scatter(
                            mda_sentiment,
                            x_column='filing_date',
                            y_column=available_metrics[0],
                            color_column='ticker',
                            hover_data=['filing_year'],
                            title=f'MD&A {available_metrics[0].replace("_", " ").title()} Over Time'
                        )
                        
                        interactive_path = os.path.join(args.output_dir, 'mda_sentiment_interactive.html')
                        fig.write_html(interactive_path)
                        visualizations.append(interactive_path)
    
    if text_df is not None:
        logger.info("Creating text analysis visualizations")
        
        # Word frequency visualization
        if 'word' in text_df.columns and 'frequency' in text_df.columns:
            # Group by section if available
            if 'section' in text_df.columns:
                for section, section_data in text_df.groupby('section'):
                    # Get top words
                    top_words = section_data.sort_values('frequency', ascending=False).head(30)
                    
                    # Create bar chart
                    plt.figure(figsize=(12, 8))
                    plt.bar(top_words['word'], top_words['frequency'])
                    plt.title(f'Top Words in {section.replace("item_", "Item ").title()}', fontsize=16)
                    plt.xlabel('Word', fontsize=14)
                    plt.ylabel('Frequency', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    static_path = os.path.join(args.output_dir, f'{section}_top_words')
                    if args.format in ['png', 'all']:
                        plt.savefig(f"{static_path}.png", dpi=300, bbox_inches='tight')
                        visualizations.append(f"{static_path}.png")
                    if args.format in ['pdf', 'all']:
                        plt.savefig(f"{static_path}.pdf", bbox_inches='tight')
                        visualizations.append(f"{static_path}.pdf")
                    plt.close()
    
    # Create combined visualizations if multiple datasets are available
    if financial_df is not None and sentiment_df is not None:
        logger.info("Creating combined visualizations")
        
        # Check if we can merge the datasets
        if 'ticker' in financial_df.columns and 'ticker' in sentiment_df.columns and 'filing_year' in financial_df.columns and 'filing_year' in sentiment_df.columns:
            # Filter sentiment to MD&A section
            if 'section' in sentiment_df.columns:
                mda_sentiment = sentiment_df[sentiment_df['section'] == 'item_7'].copy()
                
                # Merge datasets
                if 'lexicon_net_score' in mda_sentiment.columns:
                    # Group by ticker and year
                    financial_agg = financial_df.groupby(['ticker', 'filing_year']).agg({
                        'revenue': 'mean',
                        'net_income': 'mean' if 'net_income' in financial_df.columns else None,
                        'profit_margin': 'mean' if 'profit_margin' in financial_df.columns else None
                    }).reset_index()
                    
                    sentiment_agg = mda_sentiment.groupby(['ticker', 'filing_year']).agg({
                        'lexicon_net_score': 'mean'
                    }).reset_index()
                    
                    # Merge
                    merged_df = pd.merge(financial_agg, sentiment_agg, on=['ticker', 'filing_year'])
                    
                    # Create scatter plot of sentiment vs. financial metrics
                    if 'profit_margin' in merged_df.columns:
                        # Interactive scatter plot
                        if args.format in ['html', 'all']:
                            fig = create_interactive_scatter(
                                merged_df,
                                x_column='lexicon_net_score',
                                y_column='profit_margin',
                                color_column='ticker',
                                size_column='revenue',
                                hover_data=['filing_year'],
                                title='Sentiment vs. Profit Margin'
                            )
                            
                            interactive_path = os.path.join(args.output_dir, 'sentiment_vs_profit_margin.html')
                            fig.write_html(interactive_path)
                            visualizations.append(interactive_path)
                    
                    # Create bubble chart
                    if 'revenue' in merged_df.columns and 'profit_margin' in merged_df.columns:
                        # Interactive bubble chart
                        if args.format in ['html', 'all']:
                            fig = create_bubble_chart(
                                merged_df,
                                x_column='lexicon_net_score',
                                y_column='profit_margin',
                                size_column='revenue',
                                color_column='ticker',
                                text_column='filing_year',
                                title='Financial Performance and Sentiment Analysis'
                            )
                            
                            interactive_path = os.path.join(args.output_dir, 'financial_sentiment_bubble_chart.html')
                            fig.write_html(interactive_path)
                            visualizations.append(interactive_path)
    
    logger.info(f"Created {len(visualizations)} visualizations in {args.output_dir}")
    return visualizations

def run_pipeline(args):
    """Run the full analysis pipeline."""
    logger.info(f"Running full analysis pipeline for tickers: {args.tickers}")
    
    # Create output directories
    data_dir = os.path.join(args.output_dir, 'data')
    results_dir = os.path.join(args.output_dir, 'results')
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Define file paths
    raw_file = os.path.join(data_dir, 'filings.pkl')
    processed_file = os.path.join(data_dir, 'processed_filings.pkl')
    text_file = os.path.join(results_dir, 'text_analysis.pkl')
    financial_file = os.path.join(results_dir, 'financial_analysis.pkl')
    sentiment_file = os.path.join(results_dir, 'sentiment_analysis.pkl')
    
    # Run each step
    
    # 1. Download filings
    download_args = argparse.Namespace(
        tickers=args.tickers,
        years=args.years,
        output=raw_file
    )
    filings_df = download_filings(download_args)
    
    # 2. Preprocess filings
    preprocess_args = argparse.Namespace(
        input=raw_file,
        output=processed_file,
        extract_sections=True,
        clean_text=True
    )
    processed_df = preprocess_filings(preprocess_args)
    
    # 3. Text analysis
    text_args = argparse.Namespace(
        input=processed_file,
        output=text_file,
        sections=['item_1a', 'item_7']
    )
    text_df = text_analysis(text_args)
    
    # 4. Financial analysis
    financial_args = argparse.Namespace(
        input=processed_file,
        output=financial_file
    )
    financial_df = financial_analysis(financial_args)
    
    # 5. Sentiment analysis
    sentiment_args = argparse.Namespace(
        input=processed_file,
        output=sentiment_file,
        sections=['item_7']
    )
    sentiment_df = sentiment_analysis(sentiment_args)
    
    # 6. Create visualizations
    viz_args = argparse.Namespace(
        financial=financial_file,
        sentiment=sentiment_file,
        text=text_file,
        output_dir=viz_dir,
        format='all'
    )
    visualizations = create_visualizations(viz_args)
    
    # Create summary report
    report_path = os.path.join(args.output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"10-K Analysis Report\n")
        f.write(f"=================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Companies analyzed: {', '.join(args.tickers)}\n")
        f.write(f"Years: {', '.join(map(str, args.years)) if args.years else 'All available'}\n\n")
        
        f.write(f"Data summary:\n")
        f.write(f"- Number of filings: {len(filings_df)}\n")
        f.write(f"- Date range: {filings_df['filing_date'].min()} to {filings_df['filing_date'].max()}\n\n")
        
        f.write(f"Analysis performed:\n")
        f.write(f"- Text analysis of sections: item_1a (Risk Factors), item_7 (MD&A)\n")
        f.write(f"- Financial metrics extraction\n")
        f.write(f"- Sentiment analysis of MD&A section\n\n")
        
        f.write(f"Visualizations generated:\n")
        for viz in visualizations:
            f.write(f"- {os.path.basename(viz)}\n")
    
    logger.info(f"Pipeline completed. Summary report saved to {report_path}")
    
    return {
        'filings_df': filings_df,
        'processed_df': processed_df,
        'text_df': text_df,
        'financial_df': financial_df,
        'sentiment_df': sentiment_df,
        'visualizations': visualizations,
        'report_path': report_path
    }

def main():
    """Main function."""
    args = parse_args()
    
    # Execute the appropriate command
    if args.command == 'download':
        download_filings(args)
    elif args.command == 'preprocess':
        preprocess_filings(args)
    elif args.command == 'text-analysis':
        text_analysis(args)
    elif args.command == 'financial-analysis':
        financial_analysis(args)
    elif args.command == 'sentiment-analysis':
        sentiment_analysis(args)
    elif args.command == 'visualize':
        create_visualizations(args)
    elif args.command == 'pipeline':
        run_pipeline(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())