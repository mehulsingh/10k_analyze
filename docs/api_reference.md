# API Reference

This document provides detailed information about the classes and functions in the 10-K Analysis Toolkit.

## Table of Contents

- [Data Module](#data-module)
  - [SECDataLoader](#secdataloader)
  - [FilingPreprocessor](#filingpreprocessor)
- [Analysis Module](#analysis-module)
  - [TextAnalyzer](#textanalyzer)
  - [FinancialAnalyzer](#financialanalyzer)
  - [SentimentAnalyzer](#sentimentanalyzer)
- [Visualization Module](#visualization-module)
  - [Basic Plots](#basic-plots)
  - [Advanced Plots](#advanced-plots)
- [Utils Module](#utils-module)
  - [Helpers](#helpers)
- [Command-Line Interface](#command-line-interface)

## Data Module

The `data` module provides tools for downloading and preprocessing 10-K filings.

### SECDataLoader

`SECDataLoader` is responsible for downloading 10-K filings from the SEC EDGAR database.

```python
from tenk_toolkit.data import SECDataLoader
```

#### Constructor

```python
SECDataLoader(cache_dir='./data/cache')
```

- `cache_dir` (str): Directory to cache downloaded filings

#### Methods

- `get_cik_for_ticker(ticker)`
  - Get the Central Index Key (CIK) for a given ticker symbol
  - Parameters:
    - `ticker` (str): Ticker symbol
  - Returns:
    - CIK number as string, or None if not found

- `get_filing_links(cik, filing_type='10-K', start_year=None, end_year=None)`
  - Get links to filings for a company based on CIK
  - Parameters:
    - `cik` (str): CIK number of the company
    - `filing_type` (str): Type of filing to retrieve (default: '10-K')
    - `start_year` (int): Starting year for filings
    - `end_year` (int): Ending year for filings
  - Returns:
    - List of filing links with metadata

- `get_filing_content(filing_link)`
  - Get the content of a filing from its link
  - Parameters:
    - `filing_link` (str): Link to the filing detail page
  - Returns:
    - Dictionary containing filing content and metadata

- `load_filings(tickers, years=None, filing_type='10-K')`
  - Load 10-K filings for multiple tickers and years
  - Parameters:
    - `tickers` (list): List of ticker symbols
    - `years` (list): List of years to include
    - `filing_type` (str): Type of filing to retrieve (default: '10-K')
  - Returns:
    - DataFrame containing filing data and metadata

#### Example

```python
# Create loader
loader = SECDataLoader()

# Download filings for Apple and Microsoft for 2020-2022
filings = loader.load_filings(['AAPL', 'MSFT'], years=[2020, 2021, 2022])

# Display the result
print(f"Downloaded {len(filings)} filings")
print(filings[['ticker', 'filing_year', 'company_name']].head())
```

### FilingPreprocessor

`FilingPreprocessor` is responsible for extracting sections and cleaning text from 10-K filings.

```python
from tenk_toolkit.data import FilingPreprocessor
```

#### Constructor

```python
FilingPreprocessor()
```

#### Methods

- `clean_html(html_content)`
  - Clean HTML content, removing scripts, styles, and other non-content elements
  - Parameters:
    - `html_content` (str): HTML content to clean
  - Returns:
    - Cleaned text content

- `extract_section(text, section_name)`
  - Extract a specific section from the 10-K text
  - Parameters:
    - `text` (str): Full text of the 10-K filing
    - `section_name` (str): Name of the section to extract
  - Returns:
    - Text of the requested section

- `extract_all_sections(text)`
  - Extract all sections from the 10-K text
  - Parameters:
    - `text` (str): Full text of the 10-K filing
  - Returns:
    - Dictionary with section names as keys and section texts as values

- `clean_text(text, remove_stopwords=True, lemmatize=True)`
  - Clean text by removing punctuation, numbers, and optionally stopwords, and lemmatizing
  - Parameters:
    - `text` (str): Text to clean
    - `remove_stopwords` (bool): Whether to remove stopwords
    - `lemmatize` (bool): Whether to lemmatize the text
  - Returns:
    - Cleaned text

- `process_filings(filings_df, extract_sections=True, clean_text=False)`
  - Process a DataFrame of 10-K filings
  - Parameters:
    - `filings_df` (DataFrame): DataFrame containing filings data
    - `extract_sections` (bool): Whether to extract sections from the filings
    - `clean_text` (bool): Whether to clean the text
  - Returns:
    - Processed DataFrame

#### Example

```python
# Create preprocessor
preprocessor = FilingPreprocessor()

# Process filings
processed_filings = preprocessor.process_filings(
    filings_df,
    extract_sections=True,
    clean_text=True
)

# Check the extracted sections
section_cols = [col for col in processed_filings.columns if col.startswith('section_')]
print(f"Extracted {len(section_cols)} sections")
```

## Analysis Module

The `analysis` module provides tools for analyzing the content of 10-K filings.

### TextAnalyzer

`TextAnalyzer` is responsible for analyzing text content in 10-K filings.

```python
from tenk_toolkit.analysis import TextAnalyzer
```

#### Constructor

```python
TextAnalyzer()
```

#### Methods

- `preprocess_text(text)`
  - Preprocess text for analysis
  - Parameters:
    - `text` (str): Text to preprocess
  - Returns:
    - Preprocessed text

- `get_tokens(text, remove_stopwords=True, lemmatize=True, min_length=3)`
  - Get tokens from text
  - Parameters:
    - `text` (str): Text to tokenize
    - `remove_stopwords` (bool): Whether to remove stopwords
    - `lemmatize` (bool): Whether to lemmatize tokens
    - `min_length` (int): Minimum token length to keep
  - Returns:
    - List of tokens

- `get_word_frequencies(text, top_n=50, remove_stopwords=True, lemmatize=True, min_length=3)`
  - Get word frequencies from text
  - Parameters:
    - `text` (str): Text to analyze
    - `top_n` (int): Number of top words to return
    - `remove_stopwords` (bool): Whether to remove stopwords
    - `lemmatize` (bool): Whether to lemmatize tokens
    - `min_length` (int): Minimum token length to keep
  - Returns:
    - DataFrame with words and frequencies

- `calculate_sentiment(text)`
  - Calculate sentiment scores for text
  - Parameters:
    - `text` (str): Text to analyze
  - Returns:
    - Dictionary with sentiment metrics

- `extract_topics(texts, n_topics=5, n_top_words=10, method='lda')`
  - Extract topics from a collection of texts
  - Parameters:
    - `texts` (list): List of text documents
    - `n_topics` (int): Number of topics to extract
    - `n_top_words` (int): Number of top words per topic
    - `method` (str): Topic modeling method ('lda' or 'nmf')
  - Returns:
    - Tuple of (topic model, vectorizer, topic words, document-topic matrix)

- `analyze_filings(filings_df, sections=None)`
  - Analyze filings and extract text metrics
  - Parameters:
    - `filings_df` (DataFrame): DataFrame containing filings data
    - `sections` (list): List of section names to analyze
  - Returns:
    - DataFrame with text analysis metrics

- `compare_filings(filings_df, groupby='ticker', section='full_text')`
  - Compare filings across companies or time periods
  - Parameters:
    - `filings_df` (DataFrame): DataFrame containing filings data
    - `groupby` (str): Column to group by ('ticker' or 'filing_year')
    - `section` (str): Section to analyze
  - Returns:
    - DataFrame with comparison metrics

- `plot_sentiment_trends(metrics_df, groupby='filing_year', section='full_text')`
  - Plot sentiment trends over time or across companies
  - Parameters:
    - `metrics_df` (DataFrame): DataFrame with text metrics
    - `groupby` (str): Column to group by ('ticker' or 'filing_year')
    - `section` (str): Section to analyze
  - Returns:
    - Figure with sentiment trends plot

#### Example

```python
# Create analyzer
analyzer = TextAnalyzer()

# Analyze text content
text_metrics = analyzer.analyze_filings(
    processed_filings,
    sections=['item_1a', 'item_7']
)

# Get word frequencies for risk factors section
word_freq = analyzer.get_word_frequencies(
    processed_filings.iloc[0]['section_item_1a'],
    top_n=20
)

# Extract topics from MD&A sections
texts = processed_filings['section_item_7'].tolist()
model, vectorizer, topic_words, doc_topic_matrix = analyzer.extract_topics(
    texts,
    n_topics=5,
    n_top_words=10
)
```

### FinancialAnalyzer

`FinancialAnalyzer` is responsible for extracting and analyzing financial metrics from 10-K filings.

```python
from tenk_toolkit.analysis import FinancialAnalyzer
```

#### Constructor

```python
FinancialAnalyzer()
```

#### Methods

- `extract_tables_from_html(html_content)`
  - Extract tables from HTML content
  - Parameters:
    - `html_content` (str): HTML content to extract tables from
  - Returns:
    - List of tables as pandas DataFrames

- `extract_financial_tables(html_content, min_rows=3, min_cols=2)`
  - Extract financial tables from HTML content
  - Parameters:
    - `html_content` (str): HTML content to extract tables from
    - `min_rows` (int): Minimum number of rows for a table to be considered
    - `min_cols` (int): Minimum number of columns for a table to be considered
  - Returns:
    - List of financial tables as pandas DataFrames

- `parse_dollar_amount(text)`
  - Parse a dollar amount from text
  - Parameters:
    - `text` (str): Text containing a dollar amount
  - Returns:
    - Parsed dollar amount or None if parsing fails

- `extract_metric_from_tables(tables, metric_patterns)`
  - Extract a metric from tables using pattern matching
  - Parameters:
    - `tables` (list): List of tables
    - `metric_patterns` (list): List of regex patterns to match the metric
  - Returns:
    - Extracted metric value or None if not found

- `extract_metrics_from_filing(html_content)`
  - Extract financial metrics from a filing
  - Parameters:
    - `html_content` (str): HTML content of the filing
  - Returns:
    - Dictionary of extracted metrics

- `calculate_derived_metrics(metrics)`
  - Calculate derived financial metrics
  - Parameters:
    - `metrics` (dict): Dictionary of financial metrics
  - Returns:
    - Dictionary with additional derived metrics

- `analyze_filings(filings_df)`
  - Analyze filings and extract financial metrics
  - Parameters:
    - `filings_df` (DataFrame): DataFrame containing filings data
  - Returns:
    - DataFrame with financial metrics

- `compare_financials(metrics_df, companies=None, years=None)`
  - Compare financial metrics across companies and years
  - Parameters:
    - `metrics_df` (DataFrame): DataFrame with financial metrics
    - `companies` (list): List of companies to include
    - `years` (list): List of years to include
  - Returns:
    - Dictionary with comparative analysis

#### Example

```python
# Create analyzer
analyzer = FinancialAnalyzer()

# Extract financial metrics
financial_metrics = analyzer.analyze_filings(processed_filings)

# Calculate derived metrics
metrics = {
    'revenue': 100000000,
    'net_income': 20000000,
    'total_assets': 200000000
}
derived_metrics = analyzer.calculate_derived_metrics(metrics)

# Compare financials across companies
comparison = analyzer.compare_financials(financial_metrics)
```

### SentimentAnalyzer

`SentimentAnalyzer` is responsible for analyzing sentiment in 10-K filings.

```python
from tenk_toolkit.analysis import SentimentAnalyzer
```

#### Constructor

```python
SentimentAnalyzer(load_lexicons=True)
```

- `load_lexicons` (bool): Whether to load financial sentiment lexicons

#### Methods

- `preprocess_text(text)`
  - Preprocess text for sentiment analysis
  - Parameters:
    - `text` (str): Text to preprocess
  - Returns:
    - Preprocessed text

- `extract_sentences(text)`
  - Extract sentences from text
  - Parameters:
    - `text` (str): Text to extract sentences from
  - Returns:
    - List of sentences

- `analyze_textblob_sentiment(text)`
  - Analyze sentiment using TextBlob
  - Parameters:
    - `text` (str): Text to analyze
  - Returns:
    - Dictionary with sentiment metrics

- `analyze_lexicon_sentiment(text)`
  - Analyze sentiment using financial lexicons
  - Parameters:
    - `text` (str): Text to analyze
  - Returns:
    - Dictionary with sentiment metrics

- `extract_sentiment_by_section(filing_data, section_prefix='section_')`
  - Extract sentiment metrics for each section in a filing
  - Parameters:
    - `filing_data` (Series): Filing data with section text
    - `section_prefix` (str): Prefix for section columns
  - Returns:
    - Dictionary with sentiment metrics for each section

- `analyze_filings(filings_df, sections=None)`
  - Analyze sentiment in filings
  - Parameters:
    - `filings_df` (DataFrame): DataFrame containing filings data
    - `sections` (list): List of section names to analyze
  - Returns:
    - DataFrame with sentiment metrics

- `plot_sentiment_trends(sentiment_df, section='item_7', metric='lexicon_net_score')`
  - Plot sentiment trends over time
  - Parameters:
    - `sentiment_df` (DataFrame): DataFrame with sentiment metrics
    - `section` (str): Section to analyze
    - `metric` (str): Sentiment metric to plot
  - Returns:
    - Figure with sentiment trends plot

- `plot_sentiment_comparison(sentiment_df, section='item_7', metric='lexicon_net_score')`
  - Create a comparison bar chart of sentiment across companies
  - Parameters:
    - `sentiment_df` (DataFrame): DataFrame with sentiment metrics
    - `section` (str): Section to analyze
    - `metric` (str): Sentiment metric to plot
  - Returns:
    - Figure with sentiment comparison plot

#### Example

```python
# Create analyzer
analyzer = SentimentAnalyzer()

# Analyze sentiment in MD&A sections
sentiment_metrics = analyzer.analyze_filings(
    processed_filings,
    sections=['item_7']
)

# Plot sentiment trends
fig = analyzer.plot_sentiment_trends(
    sentiment_metrics,
    section='item_7',
    metric='lexicon_net_score'
)
```

## Visualization Module

The `visualization` module provides tools for creating visualizations of 10-K filing analysis.

### Basic Plots

The `basic_plots` module provides static visualizations using matplotlib and seaborn.

```python
from tenk_toolkit.visualization import (
    plot_time_series, plot_metric_distribution, create_wordcloud,
    plot_sentiment_analysis, plot_comparative_metrics, plot_correlation_heatmap,
    plot_stacked_bar, plot_year_over_year_comparison
)
```

#### Functions

- `plot_time_series(df, date_column, value_column, company_column=None, title=None, figsize=(12, 6))`
  - Plot time series data from 10-K filings
  - Returns: Figure and axes objects

- `plot_metric_distribution(df, metric_column, group_column=None, title=None, figsize=(12, 6), kind='hist')`
  - Plot distribution of a metric from 10-K filings
  - Returns: Figure and axes objects

- `create_wordcloud(text_data, title='Word Cloud of 10-K Filings', figsize=(12, 8), background_color='white', colormap='viridis', max_words=200)`
  - Create a word cloud visualization from 10-K filing text
  - Returns: Figure and axes objects

- `plot_sentiment_analysis(df, date_column, sentiment_column, company_column=None, title=None, figsize=(14, 7))`
  - Plot sentiment analysis results from 10-K filings over time
  - Returns: Figure and axes objects

- `plot_comparative_metrics(df, metrics, company_column, title=None, figsize=(14, 8))`
  - Create a comparative bar chart for multiple metrics across companies
  - Returns: Figure and axes objects

- `plot_correlation_heatmap(df, columns=None, title='Correlation Heatmap of 10-K Metrics', figsize=(12, 10), cmap='coolwarm', annot=True)`
  - Create a correlation heatmap for 10-K filing metrics
  - Returns: Figure and axes objects

- `plot_stacked_bar(data, analysis_type, selected_year=None, selected_therapy_area=None, selected_geography=None, title=None, figsize=(14, 8), color_palette='viridis', sort_by=None)`
  - Create a stacked bar chart for 10-K filings analysis
  - Returns: Figure and axes objects

- `plot_year_over_year_comparison(df, date_column, value_column, group_column=None, title='Year-over-Year Comparison', normalize=False)`
  - Create bar chart comparing metrics year over year
  - Returns: Figure and axes objects

#### Example

```python
# Create time series plot
fig, ax = plot_time_series(
    financial_metrics,
    date_column='filing_date',
    value_column='revenue',
    company_column='ticker',
    title='Revenue Over Time by Company'
)

# Create word cloud
fig, ax = create_wordcloud(
    risk_factors_text,
    title='Risk Factors Word Cloud',
    colormap='Reds'
)

# Create comparative metrics plot
fig, ax = plot_comparative_metrics(
    financial_metrics,
    metrics=['revenue', 'net_income', 'operating_income'],
    company_column='ticker',
    title='Financial Metrics Comparison'
)
```

### Advanced Plots

The `advanced_plots` module provides interactive visualizations using Plotly.

```python
from tenk_toolkit.visualization import (
    create_interactive_time_series, create_heatmap_over_time,
    create_interactive_scatter, create_interactive_bar_chart,
    create_interactive_pie_chart, create_interactive_heatmap,
    create_radar_chart, plot_embeddings_2d, create_waterfall_chart,
    create_sunburst_chart, create_treemap_chart, create_bubble_chart,
    create_interactive_dashboard
)
```

#### Functions

- `create_interactive_time_series(df, date_column, value_columns, company_column=None, title='Interactive Time Series of 10-K Metrics')`
  - Create an interactive time series plot using Plotly
  - Returns: Plotly figure

- `create_heatmap_over_time(df, date_column, company_column, value_column, title='Heatmap of Metrics Over Time')`
  - Create a heatmap showing the evolution of a metric over time for different companies
  - Returns: Plotly figure

- `create_interactive_scatter(df, x_column, y_column, color_column=None, size_column=None, hover_data=None, title='Interactive Scatter Plot')`
  - Create an interactive scatter plot using Plotly
  - Returns: Plotly figure

- `create_interactive_bar_chart(df, x_column, y_column, color_column=None, title='Interactive Bar Chart')`
  - Create an interactive bar chart using Plotly
  - Returns: Plotly figure

- `create_interactive_pie_chart(df, names_column, values_column, title='Interactive Pie Chart')`
  - Create an interactive pie chart using Plotly
  - Returns: Plotly figure

- `create_interactive_heatmap(df, x_column, y_column, z_column, title='Interactive Heatmap')`
  - Create an interactive heatmap using Plotly
  - Returns: Plotly figure

- `create_radar_chart(df, categories, value_column, group_column, title='Radar Chart Comparison')`
  - Create a radar chart (spider plot) for comparing metrics across companies
  - Returns: Plotly figure

- `plot_embeddings_2d(embeddings, labels=None, method='pca', title=None, figsize=(12, 10))`
  - Plot 2D visualization of document embeddings from 10-K filings
  - Returns: Figure and axes objects

- `create_waterfall_chart(df, category_column, value_column, title='Waterfall Chart')`
  - Create a waterfall chart showing contributions to a total
  - Returns: Plotly figure

- `create_sunburst_chart(df, path_columns, value_column, title='Sunburst Chart')`
  - Create a sunburst chart for hierarchical data
  - Returns: Plotly figure

- `create_treemap_chart(df, path_columns, value_column, color_column=None, title='Treemap Chart')`
  - Create a treemap chart for hierarchical data
  - Returns: Plotly figure

- `create_bubble_chart(df, x_column, y_column, size_column, color_column, text_column=None, title='Bubble Chart')`
  - Create a bubble chart with multiple dimensions
  - Returns: Plotly figure

- `create_interactive_dashboard(financial_df, sentiment_df=None, topic_df=None)`
  - Create an interactive dashboard with multiple visualizations
  - Returns: Plotly figure

#### Example

```python
# Create interactive time series
fig = create_interactive_time_series(
    financial_metrics,
    date_column='filing_date',
    value_columns=['revenue', 'net_income'],
    company_column='ticker',
    title='Financial Metrics Over Time'
)

# Create interactive scatter plot
fig = create_interactive_scatter(
    merged_df,
    x_column='lexicon_net_score',
    y_column='profit_margin',
    color_column='ticker',
    size_column='revenue',
    hover_data=['filing_year'],
    title='MD&A Sentiment vs. Profit Margin'
)

# Create dashboard
fig = create_interactive_dashboard(
    financial_df=financial_metrics,
    sentiment_df=sentiment_metrics
)
```

## Utils Module

The `utils` module provides utility functions for common operations.

### Helpers

```python
from tenk_toolkit.utils import (
    ensure_dir, setup_matplotlib_style, normalize_ticker, format_number,
    filter_dataframe, get_date_range_filter, create_report_header,
    merge_dataframes, extract_year_from_date, calculate_growth_rates,
    calculate_summary_statistics, detect_outliers
)
```

#### Functions

- `ensure_dir(dir_path)`
  - Ensure a directory exists, creating it if necessary

- `setup_matplotlib_style()`
  - Set up consistent matplotlib styling for all visualizations

- `normalize_ticker(ticker)`
  - Normalize ticker symbol (uppercase, remove special characters)

- `format_number(value, precision=2, prefix='', suffix='', with_commas=True, abbreviate=False, is_percentage=False)`
  - Format a number for display with various options

- `filter_dataframe(df, filters)`
  - Apply filters to a DataFrame

- `get_date_range_filter(df, date_column, start_date=None, end_date=None)`
  - Create a date range filter mask for a DataFrame

- `create_report_header(title, subtitle=None, date=None)`
  - Create a formatted header for reports

- `merge_dataframes(df1, df2, on=None, how='inner')`
  - Merge two DataFrames with error handling

- `extract_year_from_date(df, date_column, year_column=None)`
  - Extract year from a date column in a DataFrame

- `calculate_growth_rates(df, group_column, date_column, value_column, growth_column=None)`
  - Calculate year-over-year growth rates for a value column

- `calculate_summary_statistics(df, group_columns, value_columns, include_count=True)`
  - Calculate summary statistics for value columns grouped by specified columns

- `detect_outliers(df, column, method='iqr', threshold=1.5)`
  - Detect outliers in a DataFrame column

#### Example

```python
# Format a number
formatted_revenue = format_number(
    123456789,
    prefix='$',
    abbreviate=True
)  # Returns "$123.46M"

# Calculate growth rates
growth_df = calculate_growth_rates(
    financial_metrics,
    group_column='ticker',
    date_column='filing_date',
    value_column='revenue'
)
```

## Command-Line Interface

The toolkit provides a command-line interface (CLI) for common tasks:

```bash
tenk [command] [options]
```

### Commands

- `download`: Download 10-K filings from SEC EDGAR
- `preprocess`: Preprocess 10-K filings (extract sections, clean text)
- `text-analysis`: Perform text analysis on 10-K filings
- `financial-analysis`: Extract and analyze financial metrics from 10-K filings
- `sentiment-analysis`: Analyze sentiment in 10-K filings
- `visualize`: Create visualizations from analysis results
- `pipeline`: Run the full analysis pipeline

### Example

```bash
# Download filings
tenk download --tickers AAPL MSFT --years 2020 2021 2022 --output data/filings.pkl

# Run the full pipeline
tenk pipeline --tickers AAPL MSFT --years 2020 2021 2022 --output-dir results
```

For more details, run `tenk --help` or `tenk [command] --help`.