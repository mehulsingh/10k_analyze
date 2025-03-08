# Examples

This document provides examples of how to use the 10-K Analysis Toolkit for various analysis tasks.

## Table of Contents
- [Basic Usage Examples](#basic-usage-examples)
- [Financial Analysis Examples](#financial-analysis-examples)
- [Text Analysis Examples](#text-analysis-examples)
- [Sentiment Analysis Examples](#sentiment-analysis-examples)
- [Visualization Examples](#visualization-examples)
- [Advanced Examples](#advanced-examples)

## Basic Usage Examples

### Download and Preprocess Filings

```python
from tenk_toolkit.data import SECDataLoader, FilingPreprocessor

# Download filings
loader = SECDataLoader()
filings = loader.load_filings(['AAPL', 'MSFT', 'GOOGL'], years=[2020, 2021, 2022])

# Preprocess filings
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings, extract_sections=True, clean_text=True)

# Save to file
processed_filings.to_pickle('processed_filings.pkl')
```

### Load Processed Filings

```python
import pandas as pd

# Load processed filings
processed_filings = pd.read_pickle('processed_filings.pkl')

# Display available companies and years
print(f"Companies: {processed_filings['ticker'].unique()}")
print(f"Years: {sorted(processed_filings['filing_year'].unique())}")
```

### Extract a Specific Section

```python
from tenk_toolkit.data import FilingPreprocessor

# Create preprocessor
preprocessor = FilingPreprocessor()

# Load a filing
filing_text = processed_filings.iloc[0]['filing_text']

# Extract Risk Factors section
risk_factors = preprocessor.extract_section(filing_text, 'item_1a')

print(f"Extracted {len(risk_factors)} characters from Risk Factors section")
```

## Financial Analysis Examples

### Extract Financial Metrics

```python
from tenk_toolkit.analysis import FinancialAnalyzer

# Create analyzer
analyzer = FinancialAnalyzer()

# Extract financial metrics
financial_metrics = analyzer.analyze_filings(processed_filings)

# Display metrics for the first filing
first_filing_metrics = financial_metrics.iloc[0]
print(f"Company: {first_filing_metrics['ticker']}")
print(f"Year: {first_filing_metrics['filing_year']}")
print(f"Revenue: ${first_filing_metrics['revenue']:,.0f}")
print(f"Net Income: ${first_filing_metrics['net_income']:,.0f}")
print(f"Profit Margin: {first_filing_metrics['profit_margin']:.2%}")
```

### Compare Financial Metrics Across Companies

```python
from tenk_toolkit.analysis import FinancialAnalyzer
import matplotlib.pyplot as plt
from tenk_toolkit.visualization import plot_comparative_metrics

# Create analyzer
analyzer = FinancialAnalyzer()

# Extract financial metrics
financial_metrics = analyzer.analyze_filings(processed_filings)

# Group by company and calculate mean metrics
company_metrics = financial_metrics.groupby('ticker').mean()

# Compare metrics
comparison = analyzer.compare_financials(financial_metrics)

# Plot comparative metrics
fig, ax = plot_comparative_metrics(
    financial_metrics,
    metrics=['revenue', 'net_income', 'operating_income'],
    company_column='ticker',
    title='Financial Metrics Comparison by Company'
)
plt.show()
```

### Calculate Financial Growth Rates

```python
from tenk_toolkit.analysis import FinancialAnalyzer
from tenk_toolkit.utils import calculate_growth_rates
import matplotlib.pyplot as plt

# Create analyzer
analyzer = FinancialAnalyzer()

# Extract financial metrics
financial_metrics = analyzer.analyze_filings(processed_filings)

# Calculate growth rates
growth_df = calculate_growth_rates(
    financial_metrics,
    group_column='ticker',
    date_column='filing_date',
    value_column='revenue',
    growth_column='revenue_growth'
)

# Plot growth rates
plt.figure(figsize=(12, 6))
for ticker, group in growth_df.groupby('ticker'):
    plt.plot(group['filing_date'], group['revenue_growth'] * 100, marker='o', label=ticker)
plt.title('Revenue Growth Rates', fontsize=16, fontweight='bold')
plt.xlabel('Filing Date', fontsize=14)
plt.ylabel('Growth Rate (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
```

## Text Analysis Examples

### Word Frequency Analysis

```python
from tenk_toolkit.analysis import TextAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Create analyzer
analyzer = TextAnalyzer()

# Get word frequencies for Risk Factors section
risk_text = processed_filings[processed_filings['ticker'] == 'AAPL'].iloc[0]['section_item_1a']
word_freq = analyzer.get_word_frequencies(risk_text, top_n=20, remove_stopwords=True)

# Plot word frequencies
plt.figure(figsize=(12, 6))
sns.barplot(x='frequency', y='word', data=word_freq)
plt.title('Top 20 Words in Risk Factors Section', fontsize=16, fontweight='bold')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Word', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

### Create Word Cloud

```python
from tenk_toolkit.visualization import create_wordcloud
import matplotlib.pyplot as plt

# Create word cloud for Risk Factors section
risk_text = processed_filings[processed_filings['ticker'] == 'AAPL'].iloc[0]['section_item_1a']
fig, ax = create_wordcloud(
    risk_text,
    title='Risk Factors Word Cloud - Apple',
    colormap='Reds',
    max_words=100
)
plt.show()
```

### Topic Modeling

```python
from tenk_toolkit.analysis import TextAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create analyzer
analyzer = TextAnalyzer()

# Extract MD&A sections
mda_texts = processed_filings['section_item_7'].dropna().tolist()

# Perform topic modeling
model, vectorizer, topic_words, doc_topic_matrix = analyzer.extract_topics(
    mda_texts,
    n_topics=5,
    n_top_words=10,
    method='lda'
)

# Visualize topics
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, (topic_idx, words) in enumerate(topic_words):
    # Create word importance dictionary
    word_importance = {word: 1/(j+1) for j, word in enumerate(words)}
    
    # Create word cloud
    wordcloud = WordCloud(
        background_color='white',
        colormap=f'Blues_{i+3}',
        max_words=10
    ).generate_from_frequencies(word_importance)
    
    # Add to subplot
    ax = axes[i]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Topic {topic_idx+1}', fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle('Topics in MD&A Sections', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```

## Sentiment Analysis Examples

### Analyze Sentiment in MD&A Section

```python
from tenk_toolkit.analysis import SentimentAnalyzer
import matplotlib.pyplot as plt
from tenk_toolkit.visualization import plot_sentiment_analysis

# Create analyzer
analyzer = SentimentAnalyzer()

# Analyze sentiment in MD&A sections
sentiment_metrics = analyzer.analyze_filings(processed_filings, sections=['item_7'])

# Filter to MD&A section
mda_sentiment = sentiment_metrics[sentiment_metrics['section'] == 'item_7']

# Plot sentiment over time
fig, ax = plot_sentiment_analysis(
    mda_sentiment,
    date_column='filing_date',
    sentiment_column='lexicon_net_score',
    company_column='ticker',
    title='MD&A Sentiment Over Time by Company'
)
plt.show()
```

### Compare Positive and Negative Scores

```python
from tenk_toolkit.analysis import SentimentAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Create analyzer
analyzer = SentimentAnalyzer()

# Analyze sentiment in Risk Factors section
sentiment_metrics = analyzer.analyze_filings(processed_filings, sections=['item_1a'])

# Filter to Risk Factors section
risk_sentiment = sentiment_metrics[sentiment_metrics['section'] == 'item_1a']

# Calculate average positive and negative scores by company
sentiment_by_company = risk_sentiment.groupby('ticker')[
    ['lexicon_positive_score', 'lexicon_negative_score']
].mean().reset_index()

# Melt for easier plotting
sentiment_melted = pd.melt(
    sentiment_by_company,
    id_vars='ticker',
    value_vars=['lexicon_positive_score', 'lexicon_negative_score'],
    var_name='score_type',
    value_name='score'
)

# Rename for better display
sentiment_melted['score_type'] = sentiment_melted['score_type'].map({
    'lexicon_positive_score': 'Positive',
    'lexicon_negative_score': 'Negative'
})

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='ticker', y='score', hue='score_type', data=sentiment_melted)
plt.title('Positive vs. Negative Scores in Risk Factors by Company', fontsize=16, fontweight='bold')
plt.xlabel('Company', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.legend(title='Score Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

### Sentiment Trend Analysis

```python
from tenk_toolkit.analysis import SentimentAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

# Create analyzer
analyzer = SentimentAnalyzer()

# Analyze sentiment in MD&A sections
sentiment_metrics = analyzer.analyze_filings(processed_filings, sections=['item_7'])

# Calculate average sentiment by year for each company
sentiment_by_year = sentiment_metrics.groupby(['ticker', 'filing_year'])['lexicon_net_score'].mean().reset_index()

# Plot sentiment trends
plt.figure(figsize=(12, 6))
for ticker, group in sentiment_by_year.groupby('ticker'):
    plt.plot(group['filing_year'], group['lexicon_net_score'], marker='o', linewidth=2, label=ticker)
plt.title('MD&A Sentiment Trends by Company', fontsize=16, fontweight='bold')
plt.xlabel('Filing Year', fontsize=14)
plt.ylabel('Net Sentiment Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
```

## Visualization Examples

### Interactive Time Series

```python
from tenk_toolkit.visualization import create_interactive_time_series

# Create interactive time series plot
fig = create_interactive_time_series(
    financial_metrics,
    date_column='filing_date',
    value_columns=['revenue', 'net_income', 'operating_income'],
    company_column='ticker',
    title='Financial Metrics Over Time'
)

# Show the figure
fig.show()

# Save to HTML file
fig.write_html('financial_metrics.html')
```

### Interactive Scatter Plot

```python
from tenk_toolkit.visualization import create_interactive_scatter

# Combine financial and sentiment metrics
sentiment_by_company_year = sentiment_metrics.groupby(['ticker', 'filing_year'])['lexicon_net_score'].mean().reset_index()
financial_by_company_year = financial_metrics.groupby(['ticker', 'filing_year'])[['revenue', 'profit_margin']].mean().reset_index()
merged_df = pd.merge(financial_by_company_year, sentiment_by_company_year, on=['ticker', 'filing_year'])

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

# Show the figure
fig.show()

# Save to HTML file
fig.write_html('sentiment_vs_profit_margin.html')
```

### Correlation Heatmap

```python
from tenk_toolkit.visualization import plot_correlation_heatmap
import numpy as np

# Create correlation heatmap
num_cols = financial_metrics.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in num_cols if col not in ['filing_year']]

fig, ax = plot_correlation_heatmap(
    financial_metrics,
    columns=num_cols,
    title='Correlation Heatmap of Financial Metrics'
)
plt.show()
```

### Year-over-Year Comparison

```python
from tenk_toolkit.visualization import plot_year_over_year_comparison

# Create year-over-year comparison
fig, ax = plot_year_over_year_comparison(
    financial_metrics,
    date_column='filing_date',
    value_column='revenue',
    group_column='ticker',
    title='Year-over-Year Revenue Comparison',
    normalize=False
)
plt.show()

# Create normalized comparison (percentage change)
fig, ax = plot_year_over_year_comparison(
    financial_metrics,
    date_column='filing_date',
    value_column='revenue',
    group_column='ticker',
    title='Year-over-Year Revenue Growth (%)',
    normalize=True
)
plt.show()
```

### Comprehensive Dashboard

```python
from tenk_toolkit.visualization import create_interactive_dashboard

# Create interactive dashboard
dashboard = create_interactive_dashboard(
    financial_df=financial_metrics,
    sentiment_df=sentiment_metrics
)

# Show the dashboard
dashboard.show()

# Save to HTML file
dashboard.write_html('10k_analysis_dashboard.html')
```

## Advanced Examples

### Combining Financial and Sentiment Analysis

```python
from tenk_toolkit.data import SECDataLoader, FilingPreprocessor
from tenk_toolkit.analysis import FinancialAnalyzer, SentimentAnalyzer
from tenk_toolkit.visualization import create_bubble_chart
import pandas as pd

# Download and process filings
loader = SECDataLoader()
filings = loader.load_filings(['AAPL', 'MSFT', 'GOOGL'], years=[2020, 2021, 2022])
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings, extract_sections=True)

# Analyze financial metrics
financial_analyzer = FinancialAnalyzer()
financial_metrics = financial_analyzer.analyze_filings(processed_filings)

# Analyze sentiment
sentiment_analyzer = SentimentAnalyzer()
sentiment_metrics = sentiment_analyzer.analyze_filings(processed_filings, sections=['item_7'])

# Combine financial and sentiment metrics
mda_sentiment = sentiment_metrics[sentiment_metrics['section'] == 'item_7']
sentiment_by_company_year = mda_sentiment.groupby(['ticker', 'filing_year'])['lexicon_net_score'].mean().reset_index()
financial_by_company_year = financial_metrics.groupby(['ticker', 'filing_year'])[['revenue', 'profit_margin']].mean().reset_index()
merged_df = pd.merge(financial_by_company_year, sentiment_by_company_year, on=['ticker', 'filing_year'])

# Create bubble chart
fig = create_bubble_chart(
    merged_df,
    x_column='lexicon_net_score',
    y_column='profit_margin',
    size_column='revenue',
    color_column='ticker',
    text_column='filing_year',
    title='Financial Performance and Sentiment Analysis'
)

# Show the figure
fig.show()
```

### Analyzing Risk Factor Changes Over Time

```python
from tenk_toolkit.data import SECDataLoader, FilingPreprocessor
from tenk_toolkit.analysis import TextAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download and process filings for a single company over multiple years
loader = SECDataLoader()
filings = loader.load_filings(['AAPL'], years=[2018, 2019, 2020, 2021, 2022])
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings, extract_sections=True)

# Extract Risk Factors sections
risk_factors = processed_filings[['ticker', 'filing_year', 'section_item_1a']].copy()
risk_factors = risk_factors.sort_values(['ticker', 'filing_year'])

# Create a CountVectorizer to convert text to vectors
vectorizer = CountVectorizer(stop_words='english', min_df=2)
vectors = vectorizer.fit_transform(risk_factors['section_item_1a'])

# Calculate similarity matrix
similarity_matrix = cosine_similarity(vectors)

# Create a DataFrame for the similarity matrix
years = risk_factors['filing_year'].tolist()
similarity_df = pd.DataFrame(similarity_matrix, index=years, columns=years)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
plt.title('Risk Factors Similarity Across Years', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Analyze word frequency changes
analyzer = TextAnalyzer()
word_freq_by_year = {}

for _, row in risk_factors.iterrows():
    year = row['filing_year']
    text = row['section_item_1a']
    word_freq = analyzer.get_word_frequencies(text, top_n=100, remove_stopwords=True)
    word_freq_by_year[year] = dict(zip(word_freq['word'], word_freq['frequency']))

# Create a DataFrame for word frequency changes
all_words = set()
for word_freq in word_freq_by_year.values():
    all_words.update(word_freq.keys())

word_freq_changes = pd.DataFrame(index=all_words)

for year, word_freq in word_freq_by_year.items():
    word_freq_changes[year] = pd.Series(word_freq)

word_freq_changes = word_freq_changes.fillna(0)

# Calculate frequency changes between first and last year
first_year = min(word_freq_by_year.keys())
last_year = max(word_freq_by_year.keys())
word_freq_changes['change'] = word_freq_changes[last_year] - word_freq_changes[first_year]

# Get top words with increasing and decreasing frequency
top_increasing = word_freq_changes.sort_values('change', ascending=False).head(20)
top_decreasing = word_freq_changes.sort_values('change', ascending=True).head(20)

# Plot top increasing words
plt.figure(figsize=(12, 6))
plt.bar(top_increasing.index, top_increasing['change'], color='green')
plt.title(f'Top Words with Increasing Frequency ({first_year} to {last_year})', fontsize=16, fontweight='bold')
plt.xlabel('Word', fontsize=14)
plt.ylabel('Frequency Change', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot top decreasing words
plt.figure(figsize=(12, 6))
plt.bar(top_decreasing.index, top_decreasing['change'], color='red')
plt.title(f'Top Words with Decreasing Frequency ({first_year} to {last_year})', fontsize=16, fontweight='bold')
plt.xlabel('Word', fontsize=14)
plt.ylabel('Frequency Change', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

### Competitive Analysis Dashboard

```python
from tenk_toolkit.data import SECDataLoader, FilingPreprocessor
from tenk_toolkit.analysis import FinancialAnalyzer, SentimentAnalyzer, TextAnalyzer
from tenk_toolkit.visualization import create_radar_chart, create_interactive_bar_chart, create_interactive_time_series
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download and process filings for multiple companies
loader = SECDataLoader()
filings = loader.load_filings(['AAPL', 'MSFT', 'GOOGL'], years=[2020, 2021, 2022])
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings, extract_sections=True)

# Analyze financial metrics
financial_analyzer = FinancialAnalyzer()
financial_metrics = financial_analyzer.analyze_filings(processed_filings)

# Analyze sentiment
sentiment_analyzer = SentimentAnalyzer()
sentiment_metrics = sentiment_analyzer.analyze_filings(processed_filings, sections=['item_1a', 'item_7'])

# Get the most recent year
latest_year = financial_metrics['filing_year'].max()
latest_metrics = financial_metrics[financial_metrics['filing_year'] == latest_year]

# Calculate normalized metrics for radar chart
categories = ['revenue', 'net_income', 'operating_income', 'profit_margin']
available_categories = [c for c in categories if c in latest_metrics.columns]

normalized_metrics = latest_metrics.copy()
for category in available_categories:
    if category in latest_metrics.columns:
        max_value = latest_metrics[category].max()
        if max_value > 0:
            normalized_metrics[category] = latest_metrics[category] / max_value

# Prepare data for radar chart
radar_data = []
for ticker, group in normalized_metrics.groupby('ticker'):
    if not group.empty:
        radar_data.append({
            'ticker': ticker,
            'category': 'revenue',
            'value': group['revenue'].iloc[0] if 'revenue' in group.columns else 0
        })
        radar_data.append({
            'ticker': ticker,
            'category': 'net_income',
            'value': group['net_income'].iloc[0] if 'net_income' in group.columns else 0
        })
        radar_data.append({
            'ticker': ticker,
            'category': 'operating_income',
            'value': group['operating_income'].iloc[0] if 'operating_income' in group.columns else 0
        })
        radar_data.append({
            'ticker': ticker,
            'category': 'profit_margin',
            'value': group['profit_margin'].iloc[0] if 'profit_margin' in group.columns else 0
        })

radar_df = pd.DataFrame(radar_data)

# Create radar chart
radar_fig = create_radar_chart(
    radar_df,
    categories=available_categories,
    value_column='value',
    group_column='ticker',
    title='Competitive Financial Performance'
)

# Filter sentiment for Risk Factors section
risk_sentiment = sentiment_metrics[sentiment_metrics['section'] == 'item_1a']
risk_sentiment_by_company = risk_sentiment.groupby('ticker')['lexicon_net_score'].mean().reset_index()

# Create bar chart for Risk Factors sentiment
sentiment_fig = create_interactive_bar_chart(
    risk_sentiment_by_company,
    x_column='ticker',
    y_column='lexicon_net_score',
    title='Risk Factors Sentiment by Company'
)

# Create time series for revenue
time_series_fig = create_interactive_time_series(
    financial_metrics,
    date_column='filing_date',
    value_columns=['revenue'],
    company_column='ticker',
    title='Revenue Over Time by Company'
)

# Create dashboard
dashboard_fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Competitive Financial Performance',
        'Risk Factors Sentiment by Company',
        'Revenue Over Time by Company',
        ''
    ],
    specs=[
        [{"type": "polar"}, {"type": "xy"}],
        [{"type": "xy", "colspan": 2}, None]
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# Add radar chart
for trace in radar_fig.data:
    dashboard_fig.add_trace(trace, row=1, col=1)

# Add sentiment bar chart
for trace in sentiment_fig.data:
    dashboard_fig.add_trace(trace, row=1, col=2)

# Add revenue time series
for trace in time_series_fig.data:
    dashboard_fig.add_trace(trace, row=2, col=1)

# Update layout
dashboard_fig.update_layout(
    title='Competitive Analysis Dashboard',
    height=800,
    width=1200,
    showlegend=True
)

# Show the dashboard
dashboard_fig.show()

# Save to HTML file
dashboard_fig.write_html('competitive_analysis_dashboard.html')
```

These examples demonstrate how to use the 10-K Analysis Toolkit for various analysis tasks. You can adapt and combine these examples to suit your specific needs.