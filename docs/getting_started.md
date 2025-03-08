# Getting Started with the 10-K Analysis Toolkit

This guide will walk you through the setup and basic usage of the 10-K Analysis Toolkit. The toolkit allows you to download, analyze, and visualize data from SEC 10-K filings.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Command-line Interface](#command-line-interface)
- [Python API](#python-api)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Common Workflows](#common-workflows)
- [Next Steps](#next-steps)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Method 1: Install from PyPI
You can install the toolkit directly from PyPI:

```bash
pip install tenk-toolkit
```

### Method 2: Install from Source
Alternatively, you can clone the repository and install from source:

```bash
# Clone the repository
git clone https://github.com/username/10k-analysis-toolkit.git
cd 10k-analysis-toolkit

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

### Verify Installation
Verify that the toolkit is installed correctly:

```bash
python -c "import tenk_toolkit; print(tenk_toolkit.__version__)"
```

This should print the version number of the toolkit.

## Basic Usage

The 10-K Analysis Toolkit can be used in three ways:
1. As a command-line interface (CLI)
2. As a Python API
3. Through Jupyter notebooks

Here's a quick overview of each approach.

## Command-line Interface

The toolkit provides a command-line interface (CLI) for common tasks:

### Download 10-K Filings
```bash
tenk download --tickers AAPL MSFT GOOGL --years 2020 2021 2022 --output data/filings.pkl
```

### Preprocess Filings
```bash
tenk preprocess --input data/filings.pkl --output data/processed_filings.pkl --extract-sections --clean-text
```

### Analyze Text
```bash
tenk text-analysis --input data/processed_filings.pkl --output data/text_analysis.pkl --sections item_1a item_7
```

### Analyze Financial Metrics
```bash
tenk financial-analysis --input data/processed_filings.pkl --output data/financial_analysis.pkl
```

### Analyze Sentiment
```bash
tenk sentiment-analysis --input data/processed_filings.pkl --output data/sentiment_analysis.pkl --sections item_7
```

### Create Visualizations
```bash
tenk visualize --financial data/financial_analysis.pkl --sentiment data/sentiment_analysis.pkl --output-dir visualizations
```

### Run Full Pipeline
```bash
tenk pipeline --tickers AAPL MSFT GOOGL --years 2020 2021 2022 --output-dir results
```

Run `tenk --help` for more information about the CLI commands and options.

## Python API

You can also use the toolkit as a Python API in your own scripts:

```python
from tenk_toolkit.data import SECDataLoader, FilingPreprocessor
from tenk_toolkit.analysis import TextAnalyzer, FinancialAnalyzer, SentimentAnalyzer
from tenk_toolkit.visualization import create_interactive_dashboard

# Download filings
loader = SECDataLoader()
filings = loader.load_filings(['AAPL', 'MSFT'], years=[2020, 2021, 2022])

# Preprocess filings
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings, extract_sections=True)

# Analyze text
text_analyzer = TextAnalyzer()
text_metrics = text_analyzer.analyze_filings(processed_filings, sections=['item_1a', 'item_7'])

# Analyze financial metrics
financial_analyzer = FinancialAnalyzer()
financial_metrics = financial_analyzer.analyze_filings(processed_filings)

# Analyze sentiment
sentiment_analyzer = SentimentAnalyzer()
sentiment_metrics = sentiment_analyzer.analyze_filings(processed_filings, sections=['item_7'])

# Create a dashboard
dashboard = create_interactive_dashboard(financial_metrics, sentiment_metrics)
dashboard.write_html("dashboard.html")
```

## Jupyter Notebooks

The toolkit includes several Jupyter notebooks that demonstrate its features:

1. `notebooks/1_data_extraction.ipynb`: Download and explore 10-K filings
2. `notebooks/2_data_preprocessing.ipynb`: Extract sections and clean text
3. `notebooks/3_exploratory_analysis.ipynb`: Explore the processed data
4. `notebooks/4_text_analysis.ipynb`: Analyze text content with NLP
5. `notebooks/5_financial_analysis.ipynb`: Extract and analyze financial metrics
6. `notebooks/6_visualizations.ipynb`: Create visualizations and dashboards

To run the notebooks:

```bash
# Navigate to the project directory
cd 10k-analysis-toolkit

# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

## Common Workflows

Here are some common workflows using the toolkit:

### Analyzing a Specific Company
```python
# Download and analyze a specific company
from tenk_toolkit.data import SECDataLoader
from tenk_toolkit.analysis import FinancialAnalyzer
import matplotlib.pyplot as plt

# Download filings
loader = SECDataLoader()
filings = loader.load_filings(['AAPL'], years=[2018, 2019, 2020, 2021, 2022])

# Preprocess filings
from tenk_toolkit.data import FilingPreprocessor
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings)

# Analyze financial metrics
analyzer = FinancialAnalyzer()
metrics = analyzer.analyze_filings(processed_filings)

# Plot revenue over time
from tenk_toolkit.visualization import plot_time_series
fig, ax = plot_time_series(
    metrics,
    date_column='filing_date',
    value_column='revenue',
    title='Apple Revenue Over Time'
)
plt.show()
```

### Comparing Multiple Companies
```python
# Compare multiple companies
from tenk_toolkit.data import SECDataLoader
from tenk_toolkit.analysis import FinancialAnalyzer
from tenk_toolkit.visualization import plot_comparative_metrics
import matplotlib.pyplot as plt

# Download filings
loader = SECDataLoader()
filings = loader.load_filings(['AAPL', 'MSFT', 'GOOGL'], years=[2020, 2021, 2022])

# Process and analyze
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings)
analyzer = FinancialAnalyzer()
metrics = analyzer.analyze_filings(processed_filings)

# Compare financial metrics
fig, ax = plot_comparative_metrics(
    metrics,
    metrics=['revenue', 'net_income', 'profit_margin'],
    company_column='ticker',
    title='Financial Metrics Comparison'
)
plt.show()
```

### Analyzing Risk Factors Sentiment
```python
# Analyze sentiment in Risk Factors section
from tenk_toolkit.data import SECDataLoader, FilingPreprocessor
from tenk_toolkit.analysis import SentimentAnalyzer
from tenk_toolkit.visualization import plot_sentiment_analysis
import matplotlib.pyplot as plt

# Download and process filings
loader = SECDataLoader()
filings = loader.load_filings(['AAPL', 'MSFT', 'GOOGL'], years=[2020, 2021, 2022])
preprocessor = FilingPreprocessor()
processed_filings = preprocessor.process_filings(filings, extract_sections=True)

# Analyze sentiment
analyzer = SentimentAnalyzer()
sentiment = analyzer.analyze_filings(processed_filings, sections=['item_1a'])

# Plot sentiment over time
fig, ax = plot_sentiment_analysis(
    sentiment[sentiment['section'] == 'item_1a'],
    date_column='filing_date',
    sentiment_column='lexicon_net_score',
    company_column='ticker',
    title='Risk Factors Sentiment Over Time'
)
plt.show()
```

## Next Steps

- Learn about the [API Reference](api_reference.md) for detailed documentation of all modules and functions
- Check out the [Examples](examples.md) for more usage examples
- Explore the Jupyter notebooks in the `notebooks/` directory for in-depth tutorials