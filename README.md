# 10-K Analysis Toolkit

A comprehensive toolkit for analyzing, visualizing, and extracting insights from SEC 10-K filings.

## Features

- **Data Extraction**: Extract text and tabular data from 10-K filings
- **Text Analysis**: NLP-based analysis of narrative sections
- **Financial Analysis**: Track and compare key financial metrics
- **Sentiment Analysis**: Evaluate tone and sentiment in management discussions
- **Visualization**: Generate insightful charts and dashboards
- **Comparative Analysis**: Compare companies across industries and time periods

## Installation

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

## Quick Start

```python
from tenk_toolkit.data import SECDataLoader
from tenk_toolkit.analysis import TextAnalyzer, FinancialAnalyzer
from tenk_toolkit.visualization import create_dashboard

# Load 10-K data
loader = SECDataLoader()
filings = loader.load_filings(['AAPL', 'MSFT', 'GOOGL'], years=[2020, 2021, 2022])

# Analyze text content
text_analyzer = TextAnalyzer()
sentiment_results = text_analyzer.analyze_sentiment(filings)

# Analyze financial metrics
financial_analyzer = FinancialAnalyzer()
financial_metrics = financial_analyzer.extract_key_metrics(filings)

# Create visualization dashboard
dashboard = create_dashboard(sentiment_results, financial_metrics)
dashboard.show()
```

## Documentation

See the [documentation](docs/getting_started.md) for detailed usage instructions and examples.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.