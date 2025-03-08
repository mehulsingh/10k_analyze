"""
Module for extracting and analyzing financial metrics from 10-K filings.
"""

import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import logging
from tqdm import tqdm
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

class FinancialAnalyzer:
    """
    Class for extracting and analyzing financial metrics from 10-K filings.
    """
    
    def __init__(self):
        """
        Initialize the financial analyzer.
        """
        # Common financial metrics to extract
        self.metrics = {
            'revenue': [
                r'total\s+revenue',
                r'net\s+sales',
                r'total\s+net\s+sales',
                r'revenue,?\s+net',
                r'net\s+revenue'
            ],
            'net_income': [
                r'net\s+income',
                r'net\s+earnings',
                r'net\s+profit',
                r'net\s+income\s+\(loss\)',
                r'net\s+earnings\s+\(loss\)'
            ],
            'total_assets': [
                r'total\s+assets',
                r'assets,?\s+total'
            ],
            'total_liabilities': [
                r'total\s+liabilities',
                r'liabilities,?\s+total'
            ],
            'operating_income': [
                r'operating\s+income',
                r'income\s+from\s+operations',
                r'operating\s+profit',
                r'operating\s+income\s+\(loss\)'
            ],
            'rd_expenses': [
                r'research\s+and\s+development',
                r'r&d\s+expenses',
                r'research\s+and\s+development\s+expenses'
            ],
            'cash_flow_operations': [
                r'net\s+cash\s+(?:provided|generated)\s+by\s+operating\s+activities',
                r'cash\s+flows\s+from\s+operating\s+activities',
                r'operating\s+cash\s+flow'
            ],
            'eps': [
                r'earnings\s+per\s+share',
                r'basic\s+earnings\s+per\s+share',
                r'diluted\s+earnings\s+per\s+share',
                r'net\s+income\s+per\s+share'
            ]
        }
        
        # Units conversion factors
        self.unit_factors = {
            'thousand': 1e3,
            'thousands': 1e3,
            'million': 1e6,
            'millions': 1e6,
            'billion': 1e9,
            'billions': 1e9,
            'trillion': 1e12,
            'trillions': 1e12,
            'k': 1e3,
            'm': 1e6,
            'b': 1e9,
            't': 1e12
        }
    
    def extract_tables_from_html(self, html_content):
        """
        Extract tables from HTML content.
        
        Parameters:
        -----------
        html_content : str
            HTML content to extract tables from
            
        Returns:
        --------
        list
            List of tables as pandas DataFrames
        """
        if not isinstance(html_content, str) or not html_content:
            return []
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all tables
            table_elements = soup.find_all('table')
            
            tables = []
            for i, table_element in enumerate(table_elements):
                try:
                    # Extract rows
                    rows = []
                    for tr in table_element.find_all('tr'):
                        # Extract cells
                        cells = []
                        for cell in tr.find_all(['th', 'td']):
                            # Get text content
                            text = cell.get_text().strip()
                            # Clean up text
                            text = re.sub(r'\s+', ' ', text)
                            cells.append(text)
                        if cells:
                            rows.append(cells)
                    
                    if rows:
                        # Create DataFrame
                        df = pd.DataFrame(rows)
                        
                        # Use first row as header if it's different from other rows
                        if len(df) > 1:
                            header_row = df.iloc[0]
                            if not df.iloc[1:].equals(header_row):
                                df.columns = header_row
                                df = df.iloc[1:]
                        
                        # Reset index
                        df = df.reset_index(drop=True)
                        
                        # Add to tables list
                        tables.append({
                            'table_id': i,
                            'rows': len(df),
                            'cols': len(df.columns),
                            'dataframe': df
                        })
                except Exception as e:
                    logger.warning(f"Error extracting table {i}: {str(e)}")
                    continue
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from HTML: {str(e)}")
            return []
    
    def extract_financial_tables(self, html_content, min_rows=3, min_cols=2):
        """
        Extract financial tables from HTML content.
        
        Parameters:
        -----------
        html_content : str
            HTML content to extract tables from
        min_rows : int
            Minimum number of rows for a table to be considered
        min_cols : int
            Minimum number of columns for a table to be considered
            
        Returns:
        --------
        list
            List of financial tables as pandas DataFrames
        """
        # Extract all tables
        all_tables = self.extract_tables_from_html(html_content)
        
        # Filter tables by size
        filtered_tables = [
            table for table in all_tables
            if table['rows'] >= min_rows and table['cols'] >= min_cols
        ]
        
        # Filter tables by content (look for financial keywords)
        financial_keywords = [
            'revenue', 'income', 'earnings', 'assets', 'liabilities',
            'cash', 'expenses', 'profit', 'loss', 'balance', 'statement',
            'financial', 'fiscal', 'year', 'quarter', 'consolidated'
        ]
        
        financial_tables = []
        for table in filtered_tables:
            df = table['dataframe']
            # Convert to string and check for keywords
            table_text = df.to_string().lower()
            if any(keyword in table_text for keyword in financial_keywords):
                financial_tables.append(table)
        
        return financial_tables
    
    def parse_dollar_amount(self, text):
        """
        Parse a dollar amount from text.
        
        Parameters:
        -----------
        text : str
            Text containing a dollar amount
            
        Returns:
        --------
        float or None
            Parsed dollar amount or None if parsing fails
        """
        if not isinstance(text, str) or not text:
            return None
        
        try:
            # Clean up text
            text = text.lower().strip()
            
            # Remove common symbols and words
            text = text.replace('$', '').replace(',', '')
            text = re.sub(r'\([^)]*\)', '', text)  # Remove parentheses and contents
            
            # Check for negative indicators
            is_negative = False
            if '(' in text and ')' in text:
                is_negative = True
                text = text.replace('(', '').replace(')', '')
            elif '-' in text or 'loss' in text:
                is_negative = True
                text = text.replace('-', '')
                
            # Match the number
            number_match = re.search(r'[-+]?\d*\.\d+|\d+', text)
            if not number_match:
                return None
            
            value = float(number_match.group())
            
            # Check for units (thousand, million, billion)
            for unit, factor in self.unit_factors.items():
                if unit in text:
                    value *= factor
                    break
            
            # Apply negative if needed
            if is_negative:
                value = -value
            
            return value
            
        except Exception as e:
            logger.warning(f"Error parsing dollar amount from '{text}': {str(e)}")
            return None
    
    def extract_metric_from_tables(self, tables, metric_patterns):
        """
        Extract a metric from tables using pattern matching.
        
        Parameters:
        -----------
        tables : list
            List of tables (as dicts with 'dataframe' key)
        metric_patterns : list
            List of regex patterns to match the metric
            
        Returns:
        --------
        float or None
            Extracted metric value or None if not found
        """
        if not tables:
            return None
        
        # Combine patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in metric_patterns)
        
        # Search in each table
        for table in tables:
            df = table['dataframe']
            
            # Search in column headers
            for col in df.columns:
                if isinstance(col, str) and re.search(combined_pattern, col.lower()):
                    # Get the first non-empty value in the column
                    for value in df[col]:
                        if isinstance(value, str) and value:
                            parsed_value = self.parse_dollar_amount(value)
                            if parsed_value is not None:
                                return parsed_value
            
            # Search in row headers (first column)
            if df.shape[1] > 1:
                first_col = df.iloc[:, 0]
                for i, row_header in enumerate(first_col):
                    if isinstance(row_header, str) and re.search(combined_pattern, row_header.lower()):
                        # Get values in this row
                        row = df.iloc[i, 1:]
                        for value in row:
                            if isinstance(value, str) and value:
                                parsed_value = self.parse_dollar_amount(value)
                                if parsed_value is not None:
                                    return parsed_value
        
        return None
    
    def extract_metrics_from_filing(self, html_content):
        """
        Extract financial metrics from a filing.
        
        Parameters:
        -----------
        html_content : str
            HTML content of the filing
            
        Returns:
        --------
        dict
            Dictionary of extracted metrics
        """
        # Extract financial tables
        tables = self.extract_financial_tables(html_content)
        
        if not tables:
            logger.warning("No financial tables found in filing")
            return {}
        
        # Extract metrics
        metrics = {}
        for metric_name, patterns in self.metrics.items():
            value = self.extract_metric_from_tables(tables, patterns)
            metrics[metric_name] = value
        
        return metrics
    
    def calculate_derived_metrics(self, metrics):
        """
        Calculate derived financial metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of financial metrics
            
        Returns:
        --------
        dict
            Dictionary with additional derived metrics
        """
        derived_metrics = {}
        
        # Profit margin
        if metrics.get('revenue') and metrics.get('net_income'):
            if metrics['revenue'] != 0:
                derived_metrics['profit_margin'] = metrics['net_income'] / metrics['revenue']
        
        # ROA (Return on Assets)
        if metrics.get('net_income') and metrics.get('total_assets'):
            if metrics['total_assets'] != 0:
                derived_metrics['roa'] = metrics['net_income'] / metrics['total_assets']
        
        # Debt to assets ratio
        if metrics.get('total_liabilities') and metrics.get('total_assets'):
            if metrics['total_assets'] != 0:
                derived_metrics['debt_to_assets'] = metrics['total_liabilities'] / metrics['total_assets']
        
        # R&D as percentage of revenue
        if metrics.get('rd_expenses') and metrics.get('revenue'):
            if metrics['revenue'] != 0:
                derived_metrics['rd_to_revenue'] = metrics['rd_expenses'] / metrics['revenue']
        
        return derived_metrics
    
    def analyze_filings(self, filings_df):
        """
        Analyze filings and extract financial metrics.
        
        Parameters:
        -----------
        filings_df : pandas.DataFrame
            DataFrame containing filings data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with financial metrics
        """
        # Make a copy to avoid modifying the original
        df = filings_df.copy()
        
        # Check if we have the necessary column
        if 'filing_html' not in df.columns:
            logger.error("Required column 'filing_html' not found in DataFrame")
            return df
        
        # Initialize metrics DataFrame
        metrics_records = []
        
        # Process each filing
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting financial metrics"):
            filing_metrics = {
                'ticker': row.get('ticker', ''),
                'company_name': row.get('company_name', ''),
                'filing_date': row.get('filing_date', ''),
                'filing_year': row.get('filing_year', '')
            }
            
            # Extract metrics from HTML
            html_content = row['filing_html']
            extracted_metrics = self.extract_metrics_from_filing(html_content)
            
            # Calculate derived metrics
            derived_metrics = self.calculate_derived_metrics(extracted_metrics)
            
            # Combine all metrics
            all_metrics = {**extracted_metrics, **derived_metrics}
            
            # Add metrics to record
            filing_metrics.update(all_metrics)
            
            # Add record
            metrics_records.append(filing_metrics)
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_records)
        
        return metrics_df
    
    def compare_financials(self, metrics_df, companies=None, years=None):
        """
        Compare financial metrics across companies and years.
        
        Parameters:
        -----------
        metrics_df : pandas.DataFrame
            DataFrame with financial metrics
        companies : list
            List of companies to include (if None, include all)
        years : list
            List of years to include (if None, include all)
            
        Returns:
        --------
        dict
            Dictionary with comparative analysis
        """
        # Filter by companies and years if specified
        df = metrics_df.copy()
        
        if companies:
            df = df[df['ticker'].isin(companies)]
        
        if years:
            df = df[df['filing_year'].isin(years)]
        
        if df.empty:
            logger.warning("No data available after filtering")
            return {}
        
        # Get list of numeric metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_cols = [col for col in numeric_cols if col not in ['filing_year']]
        
        # Calculate comparative stats
        comparison = {}
        
        # Compare by company
        company_comparison = df.groupby('ticker')[metric_cols].agg(['mean', 'std', 'min', 'max']).reset_index()
        comparison['by_company'] = company_comparison
        
        # Compare by year
        if 'filing_year' in df.columns:
            year_comparison = df.groupby('filing_year')[metric_cols].agg(['mean', 'std', 'min', 'max']).reset_index()
            comparison['by_year'] = year_comparison
        
        # Calculate growth rates
        if 'filing_year' in df.columns and len(df['filing_year'].unique()) > 1:
            growth_data = []
            
            for ticker, company_data in df.groupby('ticker'):
                company_data = company_data.sort_values('filing_year')
                years = company_data['filing_year'].unique()
                
                for i in range(1, len(years)):
                    prev_year = years[i-1]
                    curr_year = years[i]
                    
                    prev_data = company_data[company_data['filing_year'] == prev_year].iloc[0]
                    curr_data = company_data[company_data['filing_year'] == curr_year].iloc[0]
                    
                    growth_record = {
                        'ticker': ticker,
                        'company_name': curr_data.get('company_name', ''),
                        'prev_year': prev_year,
                        'curr_year': curr_year
                    }
                    
                    # Calculate growth rates for each metric
                    for metric in metric_cols:
                        prev_value = prev_data.get(metric)
                        curr_value = curr_data.get(metric)
                        
                        if prev_value and curr_value and prev_value != 0:
                            growth_rate = (curr_value - prev_value) / abs(prev_value)
                            growth_record[f'{metric}_growth'] = growth_rate
                    
                    growth_data.append(growth_record)
            
            comparison['growth_rates'] = pd.DataFrame(growth_data)
        
        return comparison

# Example usage
if __name__ == "__main__":
    # Create a test DataFrame
    test_df = pd.DataFrame({
        'ticker': ['AAPL'],
        'filing_year': [2021],
        'filing_html': ['<table><tr><th>Revenue</th><td>$365.8 billion</td></tr></table>']
    })
    
    # Create analyzer and extract metrics
    analyzer = FinancialAnalyzer()
    metrics_df = analyzer.analyze_filings(test_df)
    
    print(metrics_df.head())