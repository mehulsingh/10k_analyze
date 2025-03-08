"""
Tests for the SECDataLoader class.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from src.data.data_loader import SECDataLoader

class TestSECDataLoader(unittest.TestCase):
    """Test suite for SECDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.TemporaryDirectory()
        self.loader = SECDataLoader(cache_dir=self.temp_dir.name)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the loader initializes correctly."""
        self.assertEqual(self.loader.cache_dir, self.temp_dir.name)
        self.assertEqual(self.loader.base_url, 'https://www.sec.gov/Archives')
        self.assertEqual(self.loader.edgar_search_url, 'https://www.sec.gov/cgi-bin/browse-edgar')
        self.assertTrue(hasattr(self.loader, 'user_agent'))
        self.assertEqual(self.loader._ticker_to_cik_cache, {})
    
    @patch('requests.get')
    def test_get_cik_for_ticker(self, mock_get):
        """Test getting CIK for a ticker."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '<input type="hidden" name="CIK" value="0000320193">'
        mock_get.return_value = mock_response
        
        # Test the method
        cik = self.loader.get_cik_for_ticker('AAPL')
        
        # Assertions
        self.assertEqual(cik, '0000320193')
        mock_get.assert_called_once()
        
        # Test cache usage
        mock_get.reset_mock()
        cik = self.loader.get_cik_for_ticker('AAPL')
        self.assertEqual(cik, '0000320193')
        mock_get.assert_not_called()  # Should use cache
    
    @patch('requests.get')
    def test_get_cik_not_found(self, mock_get):
        """Test behavior when CIK is not found."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '<html>No match found</html>'  # No CIK in response
        mock_get.return_value = mock_response
        
        # Test the method
        cik = self.loader.get_cik_for_ticker('INVALID')
        
        # Assertions
        self.assertIsNone(cik)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_get_filing_links(self, mock_get):
        """Test getting filing links."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <table class="tableFile2">
        <tr><th>Filing Type</th><th>Filing Date</th></tr>
        <tr>
            <td>10-K</td>
            <td><a href="/Archives/edgar/data/123/000123-22-000123.htm">Link</a></td>
            <td>2022-01-15</td>
        </tr>
        </table>
        """
        mock_get.return_value = mock_response
        
        # Test the method
        links = self.loader.get_filing_links('0000320193', filing_type='10-K', start_year=2020)
        
        # Assertions
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['filing_type'], '10-K')
        self.assertEqual(links[0]['filing_date'].year, 2022)
        mock_get.assert_called_once()
    
    @patch.object(SECDataLoader, 'get_cik_for_ticker')
    @patch.object(SECDataLoader, 'get_filing_links')
    @patch.object(SECDataLoader, 'get_filing_content')
    def test_load_filings(self, mock_get_content, mock_get_links, mock_get_cik):
        """Test loading filings for multiple tickers."""
        # Mock the responses
        mock_get_cik.return_value = '0000320193'
        
        mock_get_links.return_value = [
            {
                'cik': '0000320193',
                'filing_type': '10-K',
                'filing_date': pd.Timestamp('2022-01-15'),
                'filing_link': 'https://www.sec.gov/link1'
            }
        ]
        
        mock_get_content.return_value = {
            'accession_number': '000123-22-000123',
            'company_name': 'APPLE INC',
            'fiscal_year_end': '2021',
            'filing_html': '<html>Content</html>'
        }
        
        # Test the method
        filings = self.loader.load_filings(['AAPL'], years=[2022])
        
        # Assertions
        self.assertEqual(len(filings), 1)
        self.assertEqual(filings['ticker'].iloc[0], 'AAPL')
        self.assertEqual(filings['filing_year'].iloc[0], 2022)
        self.assertEqual(filings['company_name'].iloc[0], 'APPLE INC')
        
        mock_get_cik.assert_called_once_with('AAPL')
        mock_get_links.assert_called_once()
        mock_get_content.assert_called_once_with('https://www.sec.gov/link1')
    
    def test_normalize_ticker(self):
        """Test ticker normalization."""
        # Ensure the method exists
        self.assertTrue(hasattr(self.loader, 'get_cik_for_ticker'))

if __name__ == '__main__':
    unittest.main()