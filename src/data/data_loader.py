"""
Module for loading SEC 10-K filing data from various sources.
"""

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from tqdm import tqdm
import re
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SECDataLoader:
    """
    Class for loading 10-K filings from the SEC EDGAR database.
    """
    
    def __init__(self, cache_dir='./data/cache'):
        """
        Initialize the SEC data loader.
        
        Parameters:
        -----------
        cache_dir : str
            Directory to cache downloaded filings
        """
        self.cache_dir = cache_dir
        self.base_url = 'https://www.sec.gov/Archives'
        self.edgar_search_url = 'https://www.sec.gov/cgi-bin/browse-edgar'
        self.user_agent = 'Example Company Name AdminContact@example.com'
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache to store CIK mappings
        self._ticker_to_cik_cache = {}
        
    def get_cik_for_ticker(self, ticker):
        """
        Get the CIK number for a given ticker symbol.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        str or None
            CIK number if found, None otherwise
        """
        # Check cache first
        if ticker in self._ticker_to_cik_cache:
            return self._ticker_to_cik_cache[ticker]
        
        # Query SEC website for CIK
        try:
            params = {
                'CIK': ticker,
                'owner': 'exclude',
                'action': 'getcompany'
            }
            headers = {'User-Agent': self.user_agent}
            response = requests.get(self.edgar_search_url, params=params, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract CIK from the response
                cik_text = soup.find('input', {'name': 'CIK'})
                if cik_text:
                    cik = cik_text.get('value')
                    # Cache the result
                    self._ticker_to_cik_cache[ticker] = cik
                    return cik
            
            logger.warning(f"Could not find CIK for ticker {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {str(e)}")
            return None
    
    def get_filing_links(self, cik, filing_type='10-K', start_year=None, end_year=None):
        """
        Get links to filings for a company based on CIK.
        
        Parameters:
        -----------
        cik : str
            CIK number of the company
        filing_type : str
            Type of filing to retrieve (default: '10-K')
        start_year : int
            Starting year for filings
        end_year : int
            Ending year for filings
            
        Returns:
        --------
        list
            List of filing links with metadata
        """
        links = []
        try:
            # Format CIK with leading zeros to 10 digits
            cik_formatted = cik.zfill(10)
            
            # Parameters for the SEC query
            params = {
                'action': 'getcompany',
                'CIK': cik,
                'type': filing_type,
                'dateb': '',
                'owner': 'exclude',
                'count': '100'  # Limit to 100 most recent filings
            }
            
            headers = {'User-Agent': self.user_agent}
            
            # Get the list of filings
            response = requests.get(self.edgar_search_url, params=params, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get filing list for CIK {cik} - Status: {response.status_code}")
                return links
            
            # Parse the response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table containing filing information
            filing_table = soup.find('table', {'class': 'tableFile2'})
            
            if not filing_table:
                logger.warning(f"No filing table found for CIK {cik}")
                return links
            
            # Process each row in the table
            for row in filing_table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) >= 4:
                    filing_date_text = cells[3].text.strip()
                    try:
                        filing_date = datetime.strptime(filing_date_text, '%Y-%m-%d')
                        
                        # Filter by year if specified
                        if (start_year and filing_date.year < start_year) or \
                           (end_year and filing_date.year > end_year):
                            continue
                        
                        # Get filing details
                        filing_type_text = cells[0].text.strip()
                        
                        # Only consider exact matches for filing type
                        if filing_type_text != filing_type:
                            continue
                        
                        # Get link to the filing detail page
                        filing_link_element = cells[1].find('a', href=True)
                        if filing_link_element:
                            filing_link = 'https://www.sec.gov' + filing_link_element['href']
                            
                            # Extract the accession number from the URL if possible
                            acc_match = re.search(r'accession_number=([^&]+)', filing_link)
                            if not acc_match:
                                acc_match = re.search(r'/(\d+-\d+-\d+)', filing_link)
                            
                            acc_no = acc_match.group(1) if acc_match else None
                            
                            links.append({
                                'cik': cik,
                                'filing_type': filing_type_text,
                                'filing_date': filing_date,
                                'filing_link': filing_link,
                                'accession_number': acc_no  # Add this directly to the links data
                            })
                    except ValueError:
                        logger.warning(f"Could not parse date: {filing_date_text}")
                        continue
            
            # Sleep to avoid hitting SEC rate limits
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error getting filing links for CIK {cik}: {str(e)}")
        
        return links
    
    def get_filing_content(self, filing_link):
        """
        Get the content of a filing from its link.
        
        Parameters:
        -----------
        filing_link : str
            Link to the filing detail page
            
        Returns:
        --------
        dict
            Dictionary containing filing content and metadata
        """
        try:
            # Extract accession number from the filing link using multiple patterns
            # First try the URL parameter pattern
            acc_no_match = re.search(r'accession_number=([^&]+)', filing_link)
            
            # If that fails, try extracting from the URL path using different patterns
            if not acc_no_match:
                # Try to match the format /data/CIK/ACCESSION/ACCESSION-index.htm
                acc_no_match = re.search(r'/(\d+/\d+)(?:-index)?\.htm', filing_link)
            
            if not acc_no_match:
                # Try another pattern for accession numbers in the URL path
                acc_no_match = re.search(r'/(\d+-\d+-\d+)(?:-index)?\.htm', filing_link)
                
            if not acc_no_match:
                # Last resort - try to find anything that looks like an accession number
                acc_no_match = re.search(r'(\d+-\d+-\d+)', filing_link)
                
            if not acc_no_match:
                logger.error(f"Could not extract accession number from {filing_link}")
                # Return a minimal dictionary instead of None to avoid downstream errors
                return {
                    'accession_number': 'unknown-' + str(int(time.time())),
                    'company_name': None,
                    'fiscal_year_end': None,
                    'filing_html': '<html>Failed to retrieve content</html>'
                }
            
            acc_no = acc_no_match.group(1)
            # Clean the accession number by removing dashes and slashes
            acc_no_clean = acc_no.replace('-', '').replace('/', '')
            
            # Cache file path
            cache_file = os.path.join(self.cache_dir, f"{acc_no_clean}.html")
            
            # Check if filing is already cached
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                logger.info(f"Loaded filing {acc_no} from cache")
            else:
                # Get the filing detail page
                headers = {'User-Agent': self.user_agent}
                response = requests.get(filing_link, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"Failed to get filing detail page {filing_link} - Status: {response.status_code}")
                    return {
                        'accession_number': acc_no,
                        'company_name': None,
                        'fiscal_year_end': None,
                        'filing_html': '<html>Failed to retrieve content</html>'
                    }
                
                # Parse the detail page to find the actual filing document
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the link to the full text filing
                filing_document = None
                table = soup.find('table', {'class': 'tableFile'})
                if table:
                    for row in table.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 4 and '10-K' in cells[3].text:
                            doc_link = cells[2].find('a', href=True)
                            if doc_link:
                                filing_document = 'https://www.sec.gov' + doc_link['href']
                                break
                
                # If we can't find the document through the normal structure, try a fallback method
                if not filing_document:
                    # Look for any link that might contain the 10-K document
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if '10-K' in link.text or '.htm' in href:
                            filing_document = 'https://www.sec.gov' + href if href.startswith('/') else href
                            break
                
                if not filing_document:
                    logger.error(f"Could not find 10-K document link in {filing_link}")
                    return {
                        'accession_number': acc_no,
                        'company_name': None,
                        'fiscal_year_end': None,
                        'filing_html': '<html>Failed to find 10-K document</html>'
                    }
                
                # Get the actual filing content
                logger.info(f"Fetching document from {filing_document}")
                doc_response = requests.get(filing_document, headers=headers)
                
                if doc_response.status_code != 200:
                    logger.error(f"Failed to get filing document {filing_document} - Status: {doc_response.status_code}")
                    return {
                        'accession_number': acc_no,
                        'company_name': None,
                        'fiscal_year_end': None,
                        'filing_html': '<html>Failed to download document</html>'
                    }
                
                content = doc_response.text
                
                # Cache the content
                with open(cache_file, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(content)
                
                logger.info(f"Downloaded and cached filing {acc_no}")
                
                # Sleep to avoid hitting SEC rate limits
                time.sleep(0.1)
            
            # Extract metadata and return content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Try to extract company name
            company_name = None
            company_name_tag = soup.find('company-name') or soup.find('companyname')
            if company_name_tag:
                company_name = company_name_tag.text.strip()
            
            # Try to extract fiscal year end
            fiscal_year_end = None
            for tag in soup.find_all(['dei:documentfiscalyearfocus', 'documentfiscalyearfocus']):
                fiscal_year_end = tag.text.strip()
                break
            
            return {
                'accession_number': acc_no,
                'company_name': company_name,
                'fiscal_year_end': fiscal_year_end,
                'filing_html': content
            }
            
        except Exception as e:
            logger.error(f"Error getting filing content for {filing_link}: {str(e)}")
            # Return a minimal dictionary instead of None to ensure filing_html exists
            return {
                'accession_number': 'error-' + str(int(time.time())),
                'company_name': None,
                'fiscal_year_end': None,
                'filing_html': f'<html>Error retrieving content: {str(e)}</html>'
            }
    
    def load_filings(self, tickers, years=None, filing_type='10-K'):
        """
        Load 10-K filings for multiple tickers and years.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        years : list
            List of years to include
        filing_type : str
            Type of filing to retrieve (default: '10-K')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing filing data and metadata
        """
        if years:
            start_year = min(years)
            end_year = max(years)
        else:
            start_year = None
            end_year = None
        
        all_filings = []
        
        for ticker in tqdm(tickers, desc="Processing companies"):
            # Get CIK for ticker
            cik = self.get_cik_for_ticker(ticker)
            if not cik:
                logger.warning(f"Skipping {ticker} - CIK not found")
                continue
            
            # Get filing links
            filing_links = self.get_filing_links(
                cik, 
                filing_type=filing_type,
                start_year=start_year,
                end_year=end_year
            )
            
            if not filing_links:
                logger.warning(f"No {filing_type} filings found for {ticker} (CIK: {cik})")
                continue
            
            # Filter by specific years if provided
            if years:
                filing_links = [
                    link for link in filing_links 
                    if link['filing_date'].year in years
                ]
            
            # Get filing content for each link
            for link_data in tqdm(filing_links, desc=f"Loading filings for {ticker}", leave=False):
                content_data = self.get_filing_content(link_data['filing_link'])
                
                if content_data:
                    filing_data = {
                        'ticker': ticker,
                        'cik': cik,
                        'filing_type': link_data['filing_type'],
                        'filing_date': link_data['filing_date'],
                        'filing_year': link_data['filing_date'].year,
                        'filing_link': link_data['filing_link'],
                        'accession_number': content_data['accession_number'],
                        'company_name': content_data['company_name'],
                        'fiscal_year_end': content_data['fiscal_year_end'],
                        'filing_html': content_data['filing_html']
                    }
                    
                    all_filings.append(filing_data)
        
        # Log the number of filings found
        if len(all_filings) == 0:
            logger.error(f"No filings were successfully loaded. Check the log for errors.")
        else:
            logger.info(f"Successfully loaded {len(all_filings)} filings.")
        
        # Convert to DataFrame
        df_filings = pd.DataFrame(all_filings)
        
        # Add some derived columns if we have data
        if not df_filings.empty:
            # Sort by ticker and filing date
            df_filings = df_filings.sort_values(['ticker', 'filing_date'])
            
            # Format columns
            if 'filing_date' in df_filings.columns:
                df_filings['filing_date'] = pd.to_datetime(df_filings['filing_date'])
            
            # Log the columns for debugging
            logger.info(f"DataFrame columns: {df_filings.columns.tolist()}")
        
        return df_filings

# Example usage
if __name__ == "__main__":
    loader = SECDataLoader()
    filings = loader.load_filings(['AAPL', 'MSFT'], years=[2020, 2021, 2022])
    print(f"Loaded {len(filings)} filings")
    print(filings[['ticker', 'filing_year', 'company_name']].head())