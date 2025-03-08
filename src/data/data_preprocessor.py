"""
Module for preprocessing 10-K filing data.
"""

import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class FilingPreprocessor:
    """
    Class for preprocessing 10-K filings.
    """
    
    def __init__(self):
        """
        Initialize the filing preprocessor.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common sections in 10-K filings
        self.section_patterns = {
            'item_1': r'item\s*1\.?\s*business',
            'item_1a': r'item\s*1a\.?\s*risk\s*factors',
            'item_1b': r'item\s*1b\.?\s*unresolved\s*staff\s*comments',
            'item_2': r'item\s*2\.?\s*properties',
            'item_3': r'item\s*3\.?\s*legal\s*proceedings',
            'item_4': r'item\s*4\.?\s*mine\s*safety\s*disclosures',
            'item_5': r'item\s*5\.?\s*market\s*for\s*registrant',
            'item_6': r'item\s*6\.?\s*selected\s*financial\s*data',
            'item_7': r'item\s*7\.?\s*management(?:\'s|\s)?\s*discussion',
            'item_7a': r'item\s*7a\.?\s*quantitative\s*and\s*qualitative\s*disclosures\s*about\s*market\s*risk',
            'item_8': r'item\s*8\.?\s*financial\s*statements',
            'item_9': r'item\s*9\.?\s*changes\s*in\s*and\s*disagreements\s*with\s*accountants',
            'item_9a': r'item\s*9a\.?\s*controls\s*and\s*procedures',
            'item_9b': r'item\s*9b\.?\s*other\s*information',
            'item_10': r'item\s*10\.?\s*directors',
            'item_11': r'item\s*11\.?\s*executive\s*compensation',
            'item_12': r'item\s*12\.?\s*security\s*ownership',
            'item_13': r'item\s*13\.?\s*certain\s*relationships',
            'item_14': r'item\s*14\.?\s*principal\s*accounting\s*fees',
            'item_15': r'item\s*15\.?\s*exhibits,?\s*financial\s*statement\s*schedules'
        }
    
    def clean_html(self, html_content):
        """
        Clean HTML content, removing scripts, styles, and other non-content elements.
        
        Parameters:
        -----------
        html_content : str
            HTML content to clean
            
        Returns:
        --------
        str
            Cleaned HTML content
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(['script', 'style', 'meta', 'link', 'head']):
                element.decompose()
            
            # Get text content
            text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {str(e)}")
            return ""
    
    def extract_section(self, text, section_name):
        """
        Extract a specific section from the 10-K text.
        
        Parameters:
        -----------
        text : str
            Full text of the 10-K filing
        section_name : str
            Name of the section to extract
            
        Returns:
        --------
        str
            Text of the requested section
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Get the pattern for the requested section
        if section_name not in self.section_patterns:
            logger.warning(f"Unknown section: {section_name}")
            return ""
        
        start_pattern = self.section_patterns[section_name]
        
        # Find all section names in order of appearance
        section_positions = []
        for sec_name, pattern in self.section_patterns.items():
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                section_positions.append((match.start(), sec_name))
        
        # Sort by position
        section_positions.sort()
        
        # Find the start position of the requested section
        start_pos = -1
        for pos, name in section_positions:
            if name == section_name:
                start_pos = pos
                break
        
        if start_pos == -1:
            logger.warning(f"Section {section_name} not found")
            return ""
        
        # Find the end position (start of the next section)
        end_pos = len(text)
        for pos, name in section_positions:
            if pos > start_pos:
                end_pos = pos
                break
        
        # Extract the section text
        section_text = text[start_pos:end_pos].strip()
        
        return section_text
    
    def extract_all_sections(self, text):
        """
        Extract all sections from the 10-K text.
        
        Parameters:
        -----------
        text : str
            Full text of the 10-K filing
            
        Returns:
        --------
        dict
            Dictionary with section names as keys and section texts as values
        """
        sections = {}
        
        for section_name in self.section_patterns:
            section_text = self.extract_section(text, section_name)
            sections[section_name] = section_text
        
        return sections
    
    def clean_text(self, text, remove_stopwords=True, lemmatize=True):
        """
        Clean text by removing punctuation, numbers, and optionally stopwords, and lemmatizing.
        
        Parameters:
        -----------
        text : str
            Text to clean
        remove_stopwords : bool
            Whether to remove stopwords
        lemmatize : bool
            Whether to lemmatize the text
            
        Returns:
        --------
        str
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if requested
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def process_filings(self, filings_df, extract_sections=True, clean_text=False):
        """
        Process a DataFrame of 10-K filings.
        
        Parameters:
        -----------
        filings_df : pandas.DataFrame
            DataFrame containing filings data
        extract_sections : bool
            Whether to extract sections from the filings
        clean_text : bool
            Whether to clean the text
            
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = filings_df.copy()
        
        # Check if we have the necessary column
        if 'filing_html' not in df.columns:
            logger.error("Required column 'filing_html' not found in DataFrame")
            return df
        
        # Clean HTML
        logger.info("Cleaning HTML content...")
        df['filing_text'] = df['filing_html'].apply(self.clean_html)
        
        # Extract sections if requested
        if extract_sections:
            logger.info("Extracting sections...")
            
            # Extract each section into a separate column
            for section_name in self.section_patterns:
                column_name = f"section_{section_name}"
                df[column_name] = df['filing_text'].apply(
                    lambda x: self.extract_section(x, section_name)
                )
            
            # Count characters in each section for quick validation
            for section_name in self.section_patterns:
                column_name = f"section_{section_name}"
                char_count_col = f"{column_name}_chars"
                df[char_count_col] = df[column_name].str.len()
        
        # Clean text if requested
        if clean_text:
            logger.info("Cleaning text...")
            
            # Clean the full filing text
            df['clean_text'] = df['filing_text'].apply(
                lambda x: self.clean_text(x)
            )
            
            # Clean each section text if sections were extracted
            if extract_sections:
                for section_name in self.section_patterns:
                    column_name = f"section_{section_name}"
                    clean_col = f"clean_{section_name}"
                    df[clean_col] = df[column_name].apply(
                        lambda x: self.clean_text(x) if isinstance(x, str) else ""
                    )
        
        return df

# Example usage
if __name__ == "__main__":
    # Load filings DataFrame (placeholder for example)
    filings_df = pd.DataFrame({'filing_html': ['<html><body>Sample 10-K text...</body></html>']})
    
    # Create preprocessor and process filings
    preprocessor = FilingPreprocessor()
    processed_df = preprocessor.process_filings(filings_df)
    
    print(processed_df.columns)