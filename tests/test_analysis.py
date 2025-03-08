"""
Tests for the analysis modules (text_analysis, financial_analysis, sentiment_analysis).
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
from bs4 import BeautifulSoup

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.analysis.text_analysis import TextAnalyzer
from src.analysis.financial_analysis import FinancialAnalyzer
from src.analysis.sentiment_analysis import SentimentAnalyzer

class TestTextAnalyzer(unittest.TestCase):
    """Test suite for TextAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalyzer()
        
        # Sample text for testing
        self.sample_text = (
            "The Company faces significant competition in the technology market. "
            "Our growth strategy includes expansion into new markets through innovation "
            "and strategic acquisitions. There are risks associated with these strategies "
            "including regulatory challenges, integration difficulties, and market uncertainties. "
            "Despite these risks, we believe our strong financial position and experienced "
            "management team will help us navigate these challenges successfully."
        )
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        processed_text = self.analyzer.preprocess_text(self.sample_text)
        self.assertIsInstance(processed_text, str)
        self.assertEqual(processed_text, self.sample_text.lower())
        
        # Test with empty text
        self.assertEqual(self.analyzer.preprocess_text(""), "")
        self.assertEqual(self.analyzer.preprocess_text(None), "")
    
    def test_get_word_frequencies(self):
        """Test word frequency extraction."""
        word_freq = self.analyzer.get_word_frequencies(
            self.sample_text, 
            top_n=5, 
            remove_stopwords=True
        )
        
        # Check the result format
        self.assertIsInstance(word_freq, pd.DataFrame)
        self.assertIn('word', word_freq.columns)
        self.assertIn('frequency', word_freq.columns)
        self.assertLessEqual(len(word_freq), 5)  # Should have at most 5 words
        
        # The word "risks" should be in the top words
        self.assertIn('risks', word_freq['word'].values)
    
    def test_analyze_textblob_sentiment(self):
        """Test sentiment analysis using TextBlob."""
        sentiment = self.analyzer.analyze_textblob_sentiment(self.sample_text)
        
        # Check the result format
        self.assertIsInstance(sentiment, dict)
        self.assertIn('polarity', sentiment)
        self.assertIn('subjectivity', sentiment)
        
        # The sample text has mixed sentiment but is slightly positive
        self.assertGreater(sentiment['polarity'], -0.5)
        self.assertLess(sentiment['polarity'], 0.5)
    
    def test_extract_topics(self):
        """Test topic extraction."""
        texts = [
            "The company faces regulatory risks in its operations.",
            "Revenue growth was strong in the technology segment.",
            "Our research and development efforts led to new product innovations.",
            "The market competition remains intense in all segments.",
            "We acquired three new companies to expand our market presence."
        ]
        
        n_topics = 2
        n_top_words = 5
        
        model, vectorizer, topic_words, doc_topic_matrix = self.analyzer.extract_topics(
            texts, 
            n_topics=n_topics, 
            n_top_words=n_top_words, 
            method='lda'
        )
        
        # Check the result format
        self.assertEqual(len(topic_words), n_topics)
        self.assertEqual(doc_topic_matrix.shape, (len(texts), n_topics))
        
        # Each topic should have the correct number of words
        for topic_idx, words in topic_words:
            self.assertEqual(len(words), n_top_words)


class TestFinancialAnalyzer(unittest.TestCase):
    """Test suite for FinancialAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FinancialAnalyzer()
        
        # Sample HTML with a financial table
        self.sample_html = """
        <html>
        <body>
            <table>
                <tr>
                    <th>Item</th>
                    <th>2022</th>
                    <th>2021</th>
                </tr>
                <tr>
                    <td>Revenue</td>
                    <td>$100,000,000</td>
                    <td>$90,000,000</td>
                </tr>
                <tr>
                    <td>Net Income</td>
                    <td>$20,000,000</td>
                    <td>$18,000,000</td>
                </tr>
            </table>
        </body>
        </html>
        """
    
    def test_extract_tables_from_html(self):
        """Test table extraction from HTML."""
        tables = self.analyzer.extract_tables_from_html(self.sample_html)
        
        # Check that we extracted at least one table
        self.assertGreaterEqual(len(tables), 1)
        
        # Check the table structure
        first_table = tables[0]
        self.assertIsInstance(first_table, dict)
        self.assertIn('dataframe', first_table)
        self.assertIsInstance(first_table['dataframe'], pd.DataFrame)
        self.assertEqual(first_table['rows'], 3)  # Header + 2 data rows
    
    def test_parse_dollar_amount(self):
        """Test parsing of dollar amounts."""
        cases = [
            ("$100,000,000", 100000000),
            ("$90,000,000", 90000000),
            ("$20 million", 20000000),
            ("$5.2 billion", 5200000000),
            ("($10,000)", -10000),  # Negative amount in parentheses
            ("", None),
            (None, None)
        ]
        
        for text, expected in cases:
            result = self.analyzer.parse_dollar_amount(text)
            self.assertEqual(result, expected, f"Failed to parse {text}")
    
    def test_calculate_derived_metrics(self):
        """Test calculation of derived financial metrics."""
        metrics = {
            'revenue': 100000000,
            'net_income': 20000000,
            'total_assets': 200000000,
            'total_liabilities': 100000000,
            'rd_expenses': 10000000
        }
        
        derived = self.analyzer.calculate_derived_metrics(metrics)
        
        # Check the derived metrics
        self.assertIsInstance(derived, dict)
        self.assertEqual(derived['profit_margin'], 0.2)  # 20M / 100M
        self.assertEqual(derived['roa'], 0.1)  # 20M / 200M
        self.assertEqual(derived['debt_to_assets'], 0.5)  # 100M / 200M
        self.assertEqual(derived['rd_to_revenue'], 0.1)  # 10M / 100M


class TestSentimentAnalyzer(unittest.TestCase):
    """Test suite for SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        
        # Sample texts with different sentiments
        self.positive_text = (
            "We are pleased to report strong performance across all business segments. "
            "Revenue increased by 15% and profitability improved significantly. "
            "Our strategic investments have yielded excellent returns, and we are "
            "well-positioned for continued growth in the coming year."
        )
        
        self.negative_text = (
            "The company faces significant challenges in the current market environment. "
            "Revenue declined by 10% due to increased competition and regulatory pressures. "
            "We have encountered difficulties in our expansion efforts, and the outlook "
            "remains uncertain for the next fiscal year."
        )
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        processed_text = self.analyzer.preprocess_text(self.positive_text)
        self.assertIsInstance(processed_text, str)
        self.assertEqual(processed_text, self.positive_text.lower())
    
    def test_analyze_textblob_sentiment(self):
        """Test sentiment analysis using TextBlob."""
        # Test positive text
        positive_sentiment = self.analyzer.analyze_textblob_sentiment(self.positive_text)
        self.assertGreater(positive_sentiment['polarity'], 0)
        
        # Test negative text
        negative_sentiment = self.analyzer.analyze_textblob_sentiment(self.negative_text)
        self.assertLess(negative_sentiment['polarity'], 0)
    
    def test_analyze_lexicon_sentiment(self):
        """Test sentiment analysis using financial lexicons."""
        # Test positive text
        positive_sentiment = self.analyzer.analyze_lexicon_sentiment(self.positive_text)
        
        # Check the structure of the result
        self.assertIsInstance(positive_sentiment, dict)
        self.assertIn('positive_score', positive_sentiment)
        self.assertIn('negative_score', positive_sentiment)
        self.assertIn('net_score', positive_sentiment)
        
        # For the positive text, positive score should be higher than negative
        self.assertGreater(positive_sentiment['positive_score'], positive_sentiment['negative_score'])
        
        # Test negative text
        negative_sentiment = self.analyzer.analyze_lexicon_sentiment(self.negative_text)
        
        # For the negative text, negative score should be higher than positive
        self.assertGreater(negative_sentiment['negative_score'], negative_sentiment['positive_score'])
    
    def test_extract_sentiment_by_section(self):
        """Test sentiment extraction by section."""
        # Create a mock filing with sections
        filing_data = pd.Series({
            'section_item_1a': self.negative_text,  # Risk factors - negative
            'section_item_7': self.positive_text    # MD&A - positive
        })
        
        sentiment_data = self.analyzer.extract_sentiment_by_section(filing_data)
        
        # Check that we got sentiment for both sections
        self.assertEqual(len(sentiment_data), 2)
        self.assertIn('item_1a', sentiment_data)
        self.assertIn('item_7', sentiment_data)
        
        # Risk factors should be negative, MD&A should be positive
        self.assertLess(sentiment_data['item_1a']['lexicon_net_score'], 0)
        self.assertGreater(sentiment_data['item_7']['lexicon_net_score'], 0)


if __name__ == '__main__':
    unittest.main()