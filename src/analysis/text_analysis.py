"""
Module for analyzing text content in 10-K filings.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import logging
from collections import Counter
from textblob import TextBlob
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextAnalyzer:
    """
    Class for analyzing text content in 10-K filings.
    """
    
    def __init__(self):
        """
        Initialize the text analyzer.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Financial domain-specific stopwords
        self.financial_stopwords = {
            'company', 'corporation', 'inc', 'year', 'quarter', 'financial',
            'report', 'annual', 'fiscal', 'form', 'item', 'filing',
            'pursuant', 'exchange', 'commission', 'act', 'section',
            'statement', 'stock', 'shares', 'common', 'preferred',
            'outstanding', 'table', 'following', 'date', 'period', 'ended'
        }
        
        # Combined stopwords
        self.all_stopwords = self.stop_words.union(self.financial_stopwords)
        
        # Risk-related terms
        self.risk_terms = {
            'risk', 'uncertainty', 'adverse', 'negative', 'decline',
            'decrease', 'loss', 'penalty', 'litigation', 'lawsuit',
            'claim', 'regulatory', 'regulation', 'compliance', 'failure',
            'unable', 'difficult', 'challenge', 'competitor', 'competition',
            'hazard', 'disaster', 'disruption', 'interruption', 'shortage',
            'volatility', 'fluctuation', 'deterioration', 'recession',
            'inflation', 'deflation', 'pandemic', 'epidemic', 'outbreak',
            'breach', 'violation', 'penalty', 'fine', 'lawsuit', 'sue',
            'cybersecurity', 'hack', 'virus', 'malware', 'liability'
        }
        
        # Positive business terms
        self.positive_terms = {
            'growth', 'increase', 'profit', 'success', 'innovation',
            'opportunity', 'advantage', 'improvement', 'efficient',
            'strong', 'leadership', 'expand', 'gain', 'enhance',
            'achieve', 'exceed', 'outperform', 'progress', 'robust',
            'favorable', 'positive', 'benefit', 'strength', 'strategic',
            'profitable', 'momentum', 'resilient', 'sustainability'
        }
    
    def preprocess_text(self, text):
        """
        Preprocess text for analysis.
        
        Parameters:
        -----------
        text : str
            Text to preprocess
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if not isinstance(text, str) or not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_tokens(self, text, remove_stopwords=True, lemmatize=True, min_length=3):
        """
        Get tokens from text.
        
        Parameters:
        -----------
        text : str
            Text to tokenize
        remove_stopwords : bool
            Whether to remove stopwords
        lemmatize : bool
            Whether to lemmatize tokens
        min_length : int
            Minimum token length to keep
            
        Returns:
        --------
        list
            List of tokens
        """
        if not isinstance(text, str) or not text:
            return []
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter by length
        tokens = [token for token in tokens if len(token) >= min_length]
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.all_stopwords]
        
        # Lemmatize if requested
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def get_word_frequencies(self, text, top_n=50, remove_stopwords=True, lemmatize=True, min_length=3):
        """
        Get word frequencies from text.
        
        Parameters:
        -----------
        text : str
            Text to analyze
        top_n : int
            Number of top words to return
        remove_stopwords : bool
            Whether to remove stopwords
        lemmatize : bool
            Whether to lemmatize tokens
        min_length : int
            Minimum token length to keep
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with words and frequencies
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get tokens
        tokens = self.get_tokens(
            processed_text,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
            min_length=min_length
        )
        
        # Count tokens
        word_counts = Counter(tokens)
        
        # Convert to DataFrame
        df_words = pd.DataFrame([
            {'word': word, 'frequency': count}
            for word, count in word_counts.most_common(top_n)
        ])
        
        return df_words
    
    def calculate_sentiment(self, text):
        """
        Calculate sentiment scores for text.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary with sentiment metrics
        """
        if not isinstance(text, str) or not text:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'positive_score': 0,
                'negative_score': 0,
                'sentence_count': 0,
                'word_count': 0,
                'avg_sentence_length': 0
            }
        
        # Calculate overall sentiment using TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Get sentences
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Get word count
        words = word_tokenize(text)
        word_count = len(words)
        
        # Calculate average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Calculate positive and negative word frequencies
        processed_text = self.preprocess_text(text)
        tokens = self.get_tokens(processed_text, remove_stopwords=False, lemmatize=True)
        
        # Count risk and positive terms
        risk_count = sum(1 for token in tokens if token in self.risk_terms)
        positive_count = sum(1 for token in tokens if token in self.positive_terms)
        
        # Normalize by token count
        token_count = len(tokens) if tokens else 1
        negative_score = risk_count / token_count
        positive_score = positive_count / token_count
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'avg_sentence_length': avg_sentence_length
        }
    
    def extract_topics(self, texts, n_topics=5, n_top_words=10, method='lda'):
        """
        Extract topics from a collection of texts.
        
        Parameters:
        -----------
        texts : list
            List of text documents
        n_topics : int
            Number of topics to extract
        n_top_words : int
            Number of top words per topic
        method : str
            Topic modeling method ('lda' or 'nmf')
            
        Returns:
        --------
        tuple
            (topic model, vectorizer, topic words, document-topic matrix)
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) if isinstance(text, str) else "" for text in texts]
        
        # Create a vectorizer
        if method.lower() == 'nmf':
            # For NMF, use TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.85,
                stop_words=list(self.all_stopwords)
            )
        else:
            # For LDA, use CountVectorizer
            vectorizer = CountVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.85,
                stop_words=list(self.all_stopwords)
            )
        
        # Create document-term matrix
        dtm = vectorizer.fit_transform(processed_texts)
        
        # Create topic model
        if method.lower() == 'nmf':
            model = NMF(n_components=n_topics, random_state=42)
        else:
            model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
        
        # Fit model
        doc_topic_matrix = model.fit_transform(dtm)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top words for each topic
        topic_words = []
        for topic_idx, topic in enumerate(model.components_):
            top_word_indices = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topic_words.append((topic_idx, top_words))
        
        return model, vectorizer, topic_words, doc_topic_matrix
    
    def analyze_filings(self, filings_df, sections=None):
        """
        Analyze filings and extract text metrics.
        
        Parameters:
        -----------
        filings_df : pandas.DataFrame
            DataFrame containing filings data
        sections : list
            List of section names to analyze (if None, analyze full text)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with text analysis metrics
        """
        # Create a copy to avoid modifying the original
        df = filings_df.copy()
        
        # Check if we have the necessary columns
        if 'filing_text' not in df.columns:
            if 'filing_html' in df.columns:
                logger.warning("Using 'filing_html' column instead of 'filing_text'")
                text_column = 'filing_html'
            else:
                logger.error("Required columns not found in DataFrame")
                return df
        else:
            text_column = 'filing_text'
        
        # Define sections to analyze
        if sections is None:
            # Analyze full text only
            sections_to_analyze = [text_column]
        else:
            # Analyze specified sections
            sections_to_analyze = [f"section_{section}" for section in sections]
            # Check if section columns exist
            missing_sections = [col for col in sections_to_analyze if col not in df.columns]
            if missing_sections:
                logger.warning(f"Missing section columns: {missing_sections}")
                # Filter out missing sections
                sections_to_analyze = [col for col in sections_to_analyze if col in df.columns]
        
        # Initialize metrics DataFrame
        metrics_records = []
        
        # Process each filing
        for _, row in df.iterrows():
            filing_metrics = {
                'ticker': row.get('ticker', ''),
                'filing_date': row.get('filing_date', ''),
                'filing_year': row.get('filing_year', '')
            }
            
            # Process each section
            for section in sections_to_analyze:
                section_text = row.get(section, '')
                
                if not isinstance(section_text, str) or not section_text:
                    continue
                
                # Get sentiment metrics
                sentiment_metrics = self.calculate_sentiment(section_text)
                
                # Add metrics with section prefix
                section_name = section.replace('section_', '') if 'section_' in section else 'full_text'
                for key, value in sentiment_metrics.items():
                    filing_metrics[f"{section_name}_{key}"] = value
                
                # Get word frequencies
                try:
                    word_freq = self.get_word_frequencies(section_text, top_n=20)
                    
                    # Count risk and positive terms in top words
                    risk_terms_in_top = sum(1 for word in word_freq['word'] if word in self.risk_terms)
                    positive_terms_in_top = sum(1 for word in word_freq['word'] if word in self.positive_terms)
                    
                    filing_metrics[f"{section_name}_risk_terms_in_top20"] = risk_terms_in_top
                    filing_metrics[f"{section_name}_positive_terms_in_top20"] = positive_terms_in_top
                except Exception as e:
                    logger.error(f"Error getting word frequencies: {str(e)}")
            
            # Add record
            metrics_records.append(filing_metrics)
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_records)
        
        return metrics_df
    
    def compare_filings(self, filings_df, groupby='ticker', section='full_text'):
        """
        Compare filings across companies or time periods.
        
        Parameters:
        -----------
        filings_df : pandas.DataFrame
            DataFrame containing filings data
        groupby : str
            Column to group by ('ticker' or 'filing_year')
        section : str
            Section to analyze
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with comparison metrics
        """
        # Get metrics for the filings
        metrics_df = self.analyze_filings(filings_df)
        
        # Check if we have the necessary metrics
        required_cols = [
            f"{section}_polarity",
            f"{section}_subjectivity",
            f"{section}_positive_score",
            f"{section}_negative_score"
        ]
        
        missing_cols = [col for col in required_cols if col not in metrics_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Group by the specified column
        grouped = metrics_df.groupby(groupby)
        
        # Calculate aggregate metrics
        agg_metrics = grouped.agg({
            f"{section}_polarity": ['mean', 'std'],
            f"{section}_subjectivity": ['mean', 'std'],
            f"{section}_positive_score": ['mean', 'std'],
            f"{section}_negative_score": ['mean', 'std'],
            f"{section}_word_count": ['mean', 'std'],
            f"{section}_sentence_count": ['mean']
        })
        
        # Flatten column names
        agg_metrics.columns = [f"{col[0]}_{col[1]}" for col in agg_metrics.columns]
        
        # Reset index
        agg_metrics = agg_metrics.reset_index()
        
        return agg_metrics
    
    def plot_sentiment_trends(self, metrics_df, groupby='filing_year', section='full_text'):
        """
        Plot sentiment trends over time or across companies.
        
        Parameters:
        -----------
        metrics_df : pandas.DataFrame
            DataFrame with text metrics
        groupby : str
            Column to group by ('ticker' or 'filing_year')
        section : str
            Section to analyze
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with sentiment trends plot
        """
        # Check if we have the necessary columns
        required_cols = [
            groupby,
            f"{section}_polarity",
            f"{section}_positive_score",
            f"{section}_negative_score"
        ]
        
        missing_cols = [col for col in required_cols if col not in metrics_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot polarity
        sns.lineplot(
            data=metrics_df,
            x=groupby,
            y=f"{section}_polarity",
            marker='o',
            ax=axs[0]
        )
        axs[0].set_ylabel('Polarity Score')
        axs[0].set_title(f'Sentiment Polarity for {section}')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot positive and negative scores
        sns.lineplot(
            data=metrics_df,
            x=groupby,
            y=f"{section}_positive_score",
            marker='o',
            label='Positive Score',
            ax=axs[1]
        )
        sns.lineplot(
            data=metrics_df,
            x=groupby,
            y=f"{section}_negative_score",
            marker='o',
            label='Negative Score',
            ax=axs[1]
        )
        axs[1].set_ylabel('Score')
        axs[1].set_title(f'Positive and Negative Scores for {section}')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        # Set common xlabel
        if groupby == 'filing_year':
            xlabel = 'Filing Year'
        else:
            xlabel = 'Company'
        
        axs[1].set_xlabel(xlabel)
        
        plt.tight_layout()
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create a test DataFrame
    test_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'filing_year': [2021, 2021],
        'filing_text': [
            "This has been a strong year with positive growth and innovation.",
            "We face risks and uncertainties in the market, but remain competitive."
        ]
    })
    
    # Create analyzer and analyze text
    analyzer = TextAnalyzer()
    metrics_df = analyzer.analyze_filings(test_df)
    
    print(metrics_df.head())