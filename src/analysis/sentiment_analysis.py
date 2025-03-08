"""
Module for sentiment analysis of 10-K filings.
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging
from tqdm import tqdm
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
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SentimentAnalyzer:
    """
    Class for analyzing sentiment in 10-K filings.
    """
    
    def __init__(self, load_lexicons=True):
        """
        Initialize the sentiment analyzer.
        
        Parameters:
        -----------
        load_lexicons : bool
            Whether to load the financial sentiment lexicons
        """
        self.stop_words = set(stopwords.words('english'))
        
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
        
        # Load financial sentiment lexicons
        self.lexicons = {}
        if load_lexicons:
            self._load_lexicons()
    
    def _load_lexicons(self):
        """
        Load financial sentiment lexicons.
        
        Note: This is a placeholder. In a real implementation, you would
        load actual financial sentiment lexicons from files.
        """
        # Loughran-McDonald financial sentiment lexicon (simplified)
        # In a real implementation, load this from a file
        self.lexicons['lm_positive'] = {
            'able', 'above', 'accomplish', 'accomplishment', 'achievement',
            'advantage', 'beneficial', 'benefit', 'best', 'better',
            'bolster', 'boom', 'boosted', 'breakthrough', 'brilliant',
            'collaborate', 'confident', 'constructive', 'creative', 'delight',
            'delivered', 'dependable', 'desirable', 'desired', 'despite',
            'effective', 'efficient', 'enhance', 'enjoy', 'enthusiasm',
            'excellent', 'exceptional', 'excited', 'exclusive', 'favorable',
            'gain', 'good', 'great', 'greater', 'growth', 'guarantee',
            'high', 'higher', 'highest', 'honor', 'ideal', 'improve',
            'improvement', 'increase', 'incredible', 'innovative',
            'leading', 'lucrative', 'meritorious', 'opportunities',
            'optimistic', 'outstanding', 'perfect', 'pleased', 'positive',
            'potential', 'premier', 'premium', 'profit', 'profitability',
            'progress', 'prominent', 'prospered', 'prosperous', 'record',
            'reward', 'satisfaction', 'satisfied', 'solid', 'solution',
            'strength', 'strong', 'stronger', 'strongest', 'succeed',
            'success', 'successful', 'superior', 'surpass', 'transparency',
            'tremendous', 'up', 'upside', 'upturn', 'valuable', 'win'
        }
        
        self.lexicons['lm_negative'] = {
            'abnormal', 'abuse', 'adverse', 'against', 'alarming',
            'animosity', 'annoy', 'annoyance', 'anxieties', 'anxiety',
            'apathy', 'bad', 'catastrophe', 'caution', 'challenge',
            'catastrophic', 'concern', 'concerned', 'concerns', 'conflict',
            'confused', 'confusing', 'constraint', 'crash', 'crime',
            'crisis', 'critical', 'criticism', 'damage', 'damages',
            'damaging', 'decline', 'decreased', 'defect', 'deficit',
            'delays', 'delinquent', 'denied', 'depressed', 'depression',
            'deteriorate', 'deteriorating', 'difficulty', 'disagree',
            'disappointed', 'disappointment', 'disaster', 'dismiss',
            'dispute', 'disruption', 'downturn', 'drop', 'emergency',
            'fail', 'failed', 'failing', 'failure', 'fear', 'feared',
            'fractious', 'fraud', 'grievance', 'harmful', 'harsh',
            'hazard', 'hindrance', 'hostile', 'hurt', 'illegal',
            'impossibility', 'impossible', 'improper', 'imprudent',
            'inability', 'inadequate', 'incapable', 'ineffective',
            'inefficiency', 'infraction', 'infringement', 'injunction',
            'insolvent', 'instability', 'insufficient', 'interference',
            'interruption', 'lack', 'lawsuit', 'liabilities', 'liability',
            'limitation', 'liquidate', 'liquidation', 'litigation',
            'loss', 'losses', 'manipulation', 'misled', 'misrepresentation',
            'misstatement', 'mistreatment', 'negative', 'negligence',
            'negligent', 'objection', 'obstacle', 'obsolete', 'penalties',
            'penalty', 'peril', 'pessimistic', 'plaintiff', 'poor',
            'poverty', 'problem', 'problems', 'recession', 'risk',
            'risks', 'risky', 'serious', 'severely', 'shortage',
            'shortfall', 'shrinking', 'slump', 'sluggish', 'subdued',
            'suffer', 'suffered', 'suffers', 'susceptibility', 'threat',
            'threats', 'trouble', 'turmoil', 'unable', 'unacceptable',
            'undesirable', 'undocumented', 'unfavorable', 'unforeseen',
            'unfortunate', 'unpaid', 'unprofitable', 'unreliable',
            'unsuccessful', 'unsure', 'violation', 'warn', 'warning',
            'weakness', 'worst', 'worthless', 'write-down', 'writedown'
        }
        
        self.lexicons['lm_uncertainty'] = {
            'almost', 'ambiguity', 'ambiguous', 'appear', 'appeared',
            'appears', 'approximate', 'approximation', 'around',
            'assumption', 'believe', 'believed', 'believes', 'can',
            'clarification', 'conceivable', 'conditional', 'conceivably',
            'confuse', 'confuses', 'confusing', 'confusingly', 'confusion',
            'contingency', 'could', 'depend', 'depended', 'depending',
            'depends', 'doubt', 'doubtful', 'downside', 'exposure',
            'exposures', 'fluctuate', 'fluctuated', 'fluctuates',
            'fluctuating', 'fluctuation', 'hesitant', 'imprecise',
            'incompleteness', 'indefinite', 'indefinitely', 'indeterminable',
            'inexact', 'instability', 'intangible', 'likelihood', 'may',
            'maybe', 'might', 'nearly', 'occasionally', 'possibility',
            'possible', 'possibly', 'precaution', 'precautionary',
            'predict', 'predictability', 'predicted', 'predicting',
            'prediction', 'predictions', 'predictive', 'risk', 'risked',
            'riskier', 'riskiness', 'risking', 'risky', 'rough', 'roughly',
            'rumor', 'seems', 'seldom', 'seldomly', 'sometime', 'sometimes',
            'somewhat', 'somewhere', 'speculate', 'speculated', 'speculates',
            'speculating', 'speculation', 'speculations', 'speculative',
            'sporadic', 'sudden', 'suddenly', 'suggest', 'suggested',
            'suggesting', 'suggests', 'susceptibility', 'tending', 'tentative',
            'turbulence', 'uncertain', 'uncertainly', 'uncertainties',
            'uncertainty', 'unclear', 'unconfirmed', 'undecided',
            'undefined', 'understand', 'understood', 'understands',
            'unknown', 'unobservable', 'unplanned', 'unpredictable',
            'unpredictably', 'unpredicted', 'unlikely', 'unproven',
            'unquantifiable', 'unreliable', 'unseasonably', 'unusual',
            'unusually', 'unwritten', 'vagaries', 'vague', 'vaguely',
            'vagueness', 'variability', 'variable', 'variance', 'variant',
            'variation', 'varied', 'varies', 'vary', 'varying', 'volatile',
            'volatility'
        }
        
        self.lexicons['lm_litigious'] = {
            'abrogate', 'abrogated', 'abrogates', 'abrogating', 'abrogation',
            'absolve', 'acquit', 'acquits', 'acquittal', 'acquitted',
            'adjudge', 'adjudged', 'adjudges', 'adjudging', 'adjudicate',
            'adjudicated', 'adjudicates', 'adjudicating', 'adjudication',
            'admissibility', 'admissible', 'allegation', 'allegations',
            'allege', 'alleged', 'allegedly', 'alleges', 'alleging',
            'amicus', 'annul', 'annulled', 'annulling', 'annulment',
            'annuls', 'appeal', 'appealed', 'appealing', 'appeals',
            'appellant', 'appellate', 'appellees', 'arbitrate', 'arbitrated',
            'arbitrates', 'arbitrating', 'arbitration', 'arbitrations',
            'attest', 'attested', 'attesting', 'attestation', 'attorney',
            'attorneys', 'certiorari', 'claimant', 'claimants', 'complaint',
            'complaints', 'confess', 'confessed', 'confesses', 'confessing',
            'confession', 'confessions', 'constitutionality', 'contractual',
            'contradict', 'contradicted', 'contradicting', 'contradiction',
            'contradictory', 'contradicts', 'convict', 'convicted',
            'convicting', 'conviction', 'convictions', 'counterclaim',
            'counterclaimed', 'counterclaiming', 'counterclaims', 'court',
            'courts', 'criminal', 'damages', 'decree', 'decreed', 'decreeing',
            'decrees', 'defendant', 'defendants', 'depose', 'deposed',
            'deposes', 'deposing', 'deposition', 'depositions', 'evidence',
            'exculpate', 'exculpated', 'exculpates', 'exculpating',
            'exculpation', 'exculpatory', 'felonies', 'felonious', 'felony',
            'grievance', 'grievances', 'indictment', 'indictments', 'injunction',
            'injunctions', 'interrogate', 'interrogated', 'interrogates',
            'interrogating', 'interrogation', 'interrogations', 'interrogatories',
            'interrogatory', 'judgment', 'judgments', 'judicial', 'jurisdiction',
            'jurisdictions', 'lawsuit', 'lawsuits', 'lawyer', 'lawyers',
            'legal', 'legality', 'legally', 'legislate', 'legislated',
            'legislates', 'legislating', 'legislation', 'legislations',
            'legislator', 'legislators', 'liabilities', 'liability',
            'liable', 'litigant', 'litigants', 'litigate', 'litigated',
            'litigates', 'litigating', 'litigation', 'litigations', 'litigious',
            'misdemeanor', 'misdemeanors', 'negligence', 'negligent',
            'objection', 'objections', 'overturn', 'overturned', 'overturning',
            'overturns', 'plaintiff', 'plaintiffs', 'pleading', 'pleadings',
            'prosecute', 'prosecuted', 'prosecutes', 'prosecuting', 'prosecution',
            'prosecutions', 'quash', 'quashed', 'quashes', 'quashing',
            'regulation', 'regulations', 'regulatory', 'remand', 'remanded',
            'remanding', 'remands', 'restitution', 'restrain', 'restrained',
            'restraining', 'restraint', 'restraints', 'verdict', 'verdicts'
        }
        
        logger.info("Loaded financial sentiment lexicons")
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
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
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters that aren't relevant for sentiment
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def extract_sentences(self, text):
        """
        Extract sentences from text.
        
        Parameters:
        -----------
        text : str
            Text to extract sentences from
            
        Returns:
        --------
        list
            List of sentences
        """
        if not isinstance(text, str) or not text:
            return []
        
        # Extract sentences
        sentences = sent_tokenize(text)
        
        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def analyze_textblob_sentiment(self, text):
        """
        Analyze sentiment using TextBlob.
        
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
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Create TextBlob
        blob = TextBlob(text)
        
        # Calculate polarity and subjectivity
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Extract sentences
        sentences = self.extract_sentences(text)
        
        # Analyze sentence sentiment
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for sentence in sentences:
            sentence_blob = TextBlob(sentence)
            sentence_polarity = sentence_blob.sentiment.polarity
            
            if sentence_polarity > 0.1:
                positive_count += 1
            elif sentence_polarity < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate percentages
        total_sentences = len(sentences) if sentences else 1
        positive_pct = positive_count / total_sentences
        negative_pct = negative_count / total_sentences
        neutral_pct = neutral_count / total_sentences
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'positive': positive_pct,
            'negative': negative_pct,
            'neutral': neutral_pct
        }
    
    def analyze_lexicon_sentiment(self, text):
        """
        Analyze sentiment using financial lexicons.
        
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
                'positive_count': 0,
                'negative_count': 0,
                'uncertainty_count': 0,
                'litigious_count': 0,
                'positive_score': 0,
                'negative_score': 0,
                'uncertainty_score': 0,
                'litigious_score': 0,
                'net_score': 0
            }
        
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.all_stopwords]
        
        # Count words in each category
        positive_count = sum(1 for token in tokens if token in self.lexicons['lm_positive'])
        negative_count = sum(1 for token in tokens if token in self.lexicons['lm_negative'])
        uncertainty_count = sum(1 for token in tokens if token in self.lexicons['lm_uncertainty'])
        litigious_count = sum(1 for token in tokens if token in self.lexicons['lm_litigious'])
        
        # Calculate scores (normalized by token count)
        token_count = len(tokens) if tokens else 1
        positive_score = positive_count / token_count
        negative_score = negative_count / token_count
        uncertainty_score = uncertainty_count / token_count
        litigious_score = litigious_count / token_count
        
        # Calculate net sentiment score
        net_score = positive_score - negative_score
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'uncertainty_count': uncertainty_count,
            'litigious_count': litigious_count,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'uncertainty_score': uncertainty_score,
            'litigious_score': litigious_score,
            'net_score': net_score
        }
    
    def extract_sentiment_by_section(self, filing_data, section_prefix='section_'):
        """
        Extract sentiment metrics for each section in a filing.
        
        Parameters:
        -----------
        filing_data : pandas Series or dict
            Filing data with section text
        section_prefix : str
            Prefix for section columns
            
        Returns:
        --------
        dict
            Dictionary with sentiment metrics for each section
        """
        # Get section columns
        section_cols = [col for col in filing_data.index if col.startswith(section_prefix)]
        
        sentiment_data = {}
        
        for section_col in section_cols:
            section_text = filing_data.get(section_col, '')
            
            if not isinstance(section_text, str) or not section_text:
                continue
            
            # Get section name
            section_name = section_col.replace(section_prefix, '')
            
            # Analyze sentiment using both methods
            textblob_sentiment = self.analyze_textblob_sentiment(section_text)
            lexicon_sentiment = self.analyze_lexicon_sentiment(section_text)
            
            # Combine sentiment data
            section_sentiment = {
                'section': section_name,
                **{f'textblob_{k}': v for k, v in textblob_sentiment.items()},
                **{f'lexicon_{k}': v for k, v in lexicon_sentiment.items()}
            }
            
            sentiment_data[section_name] = section_sentiment
        
        return sentiment_data
    
    def analyze_filings(self, filings_df, sections=None):
        """
        Analyze sentiment in filings.
        
        Parameters:
        -----------
        filings_df : pandas.DataFrame
            DataFrame containing filings data
        sections : list
            List of section names to analyze (if None, analyze all sections)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment metrics
        """
        # Make a copy to avoid modifying the original
        df = filings_df.copy()
        
        # Initialize results
        sentiment_records = []
        
        # Process each filing
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiment"):
            filing_info = {
                'ticker': row.get('ticker', ''),
                'company_name': row.get('company_name', ''),
                'filing_date': row.get('filing_date', ''),
                'filing_year': row.get('filing_year', '')
            }
            
            # Extract sentiment by section
            section_sentiments = self.extract_sentiment_by_section(row)
            
            # Filter sections if specified
            if sections:
                section_sentiments = {k: v for k, v in section_sentiments.items() if k in sections}
            
            # Create a record for each section
            for section_name, sentiment_data in section_sentiments.items():
                record = {
                    **filing_info,
                    'section': section_name,
                    **sentiment_data
                }
                sentiment_records.append(record)
        
        # Create DataFrame
        sentiment_df = pd.DataFrame(sentiment_records)
        
        return sentiment_df
    
    def plot_sentiment_trends(self, sentiment_df, section='item_7', metric='lexicon_net_score'):
        """
        Plot sentiment trends over time.
        
        Parameters:
        -----------
        sentiment_df : pandas.DataFrame
            DataFrame with sentiment metrics
        section : str
            Section to analyze
        metric : str
            Sentiment metric to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Sentiment trends plot
        """
        # Filter data
        filtered_df = sentiment_df[sentiment_df['section'] == section].copy()
        
        # Ensure filing_date is datetime
        if 'filing_date' in filtered_df.columns:
            filtered_df['filing_date'] = pd.to_datetime(filtered_df['filing_date'])
        
        # Check if we have the metric
        if metric not in filtered_df.columns:
            logger.error(f"Metric '{metric}' not found in DataFrame")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment by company
        for ticker, company_data in filtered_df.groupby('ticker'):
            company_data = company_data.sort_values('filing_date')
            ax.plot(
                company_data['filing_date'],
                company_data[metric],
                marker='o',
                linewidth=2,
                label=ticker
            )
        
        # Set labels and title
        metric_name = metric.replace('lexicon_', '').replace('textblob_', '').replace('_', ' ').title()
        section_name = section.replace('item_', 'Item ').title()
        ax.set_title(f'{metric_name} for {section_name} Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Filing Date', fontsize=14)
        ax.set_ylabel(metric_name, fontsize=14)
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add horizontal line at y=0 for net scores
        if 'net_score' in metric or 'polarity' in metric:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig
    
    def plot_sentiment_comparison(self, sentiment_df, section='item_7', metric='lexicon_net_score'):
        """
        Create a comparison bar chart of sentiment across companies.
        
        Parameters:
        -----------
        sentiment_df : pandas.DataFrame
            DataFrame with sentiment metrics
        section : str
            Section to analyze
        metric : str
            Sentiment metric to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Sentiment comparison plot
        """
        # Filter data
        filtered_df = sentiment_df[sentiment_df['section'] == section].copy()
        
        # Group by company and get the most recent filing
        if 'filing_date' in filtered_df.columns:
            filtered_df['filing_date'] = pd.to_datetime(filtered_df['filing_date'])
            latest_filings = filtered_df.sort_values('filing_date').groupby('ticker').tail(1)
        else:
            latest_filings = filtered_df.groupby('ticker').first()
            latest_filings = latest_filings.reset_index()
        
        # Check if we have the metric
        if metric not in latest_filings.columns:
            logger.error(f"Metric '{metric}' not found in DataFrame")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by metric value
        latest_filings = latest_filings.sort_values(metric)
        
        # Set bar colors based on sentiment (red for negative, green for positive)
        colors = ['#d73027' if val < 0 else '#4575b4' for val in latest_filings[metric]]
        
        # Plot bars
        ax.bar(
            latest_filings['ticker'],
            latest_filings[metric],
            color=colors,
            alpha=0.8
        )
        
        # Set labels and title
        metric_name = metric.replace('lexicon_', '').replace('textblob_', '').replace('_', ' ').title()
        section_name = section.replace('item_', 'Item ').title()
        ax.set_title(f'{metric_name} for {section_name} by Company', fontsize=16, fontweight='bold')
        ax.set_xlabel('Company', fontsize=14)
        ax.set_ylabel(metric_name, fontsize=14)
        
        # Add horizontal line at y=0 for net scores
        if 'net_score' in metric or 'polarity' in metric:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create a test DataFrame
    test_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'filing_year': [2021, 2021],
        'section_item_7': [
            "This has been a strong year with positive growth and innovation.",
            "We face risks and uncertainties in the market, but remain competitive."
        ]
    })
    
    # Create analyzer and analyze sentiment
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_filings(test_df)
    
    print(sentiment_df.head())