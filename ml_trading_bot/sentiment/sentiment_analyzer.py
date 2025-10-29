"""
Sentiment Analysis Module using FinBERT and other NLP techniques.
Analyzes news, social media, and other text sources for market sentiment.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime, timedelta


class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer for financial text.
    Uses FinBERT for financial-specific sentiment and TextBlob as fallback.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model name (default: FinBERT)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load FinBERT model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Create pipeline for easier inference
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            self.finbert_loaded = True
            print(f"FinBERT model loaded successfully: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load FinBERT model: {e}")
            print("Falling back to TextBlob for sentiment analysis")
            self.finbert_loaded = False

        # Financial keywords for contextual weighting
        self.bullish_keywords = [
            'bullish', 'surge', 'rally', 'breakout', 'gains', 'profit',
            'growth', 'beat', 'outperform', 'upgrade', 'strong', 'positive',
            'buy', 'long', 'moon', 'rocket', 'all-time high', 'ath'
        ]

        self.bearish_keywords = [
            'bearish', 'crash', 'plunge', 'decline', 'loss', 'miss',
            'downgrade', 'weak', 'negative', 'sell', 'short', 'dump',
            'panic', 'fear', 'collapse', 'recession'
        ]

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dict with 'score' (-1 to 1), 'confidence' (0 to 1), and 'label'
        """
        if not text or len(text.strip()) < 3:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'neutral'}

        # Clean text
        text = self._preprocess_text(text)

        if self.finbert_loaded:
            return self._analyze_with_finbert(text)
        else:
            return self._analyze_with_textblob(text)

    def _analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT.

        Returns:
            Dict with normalized sentiment score and confidence
        """
        try:
            # Truncate text to model's max length
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            # Get prediction
            result = self.sentiment_pipeline(text)[0]

            # FinBERT outputs: positive, negative, neutral
            label = result['label'].lower()
            confidence = result['score']

            # Convert to normalized score (-1 to 1)
            if label == 'positive':
                score = confidence
            elif label == 'negative':
                score = -confidence
            else:  # neutral
                score = 0.0

            return {
                'score': score,
                'confidence': confidence,
                'label': label
            }

        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return self._analyze_with_textblob(text)

    def _analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Fallback sentiment analysis using TextBlob.

        Returns:
            Dict with sentiment score and confidence
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        # Use subjectivity as confidence proxy (more subjective = more confident sentiment)
        confidence = subjectivity

        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'score': polarity,
            'confidence': confidence,
            'label': label
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment dictionaries
        """
        return [self.analyze_text(text) for text in texts]

    def aggregate_sentiment(
        self,
        sentiments: List[Dict[str, float]],
        method: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Aggregate multiple sentiment scores.

        Args:
            sentiments: List of sentiment dictionaries
            method: 'weighted' (by confidence) or 'simple' (unweighted average)

        Returns:
            Aggregated sentiment dictionary
        """
        if not sentiments:
            return {'score': 0.0, 'confidence': 0.0, 'label': 'neutral'}

        if method == 'weighted':
            # Weight by confidence
            total_weight = sum(s['confidence'] for s in sentiments)
            if total_weight == 0:
                avg_score = 0.0
            else:
                avg_score = sum(s['score'] * s['confidence'] for s in sentiments) / total_weight

            avg_confidence = np.mean([s['confidence'] for s in sentiments])

        else:  # simple
            avg_score = np.mean([s['score'] for s in sentiments])
            avg_confidence = np.mean([s['confidence'] for s in sentiments])

        # Determine label
        if avg_score > 0.1:
            label = 'positive'
        elif avg_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'score': avg_score,
            'confidence': avg_confidence,
            'label': label,
            'count': len(sentiments)
        }

    def analyze_with_context(
        self,
        text: str,
        ticker: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Analyze sentiment with additional context weighting.

        Args:
            text: Text to analyze
            ticker: Stock/crypto ticker symbol
            keywords: Additional keywords to weight

        Returns:
            Enhanced sentiment dictionary
        """
        # Base sentiment
        sentiment = self.analyze_text(text)

        # Check for ticker mentions
        if ticker and ticker.upper() in text.upper():
            sentiment['ticker_mentioned'] = True
            # Boost confidence if ticker is mentioned
            sentiment['confidence'] = min(1.0, sentiment['confidence'] * 1.2)

        # Check for financial keywords
        text_lower = text.lower()
        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)

        if keywords:
            for kw in keywords:
                if kw.lower() in text_lower:
                    sentiment['confidence'] = min(1.0, sentiment['confidence'] * 1.1)

        sentiment['bullish_keywords'] = bullish_count
        sentiment['bearish_keywords'] = bearish_count

        # Adjust score based on keyword balance
        if bullish_count > bearish_count:
            sentiment['score'] = min(1.0, sentiment['score'] * 1.1)
        elif bearish_count > bullish_count:
            sentiment['score'] = max(-1.0, sentiment['score'] * 1.1)

        return sentiment

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?$%]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def get_sentiment_signal(
        self,
        recent_sentiments: List[Dict[str, float]],
        threshold: float = 0.3
    ) -> str:
        """
        Generate trading signal based on sentiment trend.

        Args:
            recent_sentiments: List of recent sentiment scores (chronological)
            threshold: Minimum score magnitude for signal

        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if not recent_sentiments or len(recent_sentiments) < 3:
            return 'NEUTRAL'

        # Calculate weighted average (recent data more important)
        weights = np.linspace(0.5, 1.0, len(recent_sentiments))
        scores = np.array([s['score'] for s in recent_sentiments])
        weighted_avg = np.average(scores, weights=weights)

        # Check trend
        if len(recent_sentiments) >= 5:
            recent_trend = np.mean(scores[-3:]) - np.mean(scores[:3])
        else:
            recent_trend = 0

        # Generate signal
        if weighted_avg > threshold and recent_trend > 0:
            return 'BULLISH'
        elif weighted_avg < -threshold and recent_trend < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources (news, Twitter, Reddit, etc.)
    """

    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.source_weights = {
            'news': 1.0,
            'twitter': 0.7,
            'reddit': 0.6,
            'earnings': 1.2,
            'analyst': 1.1
        }

    def aggregate_multi_source(
        self,
        source_sentiments: Dict[str, List[Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        Aggregate sentiment from multiple sources with weighting.

        Args:
            source_sentiments: Dict mapping source name to list of sentiments

        Returns:
            Aggregated sentiment with source breakdown
        """
        all_sentiments = []
        source_scores = {}

        for source, sentiments in source_sentiments.items():
            if not sentiments:
                continue

            # Aggregate per source
            source_agg = self.analyzer.aggregate_sentiment(sentiments)
            source_scores[source] = source_agg

            # Apply source weight
            weight = self.source_weights.get(source, 0.5)
            for sentiment in sentiments:
                weighted_sentiment = sentiment.copy()
                weighted_sentiment['confidence'] *= weight
                all_sentiments.append(weighted_sentiment)

        # Overall aggregation
        if all_sentiments:
            overall = self.analyzer.aggregate_sentiment(all_sentiments)
            overall['sources'] = source_scores
            return overall
        else:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'label': 'neutral',
                'sources': {}
            }

    def get_sentiment_time_series(
        self,
        sentiments: List[Tuple[datetime, Dict[str, float]]],
        window_hours: int = 24
    ) -> List[Dict[str, float]]:
        """
        Create time-series of sentiment scores.

        Args:
            sentiments: List of (timestamp, sentiment) tuples
            window_hours: Rolling window size in hours

        Returns:
            List of time-windowed sentiment aggregations
        """
        if not sentiments:
            return []

        # Sort by timestamp
        sentiments = sorted(sentiments, key=lambda x: x[0])

        time_series = []
        window_delta = timedelta(hours=window_hours)

        current_time = sentiments[-1][0]  # Most recent
        end_time = sentiments[0][0]  # Oldest

        while current_time >= end_time:
            window_start = current_time - window_delta

            # Get sentiments in window
            window_sentiments = [
                s for t, s in sentiments
                if window_start <= t <= current_time
            ]

            if window_sentiments:
                agg = self.analyzer.aggregate_sentiment(window_sentiments)
                agg['timestamp'] = current_time
                time_series.append(agg)

            # Move window back
            current_time -= timedelta(hours=1)

        return list(reversed(time_series))
