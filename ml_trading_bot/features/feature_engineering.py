"""
Feature Engineering Framework.
Creates technical indicators and sentiment features for ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas_ta as ta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import MarketData, SentimentData, Feature, get_db


class TechnicalFeatureEngineer:
    """
    Creates technical indicator features from price data.
    """

    def __init__(self):
        pass

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators
        """
        if df.empty or len(df) < 100:
            return df

        df = df.copy()

        # RSI (Relative Strength Index)
        df['rsi_14'] = ta.rsi(df['close'], length=14)

        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']

        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Moving Averages
        df['ma_20'] = ta.sma(df['close'], length=20)
        df['ma_50'] = ta.sma(df['close'], length=50)
        df['ma_100'] = ta.sma(df['close'], length=100)

        # EMA
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)

        # ATR (Average True Range) - Volatility
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # ADX (Average Directional Index) - Trend Strength
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx_14'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']

        # CCI (Commodity Channel Index)
        df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)

        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            df['stochastic_k'] = stoch['STOCHk_14_3_3']
            df['stochastic_d'] = stoch['STOCHd_14_3_3']

        # OBV (On-Balance Volume)
        if 'volume' in df.columns and df['volume'].notna().any():
            df['obv'] = ta.obv(df['close'], df['volume'])

        return df

    def calculate_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price patterns and levels.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with pattern features
        """
        if df.empty or len(df) < 20:
            return df

        df = df.copy()

        # Momentum
        df['momentum_10'] = df['close'].pct_change(10) * 100
        df['momentum_20'] = df['close'].pct_change(20) * 100

        # Volatility (rolling std)
        df['volatility_10'] = df['close'].pct_change().rolling(10).std() * 100
        df['volatility_20'] = df['close'].pct_change().rolling(20).std() * 100

        # Support and Resistance (using rolling min/max)
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()

        # Distance from support/resistance
        df['dist_from_support'] = (df['close'] - df['support_level']) / df['close'] * 100
        df['dist_from_resistance'] = (df['resistance_level'] - df['close']) / df['close'] * 100

        # Price position in range
        df['price_range_position'] = (
            (df['close'] - df['support_level']) /
            (df['resistance_level'] - df['support_level'])
        )

        # Candle patterns
        df['candle_body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']

        # Bullish/Bearish
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        return df

    def calculate_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-over and interaction features.

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with cross features
        """
        if df.empty:
            return df

        df = df.copy()

        # MA crosses
        if 'ma_20' in df.columns and 'ma_50' in df.columns:
            df['ma_20_50_diff'] = df['ma_20'] - df['ma_50']
            df['ma_20_above_50'] = (df['ma_20'] > df['ma_50']).astype(int)

        if 'ma_50' in df.columns and 'ma_100' in df.columns:
            df['ma_50_100_diff'] = df['ma_50'] - df['ma_100']
            df['ma_50_above_100'] = (df['ma_50'] > df['ma_100']).astype(int)

        # Price vs MA
        if 'ma_20' in df.columns:
            df['price_ma20_diff'] = (df['close'] - df['ma_20']) / df['ma_20'] * 100

        if 'ma_50' in df.columns:
            df['price_ma50_diff'] = (df['close'] - df['ma_50']) / df['ma_50'] * 100

        # Bollinger Band position
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            df['bb_position'] = (
                (df['close'] - df['bb_lower']) /
                (df['bb_upper'] - df['bb_lower'])
            )

        # RSI categories
        if 'rsi_14' in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)

        return df


class SentimentFeatureEngineer:
    """
    Creates sentiment features from text data.
    """

    def __init__(self):
        pass

    def aggregate_sentiment_features(
        self,
        sentiments: List[Dict],
        windows: List[int] = [1, 4, 24]  # hours
    ) -> Dict[str, float]:
        """
        Aggregate sentiment data into time-windowed features.

        Args:
            sentiments: List of sentiment dictionaries with timestamps
            windows: Time windows in hours

        Returns:
            Dictionary of sentiment features
        """
        if not sentiments:
            return {}

        features = {}
        now = datetime.now()

        for window_hours in windows:
            window_start = now - timedelta(hours=window_hours)

            # Filter sentiments in window
            window_sentiments = [
                s for s in sentiments
                if s.get('timestamp', now) >= window_start
            ]

            if window_sentiments:
                scores = [s['sentiment_score'] for s in window_sentiments]
                confidences = [s['confidence'] for s in window_sentiments]

                # Basic stats
                features[f'sentiment_score_{window_hours}h'] = np.mean(scores)
                features[f'sentiment_std_{window_hours}h'] = np.std(scores)
                features[f'sentiment_volume_{window_hours}h'] = len(scores)

                # Weighted by confidence
                if sum(confidences) > 0:
                    weighted_score = np.average(scores, weights=confidences)
                    features[f'sentiment_weighted_{window_hours}h'] = weighted_score

                # Sentiment trend
                if len(scores) >= 2:
                    mid = len(scores) // 2
                    recent_avg = np.mean(scores[mid:])
                    older_avg = np.mean(scores[:mid])
                    features[f'sentiment_trend_{window_hours}h'] = recent_avg - older_avg

                # Extremes
                features[f'sentiment_max_{window_hours}h'] = max(scores)
                features[f'sentiment_min_{window_hours}h'] = min(scores)

            else:
                # No data in window
                features[f'sentiment_score_{window_hours}h'] = 0.0
                features[f'sentiment_volume_{window_hours}h'] = 0

        return features


class FeatureEngineeringPipeline:
    """
    Main feature engineering pipeline.
    Combines technical and sentiment features.
    """

    def __init__(self):
        self.technical_engineer = TechnicalFeatureEngineer()
        self.sentiment_engineer = SentimentFeatureEngineer()

    def create_features_from_db(
        self,
        epic: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Create features from database data.

        Args:
            epic: Market epic
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with all features
        """
        with get_db() as db:
            # Fetch market data
            market_data = db.query(MarketData).filter(
                MarketData.epic == epic,
                MarketData.timestamp >= start_date,
                MarketData.timestamp <= end_date
            ).order_by(MarketData.timestamp).all()

            if not market_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            } for md in market_data])

            df = df.sort_values('timestamp').reset_index(drop=True)

            # Calculate technical indicators
            df = self.technical_engineer.calculate_technical_indicators(df)
            df = self.technical_engineer.calculate_price_patterns(df)
            df = self.technical_engineer.calculate_cross_features(df)

            # Fetch sentiment data
            sentiment_data = db.query(SentimentData).filter(
                SentimentData.epic == epic,
                SentimentData.timestamp >= start_date,
                SentimentData.timestamp <= end_date
            ).order_by(SentimentData.timestamp).all()

            # Add sentiment features for each timestamp
            sentiment_features_list = []

            for _, row in df.iterrows():
                timestamp = row['timestamp']

                # Get relevant sentiment data (up to this timestamp)
                relevant_sentiments = [
                    {
                        'sentiment_score': sd.sentiment_score,
                        'confidence': sd.confidence,
                        'timestamp': sd.timestamp
                    }
                    for sd in sentiment_data
                    if sd.timestamp <= timestamp
                ]

                # Calculate sentiment features
                sent_features = self.sentiment_engineer.aggregate_sentiment_features(
                    relevant_sentiments
                )
                sentiment_features_list.append(sent_features)

            # Merge sentiment features
            if sentiment_features_list:
                sent_df = pd.DataFrame(sentiment_features_list)
                df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)

            # Add target variables (future returns)
            df = self.add_target_variables(df)

            return df

    def add_target_variables(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 4, 24]  # hours
    ) -> pd.DataFrame:
        """
        Add target variables (future returns).

        Args:
            df: DataFrame with features
            horizons: Prediction horizons in hours

        Returns:
            DataFrame with targets
        """
        if df.empty:
            return df

        df = df.copy()

        for horizon in horizons:
            # Future return
            df[f'target_return_{horizon}h'] = df['close'].pct_change(horizon).shift(-horizon) * 100

            # Future direction
            df[f'target_direction_{horizon}h'] = np.where(
                df[f'target_return_{horizon}h'] > 0.1, 'UP',
                np.where(df[f'target_return_{horizon}h'] < -0.1, 'DOWN', 'NEUTRAL')
            )

        return df

    def save_features_to_db(self, epic: str, df: pd.DataFrame):
        """
        Save engineered features to database.

        Args:
            epic: Market epic
            df: DataFrame with features
        """
        with get_db() as db:
            for _, row in df.iterrows():
                # Skip if timestamp is NaT
                if pd.isna(row['timestamp']):
                    continue

                # Check if features already exist
                existing = db.query(Feature).filter(
                    Feature.epic == epic,
                    Feature.timestamp == row['timestamp']
                ).first()

                if existing:
                    # Update existing
                    for col in df.columns:
                        if col != 'timestamp' and col in row and hasattr(existing, col):
                            setattr(existing, col, float(row[col]) if pd.notna(row[col]) else None)
                else:
                    # Create new
                    feature_data = {
                        'epic': epic,
                        'timestamp': row['timestamp']
                    }

                    # Add all features
                    for col in df.columns:
                        if col != 'timestamp' and col in row:
                            # Map column names to database columns
                            if hasattr(Feature, col):
                                feature_data[col] = float(row[col]) if pd.notna(row[col]) else None

                    feature = Feature(**feature_data)
                    db.add(feature)

            db.commit()
            print(f"✓ Saved {len(df)} feature rows for {epic}")

    def create_and_save_features(
        self,
        instruments: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """
        Create and save features for multiple instruments.

        Args:
            instruments: List of epics
            start_date: Start date
            end_date: End date
        """
        for epic in instruments:
            print(f"\nProcessing features for {epic}...")
            df = self.create_features_from_db(epic, start_date, end_date)

            if not df.empty:
                self.save_features_to_db(epic, df)
                print(f"✓ Feature engineering completed for {epic}")
            else:
                print(f"⚠ No data found for {epic}")


# Example usage
if __name__ == "__main__":
    from datetime import datetime, timedelta

    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()

    # Create features for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    instruments = ["US100", "EURUSD", "GBPUSD"]

    pipeline.create_and_save_features(instruments, start_date, end_date)
