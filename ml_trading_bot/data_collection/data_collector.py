"""
Data Collection Pipeline for ML Trading Bot.
Collects market data, sentiment data, and stores in database.
"""

import asyncio
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import MarketData, SentimentData, get_db
from ml_trading_bot.sentiment import SentimentAnalyzer


class CapitalDataCollector:
    """
    Collects market data from Capital.com API.
    """

    def __init__(self, email: str, api_key: str, api_password: str, demo: bool = True):
        """
        Initialize data collector.

        Args:
            email: Capital.com account email
            api_key: API key
            api_password: API password
            demo: Use demo account (default: True)
        """
        self.email = email
        self.api_key = api_key
        self.api_password = api_password
        self.demo = demo

        if demo:
            self.base_url = "https://demo-api-capital.backend-capital.com"
        else:
            self.base_url = "https://api-capital.backend-capital.com"

        self.session_token = None
        self.security_token = None
        self.token_expires = None

    async def authenticate(self):
        """Authenticate with Capital.com API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/session",
                json={
                    "identifier": self.email,
                    "password": self.api_password
                },
                headers={
                    "X-CAP-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                }
            )

            if response.status_code == 200:
                self.session_token = response.headers.get("CST")
                self.security_token = response.headers.get("X-SECURITY-TOKEN")
                self.token_expires = datetime.now() + timedelta(minutes=10)
                print("✓ Authenticated with Capital.com")
            else:
                raise Exception(f"Authentication failed: {response.text}")

    async def ensure_authenticated(self):
        """Ensure we have a valid session token"""
        if not self.session_token or datetime.now() >= self.token_expires:
            await self.authenticate()

    async def get_price_history(
        self,
        epic: str,
        resolution: str = "MINUTE",
        max_points: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            epic: Market epic (e.g., 'US100')
            resolution: MINUTE, HOUR, DAY, WEEK
            max_points: Maximum number of data points

        Returns:
            DataFrame with OHLCV data
        """
        await self.ensure_authenticated()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/prices/{epic}",
                params={
                    "resolution": resolution,
                    "max": max_points
                },
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token
                }
            )

            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])

                if not prices:
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(prices)
                df['timestamp'] = pd.to_datetime(df['snapshotTime'])
                df = df.rename(columns={
                    'openPrice': 'open',
                    'closePrice': 'close',
                    'highPrice': 'high',
                    'lowPrice': 'low',
                    'lastTradedVolume': 'volume'
                })

                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.sort_values('timestamp')
                return df
            else:
                print(f"Error fetching price history: {response.text}")
                return pd.DataFrame()

    async def save_market_data(self, epic: str, df: pd.DataFrame, resolution: str):
        """
        Save market data to database.

        Args:
            epic: Market epic
            df: DataFrame with OHLCV data
            resolution: Time resolution
        """
        with get_db() as db:
            for _, row in df.iterrows():
                # Check if data already exists
                existing = db.query(MarketData).filter(
                    MarketData.epic == epic,
                    MarketData.timestamp == row['timestamp'],
                    MarketData.resolution == resolution
                ).first()

                if not existing:
                    market_data = MarketData(
                        epic=epic,
                        timestamp=row['timestamp'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']) if pd.notna(row['volume']) else None,
                        resolution=resolution
                    )
                    db.add(market_data)

            db.commit()
            print(f"✓ Saved {len(df)} {resolution} candles for {epic}")

    async def collect_and_store(
        self,
        instruments: List[str],
        resolution: str = "MINUTE",
        max_points: int = 1000
    ):
        """
        Collect and store market data for multiple instruments.

        Args:
            instruments: List of epics to collect
            resolution: Time resolution
            max_points: Maximum points per instrument
        """
        for epic in instruments:
            try:
                df = await self.get_price_history(epic, resolution, max_points)
                if not df.empty:
                    await self.save_market_data(epic, df, resolution)
            except Exception as e:
                print(f"Error collecting data for {epic}: {e}")


class NewsDataCollector:
    """
    Collects news data and performs sentiment analysis.
    Integrates with existing Perplexity API.
    """

    def __init__(self, perplexity_api_key: Optional[str] = None):
        """
        Initialize news collector.

        Args:
            perplexity_api_key: Perplexity API key
        """
        self.perplexity_api_key = perplexity_api_key or os.getenv('PERPLEXITY_API_KEY')
        self.sentiment_analyzer = SentimentAnalyzer()

    async def fetch_news(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Fetch news using Perplexity API.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of news articles
        """
        if not self.perplexity_api_key:
            print("Warning: Perplexity API key not configured")
            return []

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-large-128k-online",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a financial news aggregator. Provide concise summaries of recent news."
                            },
                            {
                                "role": "user",
                                "content": f"Find the latest news about {query}. Include dates and sources."
                            }
                        ]
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']

                    # Parse news from response
                    # This is a simple parser - could be enhanced
                    news_items = []
                    lines = content.split('\n')

                    for line in lines:
                        if line.strip():
                            news_items.append({
                                'text': line.strip(),
                                'source': 'perplexity',
                                'timestamp': datetime.now()
                            })

                    return news_items[:max_results]

            except Exception as e:
                print(f"Error fetching news: {e}")
                return []

    async def analyze_and_store_news(
        self,
        epic: str,
        query: str,
        max_results: int = 10
    ):
        """
        Fetch news, analyze sentiment, and store in database.

        Args:
            epic: Market epic
            query: News search query
            max_results: Maximum results to fetch
        """
        # Fetch news
        news_items = await self.fetch_news(query, max_results)

        if not news_items:
            return

        # Analyze sentiment and store
        with get_db() as db:
            for item in news_items:
                # Analyze sentiment
                sentiment = self.sentiment_analyzer.analyze_text(item['text'])

                # Store in database
                sentiment_data = SentimentData(
                    epic=epic,
                    timestamp=item.get('timestamp', datetime.now()),
                    source=item.get('source', 'unknown'),
                    text=item['text'],
                    sentiment_score=sentiment['score'],
                    confidence=sentiment['confidence'],
                    keywords=[],  # Could extract keywords here
                    url=item.get('url')
                )

                db.add(sentiment_data)

            db.commit()
            print(f"✓ Analyzed and stored {len(news_items)} news items for {epic}")


class DataCollectionPipeline:
    """
    Main data collection pipeline that coordinates market and sentiment data.
    """

    def __init__(
        self,
        capital_email: str,
        capital_api_key: str,
        capital_password: str,
        perplexity_api_key: Optional[str] = None,
        demo: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            capital_email: Capital.com email
            capital_api_key: Capital.com API key
            capital_password: Capital.com password
            perplexity_api_key: Perplexity API key
            demo: Use demo account
        """
        self.market_collector = CapitalDataCollector(
            capital_email,
            capital_api_key,
            capital_password,
            demo
        )
        self.news_collector = NewsDataCollector(perplexity_api_key)

    async def run_collection_cycle(
        self,
        instruments: List[str],
        market_resolution: str = "MINUTE",
        max_points: int = 1000,
        collect_news: bool = True
    ):
        """
        Run a complete data collection cycle.

        Args:
            instruments: List of instruments to collect
            market_resolution: Market data resolution
            max_points: Maximum market data points
            collect_news: Whether to collect news data
        """
        print(f"\n{'='*60}")
        print(f"Data Collection Cycle - {datetime.now()}")
        print(f"{'='*60}\n")

        # Collect market data
        print("Collecting market data...")
        await self.market_collector.collect_and_store(
            instruments,
            resolution=market_resolution,
            max_points=max_points
        )

        # Collect news and sentiment
        if collect_news:
            print("\nCollecting news and sentiment data...")
            for epic in instruments:
                # Create search query from epic
                query = f"{epic} market analysis trading"
                await self.news_collector.analyze_and_store_news(
                    epic,
                    query,
                    max_results=5
                )

        print(f"\n{'='*60}")
        print("Collection cycle completed!")
        print(f"{'='*60}\n")

    async def run_continuous(
        self,
        instruments: List[str],
        interval_minutes: int = 60,
        market_resolution: str = "MINUTE"
    ):
        """
        Run continuous data collection.

        Args:
            instruments: List of instruments
            interval_minutes: Collection interval in minutes
            market_resolution: Market data resolution
        """
        print(f"Starting continuous data collection (interval: {interval_minutes}m)")

        while True:
            try:
                await self.run_collection_cycle(
                    instruments,
                    market_resolution=market_resolution
                )

                print(f"Sleeping for {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                print("\nStopping data collection...")
                break
            except Exception as e:
                print(f"Error in collection cycle: {e}")
                print("Retrying in 5 minutes...")
                await asyncio.sleep(300)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Configuration
    INSTRUMENTS = ["US100", "EURUSD", "GBPUSD", "US500", "GOLD"]

    # Initialize pipeline
    pipeline = DataCollectionPipeline(
        capital_email=os.getenv('CAPITAL_EMAIL'),
        capital_api_key=os.getenv('CAPITAL_API_KEY'),
        capital_password=os.getenv('CAPITAL_PASSWORD'),
        perplexity_api_key=os.getenv('PERPLEXITY_API_KEY'),
        demo=True
    )

    # Run collection
    asyncio.run(pipeline.run_collection_cycle(INSTRUMENTS))
