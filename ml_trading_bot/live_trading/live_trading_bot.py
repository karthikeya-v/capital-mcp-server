"""
Live Trading Bot that uses ML models to execute real trades.
Continuously monitors markets, makes predictions, and executes trades on demo account.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.live_trading.trade_executor import CapitalTradeExecutor, PositionManager
from ml_trading_bot.database import get_db, Model, MarketData, SentimentData, Feature
from ml_trading_bot.features import FeatureEngineeringPipeline
from ml_trading_bot.feedback import FeedbackLoop
from ml_trading_bot.models.ml_models import RandomForestTrader, XGBoostTrader, LightGBMTrader

load_dotenv()


class LiveTradingBot:
    """
    Live trading bot that executes real trades based on ML models.
    """

    def __init__(
        self,
        instruments: List[str],
        model_ids: Optional[List[int]] = None,
        min_confidence: float = 0.65,
        max_positions: int = 3,
        risk_per_trade_percent: float = 1.0,
        max_daily_loss: float = 500.0,
        check_interval_seconds: int = 60,
        demo: bool = True
    ):
        """
        Initialize live trading bot.

        Args:
            instruments: List of instruments to trade
            model_ids: List of model IDs to use (None = use all active)
            min_confidence: Minimum confidence threshold
            max_positions: Maximum concurrent positions
            risk_per_trade_percent: Risk per trade (% of capital)
            max_daily_loss: Maximum daily loss limit
            check_interval_seconds: Seconds between checks
            demo: Use demo account
        """
        self.instruments = instruments
        self.model_ids = model_ids
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.risk_per_trade_percent = risk_per_trade_percent
        self.max_daily_loss = max_daily_loss
        self.check_interval_seconds = check_interval_seconds
        self.demo = demo

        # Initialize executor
        self.executor = CapitalTradeExecutor(
            email=os.getenv('CAPITAL_EMAIL'),
            api_key=os.getenv('CAPITAL_API_KEY'),
            api_password=os.getenv('CAPITAL_PASSWORD'),
            demo=demo
        )

        self.position_manager = PositionManager(self.executor)
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.feedback_loop = FeedbackLoop()

        # Track daily P&L
        self.daily_pnl = 0.0
        self.current_day = datetime.now().date()

        # Load models
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load active ML models"""
        with get_db() as db:
            if self.model_ids:
                models = db.query(Model).filter(
                    Model.id.in_(self.model_ids),
                    Model.is_active == True
                ).all()
            else:
                models = db.query(Model).filter(
                    Model.is_active == True
                ).all()

            print(f"\nLoading {len(models)} models...")

            for model_obj in models:
                try:
                    # Extract epic from model name
                    epic = model_obj.name.split('_')[0]

                    # Load model based on algorithm
                    if model_obj.algorithm == 'RandomForest':
                        model = RandomForestTrader()
                    elif model_obj.algorithm == 'XGBoost':
                        model = XGBoostTrader()
                    elif model_obj.algorithm == 'LightGBM':
                        model = LightGBMTrader()
                    else:
                        print(f"Unsupported algorithm: {model_obj.algorithm}")
                        continue

                    model.load_model(model_obj.model_path)

                    self.models[epic] = {
                        'model': model,
                        'model_id': model_obj.id,
                        'model_obj': model_obj
                    }

                    print(f"  ✓ Loaded {model_obj.name} for {epic}")

                except Exception as e:
                    print(f"  ✗ Failed to load {model_obj.name}: {e}")

    async def get_current_features(self, epic: str) -> Optional[pd.DataFrame]:
        """
        Get current features for an instrument.

        Args:
            epic: Market epic

        Returns:
            DataFrame with features or None
        """
        # Get recent data
        with get_db() as db:
            cutoff = datetime.now() - timedelta(hours=24)

            # Get market data
            market_data = db.query(MarketData).filter(
                MarketData.epic == epic,
                MarketData.timestamp >= cutoff
            ).order_by(MarketData.timestamp.desc()).limit(200).all()

            if not market_data:
                return None

            # Create DataFrame
            df = pd.DataFrame([{
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            } for md in reversed(market_data)])

            # Engineer features
            from ml_trading_bot.features.feature_engineering import TechnicalFeatureEngineer

            engineer = TechnicalFeatureEngineer()
            df = engineer.calculate_technical_indicators(df)
            df = engineer.calculate_price_patterns(df)
            df = engineer.calculate_cross_features(df)

            # Get sentiment
            sentiment_data = db.query(SentimentData).filter(
                SentimentData.epic == epic,
                SentimentData.timestamp >= cutoff
            ).all()

            if sentiment_data:
                from ml_trading_bot.features.feature_engineering import SentimentFeatureEngineer

                sent_engineer = SentimentFeatureEngineer()
                sentiments = [{
                    'sentiment_score': sd.sentiment_score,
                    'confidence': sd.confidence,
                    'timestamp': sd.timestamp
                } for sd in sentiment_data]

                sent_features = sent_engineer.aggregate_sentiment_features(sentiments)

                # Add to last row
                for key, value in sent_features.items():
                    df.loc[df.index[-1], key] = value

            return df.tail(1)

    def make_prediction(self, epic: str, features: pd.DataFrame) -> Optional[Dict]:
        """
        Make prediction using loaded model.

        Args:
            epic: Market epic
            features: Feature DataFrame

        Returns:
            Prediction dictionary or None
        """
        if epic not in self.models:
            return None

        model_info = self.models[epic]
        model = model_info['model']

        try:
            # Select features that model expects
            X = features[model.feature_names] if hasattr(model, 'feature_names') else features

            # Preprocess
            X_scaled, _ = model.preprocess_data(X, fit_scaler=False)

            # Predict
            prediction = model.predict(X_scaled)[0]

            # Get confidence if available
            if hasattr(model.model, 'predict_proba'):
                proba = model.model.predict_proba(X_scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.7  # Default

            # Decode prediction (0=DOWN, 1=NEUTRAL, 2=UP)
            if prediction == 2:
                signal = 'BUY'
            elif prediction == 0:
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'

            return {
                'signal': signal,
                'confidence': confidence,
                'model_id': model_info['model_id'],
                'prediction': int(prediction)
            }

        except Exception as e:
            print(f"Error making prediction for {epic}: {e}")
            return None

    async def calculate_trade_parameters(
        self,
        epic: str,
        direction: str,
        entry_price: float
    ) -> Dict:
        """
        Calculate position size, stop loss, and take profit.

        Args:
            epic: Market epic
            direction: 'BUY' or 'SELL'
            entry_price: Entry price

        Returns:
            Dict with size, stop_loss, take_profit
        """
        # Get account info
        account = await self.executor.get_account_info()
        balance = account.get('balance', 10000)

        # Get ATR for volatility-based stops
        with get_db() as db:
            recent_features = db.query(Feature).filter(
                Feature.epic == epic
            ).order_by(Feature.timestamp.desc()).first()

            atr = recent_features.atr_14 if recent_features and recent_features.atr_14 else entry_price * 0.02

        # Calculate stop loss and take profit
        if direction == 'BUY':
            stop_loss = entry_price - (2 * atr)
            take_profit = entry_price + (3 * atr)
        else:
            stop_loss = entry_price + (2 * atr)
            take_profit = entry_price - (3 * atr)

        # Calculate position size based on risk
        risk_amount = balance * (self.risk_per_trade_percent / 100)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit > 0:
            size = risk_amount / risk_per_unit
            size = max(0.1, min(size, 2.0))  # Between 0.1 and 2.0
        else:
            size = 0.1

        return {
            'size': round(size, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_amount': risk_amount
        }

    async def execute_trade_signal(self, epic: str, signal: Dict) -> bool:
        """
        Execute a trade based on signal.

        Args:
            epic: Market epic
            signal: Signal dictionary

        Returns:
            True if trade executed
        """
        if signal['signal'] == 'NEUTRAL':
            return False

        if signal['confidence'] < self.min_confidence:
            print(f"  Confidence {signal['confidence']:.2f} < {self.min_confidence}, skipping")
            return False

        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"  Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False

        # Check max positions
        positions = await self.executor.get_positions()
        if len(positions) >= self.max_positions:
            print(f"  Max positions reached: {len(positions)}/{self.max_positions}")
            return False

        # Get market info
        market = await self.executor.get_market_info(epic)
        if not market:
            return False

        snapshot = market.get('snapshot', {})
        current_price = snapshot.get('bid') if signal['signal'] == 'SELL' else snapshot.get('offer')

        if not current_price:
            return False

        # Calculate trade parameters
        params = await self.calculate_trade_parameters(
            epic,
            signal['signal'],
            current_price
        )

        # Execute trade
        print(f"\n{'='*60}")
        print(f"EXECUTING TRADE: {epic}")
        print(f"Signal: {signal['signal']}, Confidence: {signal['confidence']:.2%}")
        print(f"Entry: {current_price}, Size: {params['size']}")
        print(f"SL: {params['stop_loss']}, TP: {params['take_profit']}")
        print(f"Risk: ${params['risk_amount']:.2f}")
        print(f"{'='*60}\n")

        # Place trade
        result = await self.executor.place_trade(
            epic=epic,
            direction=signal['signal'],
            size=params['size'],
            stop_loss=params['stop_loss'],
            take_profit=params['take_profit']
        )

        if result:
            # Save to database
            trade_db_id = self.executor.save_trade_to_db(
                trade_data={
                    'dealId': result.get('dealReference'),
                    'epic': epic,
                    'direction': signal['signal'],
                    'size': params['size'],
                    'level': current_price,
                    'stopLevel': params['stop_loss'],
                    'profitLevel': params['take_profit']
                },
                model_id=signal['model_id'],
                confidence=signal['confidence'],
                is_simulated=False
            )

            # Register for monitoring
            self.position_manager.register_position(
                result.get('dealReference'),
                trade_db_id
            )

            return True

        return False

    async def scan_and_trade(self):
        """
        Scan all instruments and execute trades if signals found.
        """
        print(f"\n{'='*80}")
        print(f"LIVE TRADING SCAN - {datetime.now()}")
        print(f"{'='*80}\n")

        for epic in self.instruments:
            if epic not in self.models:
                print(f"No model for {epic}, skipping")
                continue

            print(f"\nScanning {epic}...")

            # Get current features
            features = await self.get_current_features(epic)

            if features is None or features.empty:
                print(f"  No features available for {epic}")
                continue

            # Make prediction
            signal = self.make_prediction(epic, features)

            if signal:
                print(f"  Signal: {signal['signal']}, Confidence: {signal['confidence']:.2%}")

                if signal['signal'] != 'NEUTRAL':
                    # Execute trade
                    executed = await self.execute_trade_signal(epic, signal)

                    if executed:
                        print(f"  ✓ Trade executed for {epic}")
                    else:
                        print(f"  ✗ Trade not executed")
            else:
                print(f"  No prediction available")

    async def monitor_and_update(self):
        """
        Monitor positions and update database.
        """
        # Check for new day
        if datetime.now().date() != self.current_day:
            print(f"\nNew day! Previous P&L: ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.current_day = datetime.now().date()

        # Monitor positions
        positions = await self.position_manager.monitor_positions()

        # Check and close if needed
        await self.position_manager.check_and_close_positions(max_holding_hours=24)

        # Calculate daily P&L from DB
        with get_db() as db:
            today_start = datetime.combine(self.current_day, datetime.min.time())

            from ml_trading_bot.database import Trade
            today_trades = db.query(Trade).filter(
                Trade.exit_time >= today_start,
                Trade.is_simulated == False
            ).all()

            self.daily_pnl = sum(t.pnl for t in today_trades if t.pnl)

        print(f"\nOpen Positions: {len(positions)}")
        print(f"Daily P&L: ${self.daily_pnl:.2f}")

    async def process_feedback(self):
        """
        Process feedback for closed trades.
        """
        with get_db() as db:
            from ml_trading_bot.database import Trade, TradeFeedback

            # Get closed trades without feedback
            closed_trades = db.query(Trade).filter(
                Trade.exit_time != None,
                Trade.is_simulated == False
            ).all()

            for trade in closed_trades:
                # Check if feedback exists
                existing = db.query(TradeFeedback).filter(
                    TradeFeedback.trade_id == trade.id
                ).first()

                if not existing:
                    # Process feedback
                    print(f"Processing feedback for trade {trade.trade_id}...")
                    try:
                        self.feedback_loop.process_trade_feedback(trade.id)
                    except Exception as e:
                        print(f"Error processing feedback: {e}")

    async def run_continuous(self):
        """
        Run continuous live trading loop.
        """
        print(f"\n{'='*80}")
        print("LIVE TRADING BOT STARTED")
        print(f"Mode: {'DEMO' if self.demo else 'LIVE'}")
        print(f"Instruments: {', '.join(self.instruments)}")
        print(f"Models loaded: {len(self.models)}")
        print(f"Min confidence: {self.min_confidence:.0%}")
        print(f"Check interval: {self.check_interval_seconds}s")
        print(f"{'='*80}\n")

        # Authenticate
        await self.executor.authenticate()

        cycle = 0

        while True:
            try:
                cycle += 1
                print(f"\n{'#'*80}")
                print(f"CYCLE {cycle}")
                print(f"{'#'*80}")

                # Monitor existing positions
                await self.monitor_and_update()

                # Scan for new trades
                await self.scan_and_trade()

                # Process feedback (every 10 cycles)
                if cycle % 10 == 0:
                    await self.process_feedback()

                # Sleep
                print(f"\nSleeping for {self.check_interval_seconds}s...")
                await asyncio.sleep(self.check_interval_seconds)

            except KeyboardInterrupt:
                print("\n\nStopping live trading bot...")
                break

            except Exception as e:
                print(f"\nError in trading loop: {e}")
                print("Retrying in 60 seconds...")
                await asyncio.sleep(60)


# Example usage
if __name__ == "__main__":
    # Initialize bot
    bot = LiveTradingBot(
        instruments=['US100', 'EURUSD', 'GBPUSD'],
        min_confidence=0.65,
        max_positions=3,
        risk_per_trade_percent=1.0,
        max_daily_loss=500.0,
        check_interval_seconds=60,
        demo=True  # ALWAYS USE DEMO FOR TESTING
    )

    # Run
    asyncio.run(bot.run_continuous())
