"""
ML Trading Bot Orchestrator.
Main entry point that coordinates all components.
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Optional
import os
from dotenv import load_dotenv

from database import init_database, get_db, Model
from data_collection import DataCollectionPipeline
from features import FeatureEngineeringPipeline
from models import ModelTrainingPipeline
from backtesting import BacktestEngine
from feedback import FeedbackLoop
from continuous_learning import AutoRetrainer, ModelPerformanceMonitor

load_dotenv()


class MLTradingBotOrchestrator:
    """
    Main orchestrator for the ML trading bot system.
    Coordinates data collection, feature engineering, model training,
    backtesting, and continuous learning.
    """

    def __init__(
        self,
        instruments: List[str],
        demo: bool = True
    ):
        """
        Initialize orchestrator.

        Args:
            instruments: List of instruments to trade
            demo: Use demo account
        """
        self.instruments = instruments
        self.demo = demo

        # Initialize components
        self.data_pipeline = DataCollectionPipeline(
            capital_email=os.getenv('CAPITAL_EMAIL'),
            capital_api_key=os.getenv('CAPITAL_API_KEY'),
            capital_password=os.getenv('CAPITAL_PASSWORD'),
            perplexity_api_key=os.getenv('PERPLEXITY_API_KEY'),
            demo=demo
        )

        self.feature_pipeline = FeatureEngineeringPipeline()
        self.training_pipeline = ModelTrainingPipeline()
        self.backtest_engine = BacktestEngine()
        self.feedback_loop = FeedbackLoop()
        self.auto_retrainer = AutoRetrainer()
        self.performance_monitor = ModelPerformanceMonitor()

    async def initialize_system(self):
        """Initialize database and system components"""
        print("="*80)
        print("ML TRADING BOT - INITIALIZATION")
        print("="*80 + "\n")

        # Initialize database
        print("Initializing database...")
        init_database()
        print("✓ Database initialized\n")

    async def collect_historical_data(
        self,
        days: int = 90,
        resolution: str = "MINUTE"
    ):
        """
        Collect historical data for all instruments.

        Args:
            days: Days of historical data
            resolution: Data resolution
        """
        print(f"\n{'='*80}")
        print(f"COLLECTING HISTORICAL DATA ({days} days)")
        print(f"{'='*80}\n")

        await self.data_pipeline.run_collection_cycle(
            self.instruments,
            market_resolution=resolution,
            max_points=days * 1440 if resolution == "MINUTE" else days * 24,
            collect_news=True
        )

    async def engineer_features(
        self,
        days: int = 90
    ):
        """
        Engineer features for all instruments.

        Args:
            days: Days to process
        """
        print(f"\n{'='*80}")
        print(f"ENGINEERING FEATURES ({days} days)")
        print(f"{'='*80}\n")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        self.feature_pipeline.create_and_save_features(
            self.instruments,
            start_date,
            end_date
        )

    async def train_all_models(
        self,
        model_types: List[str] = ['RandomForest', 'XGBoost', 'LightGBM'],
        lookback_days: int = 90
    ):
        """
        Train models for all instruments.

        Args:
            model_types: Types of models to train
            lookback_days: Days of data to use
        """
        print(f"\n{'='*80}")
        print(f"TRAINING MODELS")
        print(f"{'='*80}\n")

        results = self.training_pipeline.batch_train_all_instruments(
            self.instruments,
            model_types=model_types,
            lookback_days=lookback_days
        )

        return results

    async def backtest_all_models(
        self,
        test_days: int = 30
    ):
        """
        Backtest all active models.

        Args:
            test_days: Days to backtest
        """
        print(f"\n{'='*80}")
        print(f"BACKTESTING MODELS ({test_days} days)")
        print(f"{'='*80}\n")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days)

        with get_db() as db:
            active_models = db.query(Model).filter(
                Model.is_active == True
            ).all()

            print(f"Found {len(active_models)} active models\n")

            for model_obj in active_models:
                try:
                    # Extract epic from model name
                    epic = model_obj.name.split('_')[0]

                    print(f"\nBacktesting {model_obj.name}...")

                    # Load model
                    from models.ml_models import RandomForestTrader, XGBoostTrader, LightGBMTrader

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

                    # Run backtest
                    metrics = self.backtest_engine.run_backtest(
                        epic,
                        model,
                        start_date,
                        end_date
                    )

                except Exception as e:
                    print(f"Error backtesting {model_obj.name}: {e}")

    async def run_complete_workflow(
        self,
        collect_data: bool = True,
        engineer_features: bool = True,
        train_models: bool = True,
        backtest: bool = True,
        days: int = 90
    ):
        """
        Run complete ML bot workflow.

        Args:
            collect_data: Whether to collect data
            engineer_features: Whether to engineer features
            train_models: Whether to train models
            backtest: Whether to run backtests
            days: Days of historical data
        """
        print("\n" + "="*80)
        print("ML TRADING BOT - COMPLETE WORKFLOW")
        print("="*80 + "\n")

        # Initialize
        await self.initialize_system()

        # Collect data
        if collect_data:
            await self.collect_historical_data(days=days)

        # Engineer features
        if engineer_features:
            await self.engineer_features(days=days)

        # Train models
        if train_models:
            await self.train_all_models(lookback_days=days)

        # Backtest
        if backtest:
            await self.backtest_all_models(test_days=min(30, days // 3))

        print("\n" + "="*80)
        print("WORKFLOW COMPLETED!")
        print("="*80 + "\n")

    async def run_continuous_learning(
        self,
        check_interval_hours: int = 24
    ):
        """
        Run continuous learning system.

        Args:
            check_interval_hours: Hours between checks
        """
        print("\n" + "="*80)
        print("STARTING CONTINUOUS LEARNING SYSTEM")
        print("="*80 + "\n")

        # Start continuous data collection in background
        data_collection_task = asyncio.create_task(
            self.data_pipeline.run_continuous(
                self.instruments,
                interval_minutes=60
            )
        )

        # Run auto-retraining
        try:
            self.auto_retrainer.run_continuous(
                check_interval_hours=check_interval_hours
            )
        except KeyboardInterrupt:
            print("\nStopping continuous learning...")
            data_collection_task.cancel()

    def generate_system_report(self) -> dict:
        """Generate comprehensive system status report"""
        print("\n" + "="*80)
        print("SYSTEM STATUS REPORT")
        print("="*80 + "\n")

        with get_db() as db:
            # Count data
            from database import MarketData, SentimentData, Feature, Trade, Model

            market_data_count = db.query(MarketData).count()
            sentiment_data_count = db.query(SentimentData).count()
            feature_count = db.query(Feature).count()
            trade_count = db.query(Trade).count()
            model_count = db.query(Model).count()
            active_models = db.query(Model).filter(Model.is_active == True).count()

            print(f"Database Statistics:")
            print(f"  Market Data Points: {market_data_count:,}")
            print(f"  Sentiment Records: {sentiment_data_count:,}")
            print(f"  Feature Rows: {feature_count:,}")
            print(f"  Trades: {trade_count:,}")
            print(f"  Total Models: {model_count}")
            print(f"  Active Models: {active_models}\n")

            # Model performance
            print("Model Performance:")
            models = db.query(Model).filter(Model.is_active == True).all()

            for model in models:
                print(f"\n  {model.name} (v{model.version}):")
                print(f"    Algorithm: {model.algorithm}")
                print(f"    Accuracy: {model.accuracy*100:.2f}%" if model.accuracy else "    Accuracy: N/A")
                print(f"    F1 Score: {model.f1_score:.3f}" if model.f1_score else "    F1 Score: N/A")
                print(f"    Trained: {model.trained_at}")

            print("\n" + "="*80 + "\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ML Trading Bot')
    parser.add_argument(
        '--mode',
        choices=['init', 'train', 'backtest', 'continuous', 'report', 'full'],
        default='full',
        help='Operation mode'
    )
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=['US100', 'EURUSD', 'GBPUSD', 'US500', 'GOLD'],
        help='Instruments to trade'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Days of historical data'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        default=True,
        help='Use demo account'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    bot = MLTradingBotOrchestrator(
        instruments=args.instruments,
        demo=args.demo
    )

    # Run based on mode
    if args.mode == 'init':
        await bot.initialize_system()

    elif args.mode == 'train':
        await bot.initialize_system()
        await bot.collect_historical_data(days=args.days)
        await bot.engineer_features(days=args.days)
        await bot.train_all_models(lookback_days=args.days)

    elif args.mode == 'backtest':
        await bot.backtest_all_models(test_days=min(30, args.days // 3))

    elif args.mode == 'continuous':
        await bot.initialize_system()
        await bot.run_continuous_learning(check_interval_hours=24)

    elif args.mode == 'report':
        bot.generate_system_report()

    elif args.mode == 'full':
        await bot.run_complete_workflow(
            collect_data=True,
            engineer_features=True,
            train_models=True,
            backtest=True,
            days=args.days
        )


if __name__ == "__main__":
    asyncio.run(main())
