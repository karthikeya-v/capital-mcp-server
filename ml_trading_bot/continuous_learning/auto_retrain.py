"""
Continuous Learning System.
Automatically retrains models based on performance degradation and new data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import (
    Model, ModelPerformanceLog, Trade, Feature,
    get_db
)
from ml_trading_bot.models import ModelTrainingPipeline
from ml_trading_bot.feedback import FeedbackLoop
from ml_trading_bot.features import FeatureEngineeringPipeline


class ModelPerformanceMonitor:
    """
    Monitors model performance and triggers retraining when needed.
    """

    def __init__(
        self,
        performance_threshold: float = 0.5,  # Min win rate
        min_trades_for_eval: int = 20,
        evaluation_window_days: int = 7
    ):
        """
        Initialize performance monitor.

        Args:
            performance_threshold: Minimum acceptable win rate
            min_trades_for_eval: Minimum trades needed for evaluation
            evaluation_window_days: Window for performance evaluation
        """
        self.performance_threshold = performance_threshold
        self.min_trades_for_eval = min_trades_for_eval
        self.evaluation_window_days = evaluation_window_days
        self.feedback_loop = FeedbackLoop()

    def check_model_health(self, model_id: int) -> Dict:
        """
        Check if model needs retraining.

        Args:
            model_id: Model database ID

        Returns:
            Health status dictionary
        """
        with get_db() as db:
            model = db.query(Model).filter(Model.id == model_id).first()

            if not model:
                return {'status': 'ERROR', 'message': 'Model not found'}

            # Get recent trades
            cutoff_date = datetime.now() - timedelta(days=self.evaluation_window_days)

            trades = db.query(Trade).filter(
                Trade.model_id == model_id,
                Trade.entry_time >= cutoff_date
            ).all()

            if len(trades) < self.min_trades_for_eval:
                return {
                    'status': 'INSUFFICIENT_DATA',
                    'message': f'Only {len(trades)} trades, need {self.min_trades_for_eval}',
                    'needs_retraining': False
                }

            # Calculate performance
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = wins / len(trades)

            # Check performance degradation
            needs_retraining = win_rate < self.performance_threshold

            # Check model age
            days_since_training = (datetime.now() - model.trained_at).days
            is_old = days_since_training > 30  # Retrain if older than 30 days

            # Get performance trend
            perf_logs = db.query(ModelPerformanceLog).filter(
                ModelPerformanceLog.model_id == model_id
            ).order_by(ModelPerformanceLog.timestamp.desc()).limit(5).all()

            if len(perf_logs) >= 3:
                recent_win_rates = [log.win_rate for log in perf_logs]
                trend = np.mean(np.diff(recent_win_rates))
                is_declining = trend < -5  # Declining by more than 5% avg
            else:
                is_declining = False

            return {
                'status': 'HEALTHY' if not needs_retraining else 'DEGRADED',
                'model_id': model_id,
                'model_name': model.name,
                'trades_evaluated': len(trades),
                'win_rate': win_rate * 100,
                'threshold': self.performance_threshold * 100,
                'days_since_training': days_since_training,
                'needs_retraining': needs_retraining or is_old or is_declining,
                'reasons': [
                    'Low win rate' if needs_retraining else None,
                    'Model is old' if is_old else None,
                    'Performance declining' if is_declining else None
                ],
                'recommendation': 'RETRAIN' if (needs_retraining or is_old or is_declining) else 'CONTINUE'
            }

    def monitor_all_active_models(self) -> List[Dict]:
        """
        Monitor all active models.

        Returns:
            List of health status dictionaries
        """
        with get_db() as db:
            active_models = db.query(Model).filter(
                Model.is_active == True
            ).all()

            results = []

            for model in active_models:
                health = self.check_model_health(model.id)
                results.append(health)

            return results


class AutoRetrainer:
    """
    Automatically retrains models based on performance and new data.
    """

    def __init__(self):
        self.monitor = ModelPerformanceMonitor()
        self.training_pipeline = ModelTrainingPipeline()
        self.feature_pipeline = FeatureEngineeringPipeline()

    def retrain_model(
        self,
        model_id: int,
        incremental: bool = True,
        lookback_days: int = 90
    ) -> Optional[int]:
        """
        Retrain a model with latest data.

        Args:
            model_id: Model to retrain
            incremental: Use incremental learning (append new data)
            lookback_days: Days of data to use

        Returns:
            New model ID if successful
        """
        with get_db() as db:
            # Get original model
            original_model = db.query(Model).filter(Model.id == model_id).first()

            if not original_model:
                print(f"Model {model_id} not found")
                return None

            print(f"\n{'='*60}")
            print(f"RETRAINING MODEL: {original_model.name}")
            print(f"Original version: {original_model.version}")
            print(f"Algorithm: {original_model.algorithm}")
            print(f"{'='*60}\n")

            # Get model info
            epic = original_model.name.split('_')[0]  # Extract epic from name
            model_type = original_model.algorithm

            # Determine training period
            end_date = datetime.now()

            if incremental and original_model.training_period_end:
                # Only new data since last training
                start_date = original_model.training_period_end
                print(f"Using incremental training from {start_date}")
            else:
                # Full retraining
                start_date = end_date - timedelta(days=lookback_days)
                print(f"Using full retraining from {start_date}")

            # Ensure we have fresh features
            print("\nUpdating features...")
            try:
                self.feature_pipeline.create_and_save_features(
                    [epic],
                    start_date,
                    end_date
                )
            except Exception as e:
                print(f"Warning: Could not update features: {e}")

            # Get feedback and recommendations
            print("\nAnalyzing feedback...")
            recommendations = self.monitor.feedback_loop.identify_improvement_areas(model_id)

            if recommendations:
                print("\nImprovement Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['area']}: {rec['suggestion']}")

            # Train new model
            try:
                print("\nTraining new model version...")

                new_model_id = self.training_pipeline.train_and_save(
                    model_type=model_type,
                    epic=epic,
                    lookback_days=lookback_days,
                    hyperparameter_tuning=True  # Always tune on retraining
                )

                # Deactivate old model
                original_model.is_active = False
                db.commit()

                print(f"\n✓ Model retrained successfully!")
                print(f"  Old model ID: {model_id} (deactivated)")
                print(f"  New model ID: {new_model_id} (active)")

                return new_model_id

            except Exception as e:
                print(f"\n✗ Retraining failed: {e}")
                return None

    def auto_retrain_cycle(self):
        """
        Run automatic retraining cycle for all models that need it.
        """
        print(f"\n{'='*80}")
        print(f"AUTO-RETRAIN CYCLE - {datetime.now()}")
        print(f"{'='*80}\n")

        # Monitor all models
        print("Monitoring model health...")
        health_reports = self.monitor.monitor_all_active_models()

        models_to_retrain = [
            h for h in health_reports
            if h.get('needs_retraining', False)
        ]

        print(f"\nFound {len(models_to_retrain)} models needing retraining")

        if not models_to_retrain:
            print("All models are healthy!")
            return

        # Retrain models
        results = []

        for health in models_to_retrain:
            model_id = health['model_id']
            print(f"\n{'-'*60}")
            print(f"Model: {health['model_name']}")
            print(f"Win Rate: {health['win_rate']:.1f}%")
            print(f"Reasons: {', '.join([r for r in health['reasons'] if r])}")

            new_model_id = self.retrain_model(model_id)

            results.append({
                'original_id': model_id,
                'new_id': new_model_id,
                'status': 'SUCCESS' if new_model_id else 'FAILED'
            })

        # Summary
        print(f"\n{'='*80}")
        print("AUTO-RETRAIN SUMMARY")
        print(f"{'='*80}")
        print(f"Models evaluated: {len(health_reports)}")
        print(f"Models retrained: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
        print(f"Failed retrains: {sum(1 for r in results if r['status'] == 'FAILED')}")
        print(f"{'='*80}\n")

        return results

    def run_continuous(self, check_interval_hours: int = 24):
        """
        Run continuous monitoring and retraining.

        Args:
            check_interval_hours: Hours between checks
        """
        print(f"Starting continuous learning system...")
        print(f"Check interval: {check_interval_hours} hours\n")

        while True:
            try:
                self.auto_retrain_cycle()

                print(f"\nSleeping for {check_interval_hours} hours...")
                time.sleep(check_interval_hours * 3600)

            except KeyboardInterrupt:
                print("\nStopping continuous learning...")
                break
            except Exception as e:
                print(f"\nError in continuous learning cycle: {e}")
                print("Retrying in 1 hour...")
                time.sleep(3600)


class DataDriftDetector:
    """
    Detects data drift that might require model retraining.
    """

    def __init__(self):
        pass

    def detect_feature_drift(
        self,
        epic: str,
        window_days: int = 30
    ) -> Dict:
        """
        Detect drift in feature distributions.

        Args:
            epic: Market epic
            window_days: Window to compare

        Returns:
            Drift analysis
        """
        with get_db() as db:
            # Get recent features
            recent_start = datetime.now() - timedelta(days=window_days)
            recent_features = db.query(Feature).filter(
                Feature.epic == epic,
                Feature.timestamp >= recent_start
            ).all()

            # Get historical features (previous window)
            historical_start = recent_start - timedelta(days=window_days)
            historical_end = recent_start
            historical_features = db.query(Feature).filter(
                Feature.epic == epic,
                Feature.timestamp >= historical_start,
                Feature.timestamp < historical_end
            ).all()

            if not recent_features or not historical_features:
                return {'drift_detected': False, 'message': 'Insufficient data'}

            # Convert to DataFrames
            recent_df = pd.DataFrame([{
                'rsi_14': f.rsi_14,
                'volatility_10': f.volatility_10,
                'sentiment_score_24h': f.sentiment_score_24h or 0
            } for f in recent_features])

            historical_df = pd.DataFrame([{
                'rsi_14': f.rsi_14,
                'volatility_10': f.volatility_10,
                'sentiment_score_24h': f.sentiment_score_24h or 0
            } for f in historical_features])

            # Calculate drift for each feature
            drift_scores = {}

            for col in recent_df.columns:
                recent_mean = recent_df[col].mean()
                historical_mean = historical_df[col].mean()

                recent_std = recent_df[col].std()
                historical_std = historical_df[col].std()

                # Normalized difference
                if historical_std > 0:
                    mean_drift = abs(recent_mean - historical_mean) / historical_std
                else:
                    mean_drift = 0

                drift_scores[col] = mean_drift

            # Overall drift (average of individual drifts)
            avg_drift = np.mean(list(drift_scores.values()))

            # Threshold for significant drift
            drift_detected = avg_drift > 2.0  # More than 2 std deviations

            return {
                'drift_detected': drift_detected,
                'average_drift': avg_drift,
                'feature_drifts': drift_scores,
                'recommendation': 'RETRAIN' if drift_detected else 'CONTINUE'
            }


# Example usage
if __name__ == "__main__":
    # Initialize auto-retrainer
    retrainer = AutoRetrainer()

    # Run single cycle
    retrainer.auto_retrain_cycle()

    # Or run continuous
    # retrainer.run_continuous(check_interval_hours=24)
