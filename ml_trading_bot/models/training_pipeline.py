"""
ML Model Training Pipeline.
Orchestrates data loading, model training, evaluation, and persistence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import json
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import (
    Feature, Model, TrainingJob, get_db
)
from ml_trading_bot.models.ml_models import (
    RandomForestTrader, XGBoostTrader, LightGBMTrader,
    LSTMTrader, EnsembleTrader
)


class ModelTrainingPipeline:
    """
    Comprehensive training pipeline for ML trading models.
    """

    def __init__(self, models_dir: str = "./trained_models"):
        """
        Initialize training pipeline.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def load_training_data(
        self,
        epic: str,
        start_date: datetime,
        end_date: datetime,
        target_horizon: str = '4h'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from database.

        Args:
            epic: Market epic
            start_date: Start date
            end_date: End date
            target_horizon: Target prediction horizon

        Returns:
            Tuple of (features_df, target_series)
        """
        with get_db() as db:
            # Query features
            features = db.query(Feature).filter(
                Feature.epic == epic,
                Feature.timestamp >= start_date,
                Feature.timestamp <= end_date
            ).order_by(Feature.timestamp).all()

            if not features:
                raise ValueError(f"No features found for {epic} in date range")

            # Convert to DataFrame
            data = []
            for f in features:
                row = {
                    'timestamp': f.timestamp,
                    # Technical indicators
                    'rsi_14': f.rsi_14,
                    'macd': f.macd,
                    'macd_signal': f.macd_signal,
                    'macd_hist': f.macd_hist,
                    'bb_upper': f.bb_upper,
                    'bb_middle': f.bb_middle,
                    'bb_lower': f.bb_lower,
                    'bb_width': f.bb_width,
                    'ma_20': f.ma_20,
                    'ma_50': f.ma_50,
                    'ma_100': f.ma_100,
                    'atr_14': f.atr_14,
                    'adx_14': f.adx_14,
                    'cci_20': f.cci_20,
                    'stochastic_k': f.stochastic_k,
                    'stochastic_d': f.stochastic_d,
                    # Price patterns
                    'support_level': f.support_level,
                    'resistance_level': f.resistance_level,
                    'momentum_10': f.momentum_10,
                    'momentum_20': f.momentum_20,
                    'volatility_10': f.volatility_10,
                    'volatility_20': f.volatility_20,
                    # Sentiment features
                    'sentiment_score_1h': f.sentiment_score_1h,
                    'sentiment_score_4h': f.sentiment_score_4h,
                    'sentiment_score_24h': f.sentiment_score_24h,
                    'sentiment_volume_1h': f.sentiment_volume_1h,
                    'sentiment_volume_4h': f.sentiment_volume_4h,
                    'sentiment_volume_24h': f.sentiment_volume_24h,
                    # Target
                    'target_direction': f.target_direction
                }
                data.append(row)

            df = pd.DataFrame(data)
            df = df.set_index('timestamp')

            # Extract features and target
            X = df.drop('target_direction', axis=1)
            y = df['target_direction']

            return X, y

    def train_model(
        self,
        model_type: str,
        epic: str,
        start_date: datetime,
        end_date: datetime,
        hyperparameter_tuning: bool = False,
        **kwargs
    ) -> Tuple[any, Dict]:
        """
        Train a single model.

        Args:
            model_type: 'RandomForest', 'XGBoost', 'LightGBM', 'LSTM'
            epic: Market epic
            start_date: Training start date
            end_date: Training end date
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            **kwargs: Additional model parameters

        Returns:
            Tuple of (trained_model, metrics)
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type} for {epic}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*60}\n")

        # Load data
        print("Loading training data...")
        X, y = self.load_training_data(epic, start_date, end_date)
        print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")

        # Initialize model
        if model_type == 'RandomForest':
            model = RandomForestTrader(**kwargs)
        elif model_type == 'XGBoost':
            model = XGBoostTrader(**kwargs)
        elif model_type == 'LightGBM':
            model = LightGBMTrader(**kwargs)
        elif model_type == 'LSTM':
            model = LSTMTrader(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Preprocess data
        print("Preprocessing data...")
        X_scaled, y_encoded = model.preprocess_data(X, y, fit_scaler=True)

        # Split data (time-series aware)
        X_train, X_test, y_train, y_test = model.split_data(X_scaled, y_encoded)
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

        # Train model
        if hyperparameter_tuning and hasattr(model, 'hyperparameter_tuning'):
            print("\nPerforming hyperparameter tuning...")
            tuning_results = model.hyperparameter_tuning(X_train, y_train)
            print(f"✓ Best parameters: {tuning_results['best_params']}")
        else:
            print("\nTraining model...")
            model.train(X_train, y_train)

        # Evaluate model
        print("\nEvaluating model...")
        metrics = model.evaluate(X_test, y_test)

        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Feature importance
        importance = model.get_feature_importance()
        if importance:
            print("\nTop 10 Important Features:")
            sorted_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for feat, imp in sorted_features:
                print(f"  {feat}: {imp:.4f}")

        return model, metrics

    def train_ensemble(
        self,
        epic: str,
        start_date: datetime,
        end_date: datetime,
        model_types: List[str] = ['RandomForest', 'XGBoost', 'LightGBM']
    ) -> Tuple[EnsembleTrader, Dict]:
        """
        Train an ensemble of models.

        Args:
            epic: Market epic
            start_date: Start date
            end_date: End date
            model_types: List of model types to include

        Returns:
            Tuple of (ensemble_model, metrics)
        """
        print(f"\n{'='*60}")
        print(f"Training Ensemble Model for {epic}")
        print(f"Models: {', '.join(model_types)}")
        print(f"{'='*60}\n")

        # Load data once
        X, y = self.load_training_data(epic, start_date, end_date)

        # Train individual models
        models = []

        for model_type in model_types:
            if model_type == 'LSTM':
                continue  # Skip LSTM for ensemble (different preprocessing)

            model, _ = self.train_model(
                model_type,
                epic,
                start_date,
                end_date,
                hyperparameter_tuning=False
            )
            models.append(model)

        # Create ensemble
        ensemble = EnsembleTrader(models)

        # Evaluate ensemble
        print("\nEvaluating ensemble...")

        # Use first model's preprocessing (all should be the same)
        X_scaled, y_encoded = models[0].preprocess_data(X, y, fit_scaler=False)
        _, X_test, _, y_test = models[0].split_data(X_scaled, y_encoded)

        metrics = ensemble.evaluate(X_test, y_test)

        print("\nEnsemble Performance:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric, value in model_metrics.items():
                print(f"  {metric}: {value:.4f}")

        return ensemble, metrics

    def save_model_to_db(
        self,
        model: any,
        model_type: str,
        epic: str,
        metrics: Dict,
        start_date: datetime,
        end_date: datetime,
        training_samples: int
    ) -> int:
        """
        Save trained model to database.

        Args:
            model: Trained model
            model_type: Type of model
            epic: Market epic
            metrics: Performance metrics
            start_date: Training start date
            end_date: Training end date
            training_samples: Number of training samples

        Returns:
            Model ID
        """
        # Generate version
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model file
        model_filename = f"{epic}_{model_type}_{version}.joblib"
        model_path = os.path.join(self.models_dir, model_filename)

        model.save_model(model_path)

        # Save to database
        with get_db() as db:
            db_model = Model(
                name=f"{epic}_{model_type}",
                version=version,
                algorithm=model_type,
                trained_at=datetime.now(),
                training_samples=training_samples,
                training_period_start=start_date,
                training_period_end=end_date,
                hyperparameters=model.hyperparameters if hasattr(model, 'hyperparameters') else {},
                features_used=model.feature_names if hasattr(model, 'feature_names') else [],
                accuracy=metrics.get('accuracy', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', 0.0),
                model_path=model_path,
                is_active=True,
                is_production=False
            )

            db.add(db_model)
            db.commit()
            db.refresh(db_model)

            print(f"\n✓ Model saved to database (ID: {db_model.id})")
            return db_model.id

    def train_and_save(
        self,
        model_type: str,
        epic: str,
        lookback_days: int = 90,
        hyperparameter_tuning: bool = False
    ) -> int:
        """
        Complete training workflow: train and save model.

        Args:
            model_type: Type of model
            epic: Market epic
            lookback_days: Days of historical data
            hyperparameter_tuning: Whether to tune hyperparameters

        Returns:
            Model ID
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Train model
        model, metrics = self.train_model(
            model_type,
            epic,
            start_date,
            end_date,
            hyperparameter_tuning=hyperparameter_tuning
        )

        # Save to database
        X, y = self.load_training_data(epic, start_date, end_date)
        model_id = self.save_model_to_db(
            model,
            model_type,
            epic,
            metrics,
            start_date,
            end_date,
            len(X)
        )

        return model_id

    def batch_train_all_instruments(
        self,
        instruments: List[str],
        model_types: List[str] = ['RandomForest', 'XGBoost', 'LightGBM'],
        lookback_days: int = 90
    ):
        """
        Train models for all instruments.

        Args:
            instruments: List of epics
            model_types: List of model types
            lookback_days: Days of historical data
        """
        print(f"\n{'='*80}")
        print(f"BATCH TRAINING: {len(instruments)} instruments × {len(model_types)} models")
        print(f"{'='*80}\n")

        results = []

        for epic in instruments:
            for model_type in model_types:
                try:
                    model_id = self.train_and_save(
                        model_type,
                        epic,
                        lookback_days=lookback_days
                    )

                    results.append({
                        'epic': epic,
                        'model_type': model_type,
                        'model_id': model_id,
                        'status': 'SUCCESS'
                    })

                except Exception as e:
                    print(f"\n✗ Error training {model_type} for {epic}: {e}\n")
                    results.append({
                        'epic': epic,
                        'model_type': model_type,
                        'status': 'FAILED',
                        'error': str(e)
                    })

        # Summary
        print(f"\n{'='*80}")
        print("BATCH TRAINING SUMMARY")
        print(f"{'='*80}\n")

        success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
        total_count = len(results)

        print(f"Total: {total_count}")
        print(f"Success: {success_count}")
        print(f"Failed: {total_count - success_count}")

        return results


# Example usage
if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()

    # Train a single model
    model_id = pipeline.train_and_save(
        model_type='XGBoost',
        epic='US100',
        lookback_days=90,
        hyperparameter_tuning=False
    )

    print(f"\nModel ID: {model_id}")
