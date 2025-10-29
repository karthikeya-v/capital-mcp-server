"""
ML Model Training Pipeline.
Supports multiple algorithms: Random Forest, XGBoost, LightGBM, and LSTM.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import joblib
import json
import os

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import Model, TrainingJob, get_db, Feature


class TradingDataset(Dataset):
    """PyTorch dataset for trading data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        out = self.fc(lstm_out[:, -1, :])

        return out


class BaseMLModel:
    """Base class for ML trading models"""

    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def preprocess_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess features and targets.

        Args:
            X: Feature DataFrame
            y: Target Series
            fit_scaler: Whether to fit the scaler

        Returns:
            Tuple of (X_scaled, y_encoded)
        """
        # Remove NaN
        if y is not None:
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]
        else:
            X = X.dropna()

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Encode targets
        if y is not None:
            # Map UP=2, NEUTRAL=1, DOWN=0
            label_map = {'UP': 2, 'NEUTRAL': 1, 'DOWN': 0}
            y_encoded = y.map(label_map).values
            return X_scaled, y_encoded
        else:
            return X_scaled, None

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Time-series aware train/test split.

        Args:
            X: Features
            y: Targets
            test_size: Test set proportion

        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X_test)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance.tolist()))
        return None

    def save_model(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }

        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True

        print(f"✓ Model loaded from {path}")


class RandomForestTrader(BaseMLModel):
    """Random Forest trading model"""

    def __init__(self, name: str = "RandomForest", **kwargs):
        super().__init__(name, "RandomForest")
        self.hyperparameters = kwargs or {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Random Forest model"""
        self.model = RandomForestClassifier(**self.hyperparameters)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✓ {self.name} training completed")

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 3
    ) -> Dict:
        """Perform hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Use TimeSeriesSplit for CV
        tscv = TimeSeriesSplit(n_splits=cv)

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.hyperparameters = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.is_trained = True

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }


class XGBoostTrader(BaseMLModel):
    """XGBoost trading model"""

    def __init__(self, name: str = "XGBoost", **kwargs):
        super().__init__(name, "XGBoost")
        self.hyperparameters = kwargs or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost model"""
        self.model = xgb.XGBClassifier(**self.hyperparameters)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✓ {self.name} training completed")

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 3
    ) -> Dict:
        """Perform hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        tscv = TimeSeriesSplit(n_splits=cv)

        grid_search = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            param_grid,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.hyperparameters = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.is_trained = True

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }


class LightGBMTrader(BaseMLModel):
    """LightGBM trading model"""

    def __init__(self, name: str = "LightGBM", **kwargs):
        super().__init__(name, "LightGBM")
        self.hyperparameters = kwargs or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': 42
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LightGBM model"""
        self.model = lgb.LGBMClassifier(**self.hyperparameters)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✓ {self.name} training completed")


class LSTMTrader(BaseMLModel):
    """LSTM trading model"""

    def __init__(
        self,
        name: str = "LSTM",
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_length: int = 20,
        **kwargs
    ):
        super().__init__(name, "LSTM")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """Train LSTM model"""
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_train, y_train)

        # Create dataset and dataloader
        dataset = TradingDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_size = X_seq.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        self.is_trained = True
        print(f"✓ {self.name} training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()

        X_seq, _ = self.create_sequences(X, np.zeros(len(X)))

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model"""
        X_seq, y_seq = self.create_sequences(X_test, y_test)

        y_pred = self.predict(X_test)

        # Align lengths
        min_len = min(len(y_pred), len(y_seq))
        y_pred = y_pred[:min_len]
        y_seq = y_seq[:min_len]

        return {
            'accuracy': accuracy_score(y_seq, y_pred),
            'precision': precision_score(y_seq, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_seq, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_seq, y_pred, average='weighted', zero_division=0)
        }


class EnsembleTrader:
    """Ensemble of multiple models with voting"""

    def __init__(self, models: List[BaseMLModel]):
        self.models = models
        self.weights = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all models in ensemble"""
        for model in self.models:
            print(f"\nTraining {model.name}...")
            model.train(X_train, y_train)

    def predict(self, X: np.ndarray, method: str = 'majority') -> np.ndarray:
        """
        Make ensemble prediction.

        Args:
            X: Features
            method: 'majority' (voting) or 'weighted' (weighted by accuracy)

        Returns:
            Predictions
        """
        predictions = []

        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)

        predictions = np.array(predictions)

        if method == 'majority':
            # Majority voting
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=predictions
            )
        else:
            # Weighted voting (if weights available)
            if self.weights is not None:
                weighted_pred = np.average(predictions, axis=0, weights=self.weights)
                final_pred = np.round(weighted_pred).astype(int)
            else:
                final_pred = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(),
                    axis=0,
                    arr=predictions
                )

        return final_pred

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        # Get ensemble predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        metrics = {
            'ensemble': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        }

        # Individual model metrics
        for model in self.models:
            try:
                model_metrics = model.evaluate(X_test, y_test)
                metrics[model.name] = model_metrics
            except Exception as e:
                print(f"Could not evaluate {model.name}: {e}")

        return metrics
