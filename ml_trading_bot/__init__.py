"""
ML Trading Bot - Advanced Machine Learning Trading System
with Sentiment Analysis and Continuous Learning
"""

__version__ = "1.0.0"
__author__ = "ML Trading Bot Team"

from . import database
from . import data_collection
from . import sentiment
from . import features
from . import models
from . import backtesting
from . import feedback
from . import continuous_learning
from . import dashboard

__all__ = [
    'database',
    'data_collection',
    'sentiment',
    'features',
    'models',
    'backtesting',
    'feedback',
    'continuous_learning',
    'dashboard'
]
