"""Database module for ML trading bot"""

from .models import (
    Base,
    MarketData,
    SentimentData,
    Feature,
    Trade,
    Model,
    ModelPerformanceLog,
    TradeFeedback,
    TrainingJob,
    BacktestResult
)
from .db_config import (
    engine,
    Session,
    get_db,
    get_session,
    init_database,
    drop_database
)

__all__ = [
    'Base',
    'MarketData',
    'SentimentData',
    'Feature',
    'Trade',
    'Model',
    'ModelPerformanceLog',
    'TradeFeedback',
    'TrainingJob',
    'BacktestResult',
    'engine',
    'Session',
    'get_db',
    'get_session',
    'init_database',
    'drop_database'
]
