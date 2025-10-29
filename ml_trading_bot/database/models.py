"""
Database models for ML trading bot.
Stores historical data, trades, model performance, and feedback.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class MarketData(Base):
    """Historical market data (OHLCV)"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    epic = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    resolution = Column(String(20))  # MINUTE, HOUR, DAY, etc.

    __table_args__ = (
        Index('idx_epic_timestamp', 'epic', 'timestamp'),
    )

    def __repr__(self):
        return f"<MarketData(epic={self.epic}, timestamp={self.timestamp}, close={self.close})>"


class SentimentData(Base):
    """Sentiment analysis results from news/social media"""
    __tablename__ = 'sentiment_data'

    id = Column(Integer, primary_key=True)
    epic = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String(50))  # 'news', 'twitter', 'reddit', etc.
    text = Column(Text)
    sentiment_score = Column(Float)  # -1 (bearish) to +1 (bullish)
    confidence = Column(Float)  # 0 to 1
    keywords = Column(JSON)  # Extracted keywords
    url = Column(String(500))

    __table_args__ = (
        Index('idx_epic_timestamp_sent', 'epic', 'timestamp'),
    )

    def __repr__(self):
        return f"<SentimentData(epic={self.epic}, score={self.sentiment_score}, source={self.source})>"


class Feature(Base):
    """Engineered features for ML models"""
    __tablename__ = 'features'

    id = Column(Integer, primary_key=True)
    epic = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Technical indicators
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    ma_20 = Column(Float)
    ma_50 = Column(Float)
    ma_100 = Column(Float)
    atr_14 = Column(Float)
    adx_14 = Column(Float)
    cci_20 = Column(Float)
    stochastic_k = Column(Float)
    stochastic_d = Column(Float)

    # Price patterns
    support_level = Column(Float)
    resistance_level = Column(Float)
    momentum_10 = Column(Float)
    momentum_20 = Column(Float)
    volatility_10 = Column(Float)
    volatility_20 = Column(Float)

    # Sentiment features
    sentiment_score_1h = Column(Float)
    sentiment_score_4h = Column(Float)
    sentiment_score_24h = Column(Float)
    sentiment_volume_1h = Column(Integer)  # Number of mentions
    sentiment_volume_4h = Column(Integer)
    sentiment_volume_24h = Column(Integer)

    # Target variable (for supervised learning)
    target_return_1h = Column(Float)  # Actual return after 1 hour
    target_return_4h = Column(Float)  # Actual return after 4 hours
    target_return_24h = Column(Float)  # Actual return after 24 hours
    target_direction = Column(String(10))  # 'UP', 'DOWN', 'NEUTRAL'

    __table_args__ = (
        Index('idx_epic_timestamp_feat', 'epic', 'timestamp'),
    )

    def __repr__(self):
        return f"<Feature(epic={self.epic}, timestamp={self.timestamp})>"


class Trade(Base):
    """Executed trades (simulated or real)"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    trade_id = Column(String(100), unique=True)
    epic = Column(String(50), nullable=False, index=True)

    # Trade details
    direction = Column(String(10))  # 'BUY', 'SELL'
    size = Column(Float)
    entry_price = Column(Float)
    entry_time = Column(DateTime, nullable=False)

    stop_loss = Column(Float)
    take_profit = Column(Float)

    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_reason = Column(String(50))  # 'TP', 'SL', 'MANUAL', 'TIMEOUT'

    # P&L
    pnl = Column(Float)
    pnl_percent = Column(Float)

    # Risk metrics
    risk_amount = Column(Float)
    risk_reward_ratio = Column(Float)
    max_drawdown = Column(Float)

    # Model information
    model_id = Column(Integer, ForeignKey('models.id'))
    model_version = Column(String(50))
    prediction_confidence = Column(Float)

    # Execution details
    is_simulated = Column(Boolean, default=True)
    slippage = Column(Float)
    commission = Column(Float)

    # Relationships
    model = relationship("Model", back_populates="trades")
    feedback = relationship("TradeFeedback", back_populates="trade", uselist=False)

    def __repr__(self):
        return f"<Trade(id={self.trade_id}, epic={self.epic}, direction={self.direction}, pnl={self.pnl})>"


class Model(Base):
    """ML model metadata and performance"""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(50))  # 'RandomForest', 'XGBoost', 'LSTM', 'Ensemble'

    # Training details
    trained_at = Column(DateTime, nullable=False)
    training_samples = Column(Integer)
    training_period_start = Column(DateTime)
    training_period_end = Column(DateTime)

    # Hyperparameters
    hyperparameters = Column(JSON)

    # Features used
    features_used = Column(JSON)

    # Performance metrics (on validation set)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    # Trading metrics (on validation set)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)

    # Model file path
    model_path = Column(String(500))

    # Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)

    # Relationships
    trades = relationship("Trade", back_populates="model")
    performance_logs = relationship("ModelPerformanceLog", back_populates="model")

    __table_args__ = (
        Index('idx_model_name_version', 'name', 'version'),
    )

    def __repr__(self):
        return f"<Model(name={self.name}, version={self.version}, algorithm={self.algorithm})>"


class ModelPerformanceLog(Base):
    """Real-time performance tracking for deployed models"""
    __tablename__ = 'model_performance_logs'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Performance metrics (rolling window)
    window_period = Column(String(20))  # '1D', '1W', '1M'
    trades_count = Column(Integer)
    win_count = Column(Integer)
    loss_count = Column(Integer)
    win_rate = Column(Float)

    total_pnl = Column(Float)
    average_pnl = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)

    # Prediction quality
    prediction_accuracy = Column(Float)
    average_confidence = Column(Float)
    calibration_score = Column(Float)  # How well confidence matches actual outcomes

    # Relationship
    model = relationship("Model", back_populates="performance_logs")

    def __repr__(self):
        return f"<ModelPerformanceLog(model_id={self.model_id}, timestamp={self.timestamp}, win_rate={self.win_rate})>"


class TradeFeedback(Base):
    """Feedback and analysis for each trade"""
    __tablename__ = 'trade_feedback'

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=False)

    # What went right/wrong
    outcome = Column(String(20))  # 'WIN', 'LOSS', 'BREAKEVEN'

    # Analysis
    exit_optimal = Column(Boolean)  # Did we exit at the right time?
    entry_optimal = Column(Boolean)  # Did we enter at the right time?

    price_prediction_error = Column(Float)  # How far off was our prediction?
    sentiment_accuracy = Column(Float)  # Did sentiment match outcome?

    # What we learned
    lessons_learned = Column(JSON)
    feature_importance_snapshot = Column(JSON)

    # Manual notes
    notes = Column(Text)

    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    trade = relationship("Trade", back_populates="feedback")

    def __repr__(self):
        return f"<TradeFeedback(trade_id={self.trade_id}, outcome={self.outcome})>"


class TrainingJob(Base):
    """Track model training jobs"""
    __tablename__ = 'training_jobs'

    id = Column(Integer, primary_key=True)
    job_id = Column(String(100), unique=True)

    # Job details
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    status = Column(String(20))  # 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'

    # Configuration
    model_type = Column(String(50))
    config = Column(JSON)

    # Data details
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    data_period_start = Column(DateTime)
    data_period_end = Column(DateTime)

    # Results
    best_score = Column(Float)
    best_params = Column(JSON)
    cv_scores = Column(JSON)

    # Output
    model_id = Column(Integer, ForeignKey('models.id'))

    # Error handling
    error_message = Column(Text)

    def __repr__(self):
        return f"<TrainingJob(job_id={self.job_id}, status={self.status}, model_type={self.model_type})>"


class BacktestResult(Base):
    """Backtest results for model evaluation"""
    __tablename__ = 'backtest_results'

    id = Column(Integer, primary_key=True)
    backtest_id = Column(String(100), unique=True)
    model_id = Column(Integer, ForeignKey('models.id'))

    # Backtest configuration
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float)
    instruments = Column(JSON)

    # Results
    final_capital = Column(Float)
    total_return = Column(Float)
    total_return_percent = Column(Float)

    # Performance metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_percent = Column(Float)

    # Trade statistics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)

    average_win = Column(Float)
    average_loss = Column(Float)
    profit_factor = Column(Float)

    largest_win = Column(Float)
    largest_loss = Column(Float)

    # Holding period stats
    average_holding_period = Column(Float)  # in hours

    # Equity curve data
    equity_curve = Column(JSON)

    # Created timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<BacktestResult(id={self.backtest_id}, return={self.total_return_percent}%, sharpe={self.sharpe_ratio})>"
