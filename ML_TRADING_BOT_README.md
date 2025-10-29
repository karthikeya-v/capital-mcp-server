# ML Trading Bot with Sentiment Analysis & Continuous Learning

A comprehensive machine learning trading system that learns from market data and sentiment, makes simulated trades, analyzes results, and continuously improves through feedback loops.

## 🌟 Features

### Core Capabilities
- **Multi-Algorithm ML Models**: Random Forest, XGBoost, LightGBM, and LSTM
- **Sentiment Analysis**: FinBERT-powered financial sentiment from news and social media
- **Advanced Feature Engineering**: 30+ technical indicators + sentiment features
- **Realistic Backtesting**: Simulated trading with slippage, commissions, and risk management
- **Feedback Loop**: Analyzes every trade to identify mistakes and improvement areas
- **Continuous Learning**: Automatic model retraining based on performance degradation
- **Real-time Monitoring**: Web dashboard with live performance metrics
- **Risk Management**: Dynamic position sizing, stop-loss, and daily loss limits

### System Architecture

```
ml_trading_bot/
├── database/              # SQLAlchemy models (10 tables)
│   ├── models.py         # MarketData, SentimentData, Features, Trades, Models, etc.
│   └── db_config.py      # Database configuration
│
├── data_collection/       # Market & sentiment data collection
│   └── data_collector.py # Capital.com API + Perplexity integration
│
├── sentiment/            # NLP & sentiment analysis
│   └── sentiment_analyzer.py  # FinBERT + TextBlob sentiment scoring
│
├── features/             # Feature engineering
│   └── feature_engineering.py # Technical + sentiment features
│
├── models/               # ML model training
│   ├── ml_models.py      # RandomForest, XGBoost, LightGBM, LSTM, Ensemble
│   └── training_pipeline.py  # Training orchestration
│
├── backtesting/          # Simulated trading
│   └── backtest_engine.py # Realistic backtest with risk management
│
├── feedback/             # Learning from mistakes
│   └── feedback_analyzer.py  # Trade analysis & lesson extraction
│
├── continuous_learning/  # Auto-retraining
│   └── auto_retrain.py   # Performance monitoring & model retraining
│
├── dashboard/            # Web monitoring
│   └── ml_dashboard.py   # Flask dashboard with Plotly charts
│
└── ml_bot_orchestrator.py # Main entry point

```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from ml_trading_bot.database import init_database; init_database()"
```

### 2. Configuration

Create a `.env` file:

```env
# Capital.com API
CAPITAL_EMAIL=your_email@example.com
CAPITAL_API_KEY=your_api_key
CAPITAL_PASSWORD=your_api_password

# Perplexity API
PERPLEXITY_API_KEY=your_perplexity_key

# Anthropic API (optional - for Claude integration)
ANTHROPIC_API_KEY=your_anthropic_key

# Database (optional - defaults to SQLite)
DATABASE_URL=sqlite:///./ml_trading_bot.db
# For production PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/trading_bot
```

### 3. Run Complete Workflow

```bash
# Full workflow: data collection → feature engineering → training → backtesting
cd ml_trading_bot
python ml_bot_orchestrator.py --mode full --days 90

# Or use specific modes:
python ml_bot_orchestrator.py --mode train --days 90    # Only train models
python ml_bot_orchestrator.py --mode backtest           # Only backtest
python ml_bot_orchestrator.py --mode continuous         # Continuous learning
python ml_bot_orchestrator.py --mode report             # Generate report
```

### 4. Launch Dashboard

```bash
cd ml_trading_bot/dashboard
python ml_dashboard.py

# Access at: http://localhost:5001
```

## 📊 How It Works

### Phase 1: Data Collection
```python
from ml_trading_bot.data_collection import DataCollectionPipeline

pipeline = DataCollectionPipeline(
    capital_email=EMAIL,
    capital_api_key=API_KEY,
    capital_password=PASSWORD,
    perplexity_api_key=PERPLEXITY_KEY
)

# Collect market data + sentiment
await pipeline.run_collection_cycle(
    instruments=['US100', 'EURUSD', 'GOLD'],
    market_resolution='MINUTE',
    collect_news=True
)
```

**What it does:**
- Fetches OHLCV data from Capital.com API
- Gathers news using Perplexity AI
- Analyzes sentiment with FinBERT
- Stores everything in database

### Phase 2: Feature Engineering
```python
from ml_trading_bot.features import FeatureEngineeringPipeline

feature_pipeline = FeatureEngineeringPipeline()

# Create 50+ features
feature_pipeline.create_and_save_features(
    instruments=['US100'],
    start_date=start,
    end_date=end
)
```

**Features created:**
- **Technical (30+)**: RSI, MACD, Bollinger Bands, ATR, ADX, CCI, Stochastic, etc.
- **Price Patterns**: Support/resistance, momentum, volatility
- **Sentiment (6)**: 1h/4h/24h sentiment scores and volume
- **Targets**: Future returns at 1h/4h/24h horizons

### Phase 3: Model Training
```python
from ml_trading_bot.models import ModelTrainingPipeline

trainer = ModelTrainingPipeline()

# Train XGBoost model
model_id = trainer.train_and_save(
    model_type='XGBoost',
    epic='US100',
    lookback_days=90,
    hyperparameter_tuning=True  # Grid search for best params
)
```

**Supported algorithms:**
- **Random Forest**: Robust ensemble method
- **XGBoost**: Gradient boosting (high accuracy)
- **LightGBM**: Fast gradient boosting
- **LSTM**: Deep learning for sequences
- **Ensemble**: Combines multiple models

**Evaluation metrics:**
- Accuracy, Precision, Recall, F1 Score
- Sharpe Ratio, Sortino Ratio
- Win Rate, Profit Factor
- Max Drawdown

### Phase 4: Backtesting
```python
from ml_trading_bot.backtesting import BacktestEngine

backtest = BacktestEngine(
    initial_capital=10000,
    risk_per_trade_percent=1.0,  # Risk 1% per trade
    max_daily_loss=500,
    slippage_percent=0.1,
    commission_per_trade=2.0
)

# Run realistic simulation
results = backtest.run_backtest(
    epic='US100',
    model=trained_model,
    start_date=start,
    end_date=end,
    min_confidence=0.6  # Only trade with 60%+ confidence
)
```

**Realistic conditions:**
- Slippage simulation (0.1%)
- Commission per trade ($2)
- Dynamic position sizing (risk-based)
- Stop-loss and take-profit
- Daily loss limits
- No look-ahead bias

**Output:**
```
BACKTEST RESULTS
================
Total Trades: 47
Win Rate: 63.8%
Total Return: $847.23 (8.47%)
Sharpe Ratio: 1.82
Max Drawdown: $234.12 (2.34%)
Profit Factor: 2.45
```

### Phase 5: Feedback Analysis
```python
from ml_trading_bot.feedback import FeedbackLoop

feedback = FeedbackLoop()

# Analyze every trade
feedback.process_trade_feedback(trade_id=123)

# Generate performance report
report = feedback.generate_performance_report(
    model_id=1,
    window_days=30
)

# Get improvement recommendations
recommendations = feedback.identify_improvement_areas(model_id=1)
```

**What it analyzes:**
- Entry timing quality (was it a good price?)
- Exit timing quality (could we have done better?)
- Price prediction accuracy
- Recurring mistakes (lessons learned)

**Example feedback:**
```json
{
  "lessons": [
    "Entry timing was suboptimal - entered at 0.82 of recent range",
    "Exit was premature - price moved favorably after exit",
    "High confidence (0.87) but trade failed - model may be overconfident"
  ],
  "recommendations": [
    {
      "area": "Entry Timing",
      "issue": "Entry quality is 54.2%",
      "suggestion": "Add features for better entry timing or use limit orders"
    }
  ]
}
```

### Phase 6: Continuous Learning
```python
from ml_trading_bot.continuous_learning import AutoRetrainer

retrainer = AutoRetrainer()

# Monitor and retrain automatically
retrainer.run_continuous(check_interval_hours=24)
```

**Retraining triggers:**
- Win rate drops below 50%
- Model older than 30 days
- Performance declining trend
- Data drift detected

**What happens:**
1. Detects performance degradation
2. Collects new data since last training
3. Re-engineers features
4. Performs hyperparameter tuning
5. Trains new model version
6. Deactivates old model
7. Activates new model

## 📈 Dashboard

The web dashboard provides real-time monitoring:

![Dashboard Preview](dashboard_preview.png)

**Features:**
- **Overview Stats**: Total models, trades, win rate, P&L
- **Win Rate Trend**: 90-day rolling win rate chart
- **Model Comparison**: Side-by-side performance metrics
- **Active Models Table**: All models with accuracy, Sharpe, status
- **Auto-refresh**: Updates every 30 seconds

Access at `http://localhost:5001` after running:
```bash
python ml_trading_bot/dashboard/ml_dashboard.py
```

## 🎯 Usage Examples

### Example 1: Train and Backtest a Single Model

```python
import asyncio
from ml_trading_bot.ml_bot_orchestrator import MLTradingBotOrchestrator

async def main():
    bot = MLTradingBotOrchestrator(
        instruments=['US100'],
        demo=True
    )

    # Initialize
    await bot.initialize_system()

    # Collect 90 days of data
    await bot.collect_historical_data(days=90)

    # Engineer features
    await bot.engineer_features(days=90)

    # Train XGBoost model
    await bot.train_all_models(
        model_types=['XGBoost'],
        lookback_days=90
    )

    # Backtest
    await bot.backtest_all_models(test_days=30)

asyncio.run(main())
```

### Example 2: Batch Train Multiple Models

```python
from ml_trading_bot.models import ModelTrainingPipeline

pipeline = ModelTrainingPipeline()

results = pipeline.batch_train_all_instruments(
    instruments=['US100', 'EURUSD', 'GBPUSD', 'GOLD'],
    model_types=['RandomForest', 'XGBoost', 'LightGBM'],
    lookback_days=90
)

# Results: 4 instruments × 3 models = 12 trained models
```

### Example 3: Continuous Learning Loop

```python
from ml_trading_bot.continuous_learning import AutoRetrainer

retrainer = AutoRetrainer()

# Runs forever, checking every 24 hours
retrainer.run_continuous(check_interval_hours=24)
```

**What happens in continuous mode:**
1. Every 24 hours:
   - Monitors all active models
   - Checks performance metrics
   - Identifies models needing retraining
2. For degraded models:
   - Analyzes feedback
   - Retrains with latest data
   - Performs hyperparameter tuning
3. Background data collection:
   - Fetches new market data every hour
   - Updates sentiment analysis
   - Engineers new features

### Example 4: Custom Model Training

```python
from ml_trading_bot.models import XGBoostTrader
from ml_trading_bot.features import FeatureEngineeringPipeline
from datetime import datetime, timedelta

# Create features
feature_pipeline = FeatureEngineeringPipeline()
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

df_features = feature_pipeline.create_features_from_db(
    epic='US100',
    start_date=start_date,
    end_date=end_date
)

# Extract X and y
X = df_features.drop(['target_direction_4h'], axis=1)
y = df_features['target_direction_4h']

# Train model with custom hyperparameters
model = XGBoostTrader(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05
)

X_scaled, y_encoded = model.preprocess_data(X, y)
X_train, X_test, y_train, y_test = model.split_data(X_scaled, y_encoded)

model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.3f}")

# Save
model.save_model('./my_custom_model.joblib')
```

## 🔧 Advanced Configuration

### Custom Risk Management

```python
from ml_trading_bot.backtesting import BacktestEngine

backtest = BacktestEngine(
    initial_capital=50000,
    risk_per_trade_percent=0.5,  # Conservative: 0.5% risk per trade
    max_position_size=2.0,        # Max 2 lots
    max_daily_loss=1000,          # Stop trading if lose $1000 in a day
    slippage_percent=0.15,        # Higher slippage
    commission_per_trade=5.0      # Higher commission
)
```

### Custom Feature Engineering

```python
from ml_trading_bot.features import TechnicalFeatureEngineer
import pandas as pd

engineer = TechnicalFeatureEngineer()

# Add your own features
df = pd.DataFrame(market_data)
df = engineer.calculate_technical_indicators(df)
df = engineer.calculate_price_patterns(df)

# Add custom feature
df['custom_ratio'] = df['rsi_14'] / df['atr_14']
df['custom_trend'] = (df['ma_20'] > df['ma_50']).astype(int)
```

### Custom Sentiment Sources

```python
from ml_trading_bot.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze custom text
tweets = [
    "Market is looking bullish! Strong rally expected.",
    "Concerns over economic data causing sell-off.",
    "Breakout above resistance, very positive momentum!"
]

sentiments = analyzer.analyze_batch(tweets)

# Aggregate
aggregated = analyzer.aggregate_sentiment(sentiments, method='weighted')
print(f"Overall sentiment: {aggregated['score']:.3f}")
```

## 📊 Database Schema

The system uses 10 database tables:

1. **market_data**: Historical OHLCV data
2. **sentiment_data**: Sentiment analysis results
3. **features**: Engineered features (50+ columns)
4. **trades**: Executed trades (simulated or real)
5. **models**: ML model metadata and performance
6. **model_performance_logs**: Time-series performance tracking
7. **trade_feedback**: Post-trade analysis and lessons
8. **training_jobs**: Model training job tracking
9. **backtest_results**: Backtest results storage

## 🎓 Learning System

### How the bot learns from mistakes:

1. **Trade Execution** → Trade recorded in database
2. **Post-Trade Analysis** → FeedbackLoop analyzes:
   - Entry quality (did we enter at a good price?)
   - Exit quality (could we have exited better?)
   - Prediction accuracy (how far off were we?)
3. **Lesson Extraction** → Common mistakes identified:
   - "Stop loss too tight - hit 73% of the time"
   - "Model overconfident on ranging markets"
   - "Entry timing poor during high volatility"
4. **Model Retraining** → New model trained with:
   - Latest data (includes recent mistakes)
   - Adjusted hyperparameters
   - Additional features (if drift detected)
5. **A/B Testing** → New model compared to old
6. **Deployment** → Better model activated

### Performance Monitoring

The system continuously monitors:
- **Win Rate**: Must stay above 50%
- **Sharpe Ratio**: Risk-adjusted returns
- **Drawdown**: Maximum loss from peak
- **Prediction Calibration**: Confidence vs actual outcomes
- **Data Drift**: Changes in market regime

## ⚠️ Important Notes

### Simulated Trading
This system is designed for **simulated trading only**. All trades are:
- Executed in backtests (historical simulation)
- Tracked in database with `is_simulated=True`
- Never sent to live Capital.com API

To enable live trading (USE WITH EXTREME CAUTION):
1. Set `demo=False` in orchestrator
2. Modify `execute_trade` logic to actually place orders
3. Implement proper risk controls and kill switches

### Data Requirements
For good model performance, you need:
- **Minimum**: 30 days of minute-level data (~43,000 candles)
- **Recommended**: 90 days (130,000 candles)
- **Optimal**: 180+ days (260,000+ candles)

### Computational Requirements
- **CPU**: Multi-core recommended (for XGBoost/LightGBM)
- **RAM**: 8GB+ (for large datasets)
- **GPU**: Optional (accelerates LSTM training)
- **Storage**: 1GB+ for database

## 🐛 Troubleshooting

### Issue: "No features found for training"
**Solution**: Run data collection and feature engineering first:
```bash
python ml_bot_orchestrator.py --mode init
python -c "from ml_trading_bot.data_collection import DataCollectionPipeline; import asyncio; asyncio.run(pipeline.run_collection_cycle(...))"
```

### Issue: "Model training fails"
**Solution**: Check for NaN values in features:
```python
from ml_trading_bot.database import get_db, Feature
with get_db() as db:
    features = db.query(Feature).limit(1000).all()
    # Check for nulls
```

### Issue: "Dashboard shows no data"
**Solution**: Ensure database is initialized and has data:
```python
from ml_trading_bot.ml_bot_orchestrator import MLTradingBotOrchestrator
bot = MLTradingBotOrchestrator(['US100'])
bot.generate_system_report()
```

## 📚 Further Reading

- [Capital.com API Documentation](https://capital.com/api-documentation)
- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Backtesting Best Practices](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/)

## 🤝 Contributing

This is a personal project, but improvements welcome! Areas to enhance:
- Additional data sources (Twitter, Reddit, Binance, etc.)
- More ML algorithms (Transformers, Reinforcement Learning)
- Portfolio optimization (multiple simultaneous positions)
- Alternative risk models (Kelly Criterion, VaR)

## 📄 License

MIT License - Use at your own risk. Not financial advice.

## ⚡ Quick Command Reference

```bash
# Full workflow
python ml_bot_orchestrator.py --mode full --days 90

# Initialize database
python -c "from ml_trading_bot.database import init_database; init_database()"

# Train models only
python ml_bot_orchestrator.py --mode train --days 90

# Backtest only
python ml_bot_orchestrator.py --mode backtest --days 30

# Continuous learning
python ml_bot_orchestrator.py --mode continuous

# Generate report
python ml_bot_orchestrator.py --mode report

# Launch dashboard
python ml_trading_bot/dashboard/ml_dashboard.py

# Custom instruments
python ml_bot_orchestrator.py --instruments US100 EURUSD GOLD --days 90
```

---

**Built with:** Python, SQLAlchemy, scikit-learn, XGBoost, PyTorch, Transformers, Flask, Plotly

**Status:** ✅ Production-ready for simulated trading

**Last Updated:** 2025-10-29
