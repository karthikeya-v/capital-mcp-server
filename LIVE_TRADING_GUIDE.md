# Live Trading with ML Bot - Complete Guide

## 🎯 Overview

The ML Trading Bot can now execute **real trades** on your Capital.com demo account using trained ML models. It continuously monitors markets, makes predictions, executes trades, and learns from feedback.

---

## 🚀 Quick Start

### 1. Configure Environment

Ensure your `.env` file has:

```env
CAPITAL_EMAIL=your_email@example.com
CAPITAL_API_KEY=your_api_key
CAPITAL_PASSWORD=your_api_password
PERPLEXITY_API_KEY=your_perplexity_key  # Optional but recommended
```

### 2. Train Models First

Before live trading, you need trained models:

```bash
cd ml_trading_bot
python ml_bot_orchestrator.py --mode full --days 90
```

This will:
- Collect 90 days of market data
- Engineer features
- Train models for US100, EURUSD, GBPUSD, etc.
- Backtest models

### 3. Start Live Trading Bot

```bash
cd ml_trading_bot/live_trading
python live_trading_bot.py
```

The bot will:
- ✅ Load trained ML models
- ✅ Connect to Capital.com demo account
- ✅ Monitor markets every 60 seconds
- ✅ Make predictions using models
- ✅ Execute trades when confidence > 65%
- ✅ Monitor open positions
- ✅ Close positions at SL/TP or after 24 hours
- ✅ Analyze feedback from closed trades

### 4. Monitor with Enhanced Dashboard

In a separate terminal:

```bash
cd ml_trading_bot/dashboard
python enhanced_dashboard.py
```

Access at: **http://localhost:5001**

---

## 📊 Dashboard Features

### Overview Tab
- **Real-time stats**: Models, positions, trades, P&L
- **Daily P&L chart**: Last 30 days bar chart
- **Win rate trend**: 90-day moving average

### Live Trading Tab
- **Live positions**: Real-time from Capital.com API
  - Epic, direction, size, entry price
  - Current P&L (updates live)
  - Stop loss / Take profit levels
  - Holding time
- **Open trades from database**: Cross-reference

### Models Tab
- **Performance comparison**: Bar chart of all models
- **Model details table**:
  - Algorithm, accuracy, win rate, Sharpe ratio
  - Total trades, training date
  - Active/Inactive status

### Trades Tab
- **All trades**: Filter by days (7/30) or type (live/simulated)
- **Trade details**:
  - Entry/exit prices and times
  - P&L ($ and %)
  - Exit reason (TP/SL/Manual)
  - Model confidence
  - Trade type (Live/Simulated)

### Analysis Tab
- **Model health checks**:
  - Current performance vs threshold
  - Days since training
  - Recommendations (CONTINUE/RETRAIN)
  - Reasons for degradation

### Feedback Tab
- **Common lessons learned**:
  - Top 10 recurring mistakes
  - Frequency count
  - Actionable insights

---

## 🤖 How It Works

### Trading Loop

```
1. SCAN MARKETS (every 60s)
   │
   ├─ For each instrument (US100, EURUSD, etc.):
   │   │
   │   ├─ Fetch recent market data (last 24h)
   │   ├─ Calculate technical indicators (RSI, MACD, etc.)
   │   ├─ Get sentiment data
   │   └─ Engineer features (50+)
   │
   ├─ MAKE PREDICTION
   │   │
   │   ├─ Load trained ML model
   │   ├─ Preprocess features
   │   ├─ Get prediction (BUY/SELL/NEUTRAL)
   │   └─ Get confidence (0-100%)
   │
   ├─ EXECUTE TRADE (if confidence > 65%)
   │   │
   │   ├─ Check risk limits:
   │   │   • Daily loss limit not exceeded?
   │   │   • Max positions not reached?
   │   │
   │   ├─ Calculate parameters:
   │   │   • Position size (risk-based)
   │   │   • Stop loss (2x ATR)
   │   │   • Take profit (3x ATR)
   │   │
   │   ├─ Place order via Capital.com API
   │   │
   │   └─ Save to database
   │
   └─ MONITOR POSITIONS
       │
       ├─ Check if SL/TP hit
       ├─ Close if holding > 24 hours
       └─ Update P&L in database

2. ANALYZE FEEDBACK (every 10 cycles)
   │
   ├─ For closed trades:
   │   ├─ Entry quality analysis
   │   ├─ Exit quality analysis
   │   ├─ Prediction error calculation
   │   └─ Extract lessons learned
   │
   └─ Store feedback in database

3. REPEAT ♻️
```

---

## ⚙️ Configuration

### Live Trading Bot Parameters

Edit in `live_trading_bot.py` or pass to `LiveTradingBot()`:

```python
bot = LiveTradingBot(
    instruments=['US100', 'EURUSD', 'GBPUSD'],  # Markets to trade
    min_confidence=0.65,              # Min 65% confidence to trade
    max_positions=3,                  # Max 3 concurrent positions
    risk_per_trade_percent=1.0,       # Risk 1% of capital per trade
    max_daily_loss=500.0,             # Stop trading if lose $500 in a day
    check_interval_seconds=60,        # Check markets every 60 seconds
    demo=True                         # ALWAYS TRUE for safety
)
```

### Risk Management

The bot automatically:
- **Position sizes** based on stop distance and risk %
- **Stop loss** at 2x ATR below entry (for buys)
- **Take profit** at 3x ATR above entry (for buys)
- **Daily loss limit** stops trading if exceeded
- **Maximum positions** prevents over-exposure
- **Holding time limit** closes after 24 hours

---

## 📈 Example Session

```
================================================================================
LIVE TRADING BOT STARTED
Mode: DEMO
Instruments: US100, EURUSD, GBPUSD
Models loaded: 3
Min confidence: 65%
Check interval: 60s
================================================================================

Loading 3 models...
  ✓ Loaded US100_XGBoost for US100
  ✓ Loaded EURUSD_RandomForest for EURUSD
  ✓ Loaded GBPUSD_XGBoost for GBPUSD

✓ Authenticated with Capital.com

################################################################################
CYCLE 1
################################################################################

Open Positions: 0
Daily P&L: $0.00

================================================================================
LIVE TRADING SCAN - 2025-10-29 14:23:15
================================================================================

Scanning US100...
  Signal: BUY, Confidence: 72%

============================================================
EXECUTING TRADE: US100
Signal: BUY, Confidence: 72.00%
Entry: 19543.5, Size: 0.5
SL: 19500.2, TP: 19629.8
Risk: $100.00
============================================================

✓ Trade placed: US100 BUY 0.5
✓ Trade saved to database: 7a3d92e1-4567-4890-abcd-123456789abc

Scanning EURUSD...
  Signal: NEUTRAL, Confidence: 58%

Scanning GBPUSD...
  Signal: SELL, Confidence: 63%
  Confidence 0.63 < 0.65, skipping

Sleeping for 60s...

################################################################################
CYCLE 2
################################################################################

Open Positions: 1
Daily P&L: $0.00

Position: US100 BUY 0.5 @ 19543.5, P&L: $+15.23

Sleeping for 60s...

... [continues monitoring] ...

################################################################################
CYCLE 15
################################################################################

Open Positions: 1
Daily P&L: $0.00

Position: US100 BUY 0.5 @ 19543.5, P&L: $+87.45

✓ Position closed: 7a3d92e1-4567-4890-abcd-123456789abc
✓ Trade updated with exit: ..., P&L: $87.45

Processing feedback for trade 7a3d92e1-4567-4890-abcd-123456789abc...
✓ Feedback processed for trade 7a3d92e1-4567-4890-abcd-123456789abc

... [continues] ...
```

---

## 🎯 Feedback Analysis

After each trade closes, the bot analyzes:

### Entry Quality
- Was entry at a good price?
- Position in recent range (0-1)
- Score: EXCELLENT / GOOD / POOR

### Exit Quality
- Could we have exited better?
- Did price move favorably after exit?
- Optimal: True/False

### Lessons Learned

Examples:
- "Entry timing was suboptimal - entered at 0.82 of recent range"
- "Exit was premature - price moved favorably after exit"
- "Stop loss was hit - consider wider stops or better entry timing"
- "High confidence (0.87) but trade failed - model may be overconfident"

These lessons accumulate in the database and appear in the dashboard under the **Feedback** tab.

---

## 🔄 Continuous Learning Integration

The live trading system integrates with continuous learning:

1. **Performance Monitoring**: Tracks win rate, P&L, prediction accuracy
2. **Degradation Detection**: Alerts when performance drops
3. **Auto-Retraining**: Triggers model retraining with:
   - Latest market data
   - Feedback lessons
   - Hyperparameter tuning
4. **Version Management**: Deactivates old model, activates new

To enable continuous learning alongside live trading:

**Terminal 1 - Live Trading:**
```bash
python ml_trading_bot/live_trading/live_trading_bot.py
```

**Terminal 2 - Continuous Learning:**
```bash
python ml_bot_orchestrator.py --mode continuous
```

**Terminal 3 - Dashboard:**
```bash
python ml_trading_bot/dashboard/enhanced_dashboard.py
```

---

## 📊 API Endpoints (Dashboard)

### Overview
- `GET /api/stats/overview` - System overview stats
- `GET /api/stats/daily-pnl?days=30` - Daily P&L over time

### Models
- `GET /api/models` - All models
- `GET /api/models/{id}/performance` - Performance history
- `GET /api/models/{id}/trades?days=30` - Trades for model

### Trades
- `GET /api/trades?days=7&live_only=false` - All trades
- `GET /api/trades/open` - Currently open trades

### Live Positions
- `GET /api/positions/live` - Real-time from Capital.com API

### Feedback
- `GET /api/feedback/{model_id}` - Model feedback report
- `GET /api/feedback/trade/{trade_id}` - Specific trade feedback
- `GET /api/analysis/lessons?days=30` - Common lessons learned

### Health
- `GET /api/health-check` - Model health status

### Charts
- `GET /api/charts/win-rate-trend?days=90` - Win rate trend
- `GET /api/charts/model-comparison` - Model comparison

---

## ⚠️ Safety Features

### Built-in Safeguards

1. **Demo Account Only**: Configured to use demo by default
2. **Risk Limits**:
   - Maximum daily loss ($500 default)
   - Position size caps (0.1 - 2.0 lots)
   - Risk per trade percentage (1% default)
3. **Confidence Threshold**: Only trades with 65%+ confidence
4. **Position Limits**: Maximum 3 concurrent positions
5. **Time Limits**: Closes positions after 24 hours
6. **Stop Loss**: Always included (2x ATR)

### Manual Controls

**Stop the bot**: Ctrl+C in terminal

**Close all positions manually**:
```bash
# Access Capital.com web platform or app
# Or use the API directly
```

**Disable auto-trading**:
```python
# In live_trading_bot.py, set:
max_positions = 0  # Won't open new positions
```

---

## 🐛 Troubleshooting

### "Authentication failed"
**Solution**: Check your `.env` credentials are correct

### "No models loaded"
**Solution**: Train models first:
```bash
python ml_bot_orchestrator.py --mode train --days 90
```

### "No features available"
**Solution**: Collect data first:
```bash
python ml_bot_orchestrator.py --mode init
# Then run data collection
```

### "Daily loss limit reached"
**Solution**: Bot automatically stops trading for the day. Wait for next day or increase limit.

### Dashboard shows "Live executor not configured"
**Solution**: Check `.env` file has correct Capital.com credentials

### Positions not showing in dashboard
**Solution**:
1. Ensure bot is running
2. Check trades are not simulated (`is_simulated=False`)
3. Refresh dashboard (auto-refreshes every 30s)

---

## 📈 Performance Expectations

### Typical Results (90-day training)
- **Win Rate**: 55-65% (live trades may vary from backtests)
- **Average Holding**: 4-12 hours
- **Slippage**: ~0.1% on entry/exit
- **Commission**: $2 per trade
- **Daily Trades**: 1-5 depending on signals

### Factors Affecting Performance
- **Market Conditions**: Models trained on trending markets may underperform in ranging markets
- **Volatility**: High volatility increases slippage
- **Model Freshness**: Models older than 30 days may degrade
- **Data Quality**: More data = better predictions

---

## 🎓 Best Practices

### 1. Start Small
- Use demo account first
- Start with 1-2 instruments
- Use conservative risk (0.5% per trade)

### 2. Monitor Closely
- Watch dashboard for first few hours
- Check feedback analysis
- Verify trades are executing correctly

### 3. Regular Retraining
- Retrain models monthly
- After major market events
- When performance degrades

### 4. Risk Management
- Never risk more than 2% per trade
- Keep max daily loss under 5% of capital
- Diversify across instruments

### 5. Feedback Review
- Review lessons learned weekly
- Identify recurring issues
- Adjust strategy based on feedback

---

## 📚 Advanced Usage

### Custom Trading Strategy

```python
from ml_trading_bot.live_trading import LiveTradingBot

# Custom configuration
bot = LiveTradingBot(
    instruments=['US100'],           # Trade only US100
    min_confidence=0.70,              # Higher confidence threshold
    max_positions=1,                  # Only 1 position at a time
    risk_per_trade_percent=0.5,       # Conservative risk
    max_daily_loss=250.0,             # Lower daily loss limit
    check_interval_seconds=120,       # Check every 2 minutes
    demo=True
)

# Run
asyncio.run(bot.run_continuous())
```

### Specific Model IDs

```python
# Trade only with specific models
bot = LiveTradingBot(
    instruments=['US100', 'EURUSD'],
    model_ids=[1, 3],  # Only use models with ID 1 and 3
    min_confidence=0.65,
    demo=True
)
```

### API Integration

```python
from ml_trading_bot.live_trading import CapitalTradeExecutor

executor = CapitalTradeExecutor(
    email=EMAIL,
    api_key=API_KEY,
    api_password=PASSWORD,
    demo=True
)

# Authenticate
await executor.authenticate()

# Get positions
positions = await executor.get_positions()

# Place manual trade
result = await executor.place_trade(
    epic='US100',
    direction='BUY',
    size=0.5,
    stop_loss=19500,
    take_profit=19600
)
```

---

## 🚀 Production Deployment

### For Production (Live Account)

**⚠️ WARNING: Use with extreme caution!**

1. **Extensive Testing**: Test for at least 1 month on demo
2. **Small Capital**: Start with minimal capital
3. **Monitoring**: 24/7 monitoring setup
4. **Kill Switch**: Automated shutdown on anomalies
5. **Backup**: Always have manual controls

```python
# Change to live account
bot = LiveTradingBot(
    instruments=['US100'],
    demo=False,  # ⚠️ LIVE TRADING
    max_daily_loss=100.0,  # Start very small
    risk_per_trade_percent=0.5,
    max_positions=1
)
```

---

## 📞 Support

**Issues**: See dashboard **Analysis** tab for health checks
**Logs**: Check terminal output for detailed information
**Database**: Query `trades` table for all trade history

---

## ✅ Quick Checklist

Before starting live trading:

- [ ] `.env` file configured
- [ ] Models trained (at least 1)
- [ ] Database initialized
- [ ] Demo account tested
- [ ] Dashboard accessible
- [ ] Risk limits set appropriately
- [ ] Monitoring setup
- [ ] Understanding of how bot works

---

**Built with:** Python, Capital.com API, ML Models, Sentiment Analysis

**Status:** ✅ Production-ready for demo trading

**Last Updated:** 2025-10-29
