# 🤖 Automated Trading Dashboard

An AI-powered, real-time trading dashboard for Capital.com with automated trading capabilities and comprehensive risk management.

## ⚠️ CRITICAL SAFETY WARNINGS

- **START WITH DEMO**: ALWAYS use demo account for testing
- **AUTO-TRADE IS DISABLED BY DEFAULT**: You must manually enable it
- **REAL MONEY RISK**: Live trading involves real financial risk
- **NO GUARANTEES**: Past performance doesn't guarantee future results
- **MONITOR CLOSELY**: Always supervise the automated system
- **TEST EXTENSIVELY**: Test for weeks in demo before considering live

## Features

### 🎯 Core Features

1. **Real-Time Market Monitoring**
   - Continuous analysis of configured instruments
   - Technical indicators (RSI, Moving Averages, Bollinger Bands)
   - AI-powered trade predictions with confidence scores

2. **Automated Trading**
   - Automatic trade execution based on AI signals
   - Configurable confidence thresholds
   - Position sizing based on risk parameters
   - Automatic stop-loss and take-profit placement

3. **Risk Management System**
   - Maximum daily loss limits
   - Maximum position size restrictions
   - Maximum number of open positions
   - Daily trade count limits
   - Consecutive loss protection
   - Emergency stop mechanisms

4. **Web Dashboard**
   - Real-time market signals
   - Open positions tracking
   - Daily P&L monitoring
   - Performance metrics
   - One-click controls (Start/Stop bot, Enable/Disable auto-trade)

### 🛠️ New MCP Server Tools

The enhanced MCP server now includes:

| Tool | Description |
|------|-------------|
| `place_working_order` | Place limit or stop orders at specific price levels |
| `cancel_working_order` | Cancel pending orders |
| `get_trade_history` | View closed trades with performance analytics |
| `check_market_status` | Check if markets are open and view trading hours |
| `calculate_position_size` | Calculate optimal position size based on risk |

## Installation

### 1. Update Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Trading Settings

Edit `trading_config.json` to set your risk parameters:

```json
{
  "risk_management": {
    "max_daily_loss": 500,           // Maximum loss per day before stopping
    "max_daily_trades": 10,          // Maximum trades per day
    "max_open_positions": 3,         // Maximum concurrent positions
    "max_position_size": 1.0,        // Maximum lot size per trade
    "risk_per_trade_percent": 1.0,   // Risk percentage per trade
    "use_stop_loss": true,           // Always use stop losses
    "min_risk_reward_ratio": 1.5     // Minimum R:R ratio
  },
  "trading_settings": {
    "enabled": false,                // Bot enabled on startup
    "auto_trade": false,             // Auto-trade enabled on startup
    "demo_mode": true,               // Use demo account
    "trading_instruments": [         // Markets to monitor
      "US100", "EURUSD", "GBPUSD", "US500", "GOLD"
    ],
    "check_interval_seconds": 60,    // How often to check markets
    "min_confidence_threshold": 65   // Minimum confidence to trade
  }
}
```

### 3. Launch Dashboard

```bash
python3 dashboard.py
```

The dashboard will be available at: **http://localhost:5000**

## Usage

### Starting the System

1. **Launch the dashboard:**
   ```bash
   python3 dashboard.py
   ```

2. **Open your browser:**
   Navigate to http://localhost:5000

3. **Enable the trading bot:**
   - Click "START BOT" to begin market monitoring
   - The bot will analyze markets at the configured interval
   - Signals will appear in the "Market Signals" and "Pending Signals" panels

4. **Enable auto-trading (Optional):**
   - Click "ENABLE AUTO-TRADE" to allow automatic trade execution
   - The bot will execute trades when signals meet your criteria
   - **WARNING**: Only enable after thorough testing in demo mode

### Dashboard Controls

- **START/STOP BOT**: Enable/disable market monitoring
- **ENABLE/DISABLE AUTO-TRADE**: Toggle automatic trade execution
- **REFRESH**: Manually refresh the dashboard (auto-refreshes every 15s)

### Dashboard Panels

1. **Status Panel**
   - Trading bot status (Enabled/Disabled)
   - Auto-trading status (Active/Manual)
   - Account mode (Demo/Live)
   - Daily P&L
   - Trades count
   - Open positions count
   - Last update timestamp

2. **Market Signals**
   - All monitored instruments
   - Current signal (BUY/SELL/HOLD)
   - Confidence percentage
   - RSI value

3. **Pending Signals**
   - Signals above confidence threshold
   - Entry price
   - Stop loss level
   - Take profit target

4. **Open Positions**
   - Current trades
   - Direction (BUY/SELL)
   - Position size
   - Entry level
   - Current price
   - Unrealized P&L

5. **Risk Limits**
   - Current risk configuration
   - Maximum loss limits
   - Position size limits
   - Confidence thresholds

## Configuration Guide

### Risk Management Parameters

**max_daily_loss**: Maximum amount you're willing to lose in a day
- Recommended: 1-2% of account balance
- Example: For $10,000 account, set to 100-200

**max_daily_trades**: Limit on number of trades per day
- Prevents overtrading
- Recommended: 5-15 trades/day

**max_open_positions**: Maximum concurrent positions
- Prevents overexposure
- Recommended: 2-4 positions

**max_position_size**: Maximum lot size per trade
- Capital.com minimum is usually 0.1 lots
- Start small (0.1-0.5) for testing

**risk_per_trade_percent**: Percentage of account to risk per trade
- Industry standard: 1-2%
- Conservative: 0.5-1%
- Aggressive: 2-3% (not recommended)

### Trading Settings

**check_interval_seconds**: How often to scan markets
- 60 seconds for active trading
- 300 seconds (5 min) for swing trading
- Lower values = more API calls

**min_confidence_threshold**: Minimum signal strength to trade
- 50-60%: More trades, lower quality
- 65-75%: Balanced approach (recommended)
- 75%+: Fewer trades, higher quality

**trading_instruments**: Markets to monitor
- Start with 3-5 instruments
- Mix different asset classes
- Avoid highly correlated pairs

## Safety Features

### Automatic Protections

1. **Daily Loss Limit**
   - Trading stops when max daily loss is reached
   - Requires manual restart next day

2. **Position Limits**
   - Prevents opening new trades at position limit
   - Helps manage exposure

3. **Market Status Check**
   - Only trades when markets are open
   - Prevents orders during market close

4. **Consecutive Loss Protection**
   - Pauses trading after X consecutive losses
   - Default: 3 consecutive losses

5. **Session Management**
   - Automatic re-authentication
   - Handles API timeouts gracefully

### Manual Controls

- **Emergency Stop**: Click "STOP BOT" immediately stops all monitoring
- **Disable Auto-Trade**: Keeps monitoring but prevents new trades
- **Manual Position Management**: Use MCP tools to close positions manually

## Monitoring & Logs

The dashboard logs all activity to the console:

```
2025-10-27 10:30:15 - INFO - 🤖 Trading bot started
2025-10-27 10:30:16 - INFO - Session authenticated successfully
2025-10-27 10:30:45 - INFO - 🎯 Signal: BUY US100 at 72.5% confidence
2025-10-27 10:30:46 - INFO - ✅ Trade executed successfully
2025-10-27 10:31:45 - INFO - 💤 Sleeping for 60 seconds...
```

Monitor logs for:
- Authentication issues
- Trade execution results
- Risk limit triggers
- API errors

## Using New MCP Tools

### Example 1: Place Limit Order

Ask Claude in Claude Desktop:

```
"Place a buy limit order on EURUSD at 1.0850 with size 0.5,
stop 50 points away, and take profit 100 points away"
```

### Example 2: Check Trading Hours

```
"Check if the US100 market is currently open for trading"
```

### Example 3: Calculate Position Size

```
"Calculate the optimal position size for EURUSD if I want to
enter at 1.0900 with stop loss at 1.0850, risking 1% of my account"
```

### Example 4: View Trade History

```
"Show me my trade history with performance metrics"
```

## Troubleshooting

### Bot Not Starting

1. Check API credentials in `.env`
2. Verify `trading_config.json` exists and is valid
3. Ensure demo mode is set correctly
4. Check console for error messages

### No Signals Appearing

1. Check if markets are open
2. Verify instruments in config are valid
3. Lower confidence threshold temporarily
4. Check API rate limits

### Trades Not Executing

1. Verify auto-trade is enabled
2. Check risk limits aren't exceeded
3. Ensure sufficient account balance
4. Check market status (open/closed)
5. Review position size calculations

### Dashboard Not Loading

1. Check if port 5000 is available
2. Verify Flask is installed
3. Check firewall settings
4. Try accessing http://127.0.0.1:5000

## Best Practices

### Testing Phase (Recommended: 2-4 weeks)

1. **Week 1**: Monitor only, no auto-trade
   - Watch signals vs actual market movement
   - Verify risk calculations
   - Test all controls

2. **Week 2**: Enable auto-trade with very low limits
   - Max 0.1 lot size
   - Max 2-3 trades per day
   - Max $50 daily loss
   - Min 75% confidence

3. **Week 3-4**: Gradually increase limits if performing well
   - Track win rate (aim for >50%)
   - Monitor profit factor (aim for >1.5)
   - Review trade quality

### Going Live (If Applicable)

1. **Never skip demo testing**
2. **Start with absolute minimum limits**
3. **Monitor daily for first month**
4. **Keep detailed trading journal**
5. **Review and adjust monthly**
6. **Have emergency stop plan**

### General Guidelines

- Check dashboard at least 2x daily
- Review closed trades weekly
- Adjust config based on performance
- Never increase limits after losses
- Take breaks after big wins/losses
- Keep software updated

## Performance Metrics

Track these metrics to evaluate performance:

- **Win Rate**: Winning trades / Total trades (Target: >50%)
- **Profit Factor**: Total wins / Total losses (Target: >1.5)
- **Average Win/Loss**: Average profit vs average loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns

Access via: `get_trade_history` MCP tool

## API Rate Limits

Capital.com has rate limits. The bot handles this by:
- Spacing market checks (default 60s)
- Adding delays between instrument scans
- Automatic retry on failures

If you hit limits:
- Increase `check_interval_seconds`
- Reduce number of `trading_instruments`
- Add delays in code if needed

## Security

- Never commit `.env` file
- Use environment variables for credentials
- Rotate API keys regularly
- Use read-only keys for monitoring
- Restrict IP access if possible
- Keep system updated

## Support

- **Capital.com API**: [capital.com/api-documentation](https://capital.com/api-documentation)
- **Issues**: Report bugs via GitHub issues
- **Logs**: Check console output for debugging

## Disclaimer

**THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.**

- Trading financial instruments involves substantial risk of loss
- Past performance is not indicative of future results
- The AI/algorithms are for educational purposes only
- No guarantee of profitability
- Author not responsible for financial losses
- Always consult a licensed financial advisor

**By using this software, you acknowledge that you understand and accept all risks involved.**

## License

MIT License - See LICENSE file

---

**Made with ⚡ for Capital.com | Powered by AI**
