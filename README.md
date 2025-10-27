# Capital.com MCP Server

A Model Context Protocol (MCP) server that connects Claude Desktop to Capital.com's trading API, enabling market analysis and trade execution through natural language.

## ⚠️ Important Safety Warnings

- **START WITH DEMO ACCOUNT**: Always use the demo account first to test functionality
- **REAL MONEY RISK**: Switching to live mode involves real money and real risk
- **NO GUARANTEES**: Trading carries substantial risk of loss
- **REVIEW TRADES**: Always review trade suggestions before executing
- **TEST THOROUGHLY**: Test extensively in demo mode before considering live trading

## Features

### 🤖 NEW: Automated Trading Dashboard
- **AI-Powered Trading Bot**: Continuous market monitoring with automated trade execution
- **Real-Time Dashboard**: Web interface at http://localhost:5000
- **Risk Management**: Configurable limits on losses, position sizes, and trade frequency
- **Smart Signals**: AI predictions with confidence scores for multiple instruments
- **One-Click Controls**: Start/stop bot and enable/disable auto-trading
- 📖 **[Full Dashboard Documentation](DASHBOARD_README.md)**

### Market Analysis Tools
- 🔍 **Search Markets**: Find tradeable instruments (forex, indices, commodities, stocks, crypto)
- 📊 **Market Details**: Get real-time prices, spreads, and market information
- 📈 **Historical Data**: Fetch OHLC candlestick data for technical analysis
- 💰 **Account Info**: Check balance, P&L, and available funds
- 🔮 **Trade Predictions**: AI-powered technical analysis with confidence scoring
- 📍 **Market Status**: Check if markets are open and view trading hours
- 💱 **Position Sizing**: Calculate optimal position size based on risk parameters

### Trading Tools (Use with Caution)
- 📍 **Market Orders**: Place market orders with stop-loss and take-profit
- 📋 **Limit/Stop Orders**: Place pending orders at specific price levels
- ❌ **Close Positions**: Exit existing trades
- ✏️ **Update Positions**: Modify stop-loss and take-profit levels
- 📊 **Trade History**: View closed trades with performance analytics (win rate, profit factor, etc.)
- 🧮 **Risk Calculator**: Calculate position sizes based on account balance and risk tolerance

## Prerequisites

1. **Capital.com Account**
   - Sign up at [capital.com](https://capital.com)
   - Create a demo account (recommended for testing)
   - Generate API credentials from your account settings

2. **Python 3.10+**
   - Check: `python3 --version`

3. **Claude Desktop**
   - Download from [claude.ai](https://claude.ai/download)

## Installation

### 1. Clone or Create Project Directory

```bash
cd capital-mcp-server
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Credentials

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Capital.com credentials
nano .env  # or use your preferred editor
```

Update `.env` with your credentials:

```env
CAPITAL_API_KEY=your_actual_api_key
CAPITAL_API_PASSWORD=your_actual_password
CAPITAL_USE_DEMO=true  # Keep as true for demo account
```

### 4. Test the Server

```bash
# Make the server executable
chmod +x server.py

# Test run (optional)
python3 server.py
```

Press `Ctrl+C` to stop if running standalone.

### 5. Connect to Claude Desktop

#### macOS
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "capital-com": {
      "command": "/full/path/to/capital-mcp-server/venv/bin/python3",
      "args": ["/full/path/to/capital-mcp-server/server.py"],
      "env": {
        "CAPITAL_API_KEY": "your_api_key",
        "CAPITAL_API_PASSWORD": "your_password",
        "CAPITAL_USE_DEMO": "true"
      }
    }
  }
}
```

#### Windows
Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "capital-com": {
      "command": "C:\\full\\path\\to\\capital-mcp-server\\venv\\Scripts\\python.exe",
      "args": ["C:\\full\\path\\to\\capital-mcp-server\\server.py"],
      "env": {
        "CAPITAL_API_KEY": "your_api_key",
        "CAPITAL_API_PASSWORD": "your_password",
        "CAPITAL_USE_DEMO": "true"
      }
    }
  }
}
```

#### Linux
Edit `~/.config/Claude/claude_desktop_config.json` (similar to macOS).

**Important**: Replace `/full/path/to/` with the actual absolute path to your project directory.

### 6. Restart Claude Desktop

Close and reopen Claude Desktop completely. The Capital.com tools should now appear in the MCP tools section.

## Usage Examples

Once connected, you can interact with Claude using natural language:

### Market Research
```
"Search for EUR/USD and show me the current price"
"What's the latest price of gold?"
"Find all Bitcoin trading pairs"
"Show me details for the S&P 500"
```

### Technical Analysis
```
"Get the last 100 daily candles for EUR/USD and analyze the trend"
"Show me 4-hour price history for Bitcoin and identify support levels"
"Analyze the RSI and moving averages for US500"
"What's the volatility like on Gold in the past week?"
```

### Account Management
```
"Show my account balance"
"What positions do I have open?"
"List all my pending orders"
"What's my total P&L?"
```

### Trading (Demo Account Recommended)
```
"I want to buy 0.1 lots of EUR/USD with stop loss at 1.0850"
"Close my EUR/USD position"
"Update my position to move stop loss to 1.0900"
"Place a sell order on US500 with 50 point stop loss"
```

## Available Tools

| Tool | Description | Risk Level |
|------|-------------|------------|
| `search_markets` | Find tradeable instruments | ✅ Safe |
| `get_market_details` | Get price and market info | ✅ Safe |
| `get_price_history` | Fetch historical OHLC data | ✅ Safe |
| `get_account_info` | View account balance | ✅ Safe |
| `get_positions` | List open positions | ✅ Safe |
| `get_working_orders` | List pending orders | ✅ Safe |
| `predict_trade` | AI trade prediction with confidence | ✅ Safe |
| `check_market_status` | Check if market is open | ✅ Safe |
| `get_trade_history` | View trade history & analytics | ✅ Safe |
| `calculate_position_size` | Calculate optimal position size | ✅ Safe |
| `place_position` | Open new market order | ⚠️ CAUTION |
| `place_working_order` | Place limit/stop order | ⚠️ CAUTION |
| `close_position` | Close existing trade | ⚠️ CAUTION |
| `update_position` | Modify stop/profit levels | ⚠️ CAUTION |
| `cancel_working_order` | Cancel pending order | ⚠️ CAUTION |

## Automated Trading Dashboard

### Quick Start

Launch the AI-powered trading dashboard:

```bash
python3 dashboard.py
```

Access at: **http://localhost:5000**

### Features

- 🤖 **AI Trading Bot**: Continuous market analysis across multiple instruments
- 📊 **Real-Time Dashboard**: Live positions, P&L, and market signals
- ⚙️ **Risk Controls**: Configurable loss limits, position sizing, and safety stops
- 📈 **Performance Tracking**: Win rate, profit factor, and trade analytics
- 🎯 **Smart Signals**: Only trades high-confidence setups (configurable threshold)

### Configuration

Edit `trading_config.json` to customize:

```json
{
  "risk_management": {
    "max_daily_loss": 500,           // Stop trading at this loss
    "max_position_size": 1.0,        // Maximum lot size
    "risk_per_trade_percent": 1.0    // Risk 1% per trade
  },
  "trading_settings": {
    "auto_trade": false,             // Manual approval required
    "trading_instruments": [         // Markets to monitor
      "US100", "EURUSD", "GBPUSD"
    ],
    "min_confidence_threshold": 65   // Minimum signal strength
  }
}
```

### Safety Features

- ✋ **Emergency Stop**: One-click bot shutdown
- 🛡️ **Daily Loss Limits**: Auto-stops at configured loss
- 📊 **Position Limits**: Maximum concurrent positions
- ⚠️ **Consecutive Loss Protection**: Pauses after X losses
- 🔍 **Market Hours Check**: Only trades when markets are open
- 💰 **Position Sizing**: Automatic risk-based position calculation

**[📖 Full Dashboard Documentation](DASHBOARD_README.md)**

## Session Management

- Sessions automatically expire after **10 minutes of inactivity**
- The server automatically creates new sessions when needed
- If you see authentication errors, the server will reconnect automatically

## Troubleshooting

### Server Not Appearing in Claude Desktop

1. Check that the config file path is correct
2. Verify the Python path points to your virtual environment
3. Ensure the server.py path is absolute, not relative
4. Look for errors in Claude Desktop's log files:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`

### Authentication Errors

1. Verify your API credentials in `.env`
2. Check that `CAPITAL_USE_DEMO=true` for demo accounts
3. Ensure your Capital.com account is active
4. Try generating new API credentials

### API Errors

- **401 Unauthorized**: Check API credentials
- **403 Forbidden**: Verify account permissions
- **408 Timeout**: Session expired, will auto-reconnect
- **Market closed**: Check trading hours for the instrument

## Security Best Practices

1. **Never commit `.env` file** - It contains your credentials
2. **Use demo account first** - Always test with demo before live
3. **Rotate API keys regularly** - Generate new keys periodically
4. **Monitor permissions** - Review what API keys can access
5. **Use stop losses** - Always set stop losses on trades
6. **Start small** - Test with minimal position sizes

## API Rate Limits

Capital.com API has rate limits. The server handles:
- Automatic session management
- Request timeout handling
- Error response parsing

## Support

- **Capital.com API Docs**: [capital.com/api-documentation](https://capital.com/api-documentation)
- **MCP Documentation**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Issues**: Report bugs or request features via GitHub issues

## Disclaimer

This software is provided "as is" without warranty. Trading financial instruments involves risk. Past performance is not indicative of future results. The authors are not responsible for any financial losses incurred through use of this software.

**Always consult with a licensed financial advisor before making trading decisions.**

## License

MIT License - See LICENSE file for details

---

**Made for Claude Desktop** | **Powered by Capital.com API**
