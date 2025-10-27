#!/usr/bin/env python3
"""
Capital.com Automated Trading Dashboard
Real-time monitoring and AI-powered automated trading
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import httpx
from dotenv import load_dotenv
from flask import Flask, render_template_string, jsonify, request
from threading import Thread
from market_research import researcher

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Capital.com API configuration
API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true"

if USE_DEMO:
    API_URL = "https://demo-api-capital.backend-capital.com"
else:
    API_URL = "https://api-capital.backend-capital.com"

# Session management
session_token = None
cst_token = None
session_expiry = None

# Trading state
trading_state = {
    "enabled": False,
    "auto_trade": False,
    "positions": [],
    "daily_pnl": 0,
    "trades_today": 0,
    "consecutive_losses": 0,
    "last_update": None,
    "market_data": {},
    "pending_signals": [],
    "research_data": {},
    "ai_decisions": {},
    "pre_market_done": False,
    "last_research_time": None
}

# Load config
def load_config():
    """Load trading configuration"""
    try:
        with open('trading_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None

config = load_config()

# Flask app
app = Flask(__name__)

# ========== CAPITAL.COM API FUNCTIONS ==========

async def get_session():
    """Get or refresh session tokens"""
    global session_token, cst_token, session_expiry

    if session_token and session_expiry and datetime.now() < session_expiry:
        return session_token, cst_token

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/v1/session",
                headers={"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"},
                json={"identifier": IDENTIFIER, "password": API_PASSWORD, "encryptedPassword": False}
            )

            if response.status_code == 200:
                session_token = response.headers.get("X-SECURITY-TOKEN")
                cst_token = response.headers.get("CST")
                session_expiry = datetime.now() + timedelta(minutes=9)
                logger.info("Session authenticated successfully")
                return session_token, cst_token
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return None, None
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None, None


def get_headers(security_token: str, cst: str) -> dict:
    """Generate headers for authenticated requests"""
    return {
        "X-SECURITY-TOKEN": security_token,
        "CST": cst,
        "Content-Type": "application/json"
    }


# ========== TECHNICAL ANALYSIS ==========

def calculate_rsi(prices: list, period: int = 14) -> Optional[float]:
    """Calculate RSI"""
    if len(prices) < period + 1:
        return None

    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_moving_averages(prices: list) -> dict:
    """Calculate moving averages"""
    return {
        "ma_20": sum(prices[-20:]) / 20 if len(prices) >= 20 else None,
        "ma_50": sum(prices[-50:]) / 50 if len(prices) >= 50 else None,
    }


async def analyze_instrument(epic: str, security_token: str, cst: str) -> Dict:
    """Perform technical analysis on instrument"""
    headers = get_headers(security_token, cst)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get market data
            market_response = await client.get(
                f"{API_URL}/api/v1/markets/{epic}",
                headers=headers
            )

            if market_response.status_code != 200:
                return {"error": f"Failed to fetch market data for {epic}"}

            market_data = market_response.json()
            snapshot = market_data.get('snapshot', {})

            # Check if market is open
            market_status = snapshot.get('marketStatus', 'UNKNOWN')
            if market_status not in ['TRADEABLE', 'OPEN']:
                return {"error": f"Market {epic} is closed", "status": market_status}

            current_bid = snapshot.get('bid', 0)
            current_offer = snapshot.get('offer', 0)
            current_price = (current_bid + current_offer) / 2

            # Get historical data
            price_response = await client.get(
                f"{API_URL}/api/v1/prices/{epic}",
                headers=headers,
                params={"resolution": "HOUR", "max": 200}
            )

            if price_response.status_code != 200:
                return {"error": f"Failed to fetch price history for {epic}"}

            price_data = price_response.json()
            closes = [p['closePrice']['ask'] for p in price_data['prices'] if 'closePrice' in p]

            if len(closes) < 50:
                return {"error": f"Insufficient data for {epic}"}

            # Calculate indicators
            rsi = calculate_rsi(closes, 14)
            mas = calculate_moving_averages(closes)

            # Generate signal
            bullish_score = 0
            bearish_score = 0

            if mas["ma_20"] and mas["ma_50"]:
                if mas["ma_20"] > mas["ma_50"]:
                    bullish_score += 2
                else:
                    bearish_score += 2

            if rsi:
                if rsi < 30:
                    bullish_score += 3
                elif rsi > 70:
                    bearish_score += 3

            total_score = bullish_score + bearish_score
            if total_score == 0:
                prediction = "HOLD"
                confidence = 0
            else:
                if bullish_score > bearish_score:
                    prediction = "BUY"
                    confidence = min((bullish_score / (total_score + 5)) * 100, 95)
                elif bearish_score > bullish_score:
                    prediction = "SELL"
                    confidence = min((bearish_score / (total_score + 5)) * 100, 95)
                else:
                    prediction = "HOLD"
                    confidence = 50

            # Calculate suggested levels
            if prediction == "BUY":
                entry = current_offer
                stop_loss = entry - (entry * 0.015)
                take_profit = entry + (entry * 0.03)
            elif prediction == "SELL":
                entry = current_bid
                stop_loss = entry + (entry * 0.015)
                take_profit = entry - (entry * 0.03)
            else:
                entry = current_price
                stop_loss = None
                take_profit = None

            return {
                "epic": epic,
                "instrument": market_data.get('instrument', {}).get('name', epic),
                "current_price": current_price,
                "prediction": prediction,
                "confidence": round(confidence, 1),
                "rsi": round(rsi, 2) if rsi else None,
                "ma_20": round(mas["ma_20"], 2) if mas["ma_20"] else None,
                "ma_50": round(mas["ma_50"], 2) if mas["ma_50"] else None,
                "entry": round(entry, 2) if entry else None,
                "stop_loss": round(stop_loss, 2) if stop_loss else None,
                "take_profit": round(take_profit, 2) if take_profit else None,
                "market_status": market_status
            }

    except Exception as e:
        logger.error(f"Error analyzing {epic}: {e}")
        return {"error": str(e)}


# ========== TRADING EXECUTION ==========

async def execute_trade(epic: str, direction: str, size: float, stop_loss: float, take_profit: float):
    """Execute a trade"""
    security_token, cst = await get_session()
    if not security_token:
        return {"error": "Authentication failed"}

    headers = get_headers(security_token, cst)

    payload = {
        "epic": epic,
        "direction": direction,
        "size": size,
        "stopLevel": stop_loss,
        "profitLevel": take_profit
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_URL}/api/v1/positions",
                headers=headers,
                json=payload
            )

            if response.status_code in [200, 201]:
                data = response.json()
                logger.info(f"Trade executed: {direction} {size} {epic} at {stop_loss}/{take_profit}")
                trading_state["trades_today"] += 1
                return {"success": True, "data": data}
            else:
                logger.error(f"Trade failed: {response.status_code} - {response.text}")
                return {"error": response.text}

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return {"error": str(e)}


async def get_account_info():
    """Get account information"""
    security_token, cst = await get_session()
    if not security_token:
        return None

    headers = get_headers(security_token, cst)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_URL}/api/v1/accounts",
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('accounts', [{}])[0]
            return None

    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        return None


async def get_positions():
    """Get open positions"""
    security_token, cst = await get_session()
    if not security_token:
        return []

    headers = get_headers(security_token, cst)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_URL}/api/v1/positions",
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('positions', [])
            return []

    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return []


def check_risk_limits() -> Dict:
    """Check if trading is allowed based on risk limits"""
    if not config:
        return {"allowed": False, "reason": "Config not loaded"}

    risk = config['risk_management']
    emergency = config['emergency']

    # Check daily loss
    if abs(trading_state['daily_pnl']) >= risk['max_daily_loss']:
        return {"allowed": False, "reason": f"Daily loss limit reached: {trading_state['daily_pnl']}"}

    # Check max trades
    if trading_state['trades_today'] >= risk['max_daily_trades']:
        return {"allowed": False, "reason": f"Max daily trades reached: {trading_state['trades_today']}"}

    # Check max positions
    if len(trading_state['positions']) >= risk['max_open_positions']:
        return {"allowed": False, "reason": f"Max open positions reached: {len(trading_state['positions'])}"}

    # Check consecutive losses
    if trading_state['consecutive_losses'] >= emergency['pause_trading_on_consecutive_losses']:
        return {"allowed": False, "reason": f"Too many consecutive losses: {trading_state['consecutive_losses']}"}

    return {"allowed": True}


# ========== AI TRADING BOT ==========

async def trading_bot_loop():
    """Main AI trading bot loop with AI research integration"""
    logger.info("🤖 Trading bot started with AI research capabilities")

    while True:
        try:
            if not config or not trading_state["enabled"]:
                await asyncio.sleep(5)
                continue

            # ========== PRE-MARKET RESEARCH ==========
            # Run once per day at configured time
            current_hour = datetime.now().hour
            pre_market_time = int(config.get('ai_research', {}).get('pre_market_research_time', '08:00').split(':')[0])

            if config.get('ai_research', {}).get('enabled') and not trading_state['pre_market_done']:
                if current_hour >= pre_market_time:
                    logger.info("🔍 Running pre-market AI research...")
                    instruments = config['trading_settings']['trading_instruments']

                    research_results = await researcher.pre_market_research(instruments)
                    trading_state['research_data'] = research_results
                    trading_state['pre_market_done'] = True
                    trading_state['last_research_time'] = datetime.now().isoformat()

                    logger.info("✅ Pre-market research complete!")

            # Reset flag at midnight
            if current_hour == 0 and trading_state['pre_market_done']:
                trading_state['pre_market_done'] = False

            # Get session
            security_token, cst = await get_session()
            if not security_token:
                logger.error("Failed to authenticate")
                await asyncio.sleep(30)
                continue

            # Update account info
            account = await get_account_info()
            if account:
                balance = account.get('balance', {})
                trading_state['daily_pnl'] = balance.get('profitLoss', 0)

            # Update positions
            trading_state['positions'] = await get_positions()

            # Scan markets
            instruments = config['trading_settings']['trading_instruments']
            min_confidence = config['trading_settings']['min_confidence_threshold']

            trading_state['market_data'] = {}
            trading_state['pending_signals'] = []

            for epic in instruments:
                # Algorithmic analysis first
                analysis = await analyze_instrument(epic, security_token, cst)

                if "error" not in analysis:
                    trading_state['market_data'][epic] = analysis

                    # ========== AI DECISION MAKING ==========
                    # Use Claude for borderline signals or important decisions
                    use_ai = config.get('ai_research', {}).get('enabled', False)
                    use_claude = config.get('ai_research', {}).get('use_claude', False)

                    if use_ai and use_claude and researcher.should_use_claude(analysis):
                        logger.info(f"🧠 Calling Claude AI for {epic} analysis...")

                        # Generate comprehensive trading plan with AI
                        ai_plan = await researcher.generate_trading_plan(epic, analysis)

                        if not ai_plan.get('error'):
                            # Store AI decision
                            trading_state['ai_decisions'][epic] = ai_plan

                            # Override algorithmic decision with AI decision
                            analysis['prediction'] = ai_plan['decision']
                            analysis['confidence'] = ai_plan['confidence']
                            analysis['entry'] = ai_plan.get('entry', analysis.get('entry'))
                            analysis['stop_loss'] = ai_plan.get('stop_loss', analysis.get('stop_loss'))
                            analysis['take_profit'] = ai_plan.get('take_profit', analysis.get('take_profit'))
                            analysis['ai_reasoning'] = ai_plan.get('reasoning', [])

                            logger.info(f"🎯 AI Decision: {ai_plan['decision']} at {ai_plan['confidence']}% confidence")

                    # Check if signal is strong enough
                    if analysis['prediction'] != "HOLD" and analysis['confidence'] >= min_confidence:
                        trading_state['pending_signals'].append(analysis)

                        logger.info(f"🎯 Signal: {analysis['prediction']} {epic} at {analysis['confidence']}% confidence")

                        # Execute if auto-trade is enabled
                        if trading_state['auto_trade']:
                            risk_check = check_risk_limits()
                            if risk_check['allowed']:
                                # Calculate position size
                                risk_amount = balance.get('balance', 10000) * (config['risk_management']['risk_per_trade_percent'] / 100)
                                stop_distance = abs(analysis['entry'] - analysis['stop_loss']) if analysis.get('entry') and analysis.get('stop_loss') else 100
                                position_size = min(risk_amount / stop_distance, config['risk_management']['max_position_size']) if stop_distance > 0 else 0.1

                                position_size = round(position_size, 2)

                                if position_size > 0:
                                    result = await execute_trade(
                                        epic,
                                        analysis['prediction'],
                                        position_size,
                                        analysis['stop_loss'],
                                        analysis['take_profit']
                                    )

                                    if result.get('success'):
                                        logger.info(f"✅ Trade executed successfully")
                                    else:
                                        logger.error(f"❌ Trade failed: {result.get('error')}")
                            else:
                                logger.warning(f"⚠️ Trade blocked: {risk_check['reason']}")

                await asyncio.sleep(2)  # Rate limiting

            trading_state['last_update'] = datetime.now().isoformat()

            # Wait for next cycle
            check_interval = config['trading_settings']['check_interval_seconds']
            logger.info(f"💤 Sleeping for {check_interval} seconds...")
            await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Error in trading bot loop: {e}")
            await asyncio.sleep(30)


# ========== WEB DASHBOARD ==========

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Capital.com Trading Dashboard</title>
    <meta http-equiv="refresh" content="15">
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #00ff00;
            text-align: center;
            font-size: 2em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff00;
        }
        .status-panel {
            background: #1a1a1a;
            border: 2px solid #00ff00;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .status-enabled {
            color: #00ff00;
            font-weight: bold;
        }
        .status-disabled {
            color: #ff0000;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: #1a1a1a;
            border: 1px solid #00ff00;
            padding: 15px;
            border-radius: 5px;
        }
        .panel h2 {
            color: #00ffff;
            margin-top: 0;
            font-size: 1.3em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        th {
            color: #00ffff;
        }
        .buy {
            color: #00ff00;
        }
        .sell {
            color: #ff5555;
        }
        .hold {
            color: #ffff00;
        }
        .controls {
            text-align: center;
            margin: 30px 0;
        }
        button {
            background: #1a1a1a;
            border: 2px solid #00ff00;
            color: #00ff00;
            padding: 15px 30px;
            margin: 10px;
            font-size: 1.2em;
            cursor: pointer;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }
        button:hover {
            background: #00ff00;
            color: #0a0a0a;
        }
        .warning {
            color: #ffff00;
            font-weight: bold;
        }
        .error {
            color: #ff0000;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 CAPITAL.COM AI TRADING DASHBOARD</h1>

        <div class="status-panel">
            <div class="info-row">
                <span>Trading Bot:</span>
                <span class="{{ 'status-enabled' if state.enabled else 'status-disabled' }}">
                    {{ 'ENABLED' if state.enabled else 'DISABLED' }}
                </span>
            </div>
            <div class="info-row">
                <span>Auto-Trading:</span>
                <span class="{{ 'status-enabled' if state.auto_trade else 'status-disabled' }}">
                    {{ 'ACTIVE' if state.auto_trade else 'MANUAL' }}
                </span>
            </div>
            <div class="info-row">
                <span>Mode:</span>
                <span>{{ 'DEMO ACCOUNT' if demo else 'LIVE ACCOUNT' }}</span>
            </div>
            <div class="info-row">
                <span>Daily P&L:</span>
                <span class="{{ 'buy' if state.daily_pnl > 0 else 'sell' if state.daily_pnl < 0 else '' }}">
                    {{ state.daily_pnl }}
                </span>
            </div>
            <div class="info-row">
                <span>Trades Today:</span>
                <span>{{ state.trades_today }} / {{ config.risk_management.max_daily_trades }}</span>
            </div>
            <div class="info-row">
                <span>Open Positions:</span>
                <span>{{ state.positions|length }} / {{ config.risk_management.max_open_positions }}</span>
            </div>
            <div class="info-row">
                <span>Last Update:</span>
                <span>{{ state.last_update or 'Never' }}</span>
            </div>
        </div>

        <div class="controls">
            <button onclick="toggleBot()">{{ 'STOP BOT' if state.enabled else 'START BOT' }}</button>
            <button onclick="toggleAutoTrade()">{{ 'DISABLE AUTO-TRADE' if state.auto_trade else 'ENABLE AUTO-TRADE' }}</button>
            <button onclick="location.reload()">REFRESH</button>
        </div>

        <div class="grid">
            <div class="panel">
                <h2>📊 MARKET SIGNALS</h2>
                <table>
                    <tr>
                        <th>Instrument</th>
                        <th>Signal</th>
                        <th>Confidence</th>
                        <th>RSI</th>
                    </tr>
                    {% for epic, data in state.market_data.items() %}
                    <tr>
                        <td>{{ epic }}</td>
                        <td class="{{ data.prediction.lower() }}">{{ data.prediction }}</td>
                        <td>{{ data.confidence }}%</td>
                        <td>{{ data.rsi }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            <div class="panel">
                <h2>🎯 PENDING SIGNALS</h2>
                <table>
                    <tr>
                        <th>Instrument</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>Stop</th>
                        <th>Target</th>
                    </tr>
                    {% for signal in state.pending_signals %}
                    <tr>
                        <td>{{ signal.epic }}</td>
                        <td class="{{ signal.prediction.lower() }}">{{ signal.prediction }}</td>
                        <td>{{ signal.entry }}</td>
                        <td>{{ signal.stop_loss }}</td>
                        <td>{{ signal.take_profit }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>

        <div class="panel">
            <h2>📈 OPEN POSITIONS</h2>
            <table>
                <tr>
                    <th>Instrument</th>
                    <th>Direction</th>
                    <th>Size</th>
                    <th>Open Level</th>
                    <th>Current</th>
                    <th>P&L</th>
                </tr>
                {% for pos in state.positions %}
                <tr>
                    <td>{{ pos.market.instrumentName }}</td>
                    <td class="{{ 'buy' if pos.position.direction == 'BUY' else 'sell' }}">
                        {{ pos.position.direction }}
                    </td>
                    <td>{{ pos.position.size }}</td>
                    <td>{{ pos.position.level }}</td>
                    <td>{{ pos.market.bid if pos.position.direction == 'SELL' else pos.market.offer }}</td>
                    <td class="{{ 'buy' if pos.position.profit > 0 else 'sell' }}">
                        {{ pos.position.profit }}
                    </td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <div class="status-panel warning">
            <h3>⚠️ RISK LIMITS</h3>
            <div class="info-row">
                <span>Max Daily Loss:</span>
                <span>{{ config.risk_management.max_daily_loss }}</span>
            </div>
            <div class="info-row">
                <span>Max Position Size:</span>
                <span>{{ config.risk_management.max_position_size }}</span>
            </div>
            <div class="info-row">
                <span>Risk Per Trade:</span>
                <span>{{ config.risk_management.risk_per_trade_percent }}%</span>
            </div>
            <div class="info-row">
                <span>Min Confidence:</span>
                <span>{{ config.trading_settings.min_confidence_threshold }}%</span>
            </div>
        </div>
    </div>

    <script>
        function toggleBot() {
            fetch('/api/toggle_bot', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    location.reload();
                });
        }

        function toggleAutoTrade() {
            fetch('/api/toggle_auto_trade', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    location.reload();
                });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Render dashboard"""
    return render_template_string(
        DASHBOARD_HTML,
        state=trading_state,
        config=config or {},
        demo=USE_DEMO
    )


@app.route('/api/toggle_bot', methods=['POST'])
def toggle_bot():
    """Toggle trading bot on/off"""
    trading_state['enabled'] = not trading_state['enabled']
    status = "enabled" if trading_state['enabled'] else "disabled"
    logger.info(f"Trading bot {status}")
    return jsonify({"message": f"Trading bot {status}", "enabled": trading_state['enabled']})


@app.route('/api/toggle_auto_trade', methods=['POST'])
def toggle_auto_trade():
    """Toggle auto-trading on/off"""
    trading_state['auto_trade'] = not trading_state['auto_trade']
    status = "enabled" if trading_state['auto_trade'] else "disabled"
    logger.info(f"Auto-trading {status}")
    return jsonify({"message": f"Auto-trading {status}", "auto_trade": trading_state['auto_trade']})


@app.route('/api/state')
def get_state():
    """Get current trading state"""
    return jsonify(trading_state)


def run_flask():
    """Run Flask app"""
    app.run(host='0.0.0.0', port=5000, debug=False)


def run_bot():
    """Run trading bot in asyncio loop"""
    asyncio.run(trading_bot_loop())


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🚀 CAPITAL.COM AUTOMATED TRADING DASHBOARD")
    logger.info("=" * 70)
    logger.info(f"Mode: {'DEMO' if USE_DEMO else 'LIVE'} Account")
    logger.info(f"Dashboard: http://localhost:5000")
    logger.info("=" * 70)

    if not config:
        logger.error("Failed to load configuration. Exiting.")
        exit(1)

    # Start Flask in separate thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Run bot in main thread
    try:
        run_bot()
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down...")
