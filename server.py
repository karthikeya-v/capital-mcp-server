#!/usr/bin/env python3
"""
Capital.com MCP Server
Provides Claude with tools to interact with Capital.com trading API
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Capital.com API configuration
API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true"

# Select appropriate base URL
if USE_DEMO:
    API_URL = "https://demo-api-capital.backend-capital.com"
else:
    API_URL = "https://api-capital.backend-capital.com"

# Session management
session_token = None
cst_token = None
session_expiry = None

# Initialize MCP server
app = Server("capital-com")


async def get_session():
    """
    Authenticate with Capital.com API and get session tokens.
    Sessions expire after 10 minutes of inactivity.
    """
    global session_token, cst_token, session_expiry

    # Check if we have a valid session
    if session_token and session_expiry and datetime.now() < session_expiry:
        return session_token, cst_token

    # Create new session
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/v1/session",
                headers={
                    "X-CAP-API-KEY": API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "identifier": IDENTIFIER,
                    "password": API_PASSWORD,
                    "encryptedPassword": False
                }
            )

            if response.status_code == 200:
                # Extract tokens from headers
                session_token = response.headers.get("X-SECURITY-TOKEN")
                cst_token = response.headers.get("CST")

                # Set expiry to 9 minutes from now (conservative buffer)
                session_expiry = datetime.now() + timedelta(minutes=9)

                return session_token, cst_token
            else:
                raise Exception(f"Authentication failed: {response.status_code} - {response.text}")

        except Exception as e:
            raise Exception(f"Failed to create session: {str(e)}")


def get_headers(security_token: str, cst: str) -> dict:
    """Generate headers for authenticated requests"""
    return {
        "X-SECURITY-TOKEN": security_token,
        "CST": cst,
        "Content-Type": "application/json"
    }


# ========== TECHNICAL ANALYSIS FUNCTIONS ==========

def calculate_rsi(prices: list, period: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index"""
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
    """Calculate multiple moving averages"""
    return {
        "ma_20": sum(prices[-20:]) / 20 if len(prices) >= 20 else None,
        "ma_50": sum(prices[-50:]) / 50 if len(prices) >= 50 else None,
        "ma_100": sum(prices[-100:]) / 100 if len(prices) >= 100 else None,
    }


def calculate_bollinger_bands(prices: list, period: int = 20, std_dev: int = 2) -> dict:
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return {"upper": None, "middle": None, "lower": None}

    recent = prices[-period:]
    sma = sum(recent) / period
    variance = sum((x - sma) ** 2 for x in recent) / period
    std = variance ** 0.5

    return {
        "upper": sma + (std * std_dev),
        "middle": sma,
        "lower": sma - (std * std_dev)
    }


async def analyze_market_prediction(epic: str, security_token: str, cst: str) -> dict:
    """Perform comprehensive technical analysis and generate trade prediction"""
    headers = get_headers(security_token, cst)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch current market data
            market_response = await client.get(
                f"{API_URL}/api/v1/markets/{epic}",
                headers=headers
            )

            if market_response.status_code != 200:
                return {"error": f"Failed to fetch market data: {market_response.text}"}

            market_data = market_response.json()
            snapshot = market_data.get('snapshot', {})
            current_bid = snapshot.get('bid', 0)
            current_offer = snapshot.get('offer', 0)
            current_price = (current_bid + current_offer) / 2

            # Fetch historical data (1H timeframe)
            price_response = await client.get(
                f"{API_URL}/api/v1/prices/{epic}",
                headers=headers,
                params={"resolution": "HOUR", "max": 200}
            )

            if price_response.status_code != 200:
                return {"error": f"Failed to fetch price history: {price_response.text}"}

            price_data = price_response.json()
            closes = [p['closePrice']['ask'] for p in price_data['prices'] if 'closePrice' in p]

            if len(closes) < 50:
                return {"error": "Insufficient historical data for analysis"}

            # Calculate indicators
            rsi = calculate_rsi(closes, 14)
            mas = calculate_moving_averages(closes)
            bb = calculate_bollinger_bands(closes, 20, 2)

            # Scoring system
            bullish_score = 0
            bearish_score = 0
            reasons = []

            # MA Analysis
            if mas["ma_20"] and mas["ma_50"]:
                if mas["ma_20"] > mas["ma_50"]:
                    bullish_score += 2
                    reasons.append("✅ 20 MA above 50 MA (bullish trend)")
                else:
                    bearish_score += 2
                    reasons.append("❌ 20 MA below 50 MA (bearish trend)")

            # RSI Analysis
            if rsi:
                if rsi < 30:
                    bullish_score += 3
                    reasons.append(f"✅ RSI at {rsi:.1f} (oversold, potential bounce)")
                elif rsi > 70:
                    bearish_score += 3
                    reasons.append(f"❌ RSI at {rsi:.1f} (overbought, potential pullback)")
                elif 40 < rsi < 60:
                    reasons.append(f"ℹ️  RSI at {rsi:.1f} (neutral)")
                else:
                    reasons.append(f"ℹ️  RSI at {rsi:.1f}")

            # Bollinger Band Analysis
            if bb["upper"] and bb["lower"]:
                bb_position = ((current_price - bb["lower"]) / (bb["upper"] - bb["lower"])) * 100
                if bb_position > 80:
                    bearish_score += 1
                    reasons.append(f"⚠️  Price at {bb_position:.0f}% of BB range (near resistance)")
                elif bb_position < 20:
                    bullish_score += 1
                    reasons.append(f"✅ Price at {bb_position:.0f}% of BB range (near support)")

            # Price momentum
            if len(closes) >= 10:
                price_change = ((closes[-1] - closes[-10]) / closes[-10]) * 100
                if price_change > 2:
                    bullish_score += 1
                    reasons.append(f"📈 Strong upward momentum (+{price_change:.2f}%)")
                elif price_change < -2:
                    bearish_score += 1
                    reasons.append(f"📉 Strong downward momentum ({price_change:.2f}%)")

            # Determine prediction
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

            # Add warning if extreme RSI
            if rsi and (rsi > 75 or rsi < 25):
                confidence = min(confidence, 60)
                reasons.append("⚠️  Extreme RSI reduces confidence")

            # Calculate suggested levels
            if prediction == "BUY":
                entry = current_offer
                stop_loss = entry - (entry * 0.015)  # 1.5% stop
                take_profit = entry + (entry * 0.03)  # 3% target
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
                "bullish_signals": bullish_score,
                "bearish_signals": bearish_score,
                "reasoning": reasons,
                "technical_data": {
                    "rsi": round(rsi, 2) if rsi else None,
                    "ma_20": round(mas["ma_20"], 2) if mas["ma_20"] else None,
                    "ma_50": round(mas["ma_50"], 2) if mas["ma_50"] else None,
                    "bb_upper": round(bb["upper"], 2) if bb["upper"] else None,
                    "bb_lower": round(bb["lower"], 2) if bb["lower"] else None,
                },
                "suggested_trade": {
                    "entry": round(entry, 2),
                    "stop_loss": round(stop_loss, 2) if stop_loss else None,
                    "take_profit": round(take_profit, 2) if take_profit else None,
                } if prediction != "HOLD" else None
            }

    except Exception as e:
        return {"error": str(e)}


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for interacting with Capital.com"""
    return [
        Tool(
            name="search_markets",
            description="Search for available markets/instruments to trade (e.g., forex, indices, commodities, stocks, crypto)",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Search term (e.g., 'EUR', 'gold', 'bitcoin', 'tesla', 'S&P')"
                    }
                },
                "required": ["search_term"]
            }
        ),
        Tool(
            name="get_market_details",
            description="Get detailed information about a specific market including current price, spread, and trading hours",
            inputSchema={
                "type": "object",
                "properties": {
                    "epic": {
                        "type": "string",
                        "description": "Market epic/identifier (e.g., 'EURUSD', 'US500')"
                    }
                },
                "required": ["epic"]
            }
        ),
        Tool(
            name="get_price_history",
            description="Get historical price data (OHLC candles) for technical analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "epic": {
                        "type": "string",
                        "description": "Market epic/identifier"
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Timeframe for candles",
                        "enum": ["MINUTE", "MINUTE_5", "MINUTE_15", "MINUTE_30", "HOUR", "HOUR_4", "DAY", "WEEK"]
                    },
                    "max_points": {
                        "type": "integer",
                        "description": "Maximum number of data points to return (default 50, max 1000)",
                        "default": 50
                    }
                },
                "required": ["epic", "resolution"]
            }
        ),
        Tool(
            name="get_account_info",
            description="Get account information including balance, available funds, and P&L",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_positions",
            description="Get all open positions",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_working_orders",
            description="Get all pending/working orders",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="place_position",
            description="Place a new market order (BUY or SELL). ⚠️ USE WITH EXTREME CAUTION - EXECUTES REAL TRADES",
            inputSchema={
                "type": "object",
                "properties": {
                    "epic": {
                        "type": "string",
                        "description": "Market epic/identifier"
                    },
                    "direction": {
                        "type": "string",
                        "description": "Trade direction",
                        "enum": ["BUY", "SELL"]
                    },
                    "size": {
                        "type": "number",
                        "description": "Position size"
                    },
                    "stop_level": {
                        "type": "number",
                        "description": "Stop loss level (optional but recommended)"
                    },
                    "profit_level": {
                        "type": "number",
                        "description": "Take profit level (optional)"
                    },
                    "guaranteed_stop": {
                        "type": "boolean",
                        "description": "Use guaranteed stop loss (may have premium)",
                        "default": False
                    }
                },
                "required": ["epic", "direction", "size"]
            }
        ),
        Tool(
            name="close_position",
            description="Close an existing position",
            inputSchema={
                "type": "object",
                "properties": {
                    "deal_id": {
                        "type": "string",
                        "description": "The deal ID of the position to close"
                    }
                },
                "required": ["deal_id"]
            }
        ),
        Tool(
            name="update_position",
            description="Update stop loss or take profit levels on an existing position",
            inputSchema={
                "type": "object",
                "properties": {
                    "deal_id": {
                        "type": "string",
                        "description": "The deal ID of the position to update"
                    },
                    "stop_level": {
                        "type": "number",
                        "description": "New stop loss level (optional)"
                    },
                    "profit_level": {
                        "type": "number",
                        "description": "New take profit level (optional)"
                    }
                },
                "required": ["deal_id"]
            }
        ),
        Tool(
            name="predict_trade",
            description="Advanced technical analysis with AI-powered trade prediction, confidence score, and detailed reasoning for any market",
            inputSchema={
                "type": "object",
                "properties": {
                    "epic": {
                        "type": "string",
                        "description": "Market epic/identifier (e.g., 'US100', 'EURUSD', 'BITCOIN')"
                    }
                },
                "required": ["epic"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute tool calls and interact with Capital.com API"""

    try:
        # Get valid session tokens
        security_token, cst = await get_session()
        headers = get_headers(security_token, cst)

        async with httpx.AsyncClient(timeout=30.0) as client:

            # ===== MARKET SEARCH =====
            if name == "search_markets":
                response = await client.get(
                    f"{API_URL}/api/v1/markets",
                    headers=headers,
                    params={"searchTerm": arguments['search_term']}
                )

                if response.status_code == 200:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps(data, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error searching markets: {response.status_code} - {response.text}"
                    )]

            # ===== MARKET DETAILS =====
            elif name == "get_market_details":
                response = await client.get(
                    f"{API_URL}/api/v1/markets/{arguments['epic']}",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps(data, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error fetching market details: {response.status_code} - {response.text}"
                    )]

            # ===== PRICE HISTORY =====
            elif name == "get_price_history":
                max_points = arguments.get('max_points', 50)
                response = await client.get(
                    f"{API_URL}/api/v1/prices/{arguments['epic']}",
                    headers=headers,
                    params={
                        "resolution": arguments['resolution'],
                        "max": min(max_points, 1000)  # Cap at API limit
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps(data, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error fetching price history: {response.status_code} - {response.text}"
                    )]

            # ===== ACCOUNT INFO =====
            elif name == "get_account_info":
                response = await client.get(
                    f"{API_URL}/api/v1/accounts",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps(data, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error fetching account info: {response.status_code} - {response.text}"
                    )]

            # ===== GET POSITIONS =====
            elif name == "get_positions":
                response = await client.get(
                    f"{API_URL}/api/v1/positions",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps(data, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error fetching positions: {response.status_code} - {response.text}"
                    )]

            # ===== GET WORKING ORDERS =====
            elif name == "get_working_orders":
                response = await client.get(
                    f"{API_URL}/api/v1/workingorders",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=json.dumps(data, indent=2)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error fetching working orders: {response.status_code} - {response.text}"
                    )]

            # ===== PLACE POSITION =====
            elif name == "place_position":
                payload = {
                    "epic": arguments['epic'],
                    "direction": arguments['direction'],
                    "size": arguments['size']
                }

                if 'stop_level' in arguments:
                    payload['stopLevel'] = arguments['stop_level']
                if 'profit_level' in arguments:
                    payload['profitLevel'] = arguments['profit_level']
                if 'guaranteed_stop' in arguments:
                    payload['guaranteedStop'] = arguments['guaranteed_stop']

                response = await client.post(
                    f"{API_URL}/api/v1/positions",
                    headers=headers,
                    json=payload
                )

                if response.status_code in [200, 201]:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=f"✅ Position opened successfully!\n{json.dumps(data, indent=2)}"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Error placing position: {response.status_code} - {response.text}"
                    )]

            # ===== CLOSE POSITION =====
            elif name == "close_position":
                response = await client.delete(
                    f"{API_URL}/api/v1/positions/{arguments['deal_id']}",
                    headers=headers
                )

                if response.status_code in [200, 204]:
                    return [TextContent(
                        type="text",
                        text=f"✅ Position {arguments['deal_id']} closed successfully!"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Error closing position: {response.status_code} - {response.text}"
                    )]

            # ===== UPDATE POSITION =====
            elif name == "update_position":
                payload = {}
                if 'stop_level' in arguments:
                    payload['stopLevel'] = arguments['stop_level']
                if 'profit_level' in arguments:
                    payload['profitLevel'] = arguments['profit_level']

                response = await client.put(
                    f"{API_URL}/api/v1/positions/{arguments['deal_id']}",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    return [TextContent(
                        type="text",
                        text=f"✅ Position updated successfully!\n{json.dumps(data, indent=2)}"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Error updating position: {response.status_code} - {response.text}"
                    )]

            # ===== PREDICT TRADE =====
            elif name == "predict_trade":
                epic = arguments['epic']
                print(f"🔮 Analyzing {epic} for trade prediction...")

                result = await analyze_market_prediction(epic, security_token, cst)

                if "error" in result:
                    return [TextContent(
                        type="text",
                        text=f"❌ Analysis Error: {result['error']}"
                    )]

                # Format the output nicely
                output = f"""
🔮 TRADE PREDICTION ANALYSIS
{'='*60}
📊 Instrument: {result['instrument']} ({result['epic']})
💰 Current Price: {result['current_price']:.2f}

{'='*60}
PREDICTION: {result['prediction']}
CONFIDENCE: {result['confidence']:.1f}%
{'='*60}

📊 Signal Breakdown:
  • Bullish Signals: {result['bullish_signals']}
  • Bearish Signals: {result['bearish_signals']}

📈 Technical Indicators:
  • RSI: {result['technical_data']['rsi']}
  • MA(20): {result['technical_data']['ma_20']}
  • MA(50): {result['technical_data']['ma_50']}
  • BB Upper: {result['technical_data']['bb_upper']}
  • BB Lower: {result['technical_data']['bb_lower']}

💡 Reasoning:
"""
                for reason in result['reasoning']:
                    output += f"  {reason}\n"

                if result.get('suggested_trade'):
                    trade = result['suggested_trade']
                    output += f"""
{'='*60}
📍 SUGGESTED TRADE SETUP:
  Entry: {trade['entry']}
  Stop Loss: {trade['stop_loss']}
  Take Profit: {trade['take_profit']}
  Risk/Reward: 1:2
{'='*60}
"""

                output += """
⚠️  DISCLAIMER: This is algorithmic analysis for educational
   purposes only. Always do your own research and manage risk.
"""

                return [TextContent(
                    type="text",
                    text=output
                )]

            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server

    # Verify configuration
    if not API_KEY or not API_PASSWORD:
        print("ERROR: Missing API credentials. Please set CAPITAL_API_KEY and CAPITAL_API_PASSWORD in .env file")
        return

    mode = "DEMO" if USE_DEMO else "LIVE"
    print(f"Starting Capital.com MCP Server in {mode} mode...")
    print(f"API URL: {API_URL}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
