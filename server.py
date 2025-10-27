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
