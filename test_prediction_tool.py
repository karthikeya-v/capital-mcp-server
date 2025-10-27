#!/usr/bin/env python3
"""Test the prediction tool"""

import asyncio
import sys
sys.path.insert(0, '.')

from server import get_session, analyze_market_prediction

async def test_predict():
    print("Testing prediction tool...")
    print()

    # Get session
    security_token, cst = await get_session()

    if not security_token:
        print("❌ Authentication failed")
        return

    print("✅ Authenticated")
    print()

    # Test with US100
    print("Analyzing US100...")
    result = await analyze_market_prediction("US100", security_token, cst)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"""
🔮 PREDICTION RESULT:
  Instrument: {result['instrument']}
  Price: {result['current_price']:.2f}
  Prediction: {result['prediction']}
  Confidence: {result['confidence']:.1f}%

  Bullish Signals: {result['bullish_signals']}
  Bearish Signals: {result['bearish_signals']}

  RSI: {result['technical_data']['rsi']}

  Reasoning:
""")

    for reason in result['reasoning']:
        print(f"    {reason}")

    if result.get('suggested_trade'):
        trade = result['suggested_trade']
        print(f"""
  Suggested Trade:
    Entry: {trade['entry']}
    Stop Loss: {trade['stop_loss']}
    Take Profit: {trade['take_profit']}
""")

    print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_predict())
