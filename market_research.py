#!/usr/bin/env python3
"""
AI-Powered Market Research System
Uses Perplexity for news gathering and Claude for deep analysis
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# API Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


class MarketResearcher:
    """AI-powered market research and analysis"""

    def __init__(self):
        self.research_cache = {}
        self.last_research_time = {}

    async def search_news_perplexity(self, query: str) -> Dict:
        """Search for news using Perplexity AI"""
        if not PERPLEXITY_API_KEY:
            return {"error": "Perplexity API key not set"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-large-128k-online",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a financial news analyst. Provide concise, factual summaries of market-moving news with sources."
                            },
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        "temperature": 0.2,
                        "return_citations": True,
                        "search_recency_filter": "day"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "content": data['choices'][0]['message']['content'],
                        "citations": data.get('citations', []),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
                    return {"error": f"API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Error calling Perplexity API: {e}")
            return {"error": str(e)}

    async def analyze_with_claude(self, market_data: Dict, news_context: str) -> Dict:
        """Deep analysis using Claude AI"""
        if not ANTHROPIC_API_KEY:
            return {"error": "Anthropic API key not set"}

        try:
            epic = market_data.get('epic', 'UNKNOWN')
            current_price = market_data.get('current_price', 0)
            rsi = market_data.get('rsi', 'N/A')
            ma_20 = market_data.get('ma_20', 'N/A')
            ma_50 = market_data.get('ma_50', 'N/A')

            prompt = f"""You are an expert trading analyst. Analyze this market opportunity:

INSTRUMENT: {epic}
CURRENT PRICE: {current_price}

TECHNICAL ANALYSIS:
- RSI (14): {rsi}
- MA(20): {ma_20}
- MA(50): {ma_50}

NEWS CONTEXT:
{news_context}

TASK:
1. Analyze how the news impacts this market
2. Consider technical indicators in context of news
3. Provide a clear trading recommendation: BUY, SELL, or HOLD
4. Explain your reasoning in 3-5 bullet points
5. Suggest entry, stop-loss, and take-profit levels if trading
6. Rate your confidence: LOW (40-60%), MEDIUM (60-75%), HIGH (75%+)

Format your response as:
DECISION: [BUY/SELL/HOLD]
CONFIDENCE: [percentage]
REASONING:
- [point 1]
- [point 2]
- [point 3]

TRADE SETUP (if not HOLD):
Entry: [price]
Stop Loss: [price]
Take Profit: [price]
Risk/Reward: [ratio]

RISKS:
- [key risk 1]
- [key risk 2]
"""

            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            analysis_text = message.content[0].text

            # Parse Claude's response
            decision = "HOLD"
            confidence = 50
            reasoning = []
            entry = None
            stop_loss = None
            take_profit = None
            risks = []

            lines = analysis_text.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith('DECISION:'):
                    decision = line.split(':', 1)[1].strip().upper()
                elif line.startswith('CONFIDENCE:'):
                    conf_str = line.split(':', 1)[1].strip().rstrip('%')
                    try:
                        confidence = float(conf_str)
                    except:
                        confidence = 50
                elif line.startswith('REASONING:'):
                    current_section = 'reasoning'
                elif line.startswith('TRADE SETUP'):
                    current_section = 'trade'
                elif line.startswith('RISKS:'):
                    current_section = 'risks'
                elif line.startswith('Entry:'):
                    try:
                        entry = float(line.split(':', 1)[1].strip())
                    except:
                        pass
                elif line.startswith('Stop Loss:'):
                    try:
                        stop_loss = float(line.split(':', 1)[1].strip())
                    except:
                        pass
                elif line.startswith('Take Profit:'):
                    try:
                        take_profit = float(line.split(':', 1)[1].strip())
                    except:
                        pass
                elif line.startswith('- '):
                    if current_section == 'reasoning':
                        reasoning.append(line[2:])
                    elif current_section == 'risks':
                        risks.append(line[2:])

            return {
                "success": True,
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risks": risks,
                "full_analysis": analysis_text,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens
            }

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return {"error": str(e)}

    async def pre_market_research(self, instruments: List[str]) -> Dict:
        """Comprehensive pre-market research for all instruments"""
        logger.info("🔍 Starting pre-market research...")

        results = {}

        for epic in instruments:
            logger.info(f"Researching {epic}...")

            # Map epic to searchable name
            instrument_names = {
                "US100": "NASDAQ 100 / US Tech 100",
                "US500": "S&P 500",
                "EURUSD": "EUR/USD forex",
                "GBPUSD": "GBP/USD forex",
                "GOLD": "Gold / XAU",
                "BITCOIN": "Bitcoin / BTC"
            }

            instrument_name = instrument_names.get(epic, epic)

            # Search for news
            query = f"""What are the most important news and events affecting {instrument_name} today?
            Include:
            - Major economic data releases
            - Central bank announcements
            - Corporate earnings (if applicable)
            - Geopolitical events
            - Market sentiment

            Focus on information from the last 24 hours that could impact prices today."""

            news_result = await self.search_news_perplexity(query)

            if news_result.get('success'):
                results[epic] = {
                    "news": news_result['content'],
                    "citations": news_result.get('citations', []),
                    "timestamp": news_result['timestamp']
                }
                logger.info(f"✅ Research complete for {epic}")
            else:
                logger.error(f"❌ Failed to research {epic}: {news_result.get('error')}")
                results[epic] = {"error": news_result.get('error')}

            # Rate limiting
            await asyncio.sleep(2)

        # Cache results
        self.research_cache = results
        self.last_research_time['pre_market'] = datetime.now()

        logger.info("✅ Pre-market research complete!")
        return results

    async def check_breaking_news(self, epic: str) -> Optional[Dict]:
        """Check for breaking news on specific instrument"""
        # Only check if more than 15 minutes since last check
        last_check = self.last_research_time.get(f'breaking_{epic}')
        if last_check and (datetime.now() - last_check).seconds < 900:
            return None

        instrument_names = {
            "US100": "NASDAQ 100",
            "US500": "S&P 500",
            "EURUSD": "EUR/USD",
            "GBPUSD": "GBP/USD",
            "GOLD": "Gold",
            "BITCOIN": "Bitcoin"
        }

        instrument_name = instrument_names.get(epic, epic)

        query = f"Breaking news in the last 2 hours affecting {instrument_name}. Only report if there's significant market-moving news."

        result = await self.search_news_perplexity(query)

        if result.get('success'):
            self.last_research_time[f'breaking_{epic}'] = datetime.now()
            return result

        return None

    async def generate_trading_plan(self, epic: str, market_data: Dict) -> Dict:
        """Generate comprehensive trading plan with AI analysis"""
        logger.info(f"📋 Generating trading plan for {epic}...")

        # Get cached news or fetch new
        news_context = "No recent news available."
        if epic in self.research_cache:
            news_context = self.research_cache[epic].get('news', news_context)
        else:
            # Fetch quick news update
            news_result = await self.check_breaking_news(epic)
            if news_result and news_result.get('success'):
                news_context = news_result['content']

        # Analyze with Claude
        analysis = await self.analyze_with_claude(market_data, news_context)

        if analysis.get('success'):
            logger.info(f"✅ Trading plan generated: {analysis['decision']} at {analysis['confidence']}% confidence")
            return {
                "epic": epic,
                "decision": analysis['decision'],
                "confidence": analysis['confidence'],
                "reasoning": analysis['reasoning'],
                "entry": analysis.get('entry'),
                "stop_loss": analysis.get('stop_loss'),
                "take_profit": analysis.get('take_profit'),
                "risks": analysis.get('risks', []),
                "news_context": news_context[:500] + "..." if len(news_context) > 500 else news_context,
                "full_analysis": analysis['full_analysis'],
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.error(f"❌ Failed to generate trading plan: {analysis.get('error')}")
            return {"error": analysis.get('error')}

    def get_cached_research(self, epic: str) -> Optional[Dict]:
        """Get cached research results"""
        return self.research_cache.get(epic)

    def should_use_claude(self, algo_analysis: Dict) -> bool:
        """Decide if we should call Claude for this analysis"""
        confidence = algo_analysis.get('confidence', 0)
        prediction = algo_analysis.get('prediction', 'HOLD')

        # Don't use Claude for clear holds
        if prediction == 'HOLD':
            return False

        # Don't use Claude for very high confidence (algorithm is sure)
        if confidence > 80:
            return False

        # Don't use Claude for very low confidence (not worth trading)
        if confidence < 50:
            return False

        # Use Claude for borderline cases (50-80%)
        if 50 <= confidence <= 80:
            return True

        return False


# Singleton instance
researcher = MarketResearcher()


async def test_research_system():
    """Test the research system"""
    print("=" * 70)
    print("🧪 TESTING AI RESEARCH SYSTEM")
    print("=" * 70)

    # Test Perplexity
    print("\n1️⃣ Testing Perplexity API...")
    news = await researcher.search_news_perplexity(
        "What news is affecting the NASDAQ 100 today?"
    )
    if news.get('success'):
        print("✅ Perplexity working!")
        print(f"\nNews Summary:\n{news['content'][:300]}...\n")
    else:
        print(f"❌ Perplexity failed: {news.get('error')}")

    # Test Claude
    print("\n2️⃣ Testing Claude API...")
    market_data = {
        'epic': 'US100',
        'current_price': 20150.5,
        'rsi': 58.3,
        'ma_20': 20100,
        'ma_50': 19950
    }

    analysis = await researcher.analyze_with_claude(
        market_data,
        "Tech stocks showing strength. Apple announces new AI chip. Market sentiment positive."
    )

    if analysis.get('success'):
        print("✅ Claude working!")
        print(f"\nDecision: {analysis['decision']}")
        print(f"Confidence: {analysis['confidence']}%")
        print(f"Reasoning: {analysis['reasoning'][:2]}")
    else:
        print(f"❌ Claude failed: {analysis.get('error')}")

    print("\n" + "=" * 70)
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_research_system())
