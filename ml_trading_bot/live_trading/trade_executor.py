"""
Live Trade Executor for Capital.com API.
Executes real trades on demo account based on ML model predictions.
"""

import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import Trade, get_db


class CapitalTradeExecutor:
    """
    Executes trades on Capital.com API (demo or live).
    """

    def __init__(
        self,
        email: str,
        api_key: str,
        api_password: str,
        demo: bool = True
    ):
        """
        Initialize trade executor.

        Args:
            email: Capital.com email
            api_key: API key
            api_password: API password
            demo: Use demo account (default: True)
        """
        self.email = email
        self.api_key = api_key
        self.api_password = api_password
        self.demo = demo

        if demo:
            self.base_url = "https://demo-api-capital.backend-capital.com"
        else:
            self.base_url = "https://api-capital.backend-capital.com"

        self.session_token = None
        self.security_token = None
        self.token_expires = None

    async def authenticate(self):
        """Authenticate with Capital.com API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/session",
                json={
                    "identifier": self.email,
                    "password": self.api_password
                },
                headers={
                    "X-CAP-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                }
            )

            if response.status_code == 200:
                self.session_token = response.headers.get("CST")
                self.security_token = response.headers.get("X-SECURITY-TOKEN")
                self.token_expires = datetime.now() + timedelta(minutes=10)
                print("✓ Authenticated with Capital.com")
                return True
            else:
                print(f"✗ Authentication failed: {response.text}")
                return False

    async def ensure_authenticated(self):
        """Ensure we have a valid session token"""
        if not self.session_token or datetime.now() >= self.token_expires:
            await self.authenticate()

    async def get_account_info(self) -> Dict:
        """Get account information"""
        await self.ensure_authenticated()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/accounts",
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting account info: {response.text}")
                return {}

    async def get_market_info(self, epic: str) -> Optional[Dict]:
        """Get market information"""
        await self.ensure_authenticated()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/markets/{epic}",
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting market info for {epic}: {response.text}")
                return None

    async def place_trade(
        self,
        epic: str,
        direction: str,  # 'BUY' or 'SELL'
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        guaranteed_stop: bool = False
    ) -> Optional[Dict]:
        """
        Place a market order.

        Args:
            epic: Market epic
            direction: 'BUY' or 'SELL'
            size: Position size
            stop_loss: Stop loss level
            take_profit: Take profit level
            guaranteed_stop: Use guaranteed stop

        Returns:
            Trade response or None if failed
        """
        await self.ensure_authenticated()

        # Prepare request
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": guaranteed_stop
        }

        # Add stop loss if provided
        if stop_loss is not None:
            payload["stopLevel"] = stop_loss
            payload["stopDistance"] = None

        # Add take profit if provided
        if take_profit is not None:
            payload["profitLevel"] = take_profit
            payload["profitDistance"] = None

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/positions",
                json=payload,
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token,
                    "Content-Type": "application/json"
                }
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Trade placed: {epic} {direction} {size}")
                return result
            else:
                print(f"✗ Trade failed: {response.text}")
                return None

    async def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        await self.ensure_authenticated()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/positions",
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('positions', [])
            else:
                print(f"Error getting positions: {response.text}")
                return []

    async def close_position(self, deal_id: str) -> bool:
        """
        Close a position.

        Args:
            deal_id: Position deal ID

        Returns:
            True if successful
        """
        await self.ensure_authenticated()

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/api/v1/positions/{deal_id}",
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token
                }
            )

            if response.status_code == 200:
                print(f"✓ Position closed: {deal_id}")
                return True
            else:
                print(f"✗ Failed to close position {deal_id}: {response.text}")
                return False

    async def update_position(
        self,
        deal_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Update position stop loss or take profit.

        Args:
            deal_id: Position deal ID
            stop_loss: New stop loss level
            take_profit: New take profit level

        Returns:
            True if successful
        """
        await self.ensure_authenticated()

        payload = {}
        if stop_loss is not None:
            payload["stopLevel"] = stop_loss
        if take_profit is not None:
            payload["profitLevel"] = take_profit

        if not payload:
            return False

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/api/v1/positions/{deal_id}",
                json=payload,
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token,
                    "Content-Type": "application/json"
                }
            )

            if response.status_code == 200:
                print(f"✓ Position updated: {deal_id}")
                return True
            else:
                print(f"✗ Failed to update position {deal_id}: {response.text}")
                return False

    async def get_trade_history(self, days: int = 7) -> List[Dict]:
        """
        Get trade history.

        Args:
            days: Number of days to fetch

        Returns:
            List of historical trades
        """
        await self.ensure_authenticated()

        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/history/activity",
                params={
                    "from": from_date.isoformat(),
                    "to": to_date.isoformat()
                },
                headers={
                    "X-SECURITY-TOKEN": self.security_token,
                    "CST": self.session_token
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('activities', [])
            else:
                print(f"Error getting trade history: {response.text}")
                return []

    def save_trade_to_db(
        self,
        trade_data: Dict,
        model_id: Optional[int] = None,
        confidence: float = 0.0,
        is_simulated: bool = False
    ):
        """
        Save trade to database.

        Args:
            trade_data: Trade data from API response
            model_id: Model that generated the signal
            confidence: Model confidence
            is_simulated: Whether trade is simulated
        """
        with get_db() as db:
            trade = Trade(
                trade_id=trade_data.get('dealId', str(uuid.uuid4())),
                epic=trade_data.get('epic'),
                direction=trade_data.get('direction'),
                size=trade_data.get('size'),
                entry_price=trade_data.get('level') or trade_data.get('openLevel'),
                entry_time=datetime.now(),
                stop_loss=trade_data.get('stopLevel'),
                take_profit=trade_data.get('profitLevel'),
                model_id=model_id,
                prediction_confidence=confidence,
                is_simulated=is_simulated
            )

            db.add(trade)
            db.commit()
            db.refresh(trade)

            print(f"✓ Trade saved to database: {trade.trade_id}")
            return trade.id

    def update_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        exit_reason: str = 'MANUAL'
    ):
        """
        Update trade with exit information.

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            pnl: Profit/Loss
            exit_reason: Reason for exit
        """
        with get_db() as db:
            trade = db.query(Trade).filter(Trade.trade_id == trade_id).first()

            if trade:
                trade.exit_price = exit_price
                trade.exit_time = datetime.now()
                trade.pnl = pnl
                trade.pnl_percent = (pnl / (trade.entry_price * trade.size)) * 100 if trade.entry_price else 0
                trade.exit_reason = exit_reason

                db.commit()
                print(f"✓ Trade updated with exit: {trade_id}, P&L: ${pnl:.2f}")
            else:
                print(f"✗ Trade not found: {trade_id}")


class PositionManager:
    """
    Manages open positions and monitors for SL/TP.
    """

    def __init__(self, executor: CapitalTradeExecutor):
        """
        Initialize position manager.

        Args:
            executor: Trade executor instance
        """
        self.executor = executor
        self.monitored_positions = {}  # {deal_id: trade_db_id}

    async def monitor_positions(self):
        """
        Monitor all open positions and update database.
        """
        positions = await self.executor.get_positions()

        # Update database with current positions
        with get_db() as db:
            for position in positions:
                deal_id = position.get('dealId')
                epic = position.get('epic')
                direction = position.get('direction')
                size = position.get('size')
                level = position.get('level')
                profit_loss = position.get('profitLoss')

                # Check if we're tracking this position
                if deal_id not in self.monitored_positions:
                    # New position not in our DB - maybe opened manually
                    # We can add it to DB if needed
                    pass

                # Log current status
                print(f"Position: {epic} {direction} {size} @ {level}, P&L: ${profit_loss}")

        return positions

    async def check_and_close_positions(self, max_holding_hours: int = 24):
        """
        Check positions and close if needed.

        Args:
            max_holding_hours: Maximum hours to hold a position
        """
        with get_db() as db:
            # Get open trades from DB
            open_trades = db.query(Trade).filter(
                Trade.exit_time == None,
                Trade.is_simulated == False
            ).all()

            for trade in open_trades:
                # Check if position still exists
                positions = await self.executor.get_positions()
                position_exists = any(p.get('dealId') == trade.trade_id for p in positions)

                if not position_exists:
                    # Position was closed (probably by SL/TP)
                    # Try to get the closing details from history
                    history = await self.executor.get_trade_history(days=1)

                    for activity in history:
                        if activity.get('dealId') == trade.trade_id and activity.get('type') == 'POSITION_CLOSED':
                            exit_price = activity.get('level')
                            pnl = activity.get('profitLoss', 0)
                            exit_reason = 'TP' if pnl > 0 else 'SL' if pnl < 0 else 'MANUAL'

                            self.executor.update_trade_exit(
                                trade.trade_id,
                                exit_price,
                                pnl,
                                exit_reason
                            )
                            break

                else:
                    # Position still open - check if should close
                    holding_time = datetime.now() - trade.entry_time
                    if holding_time.total_seconds() / 3600 > max_holding_hours:
                        # Close position (held too long)
                        print(f"Closing {trade.epic} - held for {holding_time.total_seconds()/3600:.1f} hours")
                        await self.executor.close_position(trade.trade_id)

    def register_position(self, deal_id: str, trade_db_id: int):
        """
        Register a position for monitoring.

        Args:
            deal_id: Capital.com deal ID
            trade_db_id: Database trade ID
        """
        self.monitored_positions[deal_id] = trade_db_id


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def test_trading():
        executor = CapitalTradeExecutor(
            email=os.getenv('CAPITAL_EMAIL'),
            api_key=os.getenv('CAPITAL_API_KEY'),
            api_password=os.getenv('CAPITAL_PASSWORD'),
            demo=True
        )

        # Authenticate
        await executor.authenticate()

        # Get account info
        account = await executor.get_account_info()
        print(f"\nAccount: {account}")

        # Get positions
        positions = await executor.get_positions()
        print(f"\nOpen positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos.get('epic')} {pos.get('direction')} {pos.get('size')}")

        # Get market info
        market = await executor.get_market_info('US100')
        if market:
            print(f"\nMarket: {market.get('instrument', {}).get('name')}")

    asyncio.run(test_trading())
