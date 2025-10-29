"""
Backtesting Engine for ML Trading Bot.
Simulates trading with realistic market conditions and risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import (
    Feature, Trade, BacktestResult, Model, get_db
)


@dataclass
class Position:
    """Represents an open trading position"""
    trade_id: str
    epic: str
    direction: str  # 'BUY' or 'SELL'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    confidence: float = 0.0


@dataclass
class TradeResult:
    """Results of a closed trade"""
    trade_id: str
    epic: str
    direction: str
    size: float
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    exit_reason: str  # 'TP', 'SL', 'MANUAL', 'TIMEOUT'
    pnl: float
    pnl_percent: float
    confidence: float


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade_percent: float = 1.0,
        max_position_size: float = 1.0,
        max_daily_loss: float = 500.0,
        slippage_percent: float = 0.1,
        commission_per_trade: float = 2.0
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            risk_per_trade_percent: Risk per trade (% of capital)
            max_position_size: Maximum position size
            max_daily_loss: Maximum daily loss limit
            slippage_percent: Slippage in percent
            commission_per_trade: Commission per trade
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade_percent = risk_per_trade_percent
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.slippage_percent = slippage_percent
        self.commission_per_trade = commission_per_trade

        self.positions: List[Position] = []
        self.closed_trades: List[TradeResult] = []
        self.equity_curve = []
        self.daily_pnl = 0.0
        self.current_day = None

    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.daily_pnl = 0.0
        self.current_day = None

    def calculate_position_size(
        self,
        price: float,
        stop_loss: float,
        direction: str
    ) -> float:
        """
        Calculate position size based on risk management.

        Args:
            price: Entry price
            stop_loss: Stop loss price
            direction: 'BUY' or 'SELL'

        Returns:
            Position size
        """
        # Risk amount
        risk_amount = self.capital * (self.risk_per_trade_percent / 100)

        # Distance to stop loss
        if direction == 'BUY':
            risk_per_unit = price - stop_loss
        else:
            risk_per_unit = stop_loss - price

        if risk_per_unit <= 0:
            return 0.0

        # Calculate size
        size = risk_amount / risk_per_unit

        # Cap at max position size
        size = min(size, self.max_position_size)

        return round(size, 2)

    def apply_slippage(self, price: float, direction: str) -> float:
        """
        Apply slippage to price.

        Args:
            price: Original price
            direction: 'BUY' or 'SELL'

        Returns:
            Price with slippage
        """
        slippage = price * (self.slippage_percent / 100)

        if direction == 'BUY':
            return price + slippage
        else:
            return price - slippage

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        return self.daily_pnl <= -self.max_daily_loss

    def open_position(
        self,
        epic: str,
        direction: str,
        price: float,
        stop_loss: float,
        take_profit: float,
        timestamp: datetime,
        confidence: float = 0.0
    ) -> Optional[Position]:
        """
        Open a new position.

        Args:
            epic: Market epic
            direction: 'BUY' or 'SELL'
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            timestamp: Entry timestamp
            confidence: Model confidence

        Returns:
            Position if opened, None otherwise
        """
        # Check daily loss limit
        if self.check_daily_loss_limit():
            return None

        # Apply slippage
        entry_price = self.apply_slippage(price, direction)

        # Calculate position size
        size = self.calculate_position_size(entry_price, stop_loss, direction)

        if size <= 0:
            return None

        # Create position
        position = Position(
            trade_id=str(uuid.uuid4()),
            epic=epic,
            direction=direction,
            size=size,
            entry_price=entry_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence
        )

        self.positions.append(position)
        return position

    def close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> TradeResult:
        """
        Close a position.

        Args:
            position: Position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit

        Returns:
            TradeResult
        """
        # Apply slippage
        exit_price_with_slippage = self.apply_slippage(
            exit_price,
            'SELL' if position.direction == 'BUY' else 'BUY'
        )

        # Calculate P&L
        if position.direction == 'BUY':
            pnl = (exit_price_with_slippage - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price_with_slippage) * position.size

        # Subtract commission
        pnl -= self.commission_per_trade

        # Update capital
        self.capital += pnl
        self.daily_pnl += pnl

        # Calculate percentage
        pnl_percent = (pnl / (position.entry_price * position.size)) * 100

        # Create trade result
        trade_result = TradeResult(
            trade_id=position.trade_id,
            epic=position.epic,
            direction=position.direction,
            size=position.size,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price_with_slippage,
            exit_time=exit_time,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_percent=pnl_percent,
            confidence=position.confidence
        )

        self.closed_trades.append(trade_result)

        # Remove from open positions
        self.positions.remove(position)

        return trade_result

    def update_positions(
        self,
        timestamp: datetime,
        prices: Dict[str, Dict[str, float]]
    ):
        """
        Update all open positions (check SL/TP).

        Args:
            timestamp: Current timestamp
            prices: Dict of {epic: {'high': x, 'low': y, 'close': z}}
        """
        # Check for new day (reset daily P&L)
        if self.current_day != timestamp.date():
            self.current_day = timestamp.date()
            self.daily_pnl = 0.0

        positions_to_close = []

        for position in self.positions:
            if position.epic not in prices:
                continue

            epic_prices = prices[position.epic]
            high = epic_prices['high']
            low = epic_prices['low']
            close = epic_prices['close']

            # Check stop loss and take profit
            if position.direction == 'BUY':
                if low <= position.stop_loss:
                    # Stop loss hit
                    positions_to_close.append((position, position.stop_loss, 'SL'))
                elif high >= position.take_profit:
                    # Take profit hit
                    positions_to_close.append((position, position.take_profit, 'TP'))
            else:  # SELL
                if high >= position.stop_loss:
                    # Stop loss hit
                    positions_to_close.append((position, position.stop_loss, 'SL'))
                elif low <= position.take_profit:
                    # Take profit hit
                    positions_to_close.append((position, position.take_profit, 'TP'))

        # Close positions
        for position, exit_price, exit_reason in positions_to_close:
            self.close_position(position, exit_price, timestamp, exit_reason)

    def record_equity(self, timestamp: datetime):
        """Record current equity"""
        total_equity = self.capital

        # Add unrealized P&L from open positions (not included in this simplified version)

        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'positions': len(self.positions),
            'trades': len(self.closed_trades)
        })

    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary of metrics
        """
        if not self.closed_trades:
            return {}

        # Extract data
        pnls = [t.pnl for t in self.closed_trades]
        returns = [t.pnl_percent for t in self.closed_trades]

        # Basic stats
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = sum(1 for pnl in pnls if pnl < 0)

        # P&L stats
        total_pnl = sum(pnls)
        average_pnl = np.mean(pnls)

        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]

        average_win = np.mean(wins) if wins else 0
        average_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        # Returns-based metrics
        total_return = self.capital - self.initial_capital
        total_return_percent = (total_return / self.initial_capital) * 100

        # Sharpe ratio (simplified - assuming daily returns)
        if len(returns) > 1:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns and len(negative_returns) > 1:
            downside_std = np.std(negative_returns)
            sortino_ratio = (np.mean(returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = 0

        # Maximum drawdown
        if self.equity_curve:
            equity_values = [e['equity'] for e in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0

            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            max_drawdown_percent = (max_drawdown / peak * 100) if peak > 0 else 0
        else:
            max_drawdown = 0
            max_drawdown_percent = 0

        # Holding period
        holding_periods = [
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in self.closed_trades
        ]
        average_holding_period = np.mean(holding_periods) if holding_periods else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_pnl': average_pnl,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_percent,
            'average_holding_period': average_holding_period,
            'final_capital': self.capital
        }

    def run_backtest(
        self,
        epic: str,
        model: any,
        start_date: datetime,
        end_date: datetime,
        min_confidence: float = 0.6
    ) -> Dict:
        """
        Run backtest for a model.

        Args:
            epic: Market epic
            model: Trained ML model
            start_date: Backtest start date
            end_date: Backtest end date
            min_confidence: Minimum confidence threshold

        Returns:
            Backtest results
        """
        print(f"\n{'='*60}")
        print(f"Running Backtest for {epic}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*60}\n")

        self.reset()

        # Load features from database
        with get_db() as db:
            features = db.query(Feature).filter(
                Feature.epic == epic,
                Feature.timestamp >= start_date,
                Feature.timestamp <= end_date
            ).order_by(Feature.timestamp).all()

            if not features:
                raise ValueError(f"No features found for {epic}")

            print(f"✓ Loaded {len(features)} feature points")

            # Process each timestamp
            for i, feature in enumerate(features):
                timestamp = feature.timestamp

                # Create feature vector
                X = self._create_feature_vector(feature)

                if X is None:
                    continue

                # Get prediction
                try:
                    prediction = model.predict(X.reshape(1, -1))[0]
                    # Assume confidence from probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X.reshape(1, -1))[0]
                        confidence = max(proba)
                    else:
                        confidence = 0.7  # Default

                except Exception as e:
                    continue

                # Generate signal
                signal = self._decode_prediction(prediction)

                # Get current prices (from next feature for realistic simulation)
                if i + 1 < len(features):
                    next_feature = features[i + 1]
                    current_price = next_feature.close if hasattr(next_feature, 'close') else None
                else:
                    current_price = None

                if current_price is None:
                    continue

                # Update existing positions
                prices = {
                    epic: {
                        'high': next_feature.resistance_level or current_price,
                        'low': next_feature.support_level or current_price,
                        'close': current_price
                    }
                }
                self.update_positions(timestamp, prices)

                # Open new positions based on signal
                if signal in ['BUY', 'SELL'] and confidence >= min_confidence:
                    if len(self.positions) == 0:  # Only one position at a time
                        # Calculate SL/TP
                        atr = feature.atr_14 or (current_price * 0.02)

                        if signal == 'BUY':
                            stop_loss = current_price - (2 * atr)
                            take_profit = current_price + (3 * atr)
                        else:
                            stop_loss = current_price + (2 * atr)
                            take_profit = current_price - (3 * atr)

                        self.open_position(
                            epic,
                            signal,
                            current_price,
                            stop_loss,
                            take_profit,
                            timestamp,
                            confidence
                        )

                # Record equity
                self.record_equity(timestamp)

            # Close any remaining positions
            if self.positions and features:
                last_feature = features[-1]
                last_price = current_price

                for position in self.positions[:]:
                    self.close_position(
                        position,
                        last_price,
                        end_date,
                        'BACKTEST_END'
                    )

        # Calculate metrics
        metrics = self.calculate_metrics()

        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"Total Return: ${metrics.get('total_return', 0):.2f} ({metrics.get('total_return_percent', 0):.2f}%)")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: ${metrics.get('max_drawdown', 0):.2f} ({metrics.get('max_drawdown_percent', 0):.2f}%)")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print("="*60 + "\n")

        return metrics

    def _create_feature_vector(self, feature: Feature) -> Optional[np.ndarray]:
        """Create feature vector from Feature object"""
        try:
            features = [
                feature.rsi_14,
                feature.macd,
                feature.macd_signal,
                feature.macd_hist,
                feature.bb_width,
                feature.ma_20,
                feature.ma_50,
                feature.ma_100,
                feature.atr_14,
                feature.adx_14,
                feature.cci_20,
                feature.stochastic_k,
                feature.stochastic_d,
                feature.momentum_10,
                feature.momentum_20,
                feature.volatility_10,
                feature.volatility_20,
                feature.sentiment_score_1h or 0,
                feature.sentiment_score_4h or 0,
                feature.sentiment_score_24h or 0,
                feature.sentiment_volume_1h or 0,
                feature.sentiment_volume_4h or 0,
                feature.sentiment_volume_24h or 0
            ]

            # Check for NaN
            if any(f is None for f in features):
                return None

            return np.array(features, dtype=float)

        except Exception as e:
            return None

    def _decode_prediction(self, prediction: int) -> str:
        """Decode model prediction"""
        # 0=DOWN, 1=NEUTRAL, 2=UP
        if prediction == 2:
            return 'BUY'
        elif prediction == 0:
            return 'SELL'
        else:
            return 'NEUTRAL'
