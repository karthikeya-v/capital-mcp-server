"""
Feedback and Learning System.
Analyzes trade results to identify mistakes and improve models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import (
    Trade, TradeFeedback, Model, ModelPerformanceLog,
    Feature, get_db
)


class TradeAnalyzer:
    """
    Analyzes individual trades to understand what went right/wrong.
    """

    def __init__(self):
        pass

    def analyze_trade(
        self,
        trade: Trade,
        market_data: pd.DataFrame
    ) -> Dict:
        """
        Analyze a single trade.

        Args:
            trade: Trade object
            market_data: Market data during trade period

        Returns:
            Analysis dictionary
        """
        analysis = {
            'trade_id': trade.trade_id,
            'outcome': self._classify_outcome(trade),
            'entry_quality': None,
            'exit_quality': None,
            'price_prediction_error': None,
            'lessons': []
        }

        # Analyze entry timing
        entry_analysis = self._analyze_entry(trade, market_data)
        analysis['entry_quality'] = entry_analysis

        # Analyze exit timing
        exit_analysis = self._analyze_exit(trade, market_data)
        analysis['exit_quality'] = exit_analysis

        # Calculate prediction error
        if trade.direction == 'BUY':
            expected_move = trade.take_profit - trade.entry_price
            actual_move = trade.exit_price - trade.entry_price
        else:
            expected_move = trade.entry_price - trade.take_profit
            actual_move = trade.entry_price - trade.exit_price

        if expected_move != 0:
            prediction_error = abs(actual_move - expected_move) / abs(expected_move) * 100
        else:
            prediction_error = 0

        analysis['price_prediction_error'] = prediction_error

        # Extract lessons
        analysis['lessons'] = self._extract_lessons(trade, analysis)

        return analysis

    def _classify_outcome(self, trade: Trade) -> str:
        """Classify trade outcome"""
        if trade.pnl > 0:
            return 'WIN'
        elif trade.pnl < 0:
            return 'LOSS'
        else:
            return 'BREAKEVEN'

    def _analyze_entry(self, trade: Trade, market_data: pd.DataFrame) -> Dict:
        """Analyze entry timing quality"""
        # Get market data around entry
        entry_time = trade.entry_time
        before_entry = market_data[market_data['timestamp'] < entry_time].tail(10)
        after_entry = market_data[market_data['timestamp'] >= entry_time].head(10)

        if before_entry.empty or after_entry.empty:
            return {'quality': 'UNKNOWN', 'score': 0.5}

        # Check if entry was at a good price
        if trade.direction == 'BUY':
            # For long: better if we bought near recent lows
            recent_low = before_entry['low'].min()
            recent_high = before_entry['high'].max()
            entry_position = (trade.entry_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

            # Lower is better for buys
            if entry_position < 0.3:
                quality = 'EXCELLENT'
                score = 0.9
            elif entry_position < 0.5:
                quality = 'GOOD'
                score = 0.7
            else:
                quality = 'POOR'
                score = 0.3

        else:  # SELL
            # For short: better if we sold near recent highs
            recent_low = before_entry['low'].min()
            recent_high = before_entry['high'].max()
            entry_position = (trade.entry_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

            # Higher is better for sells
            if entry_position > 0.7:
                quality = 'EXCELLENT'
                score = 0.9
            elif entry_position > 0.5:
                quality = 'GOOD'
                score = 0.7
            else:
                quality = 'POOR'
                score = 0.3

        return {
            'quality': quality,
            'score': score,
            'entry_position': entry_position
        }

    def _analyze_exit(self, trade: Trade, market_data: pd.DataFrame) -> Dict:
        """Analyze exit timing quality"""
        # Get market data after exit
        exit_time = trade.exit_time
        after_exit = market_data[market_data['timestamp'] > exit_time].head(20)

        if after_exit.empty:
            return {'quality': 'UNKNOWN', 'score': 0.5, 'optimal': True}

        # Check if we could have done better
        if trade.direction == 'BUY':
            # For closed long: check if price went higher after exit
            max_after_exit = after_exit['high'].max()
            could_have_made = max_after_exit - trade.exit_price

            if trade.exit_reason == 'TP':
                # TP is always good
                quality = 'EXCELLENT'
                score = 1.0
                optimal = True
            elif trade.exit_reason == 'SL':
                # Check if price recovered
                if could_have_made > 0:
                    quality = 'POOR'
                    score = 0.2
                    optimal = False
                else:
                    quality = 'GOOD'
                    score = 0.7
                    optimal = True
            else:
                # Manual exit
                if could_have_made > (trade.exit_price - trade.entry_price):
                    quality = 'POOR'
                    score = 0.3
                    optimal = False
                else:
                    quality = 'GOOD'
                    score = 0.7
                    optimal = True

        else:  # SELL
            # For closed short: check if price went lower after exit
            min_after_exit = after_exit['low'].min()
            could_have_made = trade.exit_price - min_after_exit

            if trade.exit_reason == 'TP':
                quality = 'EXCELLENT'
                score = 1.0
                optimal = True
            elif trade.exit_reason == 'SL':
                if could_have_made > 0:
                    quality = 'POOR'
                    score = 0.2
                    optimal = False
                else:
                    quality = 'GOOD'
                    score = 0.7
                    optimal = True
            else:
                if could_have_made > (trade.entry_price - trade.exit_price):
                    quality = 'POOR'
                    score = 0.3
                    optimal = False
                else:
                    quality = 'GOOD'
                    score = 0.7
                    optimal = True

        return {
            'quality': quality,
            'score': score,
            'optimal': optimal
        }

    def _extract_lessons(self, trade: Trade, analysis: Dict) -> List[str]:
        """Extract lessons learned from trade"""
        lessons = []

        # Entry lessons
        if analysis['entry_quality']:
            if analysis['entry_quality']['quality'] == 'POOR':
                lessons.append(f"Entry timing was suboptimal - entered at {analysis['entry_quality']['entry_position']:.2f} of recent range")

        # Exit lessons
        if analysis['exit_quality']:
            if not analysis['exit_quality'].get('optimal', True):
                lessons.append(f"Exit was premature - price moved favorably after exit")

        # Stop loss lessons
        if trade.exit_reason == 'SL' and trade.pnl < 0:
            lessons.append("Stop loss was hit - consider wider stops or better entry timing")

        # Confidence lessons
        if trade.prediction_confidence:
            if trade.pnl < 0 and trade.prediction_confidence > 0.8:
                lessons.append(f"High confidence ({trade.prediction_confidence:.2f}) but trade failed - model may be overconfident")
            elif trade.pnl > 0 and trade.prediction_confidence < 0.6:
                lessons.append(f"Low confidence ({trade.prediction_confidence:.2f}) but trade succeeded - may be underestimating")

        return lessons


class FeedbackLoop:
    """
    Continuous feedback system that learns from trading results.
    """

    def __init__(self):
        self.analyzer = TradeAnalyzer()

    def process_trade_feedback(self, trade_id: int):
        """
        Process feedback for a single trade.

        Args:
            trade_id: Database ID of trade
        """
        with get_db() as db:
            # Get trade
            trade = db.query(Trade).filter(Trade.id == trade_id).first()

            if not trade:
                print(f"Trade {trade_id} not found")
                return

            # Get market data for analysis period
            start_time = trade.entry_time - timedelta(hours=2)
            end_time = trade.exit_time + timedelta(hours=2)

            features = db.query(Feature).filter(
                Feature.epic == trade.epic,
                Feature.timestamp >= start_time,
                Feature.timestamp <= end_time
            ).order_by(Feature.timestamp).all()

            if not features:
                print(f"No market data found for analysis")
                return

            # Create market data DataFrame
            market_data = pd.DataFrame([{
                'timestamp': f.timestamp,
                'high': f.resistance_level or 0,
                'low': f.support_level or 0,
                'close': f.close if hasattr(f, 'close') else 0
            } for f in features])

            # Analyze trade
            analysis = self.analyzer.analyze_trade(trade, market_data)

            # Check if feedback already exists
            existing_feedback = db.query(TradeFeedback).filter(
                TradeFeedback.trade_id == trade_id
            ).first()

            if existing_feedback:
                # Update existing
                existing_feedback.outcome = analysis['outcome']
                existing_feedback.entry_optimal = analysis['entry_quality']['score'] > 0.6
                existing_feedback.exit_optimal = analysis['exit_quality'].get('optimal', True)
                existing_feedback.price_prediction_error = analysis['price_prediction_error']
                existing_feedback.lessons_learned = analysis['lessons']
                existing_feedback.analyzed_at = datetime.now()
            else:
                # Create new feedback
                feedback = TradeFeedback(
                    trade_id=trade_id,
                    outcome=analysis['outcome'],
                    entry_optimal=analysis['entry_quality']['score'] > 0.6,
                    exit_optimal=analysis['exit_quality'].get('optimal', True),
                    price_prediction_error=analysis['price_prediction_error'],
                    lessons_learned=analysis['lessons'],
                    analyzed_at=datetime.now()
                )
                db.add(feedback)

            db.commit()
            print(f"✓ Feedback processed for trade {trade.trade_id}")

    def generate_performance_report(
        self,
        model_id: int,
        window_days: int = 30
    ) -> Dict:
        """
        Generate performance report for a model.

        Args:
            model_id: Model database ID
            window_days: Analysis window in days

        Returns:
            Performance report
        """
        with get_db() as db:
            # Get model
            model = db.query(Model).filter(Model.id == model_id).first()

            if not model:
                return {}

            # Get trades in window
            cutoff_date = datetime.now() - timedelta(days=window_days)

            trades = db.query(Trade).filter(
                Trade.model_id == model_id,
                Trade.entry_time >= cutoff_date
            ).all()

            if not trades:
                return {'error': 'No trades found in window'}

            # Calculate metrics
            total_trades = len(trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            losses = sum(1 for t in trades if t.pnl < 0)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(t.pnl for t in trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            # Analyze feedback
            feedbacks = db.query(TradeFeedback).join(Trade).filter(
                Trade.model_id == model_id,
                Trade.entry_time >= cutoff_date
            ).all()

            if feedbacks:
                entry_quality = sum(1 for f in feedbacks if f.entry_optimal) / len(feedbacks) * 100
                exit_quality = sum(1 for f in feedbacks if f.exit_optimal) / len(feedbacks) * 100
                avg_prediction_error = np.mean([f.price_prediction_error for f in feedbacks if f.price_prediction_error])

                # Aggregate lessons
                all_lessons = []
                for f in feedbacks:
                    if f.lessons_learned:
                        all_lessons.extend(f.lessons_learned)

                # Count lesson frequency
                lesson_counts = {}
                for lesson in all_lessons:
                    lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1

                top_lessons = sorted(
                    lesson_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

            else:
                entry_quality = 0
                exit_quality = 0
                avg_prediction_error = 0
                top_lessons = []

            report = {
                'model_id': model_id,
                'model_name': model.name,
                'window_days': window_days,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'entry_quality_percent': entry_quality,
                'exit_quality_percent': exit_quality,
                'avg_prediction_error': avg_prediction_error,
                'top_lessons': top_lessons,
                'generated_at': datetime.now()
            }

            # Save performance log
            perf_log = ModelPerformanceLog(
                model_id=model_id,
                timestamp=datetime.now(),
                window_period=f"{window_days}D",
                trades_count=total_trades,
                win_count=wins,
                loss_count=losses,
                win_rate=win_rate,
                total_pnl=total_pnl,
                average_pnl=avg_pnl,
                prediction_accuracy=entry_quality / 100  # Simplified
            )

            db.add(perf_log)
            db.commit()

            return report

    def identify_improvement_areas(self, model_id: int) -> List[Dict]:
        """
        Identify areas for model improvement.

        Args:
            model_id: Model database ID

        Returns:
            List of improvement recommendations
        """
        report = self.generate_performance_report(model_id, window_days=30)

        if not report or 'error' in report:
            return []

        recommendations = []

        # Check win rate
        if report['win_rate'] < 50:
            recommendations.append({
                'area': 'Win Rate',
                'issue': f"Win rate is low at {report['win_rate']:.1f}%",
                'suggestion': 'Consider adjusting entry criteria or improving signal quality'
            })

        # Check entry quality
        if report['entry_quality_percent'] < 60:
            recommendations.append({
                'area': 'Entry Timing',
                'issue': f"Entry quality is {report['entry_quality_percent']:.1f}%",
                'suggestion': 'Add features for better entry timing or use limit orders'
            })

        # Check exit quality
        if report['exit_quality_percent'] < 60:
            recommendations.append({
                'area': 'Exit Strategy',
                'issue': f"Exit quality is {report['exit_quality_percent']:.1f}%",
                'suggestion': 'Review stop-loss and take-profit levels, consider trailing stops'
            })

        # Check prediction error
        if report['avg_prediction_error'] > 20:
            recommendations.append({
                'area': 'Price Prediction',
                'issue': f"Average prediction error is {report['avg_prediction_error']:.1f}%",
                'suggestion': 'Retrain model with more data or add more relevant features'
            })

        # Check common lessons
        if report['top_lessons']:
            most_common_lesson = report['top_lessons'][0]
            recommendations.append({
                'area': 'Recurring Issue',
                'issue': f"Most common lesson: {most_common_lesson[0]} ({most_common_lesson[1]} times)",
                'suggestion': 'Address this recurring pattern in next model iteration'
            })

        return recommendations
