"""
Enhanced ML Trading Bot Dashboard with Live Trading and Analysis Features.
Comprehensive monitoring and visualization interface.
"""

from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import (
    get_db, Model, Trade, ModelPerformanceLog,
    TradeFeedback, BacktestResult, MarketData, SentimentData
)
from ml_trading_bot.feedback import FeedbackLoop
from ml_trading_bot.continuous_learning import ModelPerformanceMonitor
from ml_trading_bot.live_trading import CapitalTradeExecutor

load_dotenv()

app = Flask(__name__)

# Initialize live executor (for position monitoring)
try:
    live_executor = CapitalTradeExecutor(
        email=os.getenv('CAPITAL_EMAIL'),
        api_key=os.getenv('CAPITAL_API_KEY'),
        api_password=os.getenv('CAPITAL_PASSWORD'),
        demo=True
    )
except:
    live_executor = None


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)


# === MODELS API ===

@app.route('/api/models')
def get_models():
    """Get all models"""
    with get_db() as db:
        models = db.query(Model).order_by(Model.trained_at.desc()).all()

        models_data = [{
            'id': m.id,
            'name': m.name,
            'version': m.version,
            'algorithm': m.algorithm,
            'trained_at': m.trained_at.isoformat() if m.trained_at else None,
            'accuracy': m.accuracy,
            'f1_score': m.f1_score,
            'sharpe_ratio': m.sharpe_ratio,
            'win_rate': m.win_rate,
            'is_active': m.is_active,
            'is_production': m.is_production,
            'total_trades': m.total_trades
        } for m in models]

        return jsonify(models_data)


@app.route('/api/models/<int:model_id>/performance')
def get_model_performance(model_id):
    """Get performance history for a model"""
    with get_db() as db:
        perf_logs = db.query(ModelPerformanceLog).filter(
            ModelPerformanceLog.model_id == model_id
        ).order_by(ModelPerformanceLog.timestamp).all()

        data = [{
            'timestamp': log.timestamp.isoformat(),
            'win_rate': log.win_rate,
            'total_pnl': log.total_pnl,
            'sharpe_ratio': log.sharpe_ratio,
            'trades_count': log.trades_count
        } for log in perf_logs]

        return jsonify(data)


# === TRADES API ===

@app.route('/api/trades')
def get_all_trades():
    """Get all recent trades"""
    days = request.args.get('days', 7, type=int)
    live_only = request.args.get('live_only', 'false').lower() == 'true'
    cutoff_date = datetime.now() - timedelta(days=days)

    with get_db() as db:
        query = db.query(Trade).filter(Trade.entry_time >= cutoff_date)

        if live_only:
            query = query.filter(Trade.is_simulated == False)

        trades = query.order_by(Trade.entry_time.desc()).all()

        trades_data = [{
            'id': t.id,
            'trade_id': t.trade_id,
            'epic': t.epic,
            'direction': t.direction,
            'size': t.size,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'entry_time': t.entry_time.isoformat() if t.entry_time else None,
            'exit_time': t.exit_time.isoformat() if t.exit_time else None,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'exit_reason': t.exit_reason,
            'confidence': t.prediction_confidence,
            'is_simulated': t.is_simulated,
            'stop_loss': t.stop_loss,
            'take_profit': t.take_profit
        } for t in trades]

        return jsonify(trades_data)


@app.route('/api/trades/open')
def get_open_trades():
    """Get currently open trades"""
    with get_db() as db:
        trades = db.query(Trade).filter(
            Trade.exit_time == None,
            Trade.is_simulated == False
        ).all()

        trades_data = [{
            'id': t.id,
            'trade_id': t.trade_id,
            'epic': t.epic,
            'direction': t.direction,
            'size': t.size,
            'entry_price': t.entry_price,
            'entry_time': t.entry_time.isoformat() if t.entry_time else None,
            'stop_loss': t.stop_loss,
            'take_profit': t.take_profit,
            'confidence': t.prediction_confidence,
            'holding_hours': (datetime.now() - t.entry_time).total_seconds() / 3600 if t.entry_time else 0
        } for t in trades]

        return jsonify(trades_data)


# === LIVE POSITIONS API ===

@app.route('/api/positions/live')
async def get_live_positions():
    """Get live positions from Capital.com API"""
    if not live_executor:
        return jsonify({'error': 'Live executor not configured'}), 500

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        await live_executor.ensure_authenticated()
        positions = await live_executor.get_positions()

        positions_data = [{
            'dealId': p.get('dealId'),
            'epic': p.get('epic'),
            'direction': p.get('direction'),
            'size': p.get('size'),
            'level': p.get('level'),
            'stopLevel': p.get('stopLevel'),
            'profitLevel': p.get('profitLevel'),
            'profitLoss': p.get('profitLoss'),
            'createdDate': p.get('createdDate')
        } for p in positions]

        return jsonify(positions_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === FEEDBACK & ANALYSIS API ===

@app.route('/api/feedback/<int:model_id>')
def get_model_feedback(model_id):
    """Get feedback and lessons for a model"""
    feedback_loop = FeedbackLoop()
    report = feedback_loop.generate_performance_report(model_id, window_days=30)

    if 'error' in report:
        return jsonify({'error': report['error']})

    recommendations = feedback_loop.identify_improvement_areas(model_id)

    return jsonify({
        'report': report,
        'recommendations': recommendations
    })


@app.route('/api/feedback/trade/<int:trade_id>')
def get_trade_feedback(trade_id):
    """Get feedback for a specific trade"""
    with get_db() as db:
        feedback = db.query(TradeFeedback).filter(
            TradeFeedback.trade_id == trade_id
        ).first()

        if not feedback:
            return jsonify({'error': 'No feedback found'}), 404

        return jsonify({
            'outcome': feedback.outcome,
            'entry_optimal': feedback.entry_optimal,
            'exit_optimal': feedback.exit_optimal,
            'price_prediction_error': feedback.price_prediction_error,
            'sentiment_accuracy': feedback.sentiment_accuracy,
            'lessons_learned': feedback.lessons_learned,
            'notes': feedback.notes,
            'analyzed_at': feedback.analyzed_at.isoformat() if feedback.analyzed_at else None
        })


@app.route('/api/analysis/lessons')
def get_common_lessons():
    """Get most common lessons learned"""
    days = request.args.get('days', 30, type=int)
    cutoff = datetime.now() - timedelta(days=days)

    with get_db() as db:
        feedbacks = db.query(TradeFeedback).join(Trade).filter(
            Trade.entry_time >= cutoff
        ).all()

        all_lessons = []
        for f in feedbacks:
            if f.lessons_learned:
                all_lessons.extend(f.lessons_learned)

        # Count frequency
        lesson_counts = {}
        for lesson in all_lessons:
            lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1

        # Sort by frequency
        top_lessons = sorted(
            [{'lesson': k, 'count': v} for k, v in lesson_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:10]

        return jsonify(top_lessons)


# === STATISTICS API ===

@app.route('/api/stats/overview')
def get_overview_stats():
    """Get system overview statistics"""
    with get_db() as db:
        total_models = db.query(Model).count()
        active_models = db.query(Model).filter(Model.is_active == True).count()
        total_trades = db.query(Trade).count()
        market_data_points = db.query(MarketData).count()
        sentiment_records = db.query(SentimentData).count()

        # Recent trades (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        recent_trades = db.query(Trade).filter(
            Trade.entry_time >= cutoff
        ).all()

        # Live trades stats
        live_trades = [t for t in recent_trades if not t.is_simulated]
        simulated_trades = [t for t in recent_trades if t.is_simulated]

        if recent_trades:
            wins = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
            total_pnl = sum(t.pnl for t in recent_trades if t.pnl)
            win_rate = (wins / len(recent_trades) * 100) if recent_trades else 0
        else:
            total_pnl = 0
            win_rate = 0

        # Open positions
        open_positions = db.query(Trade).filter(
            Trade.exit_time == None,
            Trade.is_simulated == False
        ).count()

        return jsonify({
            'total_models': total_models,
            'active_models': active_models,
            'total_trades': total_trades,
            'recent_trades_30d': len(recent_trades),
            'live_trades_30d': len(live_trades),
            'simulated_trades_30d': len(simulated_trades),
            'open_positions': open_positions,
            'win_rate_30d': win_rate,
            'total_pnl_30d': total_pnl,
            'market_data_points': market_data_points,
            'sentiment_records': sentiment_records
        })


@app.route('/api/stats/daily-pnl')
def get_daily_pnl():
    """Get daily P&L over time"""
    days = request.args.get('days', 30, type=int)
    cutoff = datetime.now() - timedelta(days=days)

    with get_db() as db:
        trades = db.query(Trade).filter(
            Trade.exit_time >= cutoff,
            Trade.exit_time != None
        ).all()

        # Group by day
        daily_data = {}
        for trade in trades:
            day = trade.exit_time.date()
            if day not in daily_data:
                daily_data[day] = {'pnl': 0, 'trades': 0, 'wins': 0}

            if trade.pnl:
                daily_data[day]['pnl'] += trade.pnl
                daily_data[day]['trades'] += 1
                if trade.pnl > 0:
                    daily_data[day]['wins'] += 1

        # Convert to list
        result = []
        for day in sorted(daily_data.keys()):
            data = daily_data[day]
            result.append({
                'date': day.isoformat(),
                'pnl': data['pnl'],
                'trades': data['trades'],
                'wins': data['wins'],
                'win_rate': (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            })

        return jsonify(result)


# === CHARTS API ===

@app.route('/api/charts/win-rate-trend')
def get_win_rate_trend():
    """Get win rate trend across all models"""
    days = request.args.get('days', 90, type=int)
    cutoff_date = datetime.now() - timedelta(days=days)

    with get_db() as db:
        trades = db.query(Trade).filter(
            Trade.entry_time >= cutoff_date,
            Trade.exit_time != None
        ).order_by(Trade.entry_time).all()

        if not trades:
            return jsonify({'dates': [], 'win_rates': []})

        # Group by day
        df = pd.DataFrame([{
            'date': t.entry_time.date(),
            'won': 1 if t.pnl and t.pnl > 0 else 0
        } for t in trades])

        daily_stats = df.groupby('date').agg({
            'won': ['sum', 'count']
        }).reset_index()

        daily_stats.columns = ['date', 'wins', 'total']
        daily_stats['win_rate'] = (daily_stats['wins'] / daily_stats['total'] * 100)

        return jsonify({
            'dates': [d.isoformat() for d in daily_stats['date']],
            'win_rates': daily_stats['win_rate'].tolist()
        })


@app.route('/api/charts/model-comparison')
def get_model_comparison():
    """Compare performance across models"""
    with get_db() as db:
        models = db.query(Model).filter(Model.is_active == True).all()

        comparison_data = []

        for model in models:
            comparison_data.append({
                'name': model.name,
                'accuracy': (model.accuracy * 100) if model.accuracy else 0,
                'f1_score': (model.f1_score * 100) if model.f1_score else 0,
                'sharpe_ratio': model.sharpe_ratio or 0,
                'win_rate': model.win_rate or 0
            })

        return jsonify(comparison_data)


@app.route('/api/health-check')
def health_check():
    """Check health of all active models"""
    monitor = ModelPerformanceMonitor()
    health_reports = monitor.monitor_all_active_models()

    return jsonify(health_reports)


# Enhanced HTML Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Trading Bot - Enhanced Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }

        h1 {
            text-align: center;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-size: 2.8em;
            font-weight: 800;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #334155;
            padding-bottom: 10px;
        }

        .tab {
            padding: 12px 24px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }

        .tab:hover {
            background: #334155;
            transform: translateY(-2px);
        }

        .tab.active {
            background: #3b82f6;
            border-color: #3b82f6;
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
        }

        .stat-label {
            color: #94a3b8;
            font-size: 0.85em;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-value {
            font-size: 2.2em;
            font-weight: bold;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .chart-container {
            background: #1e293b;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid #334155;
        }

        .chart-title {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #60a5fa;
            font-weight: 600;
        }

        table {
            width: 100%;
            background: #1e293b;
            border-radius: 12px;
            overflow: hidden;
            border-collapse: collapse;
        }

        th {
            background: #334155;
            padding: 15px;
            text-align: left;
            color: #60a5fa;
            font-weight: 600;
        }

        td {
            padding: 15px;
            border-bottom: 1px solid #334155;
        }

        tr:hover {
            background: #334155;
        }

        .badge {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 16px;
            font-size: 0.8em;
            font-weight: 600;
        }

        .badge-active { background: #22c55e; color: white; }
        .badge-inactive { background: #64748b; color: white; }
        .badge-live { background: #3b82f6; color: white; }
        .badge-simulated { background: #8b5cf6; color: white; }
        .badge-buy { background: #10b981; color: white; }
        .badge-sell { background: #ef4444; color: white; }

        .positive { color: #22c55e; font-weight: 600; }
        .negative { color: #ef4444; font-weight: 600; }

        .lessons-list {
            background: #0f172a;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ef4444;
        }

        .lesson-item {
            padding: 12px;
            margin-bottom: 10px;
            background: #1e293b;
            border-radius: 6px;
            border-left: 3px solid #3b82f6;
        }

        .refresh-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #3b82f6;
            border-radius: 20px;
            font-size: 0.9em;
            display: none;
        }

        .refresh-indicator.show {
            display: block;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="refresh-indicator" id="refreshIndicator">🔄 Refreshing...</div>

    <div class="container">
        <h1>🤖 ML Trading Bot Dashboard</h1>
        <p class="subtitle">Real-time monitoring • Live trades • ML analysis • Continuous learning</p>

        <!-- Tabs -->
        <div class="tabs">
            <div class="tab active" onclick="showTab('overview')">📊 Overview</div>
            <div class="tab" onclick="showTab('live')">🔴 Live Trading</div>
            <div class="tab" onclick="showTab('models')">🧠 Models</div>
            <div class="tab" onclick="showTab('trades')">💼 Trades</div>
            <div class="tab" onclick="showTab('analysis')">📈 Analysis</div>
            <div class="tab" onclick="showTab('feedback')">🎯 Feedback</div>
        </div>

        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="stats-grid" id="statsGrid"></div>

            <div class="chart-container">
                <div class="chart-title">Daily P&L (30 Days)</div>
                <div id="dailyPnlChart"></div>
            </div>

            <div class="chart-container">
                <div class="chart-title">Win Rate Trend (90 Days)</div>
                <div id="winRateChart"></div>
            </div>
        </div>

        <!-- Live Trading Tab -->
        <div id="live" class="tab-content">
            <h2 style="margin-bottom: 20px; color: #60a5fa;">🔴 Live Positions</h2>

            <table id="livePositionsTable">
                <thead>
                    <tr>
                        <th>Epic</th>
                        <th>Direction</th>
                        <th>Size</th>
                        <th>Entry</th>
                        <th>Current P&L</th>
                        <th>Stop Loss</th>
                        <th>Take Profit</th>
                        <th>Holding Time</th>
                    </tr>
                </thead>
                <tbody id="livePositionsBody"></tbody>
            </table>

            <h2 style="margin: 40px 0 20px; color: #60a5fa;">📋 Open Trades (Database)</h2>
            <table id="openTradesTable">
                <thead>
                    <tr>
                        <th>Epic</th>
                        <th>Direction</th>
                        <th>Entry Time</th>
                        <th>Entry Price</th>
                        <th>Confidence</th>
                        <th>SL/TP</th>
                        <th>Holding</th>
                    </tr>
                </thead>
                <tbody id="openTradesBody"></tbody>
            </table>
        </div>

        <!-- Models Tab -->
        <div id="models" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Model Performance Comparison</div>
                <div id="modelComparisonChart"></div>
            </div>

            <h2 style="margin: 40px 0 20px; color: #60a5fa;">🧠 All Models</h2>
            <table id="modelsTable">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Algorithm</th>
                        <th>Accuracy</th>
                        <th>Win Rate</th>
                        <th>Sharpe</th>
                        <th>Trades</th>
                        <th>Trained</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="modelsTableBody"></tbody>
            </table>
        </div>

        <!-- Trades Tab -->
        <div id="trades" class="tab-content">
            <div style="margin-bottom: 20px;">
                <button onclick="loadTrades(7, false)" style="padding: 10px 20px; margin-right: 10px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer;">Last 7 Days</button>
                <button onclick="loadTrades(30, false)" style="padding: 10px 20px; margin-right: 10px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer;">Last 30 Days</button>
                <button onclick="loadTrades(30, true)" style="padding: 10px 20px; background: #10b981; color: white; border: none; border-radius: 6px; cursor: pointer;">Live Only</button>
            </div>

            <table id="tradesTable">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Epic</th>
                        <th>Direction</th>
                        <th>Entry/Exit</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Exit Reason</th>
                        <th>Type</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody id="tradesTableBody"></tbody>
            </table>
        </div>

        <!-- Analysis Tab -->
        <div id="analysis" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Model Health Check</div>
                <div id="healthCheckResults"></div>
            </div>
        </div>

        <!-- Feedback Tab -->
        <div id="feedback" class="tab-content">
            <h2 style="margin-bottom: 20px; color: #60a5fa;">🎯 Common Lessons Learned</h2>
            <div id="lessonsContainer" class="lessons-list"></div>
        </div>
    </div>

    <script>
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });

            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');

            // Load tab-specific data
            if (tabName === 'live') {
                loadLivePositions();
                loadOpenTrades();
            } else if (tabName === 'models') {
                loadModels();
            } else if (tabName === 'trades') {
                loadTrades(7, false);
            } else if (tabName === 'analysis') {
                loadHealthCheck();
            } else if (tabName === 'feedback') {
                loadLessons();
            }
        }

        // Show refresh indicator
        function showRefresh() {
            const indicator = document.getElementById('refreshIndicator');
            indicator.classList.add('show');
            setTimeout(() => indicator.classList.remove('show'), 1000);
        }

        // Load overview stats
        async function loadStats() {
            showRefresh();
            const res = await axios.get('/api/stats/overview');
            const stats = res.data;

            const statsHTML = `
                <div class="stat-card">
                    <div class="stat-label">Total Models</div>
                    <div class="stat-value">${stats.total_models}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Active Models</div>
                    <div class="stat-value">${stats.active_models}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Open Positions</div>
                    <div class="stat-value">${stats.open_positions}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Live Trades (30d)</div>
                    <div class="stat-value">${stats.live_trades_30d}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Win Rate (30d)</div>
                    <div class="stat-value ${stats.win_rate_30d >= 50 ? 'positive' : 'negative'}">
                        ${stats.win_rate_30d.toFixed(1)}%
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total P&L (30d)</div>
                    <div class="stat-value ${stats.total_pnl_30d >= 0 ? 'positive' : 'negative'}">
                        $${stats.total_pnl_30d.toFixed(2)}
                    </div>
                </div>
            `;

            document.getElementById('statsGrid').innerHTML = statsHTML;
        }

        // Load daily P&L chart
        async function loadDailyPnl() {
            const res = await axios.get('/api/stats/daily-pnl?days=30');
            const data = res.data;

            const trace = {
                x: data.map(d => d.date),
                y: data.map(d => d.pnl),
                type: 'bar',
                marker: {
                    color: data.map(d => d.pnl >= 0 ? '#22c55e' : '#ef4444')
                }
            };

            const layout = {
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                xaxis: { gridcolor: '#334155' },
                yaxis: { gridcolor: '#334155', title: 'P&L ($)' },
                margin: { l: 50, r: 20, t: 20, b: 50 }
            };

            Plotly.newPlot('dailyPnlChart', [trace], layout);
        }

        // Load win rate trend
        async function loadWinRateTrend() {
            const res = await axios.get('/api/charts/win-rate-trend?days=90');
            const data = res.data;

            const trace = {
                x: data.dates,
                y: data.win_rates,
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#60a5fa', width: 3 },
                marker: { size: 6 }
            };

            const layout = {
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                xaxis: { gridcolor: '#334155' },
                yaxis: { gridcolor: '#334155', title: 'Win Rate (%)' },
                margin: { l: 50, r: 20, t: 20, b: 50 }
            };

            Plotly.newPlot('winRateChart', [trace], layout);
        }

        // Load live positions
        async function loadLivePositions() {
            try {
                const res = await axios.get('/api/positions/live');
                const positions = res.data;

                if (positions.error) {
                    document.getElementById('livePositionsBody').innerHTML =
                        `<tr><td colspan="8" style="text-align: center; color: #ef4444;">${positions.error}</td></tr>`;
                    return;
                }

                if (positions.length === 0) {
                    document.getElementById('livePositionsBody').innerHTML =
                        '<tr><td colspan="8" style="text-align: center; color: #94a3b8;">No open positions</td></tr>';
                    return;
                }

                const tbody = document.getElementById('livePositionsBody');
                tbody.innerHTML = positions.map(p => `
                    <tr>
                        <td><strong>${p.epic}</strong></td>
                        <td><span class="badge ${p.direction === 'BUY' ? 'badge-buy' : 'badge-sell'}">${p.direction}</span></td>
                        <td>${p.size}</td>
                        <td>${p.level}</td>
                        <td class="${p.profitLoss >= 0 ? 'positive' : 'negative'}">$${p.profitLoss ? p.profitLoss.toFixed(2) : '0.00'}</td>
                        <td>${p.stopLevel || 'N/A'}</td>
                        <td>${p.profitLevel || 'N/A'}</td>
                        <td>${new Date(p.createdDate).toLocaleString()}</td>
                    </tr>
                `).join('');
            } catch (error) {
                console.error('Error loading live positions:', error);
            }
        }

        // Load open trades from DB
        async function loadOpenTrades() {
            const res = await axios.get('/api/trades/open');
            const trades = res.data;

            if (trades.length === 0) {
                document.getElementById('openTradesBody').innerHTML =
                    '<tr><td colspan="7" style="text-align: center; color: #94a3b8;">No open trades in database</td></tr>';
                return;
            }

            const tbody = document.getElementById('openTradesBody');
            tbody.innerHTML = trades.map(t => `
                <tr>
                    <td><strong>${t.epic}</strong></td>
                    <td><span class="badge ${t.direction === 'BUY' ? 'badge-buy' : 'badge-sell'}">${t.direction}</span></td>
                    <td>${new Date(t.entry_time).toLocaleString()}</td>
                    <td>${t.entry_price}</td>
                    <td>${t.confidence ? (t.confidence * 100).toFixed(0) + '%' : 'N/A'}</td>
                    <td>${t.stop_loss}/${t.take_profit}</td>
                    <td>${t.holding_hours.toFixed(1)}h</td>
                </tr>
            `).join('');
        }

        // Load models
        async function loadModels() {
            const res = await axios.get('/api/models');
            const models = res.data;

            // Model comparison chart
            const trace = {
                x: models.filter(m => m.is_active).map(m => m.name),
                y: models.filter(m => m.is_active).map(m => m.win_rate || 0),
                type: 'bar',
                marker: { color: '#60a5fa' }
            };

            const layout = {
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                xaxis: { gridcolor: '#334155' },
                yaxis: { gridcolor: '#334155', title: 'Win Rate (%)' },
                margin: { l: 50, r: 20, t: 20, b: 100 }
            };

            Plotly.newPlot('modelComparisonChart', [trace], layout);

            // Models table
            const tbody = document.getElementById('modelsTableBody');
            tbody.innerHTML = models.map(m => `
                <tr>
                    <td><strong>${m.name}</strong><br><small>v${m.version}</small></td>
                    <td>${m.algorithm}</td>
                    <td>${m.accuracy ? (m.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${m.win_rate ? m.win_rate.toFixed(1) + '%' : 'N/A'}</td>
                    <td>${m.sharpe_ratio ? m.sharpe_ratio.toFixed(2) : 'N/A'}</td>
                    <td>${m.total_trades || 0}</td>
                    <td>${m.trained_at ? new Date(m.trained_at).toLocaleDateString() : 'N/A'}</td>
                    <td>
                        <span class="badge ${m.is_active ? 'badge-active' : 'badge-inactive'}">
                            ${m.is_active ? 'Active' : 'Inactive'}
                        </span>
                    </td>
                </tr>
            `).join('');
        }

        // Load trades
        async function loadTrades(days, liveOnly) {
            const res = await axios.get(`/api/trades?days=${days}&live_only=${liveOnly}`);
            const trades = res.data;

            const tbody = document.getElementById('tradesTableBody');
            tbody.innerHTML = trades.map(t => `
                <tr>
                    <td>${new Date(t.entry_time).toLocaleString()}</td>
                    <td><strong>${t.epic}</strong></td>
                    <td><span class="badge ${t.direction === 'BUY' ? 'badge-buy' : 'badge-sell'}">${t.direction}</span></td>
                    <td>${t.entry_price} → ${t.exit_price || 'Open'}</td>
                    <td class="${t.pnl >= 0 ? 'positive' : 'negative'}">$${t.pnl ? t.pnl.toFixed(2) : '-'}</td>
                    <td class="${t.pnl_percent >= 0 ? 'positive' : 'negative'}">${t.pnl_percent ? t.pnl_percent.toFixed(2) + '%' : '-'}</td>
                    <td>${t.exit_reason || 'Open'}</td>
                    <td><span class="badge ${t.is_simulated ? 'badge-simulated' : 'badge-live'}">${t.is_simulated ? 'SIM' : 'LIVE'}</span></td>
                    <td>${t.confidence ? (t.confidence * 100).toFixed(0) + '%' : 'N/A'}</td>
                </tr>
            `).join('');
        }

        // Load health check
        async function loadHealthCheck() {
            const res = await axios.get('/api/health-check');
            const reports = res.data;

            const container = document.getElementById('healthCheckResults');
            container.innerHTML = reports.map(r => `
                <div class="lesson-item">
                    <h3 style="color: ${r.needs_retraining ? '#ef4444' : '#22c55e'}; margin-bottom: 10px;">
                        ${r.needs_retraining ? '⚠️' : '✅'} ${r.model_name}
                    </h3>
                    <p><strong>Status:</strong> ${r.status}</p>
                    <p><strong>Win Rate:</strong> ${r.win_rate ? r.win_rate.toFixed(1) + '%' : 'N/A'}</p>
                    <p><strong>Trades:</strong> ${r.trades_evaluated || 0}</p>
                    <p><strong>Days Since Training:</strong> ${r.days_since_training}</p>
                    <p><strong>Recommendation:</strong> <span style="color: ${r.recommendation === 'RETRAIN' ? '#ef4444' : '#22c55e'}">${r.recommendation}</span></p>
                    ${r.reasons ? '<p><strong>Reasons:</strong> ' + r.reasons.filter(x => x).join(', ') + '</p>' : ''}
                </div>
            `).join('');
        }

        // Load lessons
        async function loadLessons() {
            const res = await axios.get('/api/analysis/lessons?days=30');
            const lessons = res.data;

            const container = document.getElementById('lessonsContainer');
            container.innerHTML = lessons.map((l, i) => `
                <div class="lesson-item">
                    <strong style="color: #60a5fa;">#${i + 1}</strong> (${l.count} times)
                    <br>${l.lesson}
                </div>
            `).join('');
        }

        // Initialize
        async function init() {
            await loadStats();
            await loadDailyPnl();
            await loadWinRateTrend();
        }

        init();

        // Auto-refresh every 30 seconds
        setInterval(init, 30000);
    </script>
</body>
</html>
"""


if __name__ == '__main__':
    print("Starting Enhanced ML Trading Bot Dashboard...")
    print("Access dashboard at: http://localhost:5001")
    print("\nFeatures:")
    print("  • Real-time live position monitoring")
    print("  • Trade analysis and feedback")
    print("  • Model performance tracking")
    print("  • Health checks and recommendations")
    print("  • Common lessons learned")

    app.run(debug=True, host='0.0.0.0', port=5001)
