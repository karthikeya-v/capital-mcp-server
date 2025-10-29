"""
ML Trading Bot Dashboard.
Web-based monitoring and visualization interface.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_trading_bot.database import (
    get_db, Model, Trade, ModelPerformanceLog,
    TradeFeedback, BacktestResult
)
from ml_trading_bot.feedback import FeedbackLoop
from ml_trading_bot.continuous_learning import ModelPerformanceMonitor

app = Flask(__name__)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


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
            'is_production': m.is_production
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


@app.route('/api/models/<int:model_id>/trades')
def get_model_trades(model_id):
    """Get trades for a model"""
    days = request.args.get('days', 30, type=int)
    cutoff_date = datetime.now() - timedelta(days=days)

    with get_db() as db:
        trades = db.query(Trade).filter(
            Trade.model_id == model_id,
            Trade.entry_time >= cutoff_date
        ).order_by(Trade.entry_time.desc()).all()

        trades_data = [{
            'id': t.id,
            'trade_id': t.trade_id,
            'epic': t.epic,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'entry_time': t.entry_time.isoformat() if t.entry_time else None,
            'exit_time': t.exit_time.isoformat() if t.exit_time else None,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'exit_reason': t.exit_reason,
            'confidence': t.prediction_confidence
        } for t in trades]

        return jsonify(trades_data)


@app.route('/api/stats/overview')
def get_overview_stats():
    """Get system overview statistics"""
    with get_db() as db:
        from ml_trading_bot.database import MarketData, SentimentData, Feature

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

        if recent_trades:
            wins = sum(1 for t in recent_trades if t.pnl > 0)
            total_pnl = sum(t.pnl for t in recent_trades)
            win_rate = (wins / len(recent_trades) * 100) if recent_trades else 0
        else:
            total_pnl = 0
            win_rate = 0

        return jsonify({
            'total_models': total_models,
            'active_models': active_models,
            'total_trades': total_trades,
            'recent_trades_30d': len(recent_trades) if recent_trades else 0,
            'win_rate_30d': win_rate,
            'total_pnl_30d': total_pnl,
            'market_data_points': market_data_points,
            'sentiment_records': sentiment_records
        })


@app.route('/api/charts/equity-curve/<int:model_id>')
def get_equity_curve(model_id):
    """Get equity curve data"""
    days = request.args.get('days', 30, type=int)
    cutoff_date = datetime.now() - timedelta(days=days)

    with get_db() as db:
        trades = db.query(Trade).filter(
            Trade.model_id == model_id,
            Trade.entry_time >= cutoff_date
        ).order_by(Trade.entry_time).all()

        if not trades:
            return jsonify({'timestamps': [], 'equity': []})

        # Calculate cumulative equity
        equity = [0]
        timestamps = [trades[0].entry_time.isoformat()]

        for trade in trades:
            equity.append(equity[-1] + (trade.pnl or 0))
            timestamps.append(trade.exit_time.isoformat() if trade.exit_time else trade.entry_time.isoformat())

        return jsonify({
            'timestamps': timestamps,
            'equity': equity
        })


@app.route('/api/charts/win-rate-trend')
def get_win_rate_trend():
    """Get win rate trend across all models"""
    days = request.args.get('days', 90, type=int)
    cutoff_date = datetime.now() - timedelta(days=days)

    with get_db() as db:
        trades = db.query(Trade).filter(
            Trade.entry_time >= cutoff_date
        ).order_by(Trade.entry_time).all()

        if not trades:
            return jsonify({'dates': [], 'win_rates': []})

        # Group by day and calculate win rate
        df = pd.DataFrame([{
            'date': t.entry_time.date(),
            'won': 1 if t.pnl > 0 else 0
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


@app.route('/api/feedback/<int:model_id>')
def get_model_feedback(model_id):
    """Get feedback and lessons for a model"""
    feedback_loop = FeedbackLoop()
    report = feedback_loop.generate_performance_report(model_id, window_days=30)

    if 'error' in report:
        return jsonify({'error': report['error']})

    # Get improvement recommendations
    recommendations = feedback_loop.identify_improvement_areas(model_id)

    return jsonify({
        'report': report,
        'recommendations': recommendations
    })


@app.route('/api/health-check')
def health_check():
    """Check health of all active models"""
    monitor = ModelPerformanceMonitor()
    health_reports = monitor.monitor_all_active_models()

    return jsonify(health_reports)


@app.route('/api/backtests')
def get_backtests():
    """Get backtest results"""
    with get_db() as db:
        backtests = db.query(BacktestResult).order_by(
            BacktestResult.created_at.desc()
        ).limit(20).all()

        backtest_data = [{
            'id': bt.id,
            'backtest_id': bt.backtest_id,
            'model_id': bt.model_id,
            'start_date': bt.start_date.isoformat() if bt.start_date else None,
            'end_date': bt.end_date.isoformat() if bt.end_date else None,
            'total_return_percent': bt.total_return_percent,
            'sharpe_ratio': bt.sharpe_ratio,
            'max_drawdown_percent': bt.max_drawdown_percent,
            'win_rate': bt.win_rate,
            'total_trades': bt.total_trades,
            'profit_factor': bt.profit_factor
        } for bt in backtests]

        return jsonify(backtest_data)


# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Trading Bot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            color: #60a5fa;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #1e293b;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #334155;
        }
        .stat-label {
            color: #94a3b8;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #60a5fa;
        }
        .chart-container {
            background: #1e293b;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }
        .models-table {
            width: 100%;
            background: #1e293b;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 20px;
        }
        .models-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .models-table th {
            background: #334155;
            padding: 15px;
            text-align: left;
            color: #60a5fa;
        }
        .models-table td {
            padding: 15px;
            border-bottom: 1px solid #334155;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .badge-active { background: #22c55e; color: white; }
        .badge-inactive { background: #64748b; color: white; }
        .positive { color: #22c55e; }
        .negative { color: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ML Trading Bot Dashboard</h1>

        <div class="stats-grid" id="stats-grid">
            <!-- Stats will be loaded here -->
        </div>

        <div class="chart-container">
            <h2>Win Rate Trend (90 Days)</h2>
            <div id="winRateChart"></div>
        </div>

        <div class="chart-container">
            <h2>Model Performance Comparison</h2>
            <div id="modelComparisonChart"></div>
        </div>

        <div class="models-table">
            <h2 style="padding: 20px;">Active Models</h2>
            <table id="modelsTable">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Algorithm</th>
                        <th>Accuracy</th>
                        <th>Win Rate</th>
                        <th>Sharpe Ratio</th>
                        <th>Trained</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="modelsTableBody">
                    <!-- Models will be loaded here -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Load stats
        async function loadStats() {
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
                    <div class="stat-label">Total Trades</div>
                    <div class="stat-value">${stats.total_trades}</div>
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

            document.getElementById('stats-grid').innerHTML = statsHTML;
        }

        // Load models
        async function loadModels() {
            const res = await axios.get('/api/models');
            const models = res.data;

            const tbody = document.getElementById('modelsTableBody');
            tbody.innerHTML = models.map(m => `
                <tr>
                    <td><strong>${m.name}</strong><br><small>v${m.version}</small></td>
                    <td>${m.algorithm}</td>
                    <td>${m.accuracy ? (m.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${m.win_rate ? m.win_rate.toFixed(1) + '%' : 'N/A'}</td>
                    <td>${m.sharpe_ratio ? m.sharpe_ratio.toFixed(2) : 'N/A'}</td>
                    <td>${m.trained_at ? new Date(m.trained_at).toLocaleDateString() : 'N/A'}</td>
                    <td>
                        <span class="badge ${m.is_active ? 'badge-active' : 'badge-inactive'}">
                            ${m.is_active ? 'Active' : 'Inactive'}
                        </span>
                    </td>
                </tr>
            `).join('');
        }

        // Load win rate chart
        async function loadWinRateChart() {
            const res = await axios.get('/api/charts/win-rate-trend?days=90');
            const data = res.data;

            const trace = {
                x: data.dates,
                y: data.win_rates,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Win Rate',
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

        // Load model comparison chart
        async function loadModelComparison() {
            const res = await axios.get('/api/charts/model-comparison');
            const data = res.data;

            const trace = {
                x: data.map(m => m.name),
                y: data.map(m => m.win_rate),
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
        }

        // Initialize dashboard
        async function init() {
            await loadStats();
            await loadModels();
            await loadWinRateChart();
            await loadModelComparison();
        }

        init();

        // Refresh every 30 seconds
        setInterval(init, 30000);
    </script>
</body>
</html>
"""


# Save template
if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)

    with open('templates/dashboard.html', 'w') as f:
        f.write(DASHBOARD_HTML)

    print("Starting ML Trading Bot Dashboard...")
    print("Access dashboard at: http://localhost:5001")

    app.run(debug=True, host='0.0.0.0', port=5001)
