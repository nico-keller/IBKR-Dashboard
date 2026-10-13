"""Tests for portfolio_analytics.py (pure calculations + deterministic demo fallback)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from demo_data import build_demo_payload
from portfolio_analytics import (
    AnalyticsSettings,
    build_demo_price_history,
    compute_portfolio_analytics_payload,
)


def _series(seed: int, periods: int = 180) -> pd.Series:
    idx = pd.bdate_range(end='2025-12-31', periods=periods)
    x = np.arange(periods, dtype='float64')
    drift = 0.0003 + seed * 0.00005
    cyc = 0.006 * np.sin(x / (8 + seed)) + 0.003 * np.cos(x / (15 + seed))
    r = np.clip(drift + cyc, -0.08, 0.08)
    prices = 100.0 * np.cumprod(1.0 + r)
    return pd.Series(prices, index=idx)


def test_build_demo_price_history_is_deterministic():
    hist_a, bench_a = build_demo_price_history(['AAPL', 'MSFT'], 'SPY', periods=120)
    hist_b, bench_b = build_demo_price_history(['AAPL', 'MSFT'], 'SPY', periods=120)

    assert hist_a['AAPL'].equals(hist_b['AAPL'])
    assert hist_a['MSFT'].equals(hist_b['MSFT'])
    assert bench_a.equals(bench_b)


def test_compute_payload_from_demo_history_has_metrics():
    payload = build_demo_payload()
    positions = payload['positions']

    settings = AnalyticsSettings(
        lookback_years=2,
        benchmark_ticker='SPY',
        risk_free_rate=0.0,
        max_weight=0.40,
    )
    tickers = [p['ticker'] for p in positions]
    history, benchmark = build_demo_price_history(tickers, settings.benchmark_ticker)

    out = compute_portfolio_analytics_payload(positions, history, benchmark, settings)

    assert out['status'] == 'ok'
    assert out['metrics']['annual_volatility'] is not None
    assert out['metrics']['sharpe'] is not None
    assert out['risk']['var_95'] is not None
    assert out['risk']['cvar_95'] is not None

    if out['min_vol_weights']:
        assert abs(sum(out['min_vol_weights'].values()) - 1.0) < 1e-6
        assert max(out['min_vol_weights'].values()) <= settings.max_weight + 1e-6
    if out['max_sharpe_weights']:
        assert abs(sum(out['max_sharpe_weights'].values()) - 1.0) < 1e-6
        assert max(out['max_sharpe_weights'].values()) <= settings.max_weight + 1e-6


def test_missing_history_keeps_missing_weight_unchanged():
    positions = [
        {'ticker': 'AAA', 'market_value': 50_000},
        {'ticker': 'BBB', 'market_value': 30_000},
        {'ticker': 'CCC', 'market_value': 20_000},
    ]
    history = {
        'AAA': _series(1),
        'BBB': _series(2),
        # CCC intentionally missing
    }
    benchmark = _series(7)

    settings = AnalyticsSettings(
        lookback_years=2,
        benchmark_ticker='SPY',
        risk_free_rate=0.0,
        max_weight=0.70,
    )
    out = compute_portfolio_analytics_payload(positions, history, benchmark, settings)

    assert out['status'] == 'ok'
    assert out['current_weights']['CCC'] == 0.2
    assert any('CCC' in msg for msg in out['messages'])

    if out['max_sharpe_weights']:
        assert math.isclose(out['max_sharpe_weights']['CCC'], 0.2, rel_tol=0, abs_tol=1e-9)
    if out['min_vol_weights']:
        assert math.isclose(out['min_vol_weights']['CCC'], 0.2, rel_tol=0, abs_tol=1e-9)
