"""Portfolio analytics engine for optimization and risk metrics.

The public entrypoint is ``get_portfolio_analytics`` which:
- derives current market-value weights from portfolio-data positions
- fetches/caches adjusted-close history (or deterministic demo history)
- computes portfolio/risk metrics from daily returns
- computes long-only, capped min-vol and max-Sharpe suggested weights

All metrics use 252 trading days for annualisation.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cache_util import cached_fetch
from config import cfg
from net_util import fetch_parallel, fetch_with_retry

log = logging.getLogger(__name__)

_TRADING_DAYS = 252
_TTL = 3600 * 4
_MIN_OBS = 60


@dataclass(frozen=True)
class AnalyticsSettings:
    lookback_years: int
    benchmark_ticker: str
    risk_free_rate: float
    max_weight: float


def _settings() -> AnalyticsSettings:
    raw = cfg.get('analytics') or {}
    return AnalyticsSettings(
        lookback_years=max(1, int(raw.get('lookback_years', 2))),
        benchmark_ticker=str(raw.get('benchmark_ticker', 'SPY') or 'SPY'),
        risk_free_rate=float(raw.get('risk_free_rate', 0.0)),
        max_weight=min(max(float(raw.get('max_weight', 0.40)), 0.05), 1.0),
    )


def _sanitize_ticker(value: object) -> str:
    return str(value or '').strip().upper()


def _weights_from_positions(positions: list[dict]) -> dict[str, float]:
    by_ticker: dict[str, float] = {}
    for p in positions:
        ticker = _sanitize_ticker(p.get('ticker'))
        if not ticker:
            continue
        mv = p.get('market_value')
        try:
            mv_f = float(mv)
        except (TypeError, ValueError):
            mv_f = 0.0
        if mv_f > 0:
            by_ticker[ticker] = by_ticker.get(ticker, 0.0) + mv_f

    total = sum(by_ticker.values())
    if total <= 0:
        return {}
    return {t: v / total for t, v in by_ticker.items()}


def _series_from_cached_payload(payload: dict | None) -> pd.Series | None:
    if not payload:
        return None
    dates = payload.get('dates')
    prices = payload.get('prices')
    if not dates or not prices or len(dates) != len(prices):
        return None
    idx = pd.to_datetime(dates)
    s = pd.Series(prices, index=idx, dtype='float64').dropna()
    if len(s) < 2:
        return None
    return s


def _fetch_symbol_history(symbol: str, period: str) -> pd.Series | None:
    symbol = _sanitize_ticker(symbol)
    if not symbol:
        return None

    key = ('analytics_price', symbol, period)

    def _download_payload() -> dict | None:
        import yfinance as yf

        def _download() -> pd.DataFrame:
            return yf.download(symbol, period=period, auto_adjust=True, progress=False, threads=False)

        try:
            raw = fetch_with_retry(_download, retries=2, base_delay=1.0)
        except Exception as e:
            log.debug('[portfolio_analytics] history %s failed: %s', symbol, e)
            return None

        if raw is None or raw.empty:
            return None
        close = raw.get('Close')
        if close is None:
            return None

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.astype('float64').dropna()
        if len(close) < 2:
            return None

        return {
            'dates': [d.strftime('%Y-%m-%d') for d in close.index],
            'prices': [round(float(v), 8) for v in close.values],
        }

    payload = cached_fetch(key, _TTL, _download_payload)
    return _series_from_cached_payload(payload)


def build_demo_price_history(
    tickers: list[str],
    benchmark_ticker: str,
    periods: int = _TRADING_DAYS * 2,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Build deterministic synthetic price history (no network) for demo mode."""
    dates = pd.bdate_range(end='2025-12-31', periods=max(periods, _MIN_OBS + 5))
    idx = np.arange(len(dates), dtype='float64')
    base_factor = 0.00025 + 0.00075 * np.sin(idx / 31.0) + 0.0005 * np.cos(idx / 17.0)

    out: dict[str, pd.Series] = {}
    for t in tickers:
        ticker = _sanitize_ticker(t)
        if not ticker:
            continue

        digest = hashlib.sha256(ticker.encode('utf-8')).digest()
        amp = 0.006 + (digest[0] / 255.0) * 0.01
        phase = (digest[1] / 255.0) * 2.0 * math.pi
        drift = ((digest[2] / 255.0) - 0.5) * 0.0012
        cyc_a = np.sin(idx / (6 + (digest[3] % 11)) + phase)
        cyc_b = np.cos(idx / (10 + (digest[4] % 13)) + phase / 2.0)

        daily_returns = base_factor + drift + amp * (0.33 * cyc_a + 0.18 * cyc_b)
        daily_returns = np.clip(daily_returns, -0.12, 0.12)

        prices = 100.0 * np.cumprod(1.0 + daily_returns)
        out[ticker] = pd.Series(prices, index=dates, dtype='float64')

    bmk = _sanitize_ticker(benchmark_ticker) or 'SPY'
    b_digest = hashlib.sha256((bmk + '_bench').encode('utf-8')).digest()
    b_phase = (b_digest[0] / 255.0) * 2.0 * math.pi
    b_returns = base_factor * 0.9 + 0.0045 * np.sin(idx / 23.0 + b_phase)
    b_returns = np.clip(b_returns, -0.10, 0.10)
    b_prices = 100.0 * np.cumprod(1.0 + b_returns)

    return out, pd.Series(b_prices, index=dates, dtype='float64')


def _portfolio_stats(returns: pd.Series, risk_free_rate: float) -> dict[str, float | None]:
    if returns.empty:
        return {
            'annual_return': None,
            'annual_volatility': None,
            'sharpe': None,
            'sortino': None,
            'downside_deviation': None,
            'max_drawdown': None,
            'var_95': None,
            'cvar_95': None,
        }

    rf_daily = (1.0 + risk_free_rate) ** (1.0 / _TRADING_DAYS) - 1.0
    mean_d = float(returns.mean())
    vol_d = float(returns.std(ddof=0))

    ann_return = mean_d * _TRADING_DAYS
    ann_vol = vol_d * math.sqrt(_TRADING_DAYS)

    downside = np.minimum(returns.to_numpy(dtype='float64') - rf_daily, 0.0)
    downside_dev_d = float(np.sqrt(np.mean(downside * downside))) if len(downside) else 0.0
    downside_dev = downside_dev_d * math.sqrt(_TRADING_DAYS)

    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-12 else None
    sortino = (ann_return - risk_free_rate) / downside_dev if downside_dev > 1e-12 else None

    growth = (1.0 + returns).cumprod()
    drawdown = growth / growth.cummax() - 1.0
    max_drawdown = abs(float(drawdown.min())) if len(drawdown) else None

    losses = -returns.to_numpy(dtype='float64')
    var95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var95]
    cvar95 = float(tail.mean()) if len(tail) else var95

    return {
        'annual_return': ann_return,
        'annual_volatility': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'downside_deviation': downside_dev,
        'max_drawdown': max_drawdown,
        'var_95': var95,
        'cvar_95': cvar95,
    }


def _stabilize_weights(weights: np.ndarray, total: float, cap: float) -> np.ndarray:
    w = np.clip(weights.astype('float64', copy=True), 0.0, cap)
    if total <= 0:
        return np.zeros_like(w)

    for _ in range(300):
        diff = total - float(w.sum())
        if abs(diff) <= 1e-10:
            break

        if diff > 0:
            room = cap - w
            candidates = np.where(room > 1e-12)[0]
            if len(candidates) == 0:
                break
            add = diff / len(candidates)
            for i in candidates:
                w[i] = min(cap, w[i] + add)
        else:
            candidates = np.where(w > 1e-12)[0]
            if len(candidates) == 0:
                break
            sub = (-diff) / len(candidates)
            for i in candidates:
                w[i] = max(0.0, w[i] - sub)

    # Final correction for tiny floating drift.
    current = float(w.sum())
    if abs(current - total) > 1e-8:
        scale = total / current if current > 0 else 0.0
        w = np.clip(w * scale, 0.0, cap)
    return w


def _pairwise_optimize(
    start: np.ndarray,
    target_sum: float,
    cap: float,
    objective,
) -> tuple[np.ndarray, bool]:
    n = len(start)
    if n == 0:
        return start, False

    w = _stabilize_weights(start, target_sum, cap)
    best = objective(w)
    if not np.isfinite(best):
        return w, False

    step_base = min(0.05, max(0.002, target_sum / 8.0))
    for step in (step_base, step_base / 2.0, 0.01, 0.005, 0.002):
        if step <= 1e-6:
            continue
        improved = True
        rounds = 0
        while improved and rounds < 40:
            improved = False
            rounds += 1
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if w[j] < step or w[i] + step > cap + 1e-12:
                        continue
                    cand = w.copy()
                    cand[i] += step
                    cand[j] -= step
                    score = objective(cand)
                    if np.isfinite(score) and score < best - 1e-12:
                        w = cand
                        best = score
                        improved = True
    return _stabilize_weights(w, target_sum, cap), True


def _compute_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series | None) -> float | None:
    if benchmark_returns is None or benchmark_returns.empty:
        return None
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    if len(aligned) < 20:
        return None

    p = aligned.iloc[:, 0].to_numpy(dtype='float64')
    b = aligned.iloc[:, 1].to_numpy(dtype='float64')
    b_var = float(np.var(b))
    if b_var <= 1e-12:
        return None
    cov = float(np.cov(p, b, ddof=0)[0, 1])
    return cov / b_var


def compute_portfolio_analytics_payload(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    benchmark_history: pd.Series | None,
    settings: AnalyticsSettings,
) -> dict:
    current_weights = _weights_from_positions(positions)
    tickers = sorted(current_weights.keys())
    if not tickers:
        return {
            'status': 'insufficient_data',
            'messages': ['No positions available for analytics.'],
            'tickers': [],
            'current_weights': {},
            'min_vol_weights': None,
            'max_sharpe_weights': None,
            'metrics': {},
            'risk': {},
            'correlation': None,
            'meta': {
                'lookback_years': settings.lookback_years,
                'benchmark_ticker': settings.benchmark_ticker,
                'risk_free_rate': settings.risk_free_rate,
                'max_weight': settings.max_weight,
                'var_convention': 'VaR/CVaR use positive loss values (95% historical quantile).',
                'coverage_pct': 0.0,
            },
        }

    closes: dict[str, pd.Series] = {}
    missing_tickers: list[str] = []
    for t in tickers:
        series = price_history.get(t)
        if series is None:
            missing_tickers.append(t)
            continue
        s = series.astype('float64').dropna()
        if len(s) < _MIN_OBS:
            missing_tickers.append(t)
            continue
        closes[t] = s

    if not closes:
        return {
            'status': 'insufficient_data',
            'messages': ['Insufficient price history for all holdings.'],
            'tickers': tickers,
            'current_weights': current_weights,
            'min_vol_weights': None,
            'max_sharpe_weights': None,
            'metrics': {},
            'risk': {},
            'correlation': None,
            'meta': {
                'lookback_years': settings.lookback_years,
                'benchmark_ticker': settings.benchmark_ticker,
                'risk_free_rate': settings.risk_free_rate,
                'max_weight': settings.max_weight,
                'var_convention': 'VaR/CVaR use positive loss values (95% historical quantile).',
                'coverage_pct': 0.0,
            },
        }

    price_df = pd.concat(closes, axis=1).sort_index().dropna(how='all')
    returns_df = price_df.pct_change().dropna(how='all')

    eligible = [t for t in tickers if t in returns_df.columns and returns_df[t].dropna().shape[0] >= _MIN_OBS]
    coverage = float(sum(current_weights.get(t, 0.0) for t in eligible))

    messages: list[str] = []
    if missing_tickers:
        messages.append(
            'Missing/short history for: ' + ', '.join(sorted(set(missing_tickers))) + '. '
            'Those weights are kept unchanged in optimized portfolios.'
        )

    if len(eligible) == 0 or coverage <= 1e-8:
        return {
            'status': 'insufficient_data',
            'messages': messages + ['Insufficient aligned return history to compute analytics.'],
            'tickers': tickers,
            'current_weights': current_weights,
            'min_vol_weights': None,
            'max_sharpe_weights': None,
            'metrics': {},
            'risk': {},
            'correlation': None,
            'meta': {
                'lookback_years': settings.lookback_years,
                'benchmark_ticker': settings.benchmark_ticker,
                'risk_free_rate': settings.risk_free_rate,
                'max_weight': settings.max_weight,
                'var_convention': 'VaR/CVaR use positive loss values (95% historical quantile).',
                'coverage_pct': 0.0,
            },
        }

    eligible_returns = returns_df[eligible].dropna()
    if len(eligible_returns) < _MIN_OBS:
        return {
            'status': 'insufficient_data',
            'messages': messages + ['Insufficient aligned daily observations after cleaning data.'],
            'tickers': tickers,
            'current_weights': current_weights,
            'min_vol_weights': None,
            'max_sharpe_weights': None,
            'metrics': {},
            'risk': {},
            'correlation': None,
            'meta': {
                'lookback_years': settings.lookback_years,
                'benchmark_ticker': settings.benchmark_ticker,
                'risk_free_rate': settings.risk_free_rate,
                'max_weight': settings.max_weight,
                'var_convention': 'VaR/CVaR use positive loss values (95% historical quantile).',
                'coverage_pct': coverage,
            },
        }

    w_eligible = np.array([current_weights[t] for t in eligible], dtype='float64')
    w_norm = w_eligible / w_eligible.sum()

    portfolio_returns = pd.Series(
        eligible_returns.to_numpy(dtype='float64') @ w_norm,
        index=eligible_returns.index,
        dtype='float64',
    )

    stats = _portfolio_stats(portfolio_returns, settings.risk_free_rate)

    asset_vols = eligible_returns.std(ddof=0).to_numpy(dtype='float64') * math.sqrt(_TRADING_DAYS)
    port_vol = stats.get('annual_volatility') or 0.0
    if port_vol > 1e-12:
        div_ratio = float(np.dot(w_norm, asset_vols) / port_vol)
    else:
        div_ratio = None

    hhi = float(sum(w * w for w in current_weights.values()))
    beta = _compute_beta(
        portfolio_returns,
        benchmark_history.pct_change().dropna() if benchmark_history is not None else None,
    )

    mu = eligible_returns.mean().to_numpy(dtype='float64')
    cov = eligible_returns.cov().to_numpy(dtype='float64')
    target_sum = float(w_eligible.sum())

    min_vol_weights = None
    max_sharpe_weights = None

    feasible = len(eligible) >= 2 and target_sum > 1e-8 and (len(eligible) * settings.max_weight >= target_sum - 1e-8)
    if feasible:
        cap = settings.max_weight
        rf_daily = (1.0 + settings.risk_free_rate) ** (1.0 / _TRADING_DAYS) - 1.0

        def _obj_var(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        def _obj_neg_sharpe(w: np.ndarray) -> float:
            var = float(w @ cov @ w)
            if var <= 1e-14:
                return float('inf')
            ret = float(np.dot(w, mu))
            return -((ret - rf_daily) / math.sqrt(var))

        min_vec, min_ok = _pairwise_optimize(w_eligible.copy(), target_sum, cap, _obj_var)
        max_vec, max_ok = _pairwise_optimize(w_eligible.copy(), target_sum, cap, _obj_neg_sharpe)

        if min_ok:
            full = dict(current_weights)
            for i, t in enumerate(eligible):
                full[t] = float(min_vec[i])
            min_vol_weights = full
        else:
            messages.append('Min-vol optimization did not converge; showing no suggestion.')

        if max_ok:
            full = dict(current_weights)
            for i, t in enumerate(eligible):
                full[t] = float(max_vec[i])
            max_sharpe_weights = full
        else:
            messages.append('Max-Sharpe optimization did not converge; showing no suggestion.')
    else:
        messages.append('Optimization constraints are infeasible with current history coverage and max-weight cap.')

    corr_payload = None
    if len(eligible) >= 2:
        corr = eligible_returns.corr().fillna(0.0)
        corr_payload = {
            'tickers': eligible,
            'matrix': [[float(v) for v in row] for row in corr.to_numpy(dtype='float64')],
        }

    return {
        'status': 'ok',
        'messages': messages,
        'tickers': tickers,
        'current_weights': current_weights,
        'min_vol_weights': min_vol_weights,
        'max_sharpe_weights': max_sharpe_weights,
        'metrics': {
            'annual_return': stats.get('annual_return'),
            'annual_volatility': stats.get('annual_volatility'),
            'sharpe': stats.get('sharpe'),
            'sortino': stats.get('sortino'),
            'diversification_ratio': div_ratio,
            'hhi': hhi,
        },
        'risk': {
            'annual_volatility': stats.get('annual_volatility'),
            'downside_deviation': stats.get('downside_deviation'),
            'max_drawdown': stats.get('max_drawdown'),
            'var_95': stats.get('var_95'),
            'cvar_95': stats.get('cvar_95'),
            'beta': beta,
        },
        'correlation': corr_payload,
        'meta': {
            'lookback_years': settings.lookback_years,
            'benchmark_ticker': settings.benchmark_ticker,
            'risk_free_rate': settings.risk_free_rate,
            'max_weight': settings.max_weight,
            'var_convention': 'VaR/CVaR use positive loss values (95% historical quantile).',
            'coverage_pct': coverage,
            'observations': int(len(eligible_returns)),
            'eligible_tickers': eligible,
            'generated_at': datetime.now(UTC).isoformat(),
        },
    }


def _fetch_live_history(
    tickers: list[str],
    benchmark_ticker: str,
    lookback_years: int,
) -> tuple[dict[str, pd.Series], pd.Series | None]:
    period = f'{max(1, lookback_years)}y'
    symbols = sorted(set(tickers + [_sanitize_ticker(benchmark_ticker)]))
    history = fetch_parallel(symbols, lambda sym: _fetch_symbol_history(sym, period), max_workers=6)

    benchmark = _sanitize_ticker(benchmark_ticker)
    benchmark_history = history.get(benchmark)
    ticker_history = {t: history.get(t) for t in tickers}
    return ticker_history, benchmark_history


def _compute_for_cacheable_payload(portfolio_data: dict, demo_mode: bool, settings: AnalyticsSettings) -> dict:
    positions = list((portfolio_data or {}).get('positions') or [])
    weights = _weights_from_positions(positions)
    tickers = sorted(weights.keys())
    if not tickers:
        return compute_portfolio_analytics_payload(positions, {}, None, settings)

    if demo_mode:
        hist, bmk = build_demo_price_history(tickers, settings.benchmark_ticker, settings.lookback_years * _TRADING_DAYS)
    else:
        hist, bmk = _fetch_live_history(tickers, settings.benchmark_ticker, settings.lookback_years)

    return compute_portfolio_analytics_payload(positions, hist, bmk, settings)


def get_portfolio_analytics(portfolio_data: dict | None, demo_mode: bool = False) -> dict | None:
    """Return cached analytics payload for current holdings.

    Returns None when portfolio data is missing so Dash callbacks can surface the
    standard loading state via ``NotReadyError``.
    """
    if not portfolio_data or not portfolio_data.get('positions'):
        return None

    settings = _settings()
    weights = _weights_from_positions(list(portfolio_data.get('positions') or []))
    key = (
        'portfolio_analytics_payload',
        tuple(sorted((k, round(v, 8)) for k, v in weights.items())),
        settings.lookback_years,
        settings.benchmark_ticker,
        round(settings.risk_free_rate, 6),
        round(settings.max_weight, 4),
        bool(demo_mode),
    )
    return cached_fetch(key, _TTL, lambda: _compute_for_cacheable_payload(portfolio_data, demo_mode, settings))
