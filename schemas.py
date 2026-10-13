"""Typed payloads for every dcc.Store in the dashboard.

These are the contracts between ibkr_client/market_* producers and the
Dash callbacks that consume them. If you change a shape here, the
consumers light up red in the IDE.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class PositionData(TypedDict):
    ticker: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    pnl_pct: float
    allocation_pct: float
    price_stale: NotRequired[bool]
    daily_change: NotRequired[float | None]
    daily_change_pct: NotRequired[float | None]
    low_52w: NotRequired[float | None]
    high_52w: NotRequired[float | None]
    spread_pct: NotRequired[float | None]
    range_52w_pct: NotRequired[float | None]


class SummaryData(TypedDict):
    total_invested: float
    total_value: float
    total_unrealized_pnl: float
    num_positions: int
    largest_position: str
    largest_position_pct: float
    best_performer: str
    worst_performer: str
    total_pnl_pct: NotRequired[float | None]
    total_daily_pnl: NotRequired[float | None]


class AccountData(TypedDict, total=False):
    cash_chf: float
    net_liquidation: float
    gross_position_value: float
    usdchf_rate: float
    daily_pnl: float | None


class PortfolioData(TypedDict):
    summary: SummaryData
    positions: list[PositionData]
    account: AccountData
    div_data: NotRequired[dict]


class SectorGeoEntry(TypedDict, total=False):
    sector: str
    industry: str
    country: str
    longName: str
    is_etf: bool
    sector_weights: dict[str, float]


class MarketIntelData(TypedDict):
    tickers: list[str]
    sector_geo: dict[str, SectorGeoEntry]
    earnings: dict


class ValuationData(TypedDict, total=False):
    buffett: dict
    sp500_pe: dict
    cape: dict
    treasury: dict


class CorrelationData(TypedDict):
    tickers: list[str]
    matrix: list[list[float]]


class PortfolioAnalyticsMeta(TypedDict, total=False):
    lookback_years: int
    benchmark_ticker: str
    risk_free_rate: float
    max_weight: float
    var_convention: str
    coverage_pct: float
    observations: int
    eligible_tickers: list[str]
    generated_at: str


class PortfolioAnalyticsMetrics(TypedDict, total=False):
    annual_return: float | None
    annual_volatility: float | None
    sharpe: float | None
    sortino: float | None
    diversification_ratio: float | None
    hhi: float | None


class PortfolioRiskMetrics(TypedDict, total=False):
    annual_volatility: float | None
    downside_deviation: float | None
    max_drawdown: float | None
    var_95: float | None
    cvar_95: float | None
    beta: float | None


class PortfolioAnalyticsData(TypedDict):
    status: str
    messages: list[str]
    tickers: list[str]
    current_weights: dict[str, float]
    min_vol_weights: dict[str, float] | None
    max_sharpe_weights: dict[str, float] | None
    metrics: PortfolioAnalyticsMetrics
    risk: PortfolioRiskMetrics
    correlation: CorrelationData | None
    meta: PortfolioAnalyticsMeta
