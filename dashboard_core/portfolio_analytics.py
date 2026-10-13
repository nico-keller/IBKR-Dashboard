"""Derived portfolio analytics store + render callbacks."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from dashboard_core.helpers import section_label
from decorators import NotReadyError, safe_render
from ibkr_client import is_demo_mode
from portfolio_analytics import get_portfolio_analytics
from styles import (
    CARD,
    COLOR_BORDER_LIGHT,
    COLOR_BRAND,
    COLOR_GOOD,
    COLOR_SURFACE_WHITE,
    COLOR_TEXT_MID,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_STRONG,
    TABLE_HEADER_CELL,
    TABLE_WRAPPER,
)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return '—'
    return f'{value * 100:.2f}%'


def _fmt_x(value: float | None) -> str:
    if value is None:
        return '—'
    return f'{value:.2f}x'


def _fmt_num(value: float | None) -> str:
    if value is None:
        return '—'
    return f'{value:.2f}'


def _metric_row(label: str, value: str) -> html.Tr:
    return html.Tr(
        [
            html.Td(label, style={'padding': '8px 0', 'color': COLOR_TEXT_MUTED, 'fontSize': '14px'}),
            html.Td(
                value,
                style={
                    'padding': '8px 0',
                    'textAlign': 'right',
                    'color': COLOR_TEXT_STRONG,
                    'fontSize': '14px',
                    'fontWeight': '600',
                },
            ),
        ],
        style={'borderTop': f'0.5px solid {COLOR_BORDER_LIGHT}'},
    )


def register(app):
    @app.callback(
        Output('portfolio-analytics-data', 'data'),
        Input('portfolio-data', 'data'),
    )
    def populate_portfolio_analytics(data):
        if not data or not data.get('positions'):
            return None
        return get_portfolio_analytics(data, demo_mode=is_demo_mode())

    @app.callback(
        Output('portfolio-analytics-section', 'children'),
        Input('portfolio-analytics-data', 'data'),
        Input('connection-status', 'data'),
        State('portfolio-data', 'data'),
    )
    @safe_render('Portfolio Analytics')
    def render_portfolio_analytics(data, status, port_data):
        base = (status or '').split(':')[0]
        if base in ('disconnected', 'no_positions'):
            return []
        if port_data and not port_data.get('positions'):
            return []
        if data is None:
            raise NotReadyError('Loading portfolio analytics…')
        return _render_portfolio_analytics_inner(data)


def _render_portfolio_analytics_inner(data: dict):
    status = data.get('status')
    msgs = data.get('messages') or []
    meta = data.get('meta') or {}

    if status != 'ok':
        note = msgs[0] if msgs else 'Insufficient data to calculate portfolio analytics.'
        return html.Div(
            [
                section_label('Portfolio Analytics'),
                html.Div(
                    [
                        html.P(
                            'Insufficient data',
                            style={'margin': '0 0 8px', 'fontSize': '17px', 'fontWeight': '600'},
                        ),
                        html.P(note, style={'margin': '0', 'fontSize': '14px', 'color': COLOR_TEXT_MUTED}),
                    ],
                    style=CARD,
                ),
            ]
        )

    metrics = data.get('metrics') or {}
    risk = data.get('risk') or {}

    assumptions = html.P(
        (
            f"Lookback: {meta.get('lookback_years', 2)}y · Benchmark: {meta.get('benchmark_ticker', 'SPY')} "
            f"· Risk-free: {meta.get('risk_free_rate', 0.0):.2%} · Max weight cap: {meta.get('max_weight', 0.4):.0%}"
        ),
        style={'margin': '0 0 14px', 'fontSize': '13px', 'color': COLOR_TEXT_MID},
    )

    optimization_card = html.Div(
        [
            html.P('Optimization', style={'margin': '0 0 12px', 'fontWeight': '600', 'fontSize': '16px'}),
            html.Table(
                [
                    html.Tbody(
                        [
                            _metric_row('Annualized return', _fmt_pct(metrics.get('annual_return'))),
                            _metric_row('Annualized volatility', _fmt_pct(metrics.get('annual_volatility'))),
                            _metric_row('Sharpe ratio', _fmt_num(metrics.get('sharpe'))),
                            _metric_row('Sortino ratio', _fmt_num(metrics.get('sortino'))),
                            _metric_row('Diversification ratio', _fmt_num(metrics.get('diversification_ratio'))),
                            _metric_row('Concentration (HHI)', _fmt_num(metrics.get('hhi'))),
                        ]
                    )
                ],
                style={**TABLE_WRAPPER, 'fontSize': '14px'},
            ),
        ],
        style={**CARD, 'flex': '1', 'minWidth': '320px'},
    )

    risk_card = html.Div(
        [
            html.P('Risk', style={'margin': '0 0 12px', 'fontWeight': '600', 'fontSize': '16px'}),
            html.Table(
                [
                    html.Tbody(
                        [
                            _metric_row('Annualized volatility', _fmt_pct(risk.get('annual_volatility'))),
                            _metric_row('Downside deviation', _fmt_pct(risk.get('downside_deviation'))),
                            _metric_row('Maximum drawdown', _fmt_pct(risk.get('max_drawdown'))),
                            _metric_row('Historical VaR (95%)', _fmt_pct(risk.get('var_95'))),
                            _metric_row('Historical CVaR / ES (95%)', _fmt_pct(risk.get('cvar_95'))),
                            _metric_row(
                                f"Beta vs {meta.get('benchmark_ticker', 'SPY')}",
                                _fmt_num(risk.get('beta')),
                            ),
                        ]
                    )
                ],
                style={**TABLE_WRAPPER, 'fontSize': '14px'},
            ),
        ],
        style={**CARD, 'flex': '1', 'minWidth': '320px'},
    )

    current = data.get('current_weights') or {}
    min_w = data.get('min_vol_weights') or {}
    max_w = data.get('max_sharpe_weights') or {}

    rows = []
    for ticker in sorted(current, key=current.get, reverse=True):
        rows.append(
            html.Tr(
                [
                    html.Td(ticker, style={'padding': '8px 10px', 'fontWeight': '600'}),
                    html.Td(_fmt_pct(current.get(ticker)), style={'padding': '8px 10px', 'textAlign': 'right'}),
                    html.Td(_fmt_pct(min_w.get(ticker) if min_w else None), style={'padding': '8px 10px', 'textAlign': 'right'}),
                    html.Td(_fmt_pct(max_w.get(ticker) if max_w else None), style={'padding': '8px 10px', 'textAlign': 'right'}),
                ],
                style={'borderTop': f'0.5px solid {COLOR_BORDER_LIGHT}'},
            )
        )

    weights_card = html.Div(
        [
            html.P('Suggested Weights', style={'margin': '0 0 12px', 'fontWeight': '600', 'fontSize': '16px'}),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th('Ticker', style={**TABLE_HEADER_CELL, 'textAlign': 'left'}),
                                html.Th('Current', style={**TABLE_HEADER_CELL, 'textAlign': 'right'}),
                                html.Th('Min Vol', style={**TABLE_HEADER_CELL, 'textAlign': 'right'}),
                                html.Th('Max Sharpe', style={**TABLE_HEADER_CELL, 'textAlign': 'right'}),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                style={**TABLE_WRAPPER, 'fontSize': '14px'},
            ),
            html.P(
                'Estimates from historical returns only. Not investment advice.',
                style={'margin': '10px 0 0', 'fontSize': '12px', 'color': COLOR_TEXT_MID},
            ),
        ],
        style={**CARD, 'marginTop': '14px'},
    )

    selected_name = 'Max Sharpe'
    selected = max_w
    if not selected:
        selected_name = 'Min Vol'
        selected = min_w

    alloc_chart = None
    if selected:
        labels = sorted(current, key=current.get, reverse=True)
        fig = go.Figure()
        fig.add_bar(name='Current', x=labels, y=[current.get(t, 0.0) * 100 for t in labels], marker_color=COLOR_BRAND)
        fig.add_bar(
            name=selected_name,
            x=labels,
            y=[selected.get(t, 0.0) * 100 for t in labels],
            marker_color=COLOR_GOOD,
        )
        fig.update_layout(
            barmode='group',
            margin=dict(t=10, r=8, b=65, l=36),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
            height=280,
            yaxis=dict(title='% weight', gridcolor='rgba(120,120,120,0.15)'),
        )
        alloc_chart = dcc.Graph(figure=fig, config={'displayModeBar': False})
    else:
        alloc_chart = html.P(
            'Optimized allocation is unavailable for current inputs.',
            style={'margin': '18px 0', 'fontSize': '14px', 'color': COLOR_TEXT_MUTED},
        )

    corr = data.get('correlation')
    if corr and corr.get('tickers') and corr.get('matrix'):
        corr_fig = go.Figure(
            data=go.Heatmap(
                z=corr['matrix'],
                x=corr['tickers'],
                y=corr['tickers'],
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                colorbar=dict(title='Corr'),
                hovertemplate='%{y} vs %{x}: %{z:.2f}<extra></extra>',
            )
        )
        corr_fig.update_layout(
            margin=dict(t=10, r=8, b=30, l=30),
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        corr_chart = dcc.Graph(figure=corr_fig, config={'displayModeBar': False})
    else:
        corr_chart = html.P(
            'Correlation heatmap unavailable (need at least two holdings with sufficient history).',
            style={'margin': '18px 0', 'fontSize': '14px', 'color': COLOR_TEXT_MUTED},
        )

    diagnostics = [
        html.P(
            f"Coverage: {meta.get('coverage_pct', 0.0):.1%} of current market value · "
            f"Observations: {meta.get('observations', 0)}",
            style={'margin': '12px 0 0', 'fontSize': '12px', 'color': COLOR_TEXT_MID},
        ),
        html.P(
            meta.get('var_convention', ''),
            style={'margin': '4px 0 0', 'fontSize': '12px', 'color': COLOR_TEXT_MID},
        ),
    ]
    if msgs:
        diagnostics.extend(
            html.P(msg, style={'margin': '4px 0 0', 'fontSize': '12px', 'color': COLOR_TEXT_MID}) for msg in msgs
        )

    return html.Div(
        [
            section_label('Portfolio Analytics'),
            assumptions,
            html.Div([optimization_card, risk_card], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap'}),
            weights_card,
            html.Div(
                [
                    html.Div(
                        [
                            html.P(
                                f'Allocation Comparison (Current vs {selected_name})',
                                style={'margin': '0 0 10px', 'fontWeight': '600', 'fontSize': '16px'},
                            ),
                            alloc_chart,
                        ],
                        style={**CARD, 'flex': '1', 'minWidth': '360px'},
                    ),
                    html.Div(
                        [
                            html.P('Holdings Correlation', style={'margin': '0 0 10px', 'fontWeight': '600', 'fontSize': '16px'}),
                            corr_chart,
                        ],
                        style={**CARD, 'flex': '1', 'minWidth': '360px'},
                    ),
                ],
                style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'marginTop': '14px'},
            ),
            html.Div(diagnostics, style={**CARD, 'marginTop': '14px', 'background': COLOR_SURFACE_WHITE}),
        ]
    )
