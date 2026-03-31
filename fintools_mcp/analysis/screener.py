"""Stock Screener — scan a universe of tickers against technical criteria."""

from __future__ import annotations

from dataclasses import dataclass

from fintools_mcp.data import fetch_bars
from fintools_mcp.indicators.rsi import compute_rsi
from fintools_mcp.indicators.atr import compute_atr
from fintools_mcp.indicators.ema import compute_ema
from fintools_mcp.analysis.trend_score import compute_trend_score


# S&P 500 top 100 by market cap (covers ~70% of the index)
SP500_TOP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "AVGO", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "JNJ", "HD", "PG", "WMT",
    "ABBV", "NFLX", "BAC", "CRM", "CVX", "MRK", "KO", "AMD", "PEP", "TMO",
    "ORCL", "LIN", "ACN", "CSCO", "MCD", "ABT", "ADBE", "WFC", "PM", "NOW",
    "GE", "IBM", "DIS", "QCOM", "ISRG", "CAT", "TXN", "INTU", "VZ", "BKNG",
    "AMGN", "T", "AMAT", "SPGI", "GS", "PFE", "DHR", "MS", "RTX", "NEE",
    "LOW", "UBER", "HON", "BLK", "UNP", "SYK", "PLD", "ADP", "SCHW", "ELV",
    "SBUX", "DE", "BA", "MDLZ", "ADI", "LMT", "GILD", "TJX", "MMC", "BMY",
    "CB", "AXP", "VRTX", "CI", "AMT", "SO", "CME", "LRCX", "PANW", "MO",
    "FI", "KLAC", "PYPL", "SNPS", "CDNS", "ICE", "DUK", "CL", "APH", "MCK",
]

# Major ETFs for broad market screening
MAJOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA",  # indices
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",  # sectors
]


@dataclass
class ScreenResult:
    ticker: str
    price: float
    rsi: float | None
    trend_score: float | None
    trend_class: str | None
    above_50ema: bool | None
    above_200ema: bool | None
    relative_volume: float | None
    atr: float | None
    change_pct_3mo: float | None


def screen(
    tickers: list[str] | None = None,
    universe: str = "sp500",
    rsi_max: float | None = None,
    rsi_min: float | None = None,
    trend_min: float | None = None,
    trend_max: float | None = None,
    above_200ema: bool | None = None,
    above_50ema: bool | None = None,
    min_relative_volume: float | None = None,
    max_results: int = 20,
) -> list[ScreenResult]:
    """Screen stocks against multiple technical criteria.

    Args:
        tickers: Custom list of tickers to screen. Overrides universe.
        universe: "sp500" (top 100) or "etfs" (sector + index ETFs)
        rsi_max: Maximum RSI (e.g. 30 for oversold)
        rsi_min: Minimum RSI (e.g. 70 for overbought)
        trend_min: Minimum trend score (e.g. 15 for uptrend)
        trend_max: Maximum trend score (e.g. -15 for downtrend)
        above_200ema: If True, only stocks above 200 EMA
        above_50ema: If True, only stocks above 50 EMA
        min_relative_volume: Minimum relative volume (e.g. 1.5 for 50% above average)
        max_results: Maximum results to return (default 20)
    """
    if tickers:
        scan_list = tickers
    elif universe == "etfs":
        scan_list = MAJOR_ETFS
    else:
        scan_list = SP500_TOP

    results = []

    for ticker in scan_list:
        try:
            bars_1y = fetch_bars(ticker, period="1y", interval="1d")
            if not bars_1y or len(bars_1y) < 50:
                continue

            closes = [b.close for b in bars_1y]
            highs = [b.high for b in bars_1y]
            lows = [b.low for b in bars_1y]
            volumes = [b.volume for b in bars_1y]

            price = closes[-1]
            rsi = compute_rsi(closes)
            atr = compute_atr(highs, lows, closes)
            ema_50 = compute_ema(closes, 50)
            ema_200 = compute_ema(closes, 200)

            # Trend score
            ts = compute_trend_score(highs, lows, closes)
            trend_val = ts.score if ts else None
            trend_cls = ts.classification if ts else None

            # Relative volume (current vs 20-day average)
            if len(volumes) >= 21:
                vol_avg = sum(volumes[-21:-1]) / 20
                rel_vol = volumes[-1] / vol_avg if vol_avg > 0 else None
            else:
                rel_vol = None

            # 3-month change
            bars_3mo = bars_1y[-63:] if len(bars_1y) >= 63 else bars_1y
            change_3mo = ((price - bars_3mo[0].close) / bars_3mo[0].close) * 100

            # Apply filters
            if rsi_max is not None and rsi is not None and rsi > rsi_max:
                continue
            if rsi_min is not None and rsi is not None and rsi < rsi_min:
                continue
            if trend_min is not None and trend_val is not None and trend_val < trend_min:
                continue
            if trend_max is not None and trend_val is not None and trend_val > trend_max:
                continue
            if above_200ema is True and ema_200 is not None and price <= ema_200:
                continue
            if above_200ema is False and ema_200 is not None and price > ema_200:
                continue
            if above_50ema is True and ema_50 is not None and price <= ema_50:
                continue
            if above_50ema is False and ema_50 is not None and price > ema_50:
                continue
            if min_relative_volume is not None and rel_vol is not None and rel_vol < min_relative_volume:
                continue

            results.append(ScreenResult(
                ticker=ticker,
                price=round(price, 2),
                rsi=round(rsi, 1) if rsi else None,
                trend_score=round(trend_val, 0) if trend_val is not None else None,
                trend_class=trend_cls,
                above_50ema=price > ema_50 if ema_50 else None,
                above_200ema=price > ema_200 if ema_200 else None,
                relative_volume=round(rel_vol, 2) if rel_vol else None,
                atr=round(atr, 2) if atr else None,
                change_pct_3mo=round(change_3mo, 1),
            ))

        except Exception:
            continue

    # Sort by trend score (most extreme first) or RSI if screening for oversold
    if rsi_max is not None:
        results.sort(key=lambda r: r.rsi if r.rsi is not None else 999)
    elif rsi_min is not None:
        results.sort(key=lambda r: -(r.rsi if r.rsi is not None else 0))
    elif trend_min is not None:
        results.sort(key=lambda r: -(r.trend_score if r.trend_score is not None else -999))
    elif trend_max is not None:
        results.sort(key=lambda r: r.trend_score if r.trend_score is not None else 999)
    else:
        results.sort(key=lambda r: -(r.trend_score if r.trend_score is not None else 0))

    return results[:max_results]
