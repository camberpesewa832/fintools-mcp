"""fintools-mcp — Financial analysis tools for AI assistants.

MCP server exposing technical indicators, options chain analysis,
position sizing, and trade statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from fintools_mcp.data import fetch_bars, fetch_options_chain, fetch_quote
from fintools_mcp.indicators.rsi import compute_rsi
from fintools_mcp.indicators.macd import compute_macd
from fintools_mcp.indicators.atr import compute_atr
from fintools_mcp.indicators.ema import compute_ema
from fintools_mcp.indicators.fibonacci import Fibonacci
from fintools_mcp.analysis.position_sizer import calculate_position, atr_based_position
from fintools_mcp.analysis.screener import screen, SP500_TOP, MAJOR_ETFS
from fintools_mcp.analysis.trend_score import compute_trend_score
from fintools_mcp.analysis.support_resistance import find_support_resistance
from fintools_mcp.analysis.trade_stats import compute_trade_stats

mcp = FastMCP(
    "fintools",
    instructions="Financial analysis tools — technical indicators, options analysis, position sizing, and trade statistics",
)


# ---------------------------------------------------------------------------
# Tool 1: Technical Indicators
# ---------------------------------------------------------------------------

@mcp.tool()
def get_technical_indicators(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> str:
    """Get technical indicators for a stock — RSI, MACD, ATR, EMAs, and Fibonacci levels.

    Args:
        ticker: Stock symbol (e.g. AAPL, SPY, GOOGL)
        period: Data period — 1mo, 3mo, 6mo, 1y (default 3mo)
        interval: Bar interval — 1m, 5m, 15m, 1h, 1d (default 1d)
    """
    bars = fetch_bars(ticker, period=period, interval=interval)
    if not bars:
        return f"No data found for {ticker}"

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]

    rsi = compute_rsi(closes)
    macd_result = compute_macd(closes)
    atr = compute_atr(highs, lows, closes)
    ema_9 = compute_ema(closes, 9)
    ema_21 = compute_ema(closes, 21)
    ema_50 = compute_ema(closes, 50)
    ema_200 = compute_ema(closes, 200)

    fib = Fibonacci()
    fib_result = fib.compute(highs, lows, closes)

    current = bars[-1]
    result = {
        "ticker": ticker.upper(),
        "price": round(current.close, 2),
        "timestamp": current.timestamp.isoformat(),
        "interval": interval,
        "bars_analyzed": len(bars),
        "indicators": {
            "rsi_14": round(rsi, 2) if rsi else None,
            "macd": {
                "macd_line": round(macd_result.macd_line, 4) if macd_result else None,
                "signal_line": round(macd_result.signal_line, 4) if macd_result else None,
                "histogram": round(macd_result.histogram, 4) if macd_result else None,
            },
            "atr_14": round(atr, 2) if atr else None,
            "ema_9": round(ema_9, 2) if ema_9 else None,
            "ema_21": round(ema_21, 2) if ema_21 else None,
            "ema_50": round(ema_50, 2) if ema_50 else None,
            "ema_200": round(ema_200, 2) if ema_200 else None,
        },
        "trend": _assess_trend(current.close, ema_9, ema_21, ema_50, ema_200, rsi),
    }

    if fib_result:
        result["fibonacci"] = {
            "swing_high": round(fib_result.swing_high, 2),
            "swing_low": round(fib_result.swing_low, 2),
            "direction": fib_result.direction,
            "in_golden_pocket": fib_result.in_golden_pocket,
            "levels": {str(k): round(v, 2) for k, v in fib_result.levels.items()},
        }

    return json.dumps(result, indent=2)


def _assess_trend(
    price: float,
    ema_9: float | None,
    ema_21: float | None,
    ema_50: float | None,
    ema_200: float | None,
    rsi: float | None,
) -> dict:
    """Simple trend assessment based on EMA alignment and RSI."""
    signals = []

    if ema_9 and ema_21:
        if ema_9 > ema_21:
            signals.append("short_term_bullish")
        else:
            signals.append("short_term_bearish")

    if ema_50 and ema_200:
        if ema_50 > ema_200:
            signals.append("long_term_bullish")
        else:
            signals.append("long_term_bearish")

    if price and ema_200:
        if price > ema_200:
            signals.append("above_200ema")
        else:
            signals.append("below_200ema")

    rsi_condition = None
    if rsi:
        if rsi > 70:
            rsi_condition = "overbought"
        elif rsi < 30:
            rsi_condition = "oversold"
        elif rsi > 50:
            rsi_condition = "bullish_momentum"
        else:
            rsi_condition = "bearish_momentum"

    # Overall assessment
    bullish_count = sum(1 for s in signals if "bullish" in s or "above" in s)
    bearish_count = sum(1 for s in signals if "bearish" in s or "below" in s)

    if bullish_count > bearish_count:
        overall = "bullish"
    elif bearish_count > bullish_count:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "signals": signals,
        "rsi_condition": rsi_condition,
    }


# ---------------------------------------------------------------------------
# Tool 2: Stock Quote
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stock_quote(ticker: str) -> str:
    """Get current price and quote data for a stock.

    Args:
        ticker: Stock symbol (e.g. AAPL, SPY, TSLA)
    """
    quote = fetch_quote(ticker)
    return json.dumps(quote, indent=2)


# ---------------------------------------------------------------------------
# Tool 3: Options Chain Analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_options_chain(
    ticker: str,
    expiration: str = "",
    min_volume: int = 10,
    min_open_interest: int = 100,
    max_spread_pct: float = 10.0,
) -> str:
    """Analyze an options chain — find liquid contracts, assess IV, identify opportunities.

    Args:
        ticker: Stock symbol (e.g. AAPL, SPY)
        expiration: Expiration date YYYY-MM-DD (empty = nearest expiration)
        min_volume: Minimum volume filter (default 10)
        min_open_interest: Minimum open interest filter (default 100)
        max_spread_pct: Maximum bid-ask spread as % of mid price (default 10%)
    """
    exp = expiration if expiration else None
    chain = fetch_options_chain(ticker, exp)

    if not chain["calls"] and not chain["puts"]:
        return f"No options data found for {ticker}"

    def filter_and_rank(contracts: list[dict], option_type: str) -> list[dict]:
        filtered = []
        for c in contracts:
            mid = (c["bid"] + c["ask"]) / 2 if (c["bid"] + c["ask"]) > 0 else 0
            spread_pct = ((c["ask"] - c["bid"]) / mid * 100) if mid > 0 else 999

            if c["volume"] < min_volume:
                continue
            if c["open_interest"] < min_open_interest:
                continue
            if spread_pct > max_spread_pct:
                continue

            filtered.append({
                "contract": c["contract"],
                "type": option_type,
                "strike": c["strike"],
                "bid": c["bid"],
                "ask": c["ask"],
                "mid": round(mid, 2),
                "spread_pct": round(spread_pct, 1),
                "volume": c["volume"],
                "open_interest": c["open_interest"],
                "iv": round(c["iv"] * 100, 1),
                "itm": c["in_the_money"],
            })

        return sorted(filtered, key=lambda x: x["volume"], reverse=True)

    calls = filter_and_rank(chain["calls"], "call")
    puts = filter_and_rank(chain["puts"], "put")

    # IV summary
    all_ivs = [c["iv"] for c in calls + puts if c["iv"] > 0]
    avg_iv = sum(all_ivs) / len(all_ivs) if all_ivs else 0

    result = {
        "ticker": ticker.upper(),
        "underlying_price": chain["underlying_price"],
        "expiration": chain["expiration"],
        "available_expirations": chain["expirations"][:10],
        "iv_summary": {
            "average_iv": round(avg_iv, 1),
            "contracts_analyzed": len(all_ivs),
        },
        "liquid_calls": calls[:15],
        "liquid_puts": puts[:15],
        "total_call_volume": sum(c["volume"] for c in chain["calls"]),
        "total_put_volume": sum(c["volume"] for c in chain["puts"]),
        "put_call_volume_ratio": round(
            sum(c["volume"] for c in chain["puts"]) / max(sum(c["volume"] for c in chain["calls"]), 1), 2
        ),
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 4: Position Sizing
# ---------------------------------------------------------------------------

@mcp.tool()
def calculate_position_size(
    ticker: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    account_size: float = 100000.0,
    risk_pct: float = 1.5,
) -> str:
    """Calculate position size and risk/reward for a trade idea.

    Args:
        ticker: Stock symbol
        entry_price: Planned entry price
        stop_price: Stop loss price
        target_price: Profit target price
        account_size: Account equity in dollars (default $100,000)
        risk_pct: Max risk as % of account (default 1.5%)
    """
    plan = calculate_position(
        ticker=ticker,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        account_size=account_size,
        risk_pct=risk_pct,
    )
    return json.dumps(asdict(plan), indent=2)


@mcp.tool()
def calculate_atr_position(
    ticker: str,
    account_size: float = 100000.0,
    risk_pct: float = 1.5,
    stop_atr_mult: float = 2.0,
    target_atr_mult: float = 3.0,
    direction: str = "long",
    period: str = "3mo",
) -> str:
    """Calculate position size using ATR for automatic stop/target placement.

    Fetches live price and ATR, then computes position size, stop, and target.

    Args:
        ticker: Stock symbol (e.g. AAPL, SPY)
        account_size: Account equity in dollars (default $100,000)
        risk_pct: Max risk as % of account (default 1.5%)
        stop_atr_mult: ATR multiplier for stop distance (default 2.0x)
        target_atr_mult: ATR multiplier for target distance (default 3.0x)
        direction: "long" or "short" (default "long")
        period: Lookback period for ATR calculation (default 3mo)
    """
    bars = fetch_bars(ticker, period=period, interval="1d")
    if not bars:
        return f"No data found for {ticker}"

    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    atr = compute_atr(highs, lows, closes)

    if not atr:
        return f"Not enough data to compute ATR for {ticker}"

    entry_price = bars[-1].close

    plan = atr_based_position(
        ticker=ticker,
        entry_price=entry_price,
        atr=atr,
        account_size=account_size,
        risk_pct=risk_pct,
        stop_atr_mult=stop_atr_mult,
        target_atr_mult=target_atr_mult,
        direction=direction,
    )

    result = asdict(plan)
    result["atr"] = round(atr, 2)
    result["entry_price"] = round(entry_price, 2)
    result["stop_price"] = round(plan.stop_price, 2)
    result["target_price"] = round(plan.target_price, 2)

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 5: Trade Statistics
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_trades(
    pnls: list[float],
    starting_equity: float = 100000.0,
) -> str:
    """Compute performance statistics from a list of trade P&L results.

    Args:
        pnls: List of profit/loss values per trade (e.g. [150, -80, 200, -50, 300])
        starting_equity: Starting account equity for drawdown calc (default $100,000)
    """
    stats = compute_trade_stats(pnls, starting_equity)
    return json.dumps(asdict(stats), indent=2)


# ---------------------------------------------------------------------------
# Tool 6: Multi-Ticker Comparison
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_tickers(
    tickers: list[str],
    period: str = "3mo",
) -> str:
    """Compare technical setups across multiple tickers side by side.

    Args:
        tickers: List of stock symbols to compare (e.g. ["AAPL", "GOOGL", "MSFT"])
        period: Data period for analysis (default 3mo)
    """
    results = []
    for ticker in tickers[:10]:  # cap at 10
        # Use 1y for 200 EMA to have enough bars, requested period for everything else
        bars = fetch_bars(ticker, period=period, interval="1d")
        bars_1y = fetch_bars(ticker, period="1y", interval="1d") if period != "1y" else bars
        if not bars:
            results.append({"ticker": ticker.upper(), "error": "no data"})
            continue

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        rsi = compute_rsi(closes)
        atr = compute_atr(highs, lows, closes)
        ema_9 = compute_ema(closes, 9)
        ema_21 = compute_ema(closes, 21)
        ema_50 = compute_ema(closes, 50)

        # 200 EMA needs 1y of data
        closes_1y = [b.close for b in bars_1y] if bars_1y else closes
        ema_200 = compute_ema(closes_1y, 200)

        price = bars[-1].close
        change_pct = ((price - bars[0].close) / bars[0].close) * 100 if bars[0].close else 0

        # Trend assessment using same logic as get_technical_indicators
        trend_signals = []
        if ema_9 and ema_21:
            trend_signals.append("bullish" if ema_9 > ema_21 else "bearish")
        if ema_50 and ema_200:
            trend_signals.append("bullish" if ema_50 > ema_200 else "bearish")
        if price and ema_200:
            trend_signals.append("bullish" if price > ema_200 else "bearish")

        if trend_signals:
            bullish = sum(1 for s in trend_signals if s == "bullish")
            bearish = sum(1 for s in trend_signals if s == "bearish")
            trend = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
        else:
            trend = "unknown"

        results.append({
            "ticker": ticker.upper(),
            "price": round(price, 2),
            "change_pct": round(change_pct, 1),
            "rsi": round(rsi, 1) if rsi else None,
            "atr": round(atr, 2) if atr else None,
            "above_50ema": price > ema_50 if ema_50 else None,
            "above_200ema": price > ema_200 if ema_200 else None,
            "trend": trend,
        })

    return json.dumps({"comparison": results, "period": period}, indent=2)


# ---------------------------------------------------------------------------
# Tool 7: Stock Screener
# ---------------------------------------------------------------------------

@mcp.tool()
def screen_stocks(
    rsi_max: float | None = None,
    rsi_min: float | None = None,
    trend_min: float | None = None,
    trend_max: float | None = None,
    above_200ema: bool | None = None,
    above_50ema: bool | None = None,
    min_relative_volume: float | None = None,
    universe: str = "sp500",
    tickers: list[str] | None = None,
    max_results: int = 15,
) -> str:
    """Screen stocks against technical criteria — find oversold bounces, trending stocks, volume spikes, etc.

    Scans the S&P 500 top 100 (or custom tickers) and filters by RSI, trend score, EMA position, and relative volume.
    Each stock gets a Trend Score from -100 (strong downtrend) to +100 (strong uptrend).

    Args:
        rsi_max: Maximum RSI to filter for oversold stocks (e.g. 30)
        rsi_min: Minimum RSI to filter for overbought stocks (e.g. 70)
        trend_min: Minimum trend score (e.g. 15 for uptrend, 40 for strong uptrend)
        trend_max: Maximum trend score (e.g. -15 for downtrend, -40 for strong downtrend)
        above_200ema: If true, only stocks above 200-day EMA
        above_50ema: If true, only stocks above 50-day EMA
        min_relative_volume: Minimum relative volume vs 20-day avg (e.g. 1.5 = 50% above average)
        universe: "sp500" (top 100 by market cap) or "etfs" (sector + index ETFs)
        tickers: Custom list of tickers to screen (overrides universe)
        max_results: Maximum results to return (default 15)
    """
    results = screen(
        tickers=tickers,
        universe=universe,
        rsi_max=rsi_max,
        rsi_min=rsi_min,
        trend_min=trend_min,
        trend_max=trend_max,
        above_200ema=above_200ema,
        above_50ema=above_50ema,
        min_relative_volume=min_relative_volume,
        max_results=max_results,
    )

    if not results:
        return json.dumps({"matches": 0, "message": "No stocks matched the criteria"})

    output = {
        "matches": len(results),
        "criteria": {
            k: v for k, v in {
                "rsi_max": rsi_max,
                "rsi_min": rsi_min,
                "trend_min": trend_min,
                "trend_max": trend_max,
                "above_200ema": above_200ema,
                "above_50ema": above_50ema,
                "min_relative_volume": min_relative_volume,
                "universe": universe if not tickers else f"custom ({len(tickers)} tickers)",
            }.items() if v is not None
        },
        "results": [asdict(r) for r in results],
    }

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Tool 8: Support & Resistance Levels
# ---------------------------------------------------------------------------

@mcp.tool()
def get_support_resistance(
    ticker: str,
    lookback: int = 120,
    max_levels: int = 5,
) -> str:
    """Find key support and resistance levels for a stock from swing highs/lows.

    Detects pivot points, clusters nearby levels, counts touches, and ranks by strength.

    Args:
        ticker: Stock symbol (e.g. AAPL, SPY)
        lookback: Number of daily bars to analyze (default 120 = ~6 months)
        max_levels: Maximum levels per side — support and resistance (default 5)
    """
    bars = fetch_bars(ticker, period="1y", interval="1d")
    if not bars or len(bars) < 20:
        return f"Not enough data for {ticker}"

    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    current_price = closes[-1]

    levels = find_support_resistance(
        highs, lows, closes, current_price,
        lookback=lookback, max_levels=max_levels,
    )

    support = [l for l in levels if l.level_type == "support"]
    resistance = [l for l in levels if l.level_type == "resistance"]

    result = {
        "ticker": ticker.upper(),
        "price": round(current_price, 2),
        "support_levels": [asdict(l) for l in support],
        "resistance_levels": [asdict(l) for l in resistance],
        "nearest_support": round(support[0].price, 2) if support else None,
        "nearest_resistance": round(resistance[0].price, 2) if resistance else None,
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 9: Trend Score
# ---------------------------------------------------------------------------

@mcp.tool()
def get_trend_score(
    ticker: str,
) -> str:
    """Get a graduated trend score (-100 to +100) for a stock.

    Combines 5 daily indicators: close vs SMA20 (25%), close vs SMA50 (25%),
    SMA20 slope (20%), ADX direction (15%), and position in 20-day range (15%).

    Classifications: strong_uptrend (>=40), uptrend (>=15), neutral (-15 to +15),
    downtrend (<=-15), strong_downtrend (<=-40).

    Args:
        ticker: Stock symbol (e.g. AAPL, SPY)
    """
    bars = fetch_bars(ticker, period="1y", interval="1d")
    if not bars or len(bars) < 50:
        return f"Not enough data for {ticker}"

    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]

    ts = compute_trend_score(highs, lows, closes)
    if not ts:
        return f"Could not compute trend score for {ticker}"

    result = {
        "ticker": ticker.upper(),
        "price": round(closes[-1], 2),
        "trend_score": round(ts.score, 1),
        "classification": ts.classification,
        "components": {
            "close_vs_sma20": round(ts.close_vs_sma20, 1),
            "close_vs_sma50": round(ts.close_vs_sma50, 1),
            "sma20_slope": round(ts.sma20_slope, 1),
            "adx_direction": round(ts.adx_direction, 1),
            "range_position": round(ts.range_position, 1),
        },
        "values": {
            "sma20": round(ts.sma20, 2),
            "sma50": round(ts.sma50, 2) if ts.sma50 else None,
            "adx": round(ts.adx, 1) if ts.adx else None,
            "plus_di": round(ts.plus_di, 1) if ts.plus_di else None,
            "minus_di": round(ts.minus_di, 1) if ts.minus_di else None,
        },
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool: Find Breakouts
# ---------------------------------------------------------------------------

@mcp.tool()
def find_breakouts(
    exclude_symbols: str = "",
    min_trend_score: float = 30.0,
    min_rsi: float = 45.0,
    max_rsi: float = 75.0,
    max_results: int = 15,
) -> str:
    """Scan S&P 500 top 100 for breakout candidates — stocks in strong uptrends with bullish EMA stacks.

    Finds stocks the V3 breakout engine should be watching. Filters for:
    - Strong trend score (default >= 30)
    - RSI in momentum range (45-75, not overbought or oversold)
    - Price above both 50 and 200 EMA (bullish structure)
    - Sorted by trend score (strongest first)

    Args:
        exclude_symbols: Comma-separated symbols to exclude (e.g. existing watchlist)
        min_trend_score: Minimum trend score (default 30)
        min_rsi: Minimum RSI (default 45)
        max_rsi: Maximum RSI — avoid overbought (default 75)
        max_results: Max results to return (default 15)
    """
    exclude = set(s.strip().upper() for s in exclude_symbols.split(",") if s.strip())

    results = screen(
        universe="sp500",
        trend_min=min_trend_score,
        above_50ema=True,
        above_200ema=True,
        max_results=100,
    )

    # Apply RSI filter and exclusion
    filtered = []
    for r in results:
        if r.ticker in exclude:
            continue
        if r.rsi is not None and (r.rsi < min_rsi or r.rsi > max_rsi):
            continue
        filtered.append(r)

    # Compute EMA phase for each
    output = []
    for r in filtered[:max_results]:
        try:
            bars = fetch_bars(r.ticker, period="6mo", interval="1d")
            if not bars or len(bars) < 50:
                continue
            closes = [b.close for b in bars]
            ema9 = compute_ema(closes, 9)
            ema21 = compute_ema(closes, 21)
            ema50 = compute_ema(closes, 50)
            price = closes[-1]

            if price > ema9 and ema9 > ema21 and ema21 > ema50:
                ema_phase = "4-FULL BULLISH"
            elif price > ema9 and ema9 > ema21:
                ema_phase = "3-TRANSITION"
            elif price > ema9:
                ema_phase = "2-BOUNCE"
            else:
                ema_phase = "1-BEARISH"

            output.append({
                "ticker": r.ticker,
                "price": r.price,
                "trend_score": r.trend_score,
                "trend_class": r.trend_class,
                "rsi": r.rsi,
                "ema_phase": ema_phase,
                "above_50ema": r.above_50ema,
                "above_200ema": r.above_200ema,
                "change_3mo": r.change_pct_3mo,
                "relative_volume": r.relative_volume,
                "atr": r.atr,
            })
        except Exception:
            continue

    summary = {
        "scan_date": "live",
        "universe": "S&P 500 top 100",
        "filters": {
            "min_trend_score": min_trend_score,
            "rsi_range": f"{min_rsi}-{max_rsi}",
            "above_50_and_200_ema": True,
            "excluded": list(exclude) if exclude else "none",
        },
        "results_count": len(output),
        "breakout_candidates": output,
    }

    return json.dumps(summary, indent=2)


# ---------------------------------------------------------------------------
# Tool: Option Quote
# ---------------------------------------------------------------------------

@mcp.tool()
def get_option_quote(
    option_symbol: str,
    entry_price: float = 0.0,
) -> str:
    """Get a live quote + greeks for a specific option contract, with P&L if entry price provided.

    Args:
        option_symbol: OCC option symbol (e.g. AAPL260406P00250000, SPY260331C00640000)
        entry_price: Your entry premium per contract (optional — enables P&L calculation)
    """
    import re
    import math

    # Parse the OCC symbol: ROOT + YYMMDD + C/P + STRIKE(8 digits, strike*1000)
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', option_symbol.upper())
    if not match:
        return json.dumps({"error": f"Invalid option symbol: {option_symbol}. Expected format like AAPL260406P00250000"})

    root = match.group(1)
    date_str = match.group(2)
    opt_type = "put" if match.group(3) == "P" else "call"
    strike = int(match.group(4)) / 1000

    expiration = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"

    # Fetch the chain for this expiration
    try:
        chain = fetch_options_chain(root, expiration)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch chain for {root} exp {expiration}: {str(e)}"})

    contracts = chain["puts"] if opt_type == "put" else chain["calls"]

    # Find the matching contract
    found = None
    for c in contracts:
        if abs(c["strike"] - strike) < 0.01:
            found = c
            break

    if not found:
        return json.dumps({"error": f"Contract {option_symbol} not found in chain"})

    bid = found["bid"]
    ask = found["ask"]
    mid = (bid + ask) / 2 if (bid + ask) > 0 else found["last"]
    spread = ask - bid
    spread_pct = (spread / mid * 100) if mid > 0 else 0
    iv = found["iv"]

    result = {
        "symbol": option_symbol.upper(),
        "underlying": root,
        "type": opt_type,
        "strike": strike,
        "expiration": expiration,
        "underlying_price": chain["underlying_price"],
        "bid": bid,
        "ask": ask,
        "mid": round(mid, 2),
        "last": found["last"],
        "spread": round(spread, 2),
        "spread_pct": round(spread_pct, 1),
        "volume": found["volume"],
        "open_interest": found["open_interest"],
        "iv": round(iv * 100, 1) if iv < 1 else round(iv, 1),
    }

    if entry_price > 0:
        pnl_per_contract = (mid - entry_price) * 100
        pnl_pct = ((mid - entry_price) / entry_price) * 100
        result["entry_price"] = entry_price
        result["pnl_per_contract"] = round(pnl_per_contract, 2)
        result["pnl_pct"] = round(pnl_pct, 1)
        result["status"] = "profit" if mid > entry_price else "loss"

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
