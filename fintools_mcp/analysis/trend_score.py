"""Trend Score — graduated trend measurement from -100 to +100.

5 components from daily bars:
  - Close vs SMA20 (25%): distance from fast moving average
  - Close vs SMA50 (25%): distance from slow moving average
  - SMA20 slope (20%): is the 20-day MA rising or falling over 5 days?
  - ADX direction (15%): +DI vs -DI for momentum direction
  - Position in 20-day range (15%): where is price within recent range?

Classifications:
  score >= +40: STRONG UPTREND
  score >= +15: UPTREND
  -15 < score < +15: NEUTRAL
  score <= -15: DOWNTREND
  score <= -40: STRONG DOWNTREND
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrendScoreResult:
    score: float  # -100 to +100
    classification: str
    close_vs_sma20: float
    close_vs_sma50: float
    sma20_slope: float
    adx_direction: float
    range_position: float
    sma20: float
    sma50: float | None
    adx: float | None
    plus_di: float | None
    minus_di: float | None


def compute_trend_score(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    sma_fast: int = 20,
    sma_slow: int = 50,
    slope_lookback: int = 5,
    adx_period: int = 14,
) -> TrendScoreResult | None:
    """Compute graduated trend score from daily OHLC data.

    Returns None if insufficient data (need >= sma_slow bars).
    """
    if len(closes) < sma_fast:
        return None

    close = closes[-1]
    sma20 = _sma(closes, sma_fast)
    sma50 = _sma(closes, sma_slow) if len(closes) >= sma_slow else None

    # 1. Close vs SMA20 (weight 25)
    pct_from_sma20 = (close - sma20) / sma20
    comp_sma20 = _clamp(pct_from_sma20 / 0.03, -1.0, 1.0) * 25.0

    # 2. Close vs SMA50 (weight 25)
    if sma50:
        pct_from_sma50 = (close - sma50) / sma50
        comp_sma50 = _clamp(pct_from_sma50 / 0.03, -1.0, 1.0) * 25.0
    else:
        comp_sma50 = 0.0

    # 3. SMA20 slope over N days (weight 20)
    if len(closes) >= sma_fast + slope_lookback:
        sma20_ago = _sma(closes[:-slope_lookback], sma_fast)
        slope_pct = (sma20 - sma20_ago) / sma20_ago if sma20_ago else 0.0
        comp_slope = _clamp(slope_pct / 0.02, -1.0, 1.0) * 20.0
    else:
        comp_slope = 0.0

    # 4. ADX direction (weight 15)
    adx_val, plus_di, minus_di = _compute_adx(highs, lows, closes, adx_period)
    if plus_di is not None and minus_di is not None:
        di_diff = plus_di - minus_di
        comp_adx = _clamp(di_diff / 30.0, -1.0, 1.0) * 15.0
        if adx_val and adx_val > 25:
            comp_adx *= min(adx_val / 25.0, 1.5)
            comp_adx = _clamp(comp_adx, -15.0, 15.0)
    else:
        comp_adx = 0.0

    # 5. Position in 20-day range (weight 15)
    recent_highs = highs[-20:] if len(highs) >= 20 else highs
    recent_lows = lows[-20:] if len(lows) >= 20 else lows
    high_20 = max(recent_highs)
    low_20 = min(recent_lows)
    if high_20 > low_20:
        range_pos = (close - low_20) / (high_20 - low_20)
        comp_range = (range_pos - 0.5) * 2.0 * 15.0
    else:
        comp_range = 0.0

    total = _clamp(comp_sma20 + comp_sma50 + comp_slope + comp_adx + comp_range, -100.0, 100.0)

    if total >= 40:
        classification = "strong_uptrend"
    elif total >= 15:
        classification = "uptrend"
    elif total <= -40:
        classification = "strong_downtrend"
    elif total <= -15:
        classification = "downtrend"
    else:
        classification = "neutral"

    return TrendScoreResult(
        score=total,
        classification=classification,
        close_vs_sma20=comp_sma20,
        close_vs_sma50=comp_sma50,
        sma20_slope=comp_slope,
        adx_direction=comp_adx,
        range_position=comp_range,
        sma20=sma20,
        sma50=sma50,
        adx=adx_val,
        plus_di=plus_di,
        minus_di=minus_di,
    )


def _sma(values: list[float], period: int) -> float:
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-period:]) / period


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> tuple[float | None, float | None, float | None]:
    """Compute ADX, +DI, -DI from daily OHLC data."""
    n = len(closes)
    if n < 2 * period + 1 or n != len(highs) or n != len(lows):
        return None, None, None

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        plus_dm = max(high_diff, 0.0) if high_diff > low_diff else 0.0
        minus_dm = max(low_diff, 0.0) if low_diff > high_diff else 0.0

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < 2 * period:
        return None, None, None

    atr = sum(tr_list[:period])
    smooth_plus_dm = sum(plus_dm_list[:period])
    smooth_minus_dm = sum(minus_dm_list[:period])

    dx_list = []
    for i in range(period, len(tr_list)):
        atr = atr - (atr / period) + tr_list[i]
        smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period) + plus_dm_list[i]
        smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period) + minus_dm_list[i]

        plus_di = (smooth_plus_dm / atr) * 100 if atr > 0 else 0.0
        minus_di = (smooth_minus_dm / atr) * 100 if atr > 0 else 0.0

        di_sum = plus_di + minus_di
        dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0.0
        dx_list.append((dx, plus_di, minus_di))

    if len(dx_list) < period:
        return None, None, None

    adx = sum(d[0] for d in dx_list[:period]) / period
    for i in range(period, len(dx_list)):
        adx = (adx * (period - 1) + dx_list[i][0]) / period

    _, final_plus_di, final_minus_di = dx_list[-1]
    return adx, final_plus_di, final_minus_di
