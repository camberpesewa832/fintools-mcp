"""Support and Resistance — key price levels from swing highs/lows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriceLevel:
    price: float
    level_type: str  # "support" or "resistance"
    touches: int  # how many times price tested this level
    strength: str  # "strong", "moderate", "weak"


def find_support_resistance(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    current_price: float,
    lookback: int = 60,
    cluster_pct: float = 1.0,
    max_levels: int = 5,
) -> list[PriceLevel]:
    """Find support and resistance levels from swing highs and lows.

    Groups nearby pivots into clusters, counts touches, and ranks by strength.

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        current_price: Current price for classifying support vs resistance
        lookback: Number of bars to analyze (default 60)
        cluster_pct: Percentage threshold for grouping nearby levels (default 1.0%)
        max_levels: Maximum levels to return per side (default 5)
    """
    if len(highs) < 10:
        return []

    h = highs[-lookback:] if len(highs) >= lookback else highs
    l = lows[-lookback:] if len(lows) >= lookback else lows

    # Find swing highs (local maxima)
    swing_highs = []
    for i in range(2, len(h) - 2):
        if h[i] > h[i - 1] and h[i] > h[i - 2] and h[i] > h[i + 1] and h[i] > h[i + 2]:
            swing_highs.append(h[i])

    # Find swing lows (local minima)
    swing_lows = []
    for i in range(2, len(l) - 2):
        if l[i] < l[i - 1] and l[i] < l[i - 2] and l[i] < l[i + 1] and l[i] < l[i + 2]:
            swing_lows.append(l[i])

    # Combine all pivot points
    all_pivots = swing_highs + swing_lows
    if not all_pivots:
        return []

    # Cluster nearby levels
    all_pivots.sort()
    clusters: list[list[float]] = []
    current_cluster = [all_pivots[0]]

    for price in all_pivots[1:]:
        cluster_avg = sum(current_cluster) / len(current_cluster)
        if abs(price - cluster_avg) / cluster_avg * 100 <= cluster_pct:
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]
    clusters.append(current_cluster)

    # Build levels from clusters
    levels = []
    for cluster in clusters:
        avg_price = sum(cluster) / len(cluster)
        touches = len(cluster)

        if touches >= 3:
            strength = "strong"
        elif touches >= 2:
            strength = "moderate"
        else:
            strength = "weak"

        level_type = "support" if avg_price < current_price else "resistance"

        levels.append(PriceLevel(
            price=round(avg_price, 2),
            level_type=level_type,
            touches=touches,
            strength=strength,
        ))

    # Split into support and resistance, sort by proximity to current price
    support = sorted(
        [l for l in levels if l.level_type == "support"],
        key=lambda x: current_price - x.price,
    )[:max_levels]

    resistance = sorted(
        [l for l in levels if l.level_type == "resistance"],
        key=lambda x: x.price - current_price,
    )[:max_levels]

    return support + resistance
