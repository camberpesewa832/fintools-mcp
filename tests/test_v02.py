"""Tests for v0.2 features — trend score, screener, support/resistance."""

from fintools_mcp.analysis.trend_score import compute_trend_score
from fintools_mcp.analysis.support_resistance import find_support_resistance


def _trending_up(n=100):
    """Generate trending up OHLC data."""
    closes = [100 + i * 0.5 + (i % 5 - 2) * 0.3 for i in range(n)]
    highs = [c + 1.5 for c in closes]
    lows = [c - 1.0 for c in closes]
    return highs, lows, closes


def _trending_down(n=100):
    """Generate trending down OHLC data."""
    closes = [200 - i * 0.5 + (i % 5 - 2) * 0.3 for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.5 for c in closes]
    return highs, lows, closes


class TestTrendScore:
    def test_needs_minimum_data(self):
        result = compute_trend_score([100] * 5, [99] * 5, [100] * 5)
        assert result is None

    def test_uptrend(self):
        highs, lows, closes = _trending_up(100)
        result = compute_trend_score(highs, lows, closes)
        assert result is not None
        assert result.score > 0
        assert "uptrend" in result.classification

    def test_downtrend(self):
        highs, lows, closes = _trending_down(100)
        result = compute_trend_score(highs, lows, closes)
        assert result is not None
        assert result.score < 0
        assert "downtrend" in result.classification

    def test_score_range(self):
        highs, lows, closes = _trending_up(100)
        result = compute_trend_score(highs, lows, closes)
        assert -100 <= result.score <= 100

    def test_components_sum(self):
        highs, lows, closes = _trending_up(100)
        result = compute_trend_score(highs, lows, closes)
        component_sum = (
            result.close_vs_sma20 + result.close_vs_sma50 +
            result.sma20_slope + result.adx_direction + result.range_position
        )
        assert abs(component_sum - result.score) < 0.01


class TestSupportResistance:
    def test_empty_data(self):
        levels = find_support_resistance([], [], [], 100)
        assert levels == []

    def test_finds_levels(self):
        # Create data with clear swing points
        highs, lows, closes = _trending_up(80)
        # Add some swings
        for i in range(20, 80, 10):
            highs[i] = max(highs) * 0.99
        for i in range(25, 80, 10):
            lows[i] = min(lows) * 1.01

        levels = find_support_resistance(highs, lows, closes, closes[-1])
        # Should find at least some levels
        assert isinstance(levels, list)

    def test_classifies_support_resistance(self):
        highs, lows, closes = _trending_up(80)
        current = closes[-1]
        levels = find_support_resistance(highs, lows, closes, current)
        for level in levels:
            if level.price < current:
                assert level.level_type == "support"
            else:
                assert level.level_type == "resistance"

    def test_strength_labels(self):
        highs, lows, closes = _trending_up(80)
        levels = find_support_resistance(highs, lows, closes, closes[-1])
        for level in levels:
            assert level.strength in ("strong", "moderate", "weak")
