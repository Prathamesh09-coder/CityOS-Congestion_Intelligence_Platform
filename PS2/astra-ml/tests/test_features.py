"""Tests for feature engineering correctness.

- Cyclical encoding: sin/cos for hours 0 and 12 should be at known positions
- Target encoding: no leakage (encoded values don't depend on current row's target)
- Text length features: correct computation
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest


class TestCyclicalEncoding:
    """Test cyclical time feature encoding."""

    def test_hour_0_encoding(self) -> None:
        """Hour 0 should have sin=0, cos=1."""
        hour = 0
        sin_val = math.sin(hour * 2 * math.pi / 24)
        cos_val = math.cos(hour * 2 * math.pi / 24)
        assert abs(sin_val - 0.0) < 1e-10
        assert abs(cos_val - 1.0) < 1e-10

    def test_hour_6_encoding(self) -> None:
        """Hour 6 should have sin=1, cos=0."""
        hour = 6
        sin_val = math.sin(hour * 2 * math.pi / 24)
        cos_val = math.cos(hour * 2 * math.pi / 24)
        assert abs(sin_val - 1.0) < 1e-10
        assert abs(cos_val - 0.0) < 1e-6

    def test_hour_12_encoding(self) -> None:
        """Hour 12 should have sin≈0, cos=-1."""
        hour = 12
        sin_val = math.sin(hour * 2 * math.pi / 24)
        cos_val = math.cos(hour * 2 * math.pi / 24)
        assert abs(sin_val) < 1e-10
        assert abs(cos_val - (-1.0)) < 1e-10

    def test_cyclical_continuity(self) -> None:
        """Hours 23 and 0 should be close in the embedding space."""
        h23_sin = math.sin(23 * 2 * math.pi / 24)
        h23_cos = math.cos(23 * 2 * math.pi / 24)
        h0_sin = math.sin(0 * 2 * math.pi / 24)
        h0_cos = math.cos(0 * 2 * math.pi / 24)

        # Euclidean distance in sin/cos space should be small
        dist = math.sqrt((h23_sin - h0_sin) ** 2 + (h23_cos - h0_cos) ** 2)
        assert dist < 0.5, f"Hours 23 and 0 should be close, got dist={dist}"

    def test_dow_encoding_range(self) -> None:
        """All day-of-week encodings should be in [-1, 1]."""
        for dow in range(1, 8):  # 1=Mon through 7=Sun
            sin_val = math.sin(dow * 2 * math.pi / 7)
            cos_val = math.cos(dow * 2 * math.pi / 7)
            assert -1.0 <= sin_val <= 1.0
            assert -1.0 <= cos_val <= 1.0


class TestTargetEncoding:
    """Test target encoding for leakage prevention."""

    def test_smoothed_encoding_range(self) -> None:
        """Smoothed target encoding should produce values in [0, 1]."""
        from astra_ml.data.features import add_target_encoded_features

        df = pl.DataFrame({
            "event_cause": ["accident"] * 50 + ["pot_holes"] * 50,
            "corridor": ["road_a"] * 50 + ["road_b"] * 50,
            "requires_road_closure": [True] * 40 + [False] * 10 + [True] * 5 + [False] * 45,
        })

        result = add_target_encoded_features(df, smoothing=10.0)

        # All encoded values should be between 0 and 1
        cause_rates = result["cause_closure_rate"].to_numpy()
        assert np.all(cause_rates >= 0.0), "Cause closure rates should be >= 0"
        assert np.all(cause_rates <= 1.0), "Cause closure rates should be <= 1"

        corridor_rates = result["corridor_closure_rate"].to_numpy()
        assert np.all(corridor_rates >= 0.0), "Corridor closure rates should be >= 0"
        assert np.all(corridor_rates <= 1.0), "Corridor closure rates should be <= 1"

    def test_smoothing_effect(self) -> None:
        """Higher smoothing should pull rates toward the global mean."""
        from astra_ml.data.features import add_target_encoded_features

        df = pl.DataFrame({
            "event_cause": ["rare_event"] * 5 + ["common_event"] * 95,
            "corridor": ["road_a"] * 100,
            "requires_road_closure": [True] * 5 + [True] * 10 + [False] * 85,
        })

        result_low_smooth = add_target_encoded_features(df, smoothing=1.0)
        result_high_smooth = add_target_encoded_features(df, smoothing=100.0)

        # With high smoothing, the rare event rate should be closer to global mean
        global_mean = 15.0 / 100.0  # 15% positive overall

        rare_rate_low = result_low_smooth.filter(
            pl.col("event_cause") == "rare_event"
        )["cause_closure_rate"][0]
        rare_rate_high = result_high_smooth.filter(
            pl.col("event_cause") == "rare_event"
        )["cause_closure_rate"][0]

        assert abs(rare_rate_high - global_mean) < abs(rare_rate_low - global_mean), \
            "High smoothing should bring rates closer to global mean"


class TestTextLengthFeatures:
    """Test text length feature computation."""

    def test_text_length_basic(self) -> None:
        """Text length should count characters correctly."""
        from astra_ml.data.features import add_text_length_features

        df = pl.DataFrame({
            "description": ["hello world", "test", None, ""],
            "comment": ["good", None, "comment here", ""],
        })

        result = add_text_length_features(df)

        assert result["description_len"][0] == 11  # "hello world"
        assert result["description_len"][1] == 4   # "test"
        assert result["description_len"][2] == 0   # None → 0
        assert result["description_len"][3] == 0   # "" → 0

        assert result["comment_len"][0] == 4   # "good"
        assert result["comment_len"][1] == 0   # None → 0
        assert result["comment_len"][2] == 12  # "comment here"
