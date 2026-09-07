"""Fase 4 tests: sqrt impact, ADV outliers, headroom thresholds."""

import numpy as np
import pandas as pd
import pytest

from crypto_checker.capacity import (
    square_root_impact,
    linear_impact,
    compute_adv,
    capacity_impact_overlay,
)


def test_sqrt_impact_greater_than_linear_at_large_size():
    small = 0.01
    large = 0.50
    assert square_root_impact(large, 1.0, 0.02) > linear_impact(large, 1.0, 0.02)
    # At tiny size linear is (correctly) the smaller one; both agree at zero.
    assert square_root_impact(0.0, 1.0, 0.02) == 0.0
    assert square_root_impact(small, 1.0, 0.02) > linear_impact(small, 1.0, 0.02)


def test_adv_excludes_outlier_volume_days():
    normal = [1_000_000.0] * 89
    spike = [50_000_000.0]  # listing spike: >3 std above the mean
    vol = pd.Series(normal + spike)
    assert compute_adv(vol, 90, exclude_outlier_days=True) == pytest.approx(1_000_000.0)
    assert compute_adv(vol, 90, exclude_outlier_days=False) > 1_000_000.0
    with pytest.raises(ValueError, match="at least one positive"):
        compute_adv(pd.Series([], dtype=float), 90, True)


def test_headroom_status_thresholds():
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    rows = []
    for i, d in enumerate(dates):
        rows.append([str(d), "A", 100.0 + i * 0.5 + (i % 7), 10_000_000.0])
        rows.append([str(d), "B", 100.0, 10_000_000.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "quote_volume"])
    trades = pd.DataFrame([{"timestamp": dates[-1], "asset": "A", "notional": 100_000.0}])
    overlay = capacity_impact_overlay(trades, data, "quote_volume", min_headroom=1.5)
    assert overlay["status"] == "SAFE"  # participation 1% -> headroom 150
    assert overlay["max_participation_rate"] == pytest.approx(0.01)
    big = pd.DataFrame([{"timestamp": dates[-1], "asset": "A", "notional": 20_000_000.0}])
    overlay_big = capacity_impact_overlay(big, data, "quote_volume", min_headroom=1.5)
    assert overlay_big["status"] == "BREACH"  # participation 200% -> headroom 0.75
    assert overlay_big["max_impact_bps"] > overlay["max_impact_bps"]


def test_linear_model_warns_explicitly():
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    rows = []
    for d in dates:
        rows.append([str(d), "A", 100.0, 10_000_000.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "quote_volume"])
    trades = pd.DataFrame([{"timestamp": dates[-1], "asset": "A", "notional": 100_000.0}])
    with pytest.warns(UserWarning, match="deprecated"):
        capacity_impact_overlay(trades, data, "quote_volume", impact_model="linear")
