"""Fase 2 tests: CPCV splits, purge/embargo, regime concentration."""

import numpy as np
import pandas as pd
import pytest

from crypto_checker.cpcv import generate_cpcv_splits, label_regime, run_cpcv, summarize_cpcv, FoldResult


def _labels(n=100):
    return pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))


def test_cpcv_split_count_and_coverage():
    splits = generate_cpcv_splits(_labels(100), n_groups=5, n_test_groups=2, embargo_pct=0.0, purge_pct=0.0)
    assert len(splits) == 10  # C(5,2)
    for train_idx, test_idx in splits:
        assert len(test_idx) == 40
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) + len(test_idx) == 100


def test_purge_removes_overlapping_train_indices():
    splits = generate_cpcv_splits(_labels(100), n_groups=5, n_test_groups=1, embargo_pct=0.0, purge_pct=0.05)
    # purge_len = 5: every train bar within 5 positions of a test bar is gone.
    for train_idx, test_idx in splits:
        test_pos = {t.value for t in pd.DatetimeIndex(test_idx)}
        base = _labels(100)
        for t in pd.DatetimeIndex(train_idx):
            gap = min(abs((t - u).days) for u in pd.DatetimeIndex(test_idx))
            assert gap > 5, (t, gap)


def test_embargo_window_applied_after_test_period():
    splits = generate_cpcv_splits(_labels(100), n_groups=5, n_test_groups=1, embargo_pct=0.05, purge_pct=0.0)
    # embargo_len = 5: no train bar in (test_end, test_end+5d].
    for train_idx, test_idx in splits:
        test_end = pd.DatetimeIndex(test_idx).max()
        for t in pd.DatetimeIndex(train_idx):
            assert not (test_end < t <= test_end + pd.Timedelta(days=5)), t


def test_regime_concentration_flag_triggers_when_single_regime_dominates():
    def fold(i, regime, ret):
        n = 20
        eq = pd.Series(1 + np.linspace(0, ret, n))
        return FoldResult(fold_id=i, train_idx=pd.Index([]), test_idx=pd.Index([]),
                          sharpe_oos=ret * 10, test_return=ret,
                          equity_curve_train=pd.Series([1.0, 1.0]), equity_curve_oos=eq,
                          regime_label=regime)
    results = [fold(0, "low_vol", 0.30), fold(1, "high_vol", 0.01), fold(2, "high_vol", -0.02)]
    summary = summarize_cpcv(results)
    assert summary["regime_concentration_flag"] is True
    assert summary["pct_profitable_folds"] == pytest.approx(2 / 3)
    assert set(summary["sharpe_by_regime"]) == {"low_vol", "high_vol"}


def test_summarize_reports_worst_drawdowns():
    up = pd.Series([1.0, 1.1, 1.2])
    down = pd.Series([1.0, 0.7, 0.9])
    results = [FoldResult(0, pd.Index([]), pd.Index([]), 0.0, 0.2, up, up, "low_vol"),
               FoldResult(1, pd.Index([]), pd.Index([]), 0.0, -0.1, down, down, "high_vol")]
    summary = summarize_cpcv(results)
    assert summary["max_dd_walk_forward"] == pytest.approx(-0.3)
    assert summary["max_dd_oos"] == pytest.approx(-0.3)
    assert summary["n_folds"] == 2


def test_run_cpcv_end_to_end_with_trivial_strategy():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(dates))))
    data = pd.DataFrame({"timestamp": list(dates) * 2, "asset": ["A"] * 120 + ["B"] * 120,
                         "price": list(prices) + [100.0] * 120})

    class Cfg:
        cpcv_n_groups = 5
        cpcv_n_test_groups = 1
        cpcv_embargo_pct = 0.01
        cpcv_purge_pct = 0.01
        regime_label_source = "realized_vol_percentile"

    def strategy(train, test):
        rets = pd.Series(0.001, index=pd.to_datetime(sorted(test["timestamp"].unique()), utc=True))
        eq = (1 + rets).cumprod()
        return {"test_returns": rets, "train_equity": pd.Series([1.0, 1.0]), "test_equity": eq}

    results = run_cpcv(data, strategy, Cfg())
    assert len(results) == 5  # C(5,1), none empty
    summary = summarize_cpcv(results)
    assert summary["n_folds"] == 5
    assert summary["pct_profitable_folds"] == 1.0
