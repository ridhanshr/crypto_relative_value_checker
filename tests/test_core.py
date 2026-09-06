import numpy as np
import pandas as pd
import pytest
from crypto_checker.core import CheckerConfig, check_strategy
from crypto_checker.signal_audit import audit_signals
from crypto_checker.validation import run_validation
from crypto_checker.signals import build_signals
from crypto_checker.selection import walk_forward
from crypto_checker.assets import canonicalize_assets, audit_asset_continuity, canonical_asset, migration_manifest, audit_migration_discontinuities, audit_migration_collisions
from crypto_checker.preflight import validate_dataset


def test_ranking_exposure_and_costs():
    # t+1 execution: ranked at t2 on t1 signals (long A / short D, filled at
    # t2 closes 110/90); PnL accrues t2 -> t3 (A +10%, D -10% short wins).
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 4], ["2026-01-01", "B", 100, 3], ["2026-01-01", "C", 100, 2], ["2026-01-01", "D", 100, 1],
        ["2026-01-02", "A", 110, 4], ["2026-01-02", "B", 100, 3], ["2026-01-02", "C", 100, 2], ["2026-01-02", "D", 90, 1],
        ["2026-01-03", "A", 121, 4], ["2026-01-03", "B", 100, 3], ["2026-01-03", "C", 100, 2], ["2026-01-03", "D", 81, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.001, slippage_rate=0))
    assert result["ranking"].iloc[0].long_assets == "A"
    assert result["exposure"].iloc[0].gross_exposure == 2
    assert result["pnl"].iloc[1].total_pnl == pytest.approx(19960.0)
    assert result["metrics"]["periods"] == 1
    assert result["trades"].notna().all().all()


def test_rejects_invalid_values():
    data = pd.DataFrame([["2026-01-01", "A", 0, 1], ["2026-01-01", "B", 1, 0], ["2026-01-02", "A", 0, 1], ["2026-01-02", "B", 1, 0]], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="Price"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1))


def test_funding_long_is_paid_and_short_received():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2, 0.01], ["2026-01-01", "B", 100, 1, 0.01], ["2026-01-02", "A", 100, 2, 0.01], ["2026-01-02", "B", 100, 1, 0.01], ["2026-01-03", "A", 100, 2, 0.01], ["2026-01-03", "B", 100, 1, 0.01]], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1))
    assert result["pnl"].iloc[1].funding_pnl == 0


def test_rebalance_charges_turnover_and_costs():
    # t+1 execution: entry at t2 on t1 signals, flip at t3 on t2 signals.
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 4], ["2026-01-01", "B", 100, 3], ["2026-01-01", "C", 100, 2], ["2026-01-01", "D", 100, 1],
        ["2026-01-02", "A", 100, 1], ["2026-01-02", "B", 100, 2], ["2026-01-02", "C", 100, 3], ["2026-01-02", "D", 100, 4],
        ["2026-01-03", "A", 100, 1], ["2026-01-03", "B", 100, 2], ["2026-01-03", "C", 100, 3], ["2026-01-03", "D", 100, 4],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.001, slippage_rate=0.001))
    assert result["pnl"].iloc[1].turnover == pytest.approx(4.0)
    assert result["pnl"].iloc[1].fee_cost == pytest.approx(398.4)
    assert result["pnl"].iloc[1].slippage_cost == pytest.approx(398.4)
    assert len(result["trades"]) == 4


def test_signal_audit_rejects_constant_signal():
    data = pd.DataFrame([["2026-01-01", "A", 100, 1], ["2026-01-02", "A", 101, 1], ["2026-01-01", "B", 100, 2], ["2026-01-02", "B", 99, 2]], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="Signal audit failed"):
        audit_signals(data)


def test_validation_requires_long_enough_data():
    data = pd.DataFrame([["2026-01-01", "A", 100, 1], ["2026-01-02", "A", 101, 1]], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="Insufficient data"):
        run_validation(data, CheckerConfig(n_long=1, n_short=1))


def test_nan_funding_tolerated_as_post_ranking_cost():
    # Funding is a post-ranking cost model: NaN funding on an otherwise
    # active row must not remove the asset from ranking. It contributes 0.0
    # to funding PnL and is recorded for investigation.
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2, 0.01], ["2026-01-01", "B", 100, 1, 0.01],
        ["2026-01-02", "A", 100, 2, float("nan")], ["2026-01-02", "B", 100, 1, 0.01],
        ["2026-01-03", "A", 100, 2, 0.01], ["2026-01-03", "B", 100, 1, 0.01],
    ], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0))
    assert set(result["ranking"].iloc[1].long_assets.split(",")) | set(result["ranking"].iloc[1].short_assets.split(",")) == {"A", "B"}
    gaps = result["funding_gaps"]
    assert len(gaps) == 1 and gaps.iloc[0]["reason"] == "active"
    assert "funding" not in " ".join(result["violations"]["type"].tolist()) if not result["violations"].empty else True


def test_rejects_nonfinite_funding():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2, float("inf")], ["2026-01-01", "B", 100, 1, 0.0], ["2026-01-02", "A", 100, 2, float("inf")], ["2026-01-02", "B", 100, 1, 0.0]], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    with pytest.raises(ValueError, match="Funding rate"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1))


def test_require_funding_rejects_price_only_input():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1]], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="Funding rate column required"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1, require_funding=True))


def test_positions_expose_next_execution_timestamp():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1], ["2026-01-02", "A", 100, 2], ["2026-01-02", "B", 100, 1], ["2026-01-03", "A", 100, 2], ["2026-01-03", "B", 100, 1]], columns=["timestamp", "asset", "price", "signal"])
    positions = check_strategy(data, CheckerConfig(n_long=1, n_short=1))["positions"]
    assert positions.iloc[0].execution_timestamp > positions.iloc[0].signal_timestamp
    # Every decided position is filled at a strictly later bar (t+1 rule).
    decided = positions.dropna(subset=["execution_timestamp"])
    assert (decided["execution_timestamp"] > decided["signal_timestamp"]).all()


def test_max_drawdown_measured_from_initial_equity():
    # t+1 execution: position opens at t2 close (A=100); the t2 -> t3 drop
    # to 80 then drives drawdown to -0.20 from initial equity.
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1],
        ["2026-01-02", "A", 100, 2], ["2026-01-02", "B", 100, 1],
        ["2026-01-03", "A", 80, 2], ["2026-01-03", "B", 100, 1],
        ["2026-01-04", "A", 85, 2], ["2026-01-04", "B", 100, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0))
    summary = result["metrics"]
    assert summary["max_drawdown"] == pytest.approx(-0.20)


def test_momentum_signal_is_point_in_time():
    rows = [[f"2026-01-{d:02d}", "A", 100 + d, float(d)] for d in range(1, 21)]
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal"])
    built = build_signals(data)
    row = built.iloc[18]
    expected = built.iloc[4:18]["asset_return"].sum()
    assert row["momentum_14"] == pytest.approx(expected)


def test_build_signals_recomputes_stale_asset_return():
    # Regression: a stale precomputed asset_return column (e.g. merged in
    # from a partial downloader output) must not poison downstream signals.
    # build_signals must always re-derive returns from price.
    rows = [[f"2026-01-{d:02d}", "A", 100 + d] for d in range(1, 21)]
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price"])
    data["asset_return"] = float("nan")
    data["signal"] = float("nan")
    built = build_signals(data)
    assert built["asset_return"].notna().sum() == 19
    assert built["momentum_7"].notna().sum() > 0
    row = built.iloc[18]
    expected = built.iloc[11:18]["asset_return"].sum()
    assert row["momentum_7"] == pytest.approx(expected)


def test_carry_signal_is_negative_funding():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 0, 0.01], ["2026-01-02", "A", 100, 0, -0.02],
    ], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    built = build_signals(data)
    # Shifted by one bar: today's print is only known at/after close(t).
    assert pd.isna(built["carry"].iloc[0])
    assert built["carry"].iloc[1] == pytest.approx(-0.01)


def test_contemporaneous_signals_cannot_see_today():
    # Reversal, carry and residual signals must be lagged one bar so that
    # signal(t) never contains information revealed only at close(t).
    dates = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    rows = []
    for i, ts in enumerate(dates):
        rows.append([ts, "A", 100 + i, 0.001])
        rows.append([ts, "B", 100 - i, 0.001])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "funding_rate"])
    built = build_signals(data)
    a = built[built.asset == "A"].reset_index(drop=True)
    # reversal_1 at bar i must equal -(return over i-1 -> i-2), not today's move.
    assert a.loc[5, "reversal_1"] == pytest.approx(-(a.loc[4, "price"] / a.loc[3, "price"] - 1))
    assert a.loc[5, "carry"] == pytest.approx(-a.loc[4, "funding_rate"])


def test_ranking_uses_configured_signal_column():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 4, 1], ["2026-01-01", "B", 100, 3, 4],
        ["2026-01-01", "C", 100, 2, 3], ["2026-01-01", "D", 100, 1, 2],
        ["2026-01-02", "A", 100, 4, 1], ["2026-01-02", "B", 100, 3, 4],
        ["2026-01-02", "C", 100, 2, 3], ["2026-01-02", "D", 100, 1, 2],
    ], columns=["timestamp", "asset", "price", "signal", "momentum_30"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, signal_column="momentum_30"))
    assert result["ranking"].iloc[0].long_assets == "B"


def test_walk_forward_runs_on_synthetic_panel():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=520, freq="D", tz="UTC")
    frames = []
    for i in range(6):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        frames.append(pd.DataFrame({"timestamp": dates, "asset": f"A{i}", "price": prices, "signal": rng.normal(0, 0.01, len(dates)), "funding_rate": rng.normal(0, 0.0002, len(dates))}))
    data = pd.concat(frames, ignore_index=True)
    result = walk_forward(data, base_config=CheckerConfig(n_long=2, n_short=2, min_signal_gap=0), min_train_days=200, test_days=100)
    assert result["n_folds"] >= 3
    assert "walk_forward" in result and "gates" in result


def test_vol_targeting_reduces_exposure_after_vol_spike():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    frames = []
    for i in range(4):
        shocks = rng.normal(0, 0.06, 40)
        calm = rng.normal(0, 0.001, 40)
        prices = 100 * np.exp(np.cumsum(np.concatenate([shocks, calm])))
        frames.append(pd.DataFrame({"timestamp": dates, "asset": f"A{i}", "price": prices, "signal": np.arange(len(dates)) * (1 if i % 2 else -1) * 0.01}))
    data = pd.concat(frames, ignore_index=True)
    off = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0))
    on = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, vol_target_annual=0.05, vol_lookback=30))
    after_shocks = on["exposure"]["gross_exposure"].iloc[45:60]
    assert after_shocks.max() < off["exposure"]["gross_exposure"].iloc[45:60].max()


def test_low_vol_signal_point_in_time():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    rows = []
    for i, d in enumerate(dates):
        rows.append([str(d), "A", 100 * np.exp(np.sin(i) * 0.1), rng.normal()])
        rows.append([str(d), "B", 100, 0.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal"])
    built = build_signals(data)
    a = built[built.asset == "A"].reset_index(drop=True)
    expected = -a["asset_return"].iloc[13:27].std(ddof=1)
    assert built[built.asset == "A"]["low_vol_14"].iloc[27] == pytest.approx(expected)


def test_funding_surprise_uses_prior_mean():
    data = pd.DataFrame({
        "timestamp": [f"2026-01-{d:02d}" for d in range(1, 17)] * 2,
        "asset": ["A"] * 16 + ["B"] * 16,
        "price": [100] * 32,
        "signal": [0.0] * 32,
        "funding_rate": [0.01] * 15 + [0.03] + [0.0] * 16,
    })
    built = build_signals(data)
    a = built[built.asset == "A"].reset_index(drop=True)
    # Lagged one bar: surprise at bar 15 uses the 0.01 print from bar 14,
    # not the 0.03 print revealed only at bar 15's close.
    assert a["funding_surprise"].iloc[15] == pytest.approx(-(0.01 - 0.01))


def test_liquidity_tiers_charge_illiquid_assets_more():
    dates = [f"2026-01-{d:02d}" for d in range(1, 6)]
    volumes = {"LIQ": 1e7, "L2": 1e6, "L3": 1e5, "L4": 5e4, "L5": 1e4, "L6": 5e3, "L7": 1e3, "ILLQ": 1e2}
    rows = []
    for d, ts in enumerate(dates, start=1):
        top = 4 if d % 2 else -4
        for i, asset in enumerate(volumes):
            if asset == "LIQ":
                sig = top
            elif asset == "ILLQ":
                sig = -top
            else:
                sig = top * (0.5 - 0.01 * i)
            rows.append([ts, asset, 100, sig, volumes[asset]])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    tiers = ((0.75, 0.0004, 0.0005), (0.25, 0.0012, 0.002), (0.0, 0.0015, 0.004))
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.001, slippage_rate=0.001, liquidity_column="quote_volume", liquidity_tiers=tiers))
    day3 = result["pnl"].iloc[2]
    equity_before = result["pnl"].iloc[1]["equity"]
    traded = day3.turnover * equity_before
    expected_fee = traded * 0.5 * 0.0004 + traded * 0.5 * 0.0015
    assert day3.fee_cost == pytest.approx(expected_fee)
    assert day3.fee_cost / traded < 0.001


def test_invalid_liquidity_tiers_rejected():
    data = pd.DataFrame([["2026-01-01", "A", 100, 1, 10], ["2026-01-01", "B", 100, 2, 20]], columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    with pytest.raises(ValueError, match="liquidity tiers"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1, liquidity_column="quote_volume", liquidity_tiers=((0.5, 0.001, 0.001), (0.5, 0.002, 0.002))))


def test_matic_pol_are_one_canonical_asset():
    data = pd.DataFrame([
        ["2024-01-01", "MATICUSDT", 1.0, 0.1, 0.0001],
        ["2024-01-02", "MATICUSDT", 1.1, 0.2, 0.0002],
        ["2024-01-03", "POLUSDT", 1.1, 0.3, 0.0003],
        ["2024-01-04", "POLUSDT", 1.2, 0.4, 0.0004],
    ], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    result = canonicalize_assets(data)
    assert result.asset.unique().tolist() == ["POLUSDT"]
    assert len(result) == 4
    assert result.asset_return.iloc[2] == pytest.approx(0)


def test_migration_master_mapping_canonicalizes_all_known_symbols():
    expected = {"GALUSDT": "GUSDT", "GUSDT": "GUSDT", "OMNIUSDT": "NOMUSDT", "NOMUSDT": "NOMUSDT", "MATICUSDT": "POLUSDT", "POLUSDT": "POLUSDT", "NANOUSDT": "XNOUSDT", "VENUSDT": "VETUSDT", "BCCUSDT": "BCHUSDT", "ANTOLDUSDT": "ANTUSDT"}
    assert {source: canonical_asset(source) for source in expected} == expected
    assert len(migration_manifest()) == 7


def test_migration_discontinuity_requires_official_factor():
    data = pd.DataFrame([
        ["2024-01-01", "GALUSDT", 1.0], ["2024-01-02", "GUSDT", 2.0],
    ], columns=["timestamp", "asset", "price"])
    canonical = canonicalize_assets(data)
    gaps = audit_migration_discontinuities(canonical)
    assert len(gaps) == 1
    assert gaps.iloc[0].status == "REQUIRES_OFFICIAL_FACTOR"


def test_migration_collision_is_reported_before_merge():
    data = pd.DataFrame([["2024-01-01", "MATICUSDT", 1.0], ["2024-01-01", "POLUSDT", 1.0]], columns=["timestamp", "asset", "price"])
    collisions = audit_migration_collisions(data)
    assert len(collisions) == 1
    assert collisions.iloc[0].status == "MIGRATION_COLLISION"


def test_official_migration_ratio_adjusts_legacy_price():
    data = pd.DataFrame([["2024-01-01", "GALUSDT", 60.0], ["2024-01-02", "GUSDT", 1.0]], columns=["timestamp", "asset", "price"])
    result = canonicalize_assets(data)
    assert result.price.tolist() == pytest.approx([1.0, 1.0])
    assert result.price_adjustment_factor.iloc[0] == 60.0


def test_asset_continuity_reports_gaps():
    data = pd.DataFrame({"timestamp": ["2024-01-01", "2024-01-03"], "asset": ["POLUSDT", "POLUSDT"]})
    gaps = audit_asset_continuity(data)
    assert gaps.timestamp.dt.day.tolist() == [2]


def test_preflight_accepts_valid_panel():
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 1, 0.01, 1000], ["2024-01-01", "B", 100, 2, -0.01, 900],
        ["2024-01-02", "A", 101, 1, 0.01, 1000], ["2024-01-02", "B", 99, 2, -0.01, 900],
    ], columns=["timestamp", "asset", "price", "signal", "funding_rate", "quote_volume"])
    result = validate_dataset(data, require_funding=True, require_liquidity=True, min_assets=2, min_periods=2)
    assert result["valid"] is True


def test_preflight_rejects_missing_funding_and_duplicates():
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 1, None], ["2024-01-01", "A", 100, 1, None],
        ["2024-01-01", "B", 100, 2, 0.01],
    ], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    result = validate_dataset(data, require_funding=True, min_assets=2, min_periods=1)
    assert result["valid"] is False
    assert any("Duplicate" in error for error in result["errors"])
    # NaN funding no longer fails validity; it is classified (here: A has no
    # funding prints at all -> not_listed) and only warned about.
    assert not any("Funding rate" in error for error in result["errors"])
    assert any("not_listed" in warning for warning in result["warnings"])


def test_preflight_warns_active_funding_gap_without_failing():
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 1, 0.01, 500], ["2024-01-01", "B", 100, 2, 0.01, 600],
        ["2024-01-02", "A", 101, 1, None, 500], ["2024-01-02", "B", 99, 2, 0.01, 600],
        ["2024-01-03", "A", 102, 1, 0.01, 500], ["2024-01-03", "B", 98, 2, 0.01, 600],
    ], columns=["timestamp", "asset", "price", "signal", "funding_rate", "quote_volume"])
    result = validate_dataset(data, require_funding=True, min_assets=2, min_periods=2)
    assert result["valid"] is True
    assert any("FUNDING_GAP_ACTIVE" in warning for warning in result["warnings"])
    assert result["funding_gap_summary"].get("active", 0) == 1


def test_capacity_violation_is_reported():
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 2, 100], ["2024-01-01", "B", 100, 1, 100],
        ["2024-01-02", "A", 100, 1, 100], ["2024-01-02", "B", 100, 2, 100],
    ], columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, liquidity_column="quote_volume", liquidity_tiers=((0.0, 0.001, 0.001),), max_volume_participation=0.05))
    assert result["metrics"]["capacity_violations"] > 0


def test_missing_held_price_is_hard_error():
    # t+1 execution: position opens at t2 on t1 signals (long A), then A
    # disappears at t3 -> hard error.
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 2], ["2024-01-01", "B", 100, 1],
        ["2024-01-02", "A", 100, 2], ["2024-01-02", "B", 100, 1],
        ["2024-01-03", "B", 100, 2],
    ], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="Missing price for held positions"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1))


def test_cost_multiplier_scales_tiered_fees():
    dates = [f"2026-01-{d:02d}" for d in range(1, 4)]
    rows = []
    for d, ts in enumerate(dates, start=1):
        top = 4 if d % 2 else -4
        rows.append([ts, "A", 100, top, 1_000_000.0])
        rows.append([ts, "B", 100, -top, 100.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    tiers = ((0.5, 0.0004, 0.0005), (0.0, 0.0015, 0.004))
    base = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.001, slippage_rate=0.001, liquidity_column="quote_volume", liquidity_tiers=tiers))
    stressed = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.001, slippage_rate=0.001, liquidity_column="quote_volume", liquidity_tiers=tiers, cost_multiplier=2.0))
    assert stressed["pnl"].iloc[0].fee_cost == pytest.approx(2 * base["pnl"].iloc[0].fee_cost)
    assert stressed["pnl"].iloc[0].slippage_cost == pytest.approx(2 * base["pnl"].iloc[0].slippage_cost)
    assert stressed["metrics"]["total_fees"] > base["metrics"]["total_fees"]


def test_forced_exit_closes_disappeared_holding():
    # t+1 execution: long A opens at t2 on t1 signals; A disappears at t3
    # while B/C remain rankable -> forced_exit + negative weight change.
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 2], ["2024-01-01", "B", 100, 1],
        ["2024-01-02", "A", 100, 2], ["2024-01-02", "B", 100, 1], ["2024-01-02", "C", 100, 1],
        ["2024-01-03", "B", 100, 2], ["2024-01-03", "C", 100, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, delist_mode="forced_exit"))
    types = result["violations"]["type"].tolist()
    assert "forced_exit" in types
    assert (result["trades"][result["trades"].asset == "A"].weight_change < 0).any()


def test_validation_uses_continuous_equity():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    frames = []
    for i in range(4):
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        frames.append(pd.DataFrame({"timestamp": dates, "asset": f"A{i}", "price": prices, "signal": np.cos(np.arange(len(dates)) + i) * 0.01}))
    data = pd.concat(frames, ignore_index=True)
    result = run_validation(data, CheckerConfig(n_long=1, n_short=1))
    assert result["continuous_equity"] is True
    from crypto_checker.core import check_strategy as cs
    full = cs(data, CheckerConfig(n_long=1, n_short=1))
    oos_end = float(full["pnl"].equity.iloc[-1])
    assert result["performance"]["out_of_sample"]["final_equity"] == pytest.approx(oos_end)
    assert result["survivorship_note"]


def test_preflight_flags_delisting_suspects():
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 1], ["2024-01-10", "A", 100, 1],
        ["2024-01-01", "B", 100, 2], ["2024-01-30", "B", 100, 2],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = validate_dataset(data, min_assets=2, min_periods=2)
    assert any("Delisting" in warning for warning in result["warnings"])
    assert len(result["delisting_suspects"]) == 1


def test_repair_drops_zero_volume_stale_rows():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.repair_midcap import repair_klines
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 10, 1000.0], ["2024-01-02", "A", 101, 11, 900.0],
        ["2024-01-03", "A", 101, 0.0, 0.0], ["2024-01-04", "A", 101, 0.0, 0.0],
        ["2024-01-01", "B", 50, 5, 500.0], ["2024-01-02", "B", 51, 6, 400.0],
    ], columns=["timestamp", "asset", "price", "volume", "quote_volume"])
    repaired, manifest = repair_klines(data)
    assert len(repaired) == 4
    assert manifest["actions"][0]["rows_dropped"] == 2
    assert "asset_return" not in repaired.columns


def test_repair_migration_cutover_uses_calendar_date():
    # Regression: raw intraday-timestamp cutover misclassified NOM's first
    # daily candle (2025-10-01 00:00, $0.03975) as legacy and divided it by
    # 75, fabricating a +72x jump. Calendar-date cutover must leave it intact.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.repair_midcap import repair_klines
    data = pd.DataFrame([
        ["2025-09-30", "NOMUSDT", 4.275, 100.0, 1000.0],
        ["2025-10-01", "NOMUSDT", 0.03975, 200.0, 2000.0],
        ["2025-10-02", "NOMUSDT", 0.03874, 150.0, 1500.0],
    ], columns=["timestamp", "asset", "price", "volume", "quote_volume"])
    repaired, _ = repair_klines(data)
    row = repaired[repaired.timestamp == "2025-10-01"].iloc[0]
    assert row.price == pytest.approx(0.03975)
    legacy = repaired[repaired.timestamp == "2025-09-30"].iloc[0]
    assert legacy.price == pytest.approx(4.275 / 75)


def _spread_frame(tmp_path):
    import csv
    path = tmp_path / "sp.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["asset", "spread_bps"])
        writer.writerow(["A", 100.0])
        writer.writerow(["B", 2.0])
    return str(path)


def test_spread_calibrated_slippage_uses_max_of_spread_and_floor(tmp_path):
    dates = [f"2026-01-{d:02d}" for d in range(1, 4)]
    rows = []
    for d, ts in enumerate(dates, start=1):
        top = 4 if d % 2 else -4
        rows.append([ts, "A", 100, top, 1_000_000.0])
        rows.append([ts, "B", 100, -top, 100.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    tiers = ((0.0, 0.0005, 0.0005),)
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.0005, slippage_rate=0.0005, liquidity_column="quote_volume", liquidity_tiers=tiers, slippage_mode="spread", spread_csv=_spread_frame(tmp_path)))
    assert result["pnl"].iloc[0].slippage_cost == pytest.approx(100_000 * 0.01 + 100_000 * 0.0005)


def test_spread_mode_falls_back_to_floor_for_missing_asset(tmp_path):
    import csv
    path = tmp_path / "sp.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["asset", "spread_bps"])
        writer.writerow(["A", 100.0])
    dates = [f"2026-01-{d:02d}" for d in range(1, 4)]
    rows = []
    for d, ts in enumerate(dates, start=1):
        top = 4 if d % 2 else -4
        rows.append([ts, "A", 100, top, 1_000_000.0])
        rows.append([ts, "B", 100, -top, 100.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    tiers = ((0.0, 0.0005, 0.0005),)
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.0005, slippage_rate=0.0005, liquidity_column="quote_volume", liquidity_tiers=tiers, slippage_mode="spread", spread_csv=str(path)))
    assert result["pnl"].iloc[0].slippage_cost == pytest.approx(100_000 * 0.01 + 100_000 * 0.0005)
    with pytest.raises(ValueError, match="spread_csv"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1, liquidity_column="quote_volume", liquidity_tiers=tiers, slippage_mode="spread", spread_csv=str(tmp_path / "missing.csv")))


def test_turnover_flag_marks_over_threshold_segments():
    from crypto_checker.validation import _segment_metrics
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    pnl = pd.DataFrame({"timestamp": dates, "equity": 100_000 + np.arange(10) * 10.0, "return": [0.001] * 10, "turnover": [0.20] * 10})
    out = _segment_metrics(pnl, pd.DataFrame(), list(dates), 100_000.0)
    assert out["turnover_flag"] == "OVER"
    pnl2 = pnl.copy()
    pnl2["turnover"] = 0.01
    out2 = _segment_metrics(pnl2, pd.DataFrame(), list(dates), 100_000.0)
    assert out2["turnover_flag"] == "OK"


def test_invalid_slippage_mode_rejected():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1],
        ["2026-01-02", "A", 100, 2], ["2026-01-02", "B", 100, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1, slippage_mode="bogus"))


def test_classify_funding_gaps_not_listed_delisted_migration_active():
    from crypto_checker.assets import classify_funding_gaps
    data = pd.DataFrame([
        # A: NaN before first print -> not_listed; NaN after last print -> delisted.
        ["2024-01-01", "A", 100, 100.0, None],
        ["2024-01-02", "A", 100, 100.0, 0.01],
        ["2024-01-03", "A", 100, 100.0, None],
        # B: NaN between prints with price+volume -> active.
        ["2024-01-01", "B", 100, 100.0, 0.02],
        ["2024-01-02", "B", 100, 100.0, None],
        ["2024-01-03", "B", 100, 100.0, 0.02],
        # GUSDT inside GAL->G migration window (effective 2024-07-19) -> migration.
        ["2024-07-18", "GUSDT", 0.04, 100.0, 0.01],
        ["2024-07-20", "GUSDT", 0.04, 100.0, None],
        ["2024-07-25", "GUSDT", 0.04, 100.0, 0.01],
    ], columns=["timestamp", "asset", "price", "volume", "funding_rate"])
    gaps = classify_funding_gaps(data)
    by_asset = {row["asset"]: row["reason"] for _, row in gaps.iterrows()}
    assert by_asset["A"] in ("not_listed", "delisted")
    assert len(gaps[gaps.asset == "A"]) == 2
    assert set(gaps[gaps.asset == "A"]["reason"]) == {"not_listed", "delisted"}
    assert by_asset["B"] == "active"
    assert by_asset["GUSDT"] == "migration"


def test_classify_funding_gaps_empty_when_complete():
    from crypto_checker.assets import classify_funding_gaps
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 0.01], ["2024-01-02", "A", 100, 0.01],
    ], columns=["timestamp", "asset", "price", "funding_rate"])
    gaps = classify_funding_gaps(data)
    assert list(gaps.columns) == ["timestamp", "asset", "reason"]
    assert gaps.empty


def test_evaluate_deployable_uses_only_hard_gates():
    from crypto_checker.decision import HARD_GATES, evaluate_deployable
    assert set(HARD_GATES) == {"wf_positive", "oos_positive", "no_risk_violations", "no_capacity_violations", "reality_check_pass"}
    passing = {name: True for name in HARD_GATES}
    assert evaluate_deployable(passing) is True
    # Informational gates failing must not matter.
    assert evaluate_deployable({**passing, "wf_sharpe_above_one": False, "static_oos_sharpe_above_one": False}) is True
    for name in HARD_GATES:
        failing = {k: True for k in HARD_GATES}
        failing[name] = False
        assert evaluate_deployable(failing) is False
    # Missing keys fail closed.
    assert evaluate_deployable({}) is False


def test_infer_periods_per_year_buckets():
    from crypto_checker.core import infer_periods_per_year
    daily = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    assert infer_periods_per_year(daily) == 365.0
    hourly = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    assert infer_periods_per_year(hourly) == 8760.0
    four_h = pd.date_range("2026-01-01", periods=100, freq="4h", tz="UTC")
    assert infer_periods_per_year(four_h) == 2190.0
    assert infer_periods_per_year(daily[:1]) == 365.0


def test_hourly_sharpe_annualizes_with_8760():
    hours = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    rows = []
    for ts in hours:
        rows.append([ts, "A", 100.0, 1.0])
        rows.append([ts, "B", 100.0, -1.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0))
    assert result["metrics"]["periods_per_year"] == 8760.0
    rets = result["pnl"]["return"].iloc[1:].astype(float)
    expected = float(rets.mean() / rets.std(ddof=1) * (8760.0 ** 0.5)) if rets.std(ddof=1) else 0.0
    assert result["metrics"]["sharpe"] == pytest.approx(expected)


def test_drawdown_violation_tracks_running_peak():
    # Equity 100k -> 130k -> 100k: true peak-to-trough drawdown is
    # 100/130-1 = -23.1%, invisible to a fixed initial-equity peak.
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1],
        ["2026-01-02", "A", 130, 2], ["2026-01-02", "B", 100, 1],
        ["2026-01-03", "A", 100, 2], ["2026-01-03", "B", 100, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, max_drawdown_limit=0.20))
    kinds = result["violations"]["type"].tolist()
    assert "max_drawdown_limit" in kinds
    assert result["metrics"]["max_drawdown"] == pytest.approx(100 / 130 - 1)


def test_regime_labels_have_no_future_leakage():
    from crypto_checker.validation import regime_labels
    dates = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    prices = list(100 + np.arange(120) * 0.5)
    data = pd.DataFrame({"timestamp": list(dates) * 1, "asset": ["BTCUSDT"] * 120, "price": prices})
    full = regime_labels(data)
    prefix = regime_labels(data.iloc[:60])
    merged = full.iloc[:60].reset_index(drop=True)
    assert (merged["regime"] == prefix.reset_index(drop=True)["regime"]).all()


def test_benchmarks_include_honest_long_only_and_buy_hold():
    from crypto_checker.validation import benchmark_suite
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    rows = []
    for i, ts in enumerate(dates):
        rows.append([ts, "BTCUSDT", 100.0 + i])
        rows.append([ts, "ETHUSDT", 50.0 - i * 0.5])
        rows.append([ts, "XRPUSDT", 10.0 + (i % 2)])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price"])
    out = benchmark_suite(data)
    assert "btc_buy_hold" in out and "eth_buy_hold" in out
    assert out["btc_buy_hold"]["total_return"] == pytest.approx(109.0 / 100.0 - 1)
    # Long-only basket must equal the unclipped equal-weight basket.
    assert out["long_only_equal_weight"]["total_return"] == pytest.approx(out["equal_weight"]["total_return"])


def test_signal_executes_next_bar_with_documented_timing():
    # Signal observed at bar t (info <= close t) must first earn PnL over
    # t -> t+1, and execution_timestamp must be the following bar.
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1],
        ["2026-01-02", "A", 110, -2], ["2026-01-02", "B", 100, -1],
        ["2026-01-03", "A", 121, -2], ["2026-01-03", "B", 100, -1],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0))
    # Day-0 decision (long A) earns the day-0 -> day-1 move: +10% on weight 1.0.
    assert result["pnl"].iloc[1]["price_pnl"] == pytest.approx(10_000.0)
    pos = result["positions"]
    first = pos[pos.timestamp == pos.timestamp.min()].iloc[0]
    assert first["execution_timestamp"] > first["signal_timestamp"]


def _funding_gap_frame():
    # A: funding starts 2024-01-10, price rows from 2024-01-01 -> not_listed head.
    # B: funding ends 2024-01-10, price rows through 2024-01-20 -> delisted tail.
    # C: funding throughout except 2024-01-10 with price+volume -> active.
    # GUSDT: NaN on 2024-07-20 inside GAL->G migration window -> migration.
    rows = []
    for d in range(1, 21):
        ts = f"2024-01-{d:02d}"
        rows.append([ts, "A", 100 + d, 1000 + d, 0.01 if d >= 10 else float("nan")])
        rows.append([ts, "B", 200 - d, 2000 + d, 0.02 if d <= 10 else float("nan")])
        rows.append([ts, "C", 300 + d, 3000 + d, float("nan") if d == 10 else 0.03])
    for d, rate in [(18, 0.01), (20, float("nan")), (25, 0.01)]:
        rows.append([f"2024-07-{d:02d}", "GUSDT", 0.04, 5000.0, rate])
    return pd.DataFrame(rows, columns=["timestamp", "asset", "price", "volume", "funding_rate"])


def test_partition_missing_funding_splits_expected_vs_unexpected():
    from crypto_checker.binance_vision import partition_missing_funding
    expected, unexpected = partition_missing_funding(_funding_gap_frame())
    assert set(expected["reason"].unique()) <= {"not_listed", "delisted", "migration"}
    assert set(unexpected["reason"].unique()) == {"active"}
    # A head (9 rows), B tail (10 rows), GUSDT migration window (1 row).
    assert (expected["reason"] == "not_listed").sum() == 9
    assert (expected["reason"] == "delisted").sum() == 10
    assert (expected["reason"] == "migration").sum() == 1
    assert len(unexpected) == 1 and unexpected.iloc[0]["asset"] == "C"


def test_partition_missing_funding_empty_when_complete():
    from crypto_checker.binance_vision import partition_missing_funding
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 1000.0, 0.01], ["2024-01-02", "A", 101, 1100.0, 0.02],
    ], columns=["timestamp", "asset", "price", "volume", "funding_rate"])
    expected, unexpected = partition_missing_funding(data)
    assert expected.empty and unexpected.empty
    assert list(expected.columns) == ["timestamp", "asset", "reason"]


def test_expected_funding_manifest_written_next_to_reports(tmp_path):
    from crypto_checker.binance_vision import _write_expected_funding_manifest, partition_missing_funding
    expected, _ = partition_missing_funding(_funding_gap_frame())
    out = tmp_path / "data" / "midcap.csv"
    out.parent.mkdir(parents=True)
    path = _write_expected_funding_manifest(expected, str(out))
    # No reports/ sibling here -> falls back to the output directory.
    assert path == out.parent / "funding_expected_missing.csv"
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["dataset", "timestamp", "asset", "reason"]
    assert (saved["dataset"] == "midcap").all()
    assert len(saved) == len(expected)


def test_funding_frequency_report_aggregates_sub8h_and_flags_extremes():
    # Regression: sub-8h funding intervals (e.g. hourly prints during
    # volatility events) must be summed per UTC day, never rejected, and
    # the extreme daily print must be auditable from the manifest.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.fetch_funding import build_funding_frequency_report
    ts = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    rows = [{"timestamp": t, "asset": "X", "funding_rate": -0.004} for t in ts]
    rows += [
        {"timestamp": pd.Timestamp("2026-01-01 00:00", tz="UTC"), "asset": "Y", "funding_rate": 0.0001},
        {"timestamp": pd.Timestamp("2026-01-01 08:00", tz="UTC"), "asset": "Y", "funding_rate": 0.0001},
        {"timestamp": pd.Timestamp("2026-01-01 16:00", tz="UTC"), "asset": "Y", "funding_rate": 0.0001},
    ]
    funding = pd.DataFrame(rows)
    freq_report, daily = build_funding_frequency_report(funding)
    assert freq_report.loc["X", "max_settlements_per_day"] == 24
    assert freq_report.loc["Y", "max_settlements_per_day"] == 3
    assert daily[(daily.asset == "X") & (daily.date == "2026-01-01")]["funding_rate"].iloc[0] == pytest.approx(-0.096)
    assert freq_report.loc["X", "max_abs_daily_rate"] == pytest.approx(-0.096)
    assert str(freq_report.loc["X", "max_abs_daily_rate_date"]) == "2026-01-01 00:00:00+00:00"


def test_execution_uses_prior_bar_signal_only():
    # THE single rule: ranking at bar t must use raw signals from t-1.
    # A spikes raw to 10 at t2 only; B is flat 1. Decisions must flip at
    # t3 (observing the spike), never at t2 (same bar as the spike).
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 0], ["2026-01-01", "B", 100, 1],
        ["2026-01-02", "A", 100, 10], ["2026-01-02", "B", 100, 1],
        ["2026-01-03", "A", 100, 0], ["2026-01-03", "B", 100, 1],
        ["2026-01-04", "A", 100, 0], ["2026-01-04", "B", 100, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0))
    ranking = result["ranking"]
    assert ranking.iloc[0]["timestamp"] == pd.Timestamp("2026-01-02", tz="UTC")
    assert ranking.iloc[0]["long_assets"] == "B"
    assert ranking.iloc[1]["long_assets"] == "A"
    assert ranking.iloc[2]["long_assets"] == "B"


def test_execution_lag_with_no_bars_left_is_hard_error():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1]], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="No executable signals"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1))


def test_lifecycle_build_marks_migration_and_inferred_dates():
    from crypto_checker.lifecycle import build_lifecycle, active_assets
    data = pd.DataFrame([
        ["2024-07-10", "GALUSDT", 5.0], ["2024-07-11", "GALUSDT", 5.1], ["2024-07-12", "GALUSDT", 5.0],
        ["2024-07-20", "GUSDT", 0.08], ["2024-07-21", "GUSDT", 0.09], ["2024-07-22", "GUSDT", 0.08],
        ["2024-07-10", "BTCUSDT", 60000.0], ["2024-07-22", "BTCUSDT", 65000.0],
    ], columns=["timestamp", "asset", "price"])
    life = build_lifecycle(data)
    gal = life[(life["canonical"] == "GUSDT") & (life["symbol"] == "GALUSDT")].iloc[0]
    assert gal["event"] == "migration_source"
    assert str(gal["delisted_at"]) == "2024-07-19 08:00:00+00:00"
    assert "official" in gal["source"]
    g = life[(life["canonical"] == "GUSDT") & (life["symbol"] == "GUSDT")].iloc[0]
    assert g["event"] == "migration_target"
    assert str(g["listed_at"]) == "2024-07-19 08:00:00+00:00"
    btc = life[life["canonical"] == "BTCUSDT"].iloc[0]
    assert btc["event"] == "active" and pd.isna(btc["delisted_at"])
    assert "inferred_from_data" in btc["source"]
    assert active_assets(life, "2024-07-11") == ["BTCUSDT", "GUSDT"]
    assert active_assets(life, "2024-07-21") == ["BTCUSDT", "GUSDT"]


def test_lifecycle_compliance_flags_trading_outside_segment():
    import pandas as pd
    from crypto_checker.lifecycle import audit_universe_compliance
    official = pd.DataFrame([{
        "canonical": "GUSDT", "symbol": "GUSDT",
        "listed_at": pd.Timestamp("2024-07-19 08:00:00+00:00"),
        "delisted_at": pd.NaT, "event": "migration_target", "source": "official",
    }])
    data = pd.DataFrame([
        ["2024-07-11", "GUSDT", 0.08],
        ["2024-07-20", "GUSDT", 0.09],
    ], columns=["timestamp", "asset", "price"])
    violations = audit_universe_compliance(data, official)
    assert len(violations) == 1
    assert violations.iloc[0]["reason"] == "trading_before_listing"


def test_lifecycle_manifest_roundtrip_and_preflight_hook(tmp_path):
    from crypto_checker.lifecycle import build_lifecycle, write_lifecycle_manifest, load_lifecycle_manifest
    data = pd.DataFrame([
        ["2024-01-01", "AUSDT", 100.0, 2], ["2024-01-02", "AUSDT", 101.0, 2],
        ["2024-01-01", "BUSDT", 50.0, 1], ["2024-01-02", "BUSDT", 51.0, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    life = build_lifecycle(data)
    path = write_lifecycle_manifest(life, tmp_path / "lifecycle_manifest.csv")
    reloaded = load_lifecycle_manifest(path)
    assert list(reloaded.columns) == ["canonical", "symbol", "listed_at", "delisted_at", "event", "source"]
    assert len(reloaded) == len(life)
    check = validate_dataset(data)
    assert "lifecycle_manifest" in check and check["lifecycle_segments"] == len(life)
    assert any("inferred from data" in w for w in check["warnings"])


def test_deflated_sharpe_ratio_properties():
    from statistics import NormalDist
    from crypto_checker.reality_check import deflated_sharpe_ratio
    # No selection (1 trial): DSR == Probabilistic Sharpe Ratio, verifiable
    # inline without reusing the implementation.
    # Unsaturated regime (SR 0.3, T 120): selection over 200 trials must
    # visibly deflate the naive probability.
    out = deflated_sharpe_ratio(observed_sr=0.3, n_trials=1, skew=0.0, kurtosis=3.0, n_obs=120, trials_variance=0.04)
    assert out["expected_sharpe_null"] == 0.0
    # PSR with the kurtosis adjustment written out explicitly (Pearson
    # kurtosis 3 -> denominator sqrt(1 + SR^2/2)).
    expected_psr = NormalDist().cdf(0.3 * (119 ** 0.5) / ((1 + 0.3 ** 2 / 2) ** 0.5))
    assert out["dsr"] == pytest.approx(expected_psr)
    # More trials tried -> higher null bar -> lower DSR.
    wide = deflated_sharpe_ratio(observed_sr=0.3, n_trials=200, skew=0.0, kurtosis=3.0, n_obs=120, trials_variance=0.04)
    assert wide["expected_sharpe_null"] > 0.0
    assert wide["dsr"] < out["dsr"]
    # Below the null bar -> DSR under a coin flip.
    bad = deflated_sharpe_ratio(observed_sr=0.1, n_trials=200, skew=0.0, kurtosis=3.0, n_obs=500, trials_variance=0.25)
    assert bad["dsr"] < 0.5
    # Degenerate inputs fail closed, never NaN.
    empty = deflated_sharpe_ratio(observed_sr=1.0, n_trials=5, skew=0.0, kurtosis=3.0, n_obs=1, trials_variance=0.1)
    assert empty["dsr"] == 0.0


def test_capacity_curve_flags_aum_where_sensible_stops():
    from crypto_checker.capacity import capacity_curve
    rows = []
    for d in ("2026-01-01", "2026-01-02", "2026-01-03"):
        rows.append([d, "A", 100, 2, 10_000_000.0])
        rows.append([d, "B", 100, 1, 10_000_000.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    cfg = CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, liquidity_column="quote_volume", liquidity_tiers=((0.0, 0.0, 0.0),))
    report = capacity_curve(data, base_config=cfg, aum_levels=(10_000.0, 10_000_000.0))
    assert report["status"] == "ok"
    small, huge = report["levels"]
    assert small["sensible"] and small["capacity_violations"] == 0 and small["headroom_multiple"] > 1
    assert not huge["sensible"] and huge["capacity_violations"] > 0 and huge["breach_trades"] > 0 and huge["headroom_multiple"] < 1
    assert small["total_return"] == pytest.approx(huge["total_return"])


def test_capacity_refuses_without_liquidity_column():
    from crypto_checker.capacity import capacity_curve
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1],
        ["2026-01-02", "A", 100, 2], ["2026-01-02", "B", 100, 1],
    ], columns=["timestamp", "asset", "price", "signal"])
    report = capacity_curve(data, base_config=CheckerConfig(n_long=1, n_short=1))
    assert report["status"] == "no_liquidity_data"


def test_walk_forward_reports_dsr_and_ensemble():
    rng = np.random.default_rng(21)
    dates = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    frames = []
    for i in range(4):
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates))))
        frames.append(pd.DataFrame({"timestamp": dates, "asset": f"A{i}", "price": prices, "funding_rate": rng.normal(0, 0.0002, len(dates))}))
    data = pd.concat(frames, ignore_index=True)
    result = walk_forward(
        data,
        base_config=CheckerConfig(n_long=1, n_short=1, min_signal_gap=0),
        min_train_days=100, test_days=40,
        candidates=["momentum_7", "reversal_1"], n_sides_grid=(1,),
        ensemble_top_k=2,
    )
    dm = result["data_mining"]
    assert dm["n_trials"] > 0
    assert 0.0 <= dm["dsr"] <= 1.0
    assert dm["expected_sharpe_null"] >= 0.0
    ens = result["ensemble"]
    assert ens["top_k"] == 2 and len(ens["members_per_fold"]) == result["n_folds"]
    assert all(len(m["members"]) >= 1 for m in ens["members_per_fold"])
    assert "diagnostic_only" in ens["note"]


def test_timestamp_gap_triage_migration_halt_vs_unexplained():
    from crypto_checker.lifecycle import classify_timestamp_gaps
    # GAL halt block straddles the official 2024-07-19 effective date;
    # the second block is an ordinary hole with no event nearby.
    gal_dates = pd.date_range("2024-07-01", "2024-07-11", freq="D", tz="UTC").tolist()
    gal_dates += pd.date_range("2024-08-15", "2024-08-20", freq="D", tz="UTC").tolist()
    rows = [[str(d), "GALUSDT" if d < pd.Timestamp("2024-07-19", tz="UTC") else "GUSDT", 1.0] for d in gal_dates]
    rows += [[str(d), "BTCUSDT", 60000.0] for d in pd.date_range("2024-07-01", "2024-08-20", freq="D", tz="UTC") if d not in (pd.Timestamp("2024-07-20", tz="UTC"),)]
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price"])
    table = classify_timestamp_gaps(data)
    g_block = table[table["asset"] == "GUSDT"].iloc[0]
    assert g_block["reason"] == "migration_halt" and g_block["gap_days"] == 34
    btc_block = table[table["asset"] == "BTCUSDT"].iloc[0]
    assert btc_block["reason"] == "unexplained_interior" and btc_block["gap_days"] == 1


def test_preflight_strict_passes_documented_halt_fails_unexplained():
    halt_only = pd.DataFrame(
        # GAL 6.0 -> G 0.1 respects the official 1:60 factor (no discontinuity).
        [[str(d), "GALUSDT" if d < pd.Timestamp("2024-07-19", tz="UTC") else "GUSDT", 6.0 if d < pd.Timestamp("2024-07-19", tz="UTC") else 0.1, 0.0]
         for d in list(pd.date_range("2024-07-01", "2024-07-11", freq="D", tz="UTC")) + list(pd.date_range("2024-08-15", "2024-08-20", freq="D", tz="UTC"))]
        + [[str(d), "BTCUSDT", 60000.0, 0.0] for d in pd.date_range("2024-07-01", "2024-08-20", freq="D", tz="UTC")],
        columns=["timestamp", "asset", "price", "signal"],
    )
    strict = validate_dataset(halt_only, min_assets=2, min_periods=2, expected_frequency="D", allow_gaps=False)
    assert strict["valid"], strict["errors"]
    assert strict["gap_classification"]["unexplained_interior_days"] == 0
    assert any("Documented exchange halt" in w for w in strict["warnings"])
    hole = halt_only.copy()
    hole = hole[~((hole["asset"] == "BTCUSDT") & (hole["timestamp"] == "2024-07-20 00:00:00+00:00") )]
    strict_hole = validate_dataset(hole, min_assets=2, min_periods=2, expected_frequency="D", allow_gaps=False)
    assert not strict_hole["valid"]
    assert any("Unexplained interior gaps" in e for e in strict_hole["errors"])
    tolerant = validate_dataset(hole, min_assets=2, min_periods=2, expected_frequency="D", allow_gaps=True)
    assert tolerant["valid"]
    assert any("NOT valid for futures" in w for w in tolerant["warnings"])


def test_stress_repeated_forced_exits_across_chaos():
    # Behavior under stress, not one observation: B/C/D vanish on three
    # consecutive bars while each is HELD. All three must be force-exited
    # at last price, the run must complete, PnL must stay finite.
    bars = {
        "2026-01-01": {"A": 5, "B": 10, "C": 4, "D": 3, "E": 0},
        "2026-01-02": {"A": 5, "B": 1, "C": 10, "D": 4, "E": 0},
        "2026-01-03": {"A": 5, "C": 1, "D": 10, "E": 0},
        "2026-01-04": {"A": 10, "D": 1, "E": 0},
        "2026-01-05": {"A": 10, "E": 0},
        "2026-01-06": {"A": 10, "E": 0},
    }
    rows = [[ts, asset, 100, sig] for ts, sigs in bars.items() for asset, sig in sigs.items()]
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, delist_mode="forced_exit"))
    exits = result["violations"][result["violations"]["type"] == "forced_exit"]
    assert sorted(exits["asset"].tolist()) == ["B", "C", "D"]
    assert (exits["last_price"] == 100).all()
    for asset in ("B", "C", "D"):
        leg = result["trades"][result["trades"]["asset"] == asset]
        assert (leg["weight_change"] < 0).any()
    assert np.isfinite(result["pnl"]["total_pnl"]).all()
    assert result["metrics"]["periods"] == 4


def test_stress_crash_then_delist_exits_at_crashed_price():
    # 100 -> 80 (x0.8) -> 56 (x0.7) -> 47.6 (x0.85), then gone. The exit
    # must print the crashed 47.6, never a stale 100, and the crash must
    # be realized in drawdown (no silent carry).
    data = pd.DataFrame([
        ["2026-01-01", "A", 100.0, 2], ["2026-01-01", "B", 100.0, 1], ["2026-01-01", "C", 100.0, 0],
        ["2026-01-02", "A", 80.0, 2], ["2026-01-02", "B", 100.0, 1], ["2026-01-02", "C", 100.0, 0],
        ["2026-01-03", "A", 56.0, 2], ["2026-01-03", "B", 100.0, 1], ["2026-01-03", "C", 100.0, 0],
        ["2026-01-04", "A", 47.6, 2], ["2026-01-04", "B", 100.0, 1], ["2026-01-04", "C", 100.0, 0],
        ["2026-01-05", "B", 100.0, 1], ["2026-01-05", "C", 100.0, 0],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, delist_mode="forced_exit"))
    exits = result["violations"][result["violations"]["type"] == "forced_exit"]
    assert len(exits) == 1 and exits.iloc[0]["asset"] == "A"
    assert exits.iloc[0]["last_price"] == pytest.approx(47.6)
    assert result["metrics"]["max_drawdown"] < -0.30


def test_stress_volume_collapse_triggers_capacity_breach():
    # Volume x0.1 overnight with a simultaneous flip: rebalancing INTO the
    # collapsed book must breach, while the same trade in the deep book
    # (t2) stays clean. Breaches must start exactly at the collapse.
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2, 10_000_000.0], ["2026-01-01", "B", 100, 1, 10_000_000.0],
        ["2026-01-02", "A", 100, 1, 10_000_000.0], ["2026-01-02", "B", 100, 2, 10_000_000.0],
        ["2026-01-03", "A", 100, 1, 1_000_000.0], ["2026-01-03", "B", 100, 2, 1_000_000.0],
        ["2026-01-04", "A", 100, 1, 1_000_000.0], ["2026-01-04", "B", 100, 2, 1_000_000.0],
    ], columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    cfg = CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, liquidity_column="quote_volume", liquidity_tiers=((0.0, 0.0, 0.0),))
    result = check_strategy(data, cfg)
    breaches = result["violations"][result["violations"]["type"] == "capacity_limit"]
    assert len(breaches) > 0
    assert (pd.to_datetime(breaches["timestamp"], utc=True) >= pd.Timestamp("2026-01-03", tz="UTC")).all()


def test_stress_spread_widening_scales_crisis_costs(tmp_path):
    # Spread panic (100 bps) vs normal (2 bps): crisis slippage must be
    # exactly 50x on identical flow. Proves the spread channel transmits
    # stress instead of absorbing it.
    import csv
    calm, panic = tmp_path / "calm.csv", tmp_path / "panic.csv"
    for path, bps in ((calm, 2.0), (panic, 100.0)):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["asset", "spread_bps"])
            writer.writerow(["A", bps])
            writer.writerow(["B", bps])
    rows = []
    for d in ("2026-01-01", "2026-01-02", "2026-01-03"):
        rows.append([d, "A", 100, 2, 1_000_000.0])
        rows.append([d, "B", 100, 1, 1_000_000.0])
    data = pd.DataFrame(rows, columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    tiers = ((0.0, 0.0, 0.0),)
    calm_res = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, liquidity_column="quote_volume", liquidity_tiers=tiers, slippage_mode="spread", spread_csv=str(calm)))
    panic_res = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0, slippage_rate=0, liquidity_column="quote_volume", liquidity_tiers=tiers, slippage_mode="spread", spread_csv=str(panic)))
    assert panic_res["pnl"].iloc[0].slippage_cost == pytest.approx(50 * calm_res["pnl"].iloc[0].slippage_cost)
    assert panic_res["metrics"]["total_slippage"] > calm_res["metrics"]["total_slippage"]


def test_listing_manifest_adapter_and_survivorship_gap():
    from crypto_checker.lifecycle import lifecycle_from_listing_manifest, measure_survivorship_gap, active_assets
    manifest = pd.DataFrame([
        {"symbol": "GALUSDT", "listed_at": "2022-05-05", "delisted_at": "2024-07-30", "status": "NOT_TRADING", "source": "inferred_vision_first_seen"},
        {"symbol": "GUSDT", "listed_at": "2024-08-15", "delisted_at": None, "status": "TRADING", "source": "inferred_vision_first_seen"},
        {"symbol": "LUNAUSDT", "listed_at": "2021-01-28", "delisted_at": "2022-05-13", "status": "NOT_TRADING", "source": "inferred_vision_first_seen"},
        {"symbol": "BTCUSDT", "listed_at": "2019-12-31", "delisted_at": None, "status": "TRADING", "source": "inferred_vision_first_seen"},
    ])
    life = lifecycle_from_listing_manifest(manifest)
    # GAL+G collapse to one canonical with two segments; LUNA is dead.
    assert sorted(life["canonical"].unique().tolist()) == ["BTCUSDT", "GUSDT", "LUNAUSDT"]
    assert active_assets(life, "2024-07-11") == ["BTCUSDT", "GUSDT"]
    assert active_assets(life, "2024-08-20") == ["BTCUSDT", "GUSDT"]
    assert active_assets(life, "2021-06-01") == ["BTCUSDT", "LUNAUSDT"]
    # Dataset holds only BTC+G in 2024: LUNA died before the window (not
    # bias), but a dead-in-window coin would be flagged.
    data = pd.DataFrame(
        [[str(d), "BTCUSDT", 60000.0] for d in pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")]
        + [[str(d), "GUSDT", 0.08] for d in pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")],
        columns=["timestamp", "asset", "price"],
    )
    gap = measure_survivorship_gap(data, manifest)
    assert gap["n_missing_dead"] == 0  # LUNA died 2022, outside the 2024 window
    manifest2 = pd.concat([manifest, pd.DataFrame([{"symbol": "FTTUSDT", "listed_at": "2022-04-15", "delisted_at": "2024-06-01", "status": "NOT_TRADING", "source": "inferred_vision_first_seen"}])], ignore_index=True)
    gap2 = measure_survivorship_gap(data, manifest2)
    assert gap2["n_missing_dead"] == 1 and gap2["missing_dead"] == ["FTTUSDT"]
    assert 0.0 < gap2["coverage_ratio"] < 1.0


def test_preflight_survivorship_warning_with_manifest():
    manifest = pd.DataFrame([
        {"symbol": "BTCUSDT", "listed_at": "2019-12-31", "delisted_at": None, "status": "TRADING", "source": "t"},
        {"symbol": "FTTUSDT", "listed_at": "2022-04-15", "delisted_at": "2024-06-01", "status": "NOT_TRADING", "source": "t"},
    ])
    data = pd.DataFrame(
        [[str(d), "BTCUSDT", 60000.0, 0.0] for d in pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")],
        columns=["timestamp", "asset", "price", "signal"],
    )
    check = validate_dataset(data, min_assets=1, min_periods=2, listing_manifest=manifest)
    assert check["valid"]
    assert check["survivorship_gap"]["missing_dead"] == ["FTTUSDT"]
    assert any("SURVIVORSHIP_GAP" in w for w in check["warnings"])
