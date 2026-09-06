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
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 4], ["2026-01-01", "B", 100, 3], ["2026-01-01", "C", 100, 2], ["2026-01-01", "D", 100, 1],
        ["2026-01-02", "A", 110, 1], ["2026-01-02", "B", 100, 2], ["2026-01-02", "C", 100, 3], ["2026-01-02", "D", 90, 4],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.001, slippage_rate=0))
    assert result["ranking"].iloc[0].long_assets == "A"
    assert result["exposure"].iloc[0].gross_exposure == 2
    assert result["pnl"].iloc[1].total_pnl == pytest.approx(19560.8)
    assert result["metrics"]["periods"] == 1
    assert result["trades"].notna().all().all()


def test_rejects_invalid_values():
    data = pd.DataFrame([["2026-01-01", "A", 0, 1], ["2026-01-01", "B", 1, 0]], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="Price"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1))


def test_funding_long_is_paid_and_short_received():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2, 0.01], ["2026-01-01", "B", 100, 1, 0.01], ["2026-01-02", "A", 100, 2, 0.01], ["2026-01-02", "B", 100, 1, 0.01]], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1))
    assert result["pnl"].iloc[1].funding_pnl == 0


def test_rebalance_charges_turnover_and_costs():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 4], ["2026-01-01", "B", 100, 3], ["2026-01-01", "C", 100, 2], ["2026-01-01", "D", 100, 1],
        ["2026-01-02", "A", 100, 1], ["2026-01-02", "B", 100, 2], ["2026-01-02", "C", 100, 3], ["2026-01-02", "D", 100, 4],
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


def test_rejects_nan_funding():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2, 0.0], ["2026-01-01", "B", 100, 1, 0.0], ["2026-01-02", "A", 100, 2, float("nan")], ["2026-01-02", "B", 100, 1, 0.0]], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    with pytest.raises(ValueError, match="Funding rate"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1))


def test_require_funding_rejects_price_only_input():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1]], columns=["timestamp", "asset", "price", "signal"])
    with pytest.raises(ValueError, match="Funding rate column required"):
        check_strategy(data, CheckerConfig(n_long=1, n_short=1, require_funding=True))


def test_positions_expose_next_execution_timestamp():
    data = pd.DataFrame([["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1], ["2026-01-02", "A", 100, 2], ["2026-01-02", "B", 100, 1]], columns=["timestamp", "asset", "price", "signal"])
    positions = check_strategy(data, CheckerConfig(n_long=1, n_short=1))["positions"]
    assert positions.iloc[0].execution_timestamp > positions.iloc[0].signal_timestamp


def test_max_drawdown_measured_from_initial_equity():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 2], ["2026-01-01", "B", 100, 1],
        ["2026-01-02", "A", 80, 2], ["2026-01-02", "B", 100, 1],
        ["2026-01-03", "A", 85, 2], ["2026-01-03", "B", 100, 1],
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


def test_carry_signal_is_negative_funding():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 0, 0.01], ["2026-01-02", "A", 100, 0, -0.02],
    ], columns=["timestamp", "asset", "price", "signal", "funding_rate"])
    built = build_signals(data)
    assert built["carry"].tolist() == pytest.approx([-0.01, 0.02])


def test_ranking_uses_configured_signal_column():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 4, 1], ["2026-01-01", "B", 100, 3, 4],
        ["2026-01-01", "C", 100, 2, 3], ["2026-01-01", "D", 100, 1, 2],
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
    assert a["funding_surprise"].iloc[15] == pytest.approx(-(0.03 - 0.01))


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
    assert any("Funding rate" in error for error in result["errors"])


def test_capacity_violation_is_reported():
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 2, 100], ["2024-01-01", "B", 100, 1, 100],
        ["2024-01-02", "A", 100, 1, 100], ["2024-01-02", "B", 100, 2, 100],
    ], columns=["timestamp", "asset", "price", "signal", "quote_volume"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, liquidity_column="quote_volume", liquidity_tiers=((0.0, 0.001, 0.001),), max_volume_participation=0.05))
    assert result["metrics"]["capacity_violations"] > 0


def test_missing_held_price_is_hard_error():
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 2], ["2024-01-01", "B", 100, 1],
        ["2024-01-02", "B", 100, 2],
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
    data = pd.DataFrame([
        ["2024-01-01", "A", 100, 2], ["2024-01-01", "B", 100, 1],
        ["2024-01-02", "B", 100, 2], ["2024-01-02", "C", 100, 1],
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
