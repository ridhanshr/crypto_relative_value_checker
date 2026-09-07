from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .assets import canonicalize_assets, audit_migration_collisions, classify_funding_gaps
from .io import atomic_write_json


@dataclass(frozen=True)
class CheckerConfig:
    n_long: int = 2
    n_short: int = 2
    gross_exposure: float = 2.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0005
    initial_equity: float = 100_000.0
    funding_positive_paid_by_long: bool = True
    signal_lookback: int = 1
    min_signal_gap: float = 0.0
    rebalance_every: int = 1
    max_drawdown_limit: float = 0.25
    daily_loss_limit: float = 0.05
    enforce_risk_limits: bool = False
    signal_column: str = "signal"
    vol_target_annual: float = 0.0
    vol_lookback: int = 30
    vol_warmup_scale: float = 0.5
    liquidity_column: str = ""
    liquidity_lookback: int = 30
    liquidity_tiers: tuple = ()
    require_funding: bool = False
    reject_migration_collisions: bool = True
    delist_mode: str = "error"
    max_volume_participation: float = 0.05
    cost_multiplier: float = 1.0
    slippage_mode: str = "tier"
    spread_csv: str = ""
    periods_per_year: float | None = None
    # --- v2 DSR ---
    dsr_reject_threshold: float = 0.5
    dsr_high_confidence_threshold: float = 0.95
    dsr_use_skew_kurtosis_adjustment: bool = True
    dsr_count_all_trials: bool = True
    # --- v2 CPCV / walk-forward ---
    cpcv_n_groups: int = 10
    cpcv_n_test_groups: int = 2
    cpcv_embargo_pct: float = 0.01
    cpcv_purge_pct: float = 0.01
    regime_label_source: str = "realized_vol_percentile"
    cpcv_mode: str = "search"
    cpcv_full_budget_only_on_best_config: bool = True
    # --- v2 ensemble ---
    ensemble_max_pairwise_corr: float = 0.7
    ensemble_corr_metric: str = "returns"
    # --- v2 capacity ---
    capacity_impact_model: str = "sqrt"
    capacity_adv_lookback_days: int = 90
    capacity_adv_exclude_outlier_days: bool = True
    capacity_min_headroom: float = 1.5
    # --- v2 drawdown gate ---
    dd_max_threshold_wf: float = -0.25
    dd_max_threshold_oos: float = -0.20
    gate_auto_reject_on_dd: bool = True
    # --- v2 data quality (module Fase 6; fields reserved for gate wiring) ---
    data_quality_dead_unit: str = "asset_days"
    data_quality_max_unexplained_halts: int = 0
    # --- v2 gate ---
    gate_auto_reject_on_oos_sharpe_negative: bool = True
    gate_flag_review_on_ensemble_corr: bool = True


def _validate(data, signal_column):
    required = {"timestamp", "asset", "price", signal_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    if data.duplicated(["timestamp", "asset"]).any():
        raise ValueError("Duplicate (timestamp, asset) rows")
    data["price"] = pd.to_numeric(data["price"], errors="raise")
    data[signal_column] = pd.to_numeric(data[signal_column], errors="raise")
    if (data["price"] <= 0).any() or not np.isfinite(data["price"]).all():
        raise ValueError("Price must be positive and non-null")
    if data[signal_column].isna().any() or not np.isfinite(data[signal_column]).all():
        raise ValueError(f"Signal column {signal_column} must be numeric and non-null")
    if "funding_rate" in data:
        # Funding is a post-ranking cost model: NaN is tolerated here and
        # classified downstream (not-listed/delisted/migration vs active).
        # Only non-numeric strings and infinite values are rejected.
        data["funding_rate"] = pd.to_numeric(data["funding_rate"], errors="raise")
        if not np.isfinite(data["funding_rate"].dropna()).all():
            raise ValueError("Funding rate must be finite where present")
    return data.sort_values(["timestamp", "asset"]).reset_index(drop=True)


_SPREAD_CACHE = {}


def _load_spread_map(spread_csv):
    key = str(Path(spread_csv).resolve())
    try:
        mtime = Path(spread_csv).stat().st_mtime
    except OSError as exc:
        raise ValueError(f"spread_csv not readable: {spread_csv}") from exc
    cached = _SPREAD_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    frame = pd.read_csv(spread_csv)
    missing = {"asset", "spread_bps"} - set(frame.columns)
    if missing:
        raise ValueError(f"Spread CSV missing columns: {sorted(missing)}")
    spread = pd.to_numeric(frame["spread_bps"], errors="coerce")
    if spread.isna().any() or (spread < 0).any():
        raise ValueError("Spread CSV has non-numeric or negative spread_bps")
    spread_map = dict(zip(frame["asset"].astype(str), (spread / 10000.0).tolist()))
    _SPREAD_CACHE[key] = (mtime, spread_map)
    return spread_map


def infer_periods_per_year(timestamps) -> float:
    """Periods per calendar year inferred from timestamp spacing.

    Daily bars -> 365.0 (exact, preserves legacy behavior), 4h -> 2190.0,
    1h -> 8760.0; anything else is computed continuously. Used for all
    annualization (Sharpe, volatility, annualized return, vol targeting).
    """
    ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(ts) < 2:
        return 365.0
    median_secs = float(ts.diff().dt.total_seconds().median())
    if median_secs <= 0:
        return 365.0
    for bucket_secs, ppy in ((86400.0, 365.0), (14400.0, 2190.0), (3600.0, 8760.0)):
        if abs(median_secs - bucket_secs) / bucket_secs <= 0.01:
            return ppy
    return 365.25 * 86400.0 / median_secs


def _assign_liquidity_tier_rates(data, cfg):
    liq = pd.to_numeric(data[cfg.liquidity_column], errors="coerce")
    data["_liq_score"] = liq.groupby(data["asset"]).transform(lambda s: s.rolling(cfg.liquidity_lookback, min_periods=1).mean().shift(1))
    pct = data.groupby("timestamp")["_liq_score"].rank(pct=True)
    conds = [pct >= t[0] for t in cfg.liquidity_tiers]
    most_illiquid = cfg.liquidity_tiers[-1]
    data["_fee_rate"] = np.select(conds, [t[1] for t in cfg.liquidity_tiers], default=most_illiquid[1])
    data["_slip_rate"] = np.select(conds, [t[2] for t in cfg.liquidity_tiers], default=most_illiquid[2])
    if cfg.slippage_mode == "spread":
        if not cfg.spread_csv:
            raise ValueError("slippage_mode='spread' requires spread_csv")
        spread_map = _load_spread_map(cfg.spread_csv)
        quoted = data["asset"].astype(str).map(spread_map)
        data["_slip_rate"] = np.maximum(data["_slip_rate"].to_numpy(), quoted.fillna(data["_slip_rate"]).to_numpy())
    return data


def _shift_signal_for_execution(data, signal_column):
    """Enforce the single execution rule: signal(t) executes at t+1.

    The ranked signal is shifted one bar within each asset BEFORE smoothing,
    validation, or ranking, so the value ranked at bar t contains only
    information available at/before close(t-1) and is filled at close(t)
    prices. This applies uniformly to built-in factors (which already
    self-lag and therefore pay one extra bar -- the price of a single
    uniform rule), custom signal columns, and the default contemporaneous
    signal. No signal, however constructed, can ever trade on same-bar
    information. The first bar per asset has no executable signal and is
    dropped.
    """
    frame = data.copy()
    if signal_column not in frame.columns:
        # Let _validate raise the proper "Missing columns" error.
        return frame
    frame[signal_column] = frame.groupby("asset")[signal_column].transform(lambda s: s.shift(1))
    return frame.dropna(subset=[signal_column]).reset_index(drop=True)


def check_strategy(data, config=None):
    """Run the cross-sectional long-short backtest.

    Timing convention -- THE single execution rule (timestamps are candle
    OPEN times; ``price`` is the candle close):
      * signal(t) executes at t+1, enforced by shifting the ranked signal
        one bar within each asset before anything else. The value ranked
        at bar t holds only information available at/before close(t-1).
      * Positions decided at t are filled at close(t) prices and first earn
        PnL over the t -> t+1 move. No PnL is ever accrued in the same bar
        whose information produced the signal.
      * ``execution_timestamp`` in position rows is therefore always the
        timestamp following ``signal_timestamp``.
    """
    cfg = config or CheckerConfig()
    if cfg.n_long < 1 or cfg.n_short < 1 or cfg.initial_equity <= 0 or cfg.gross_exposure <= 0 or cfg.fee_rate < 0 or cfg.slippage_rate < 0 or cfg.signal_lookback < 1 or cfg.min_signal_gap < 0 or cfg.rebalance_every < 1 or not 0 < cfg.max_drawdown_limit < 1 or not 0 < cfg.daily_loss_limit < 1 or cfg.vol_target_annual < 0 or cfg.vol_lookback < 2 or not 0 < cfg.vol_warmup_scale <= 1 or cfg.liquidity_lookback < 1 or not 0 < cfg.max_volume_participation <= 1 or cfg.cost_multiplier < 0 or cfg.delist_mode not in ("error", "forced_exit") or cfg.slippage_mode not in ("tier", "spread") or (cfg.periods_per_year is not None and cfg.periods_per_year <= 0):
        raise ValueError("Invalid checker configuration")
    if cfg.liquidity_tiers:
        thresholds = [t[0] for t in cfg.liquidity_tiers]
        if any(not 0 <= th < 1 for th in thresholds) or any(t[1] < 0 or t[2] < 0 for t in cfg.liquidity_tiers) or thresholds != sorted(thresholds, reverse=True) or len(set(thresholds)) != len(thresholds):
            raise ValueError("Invalid liquidity tiers; thresholds must be unique, within [0,1), and strictly descending")
    if cfg.reject_migration_collisions and not audit_migration_collisions(data).empty:
        raise ValueError("Migration source collision detected; resolve effective-date overlap before backtest")
    # Funding is a post-ranking cost model: missing funding never removes an
    # asset from ranking. Classify gaps on raw (pre-aggregation) data so
    # duplicate collapsing cannot mask them, fill with 0.0 for PnL, and keep
    # the classification for audit. Only "active" gaps need investigation.
    funding_gaps = classify_funding_gaps(data)
    data = canonicalize_assets(data)
    if cfg.require_funding and "funding_rate" not in data.columns:
        raise ValueError("Funding rate column required for this strategy")
    if "funding_rate" in data.columns:
        data["funding_rate"] = data["funding_rate"].fillna(0.0)
    data = _shift_signal_for_execution(data, cfg.signal_column)
    if data.empty:
        raise ValueError("No executable signals: the t+1 execution lag consumed all bars; provide at least 2 bars per asset")
    data = _validate(data, cfg.signal_column)
    if cfg.signal_lookback > 1:
        data[cfg.signal_column] = data.groupby("asset")[cfg.signal_column].transform(lambda x: x.rolling(cfg.signal_lookback, min_periods=cfg.signal_lookback).mean())
        data = data.dropna(subset=[cfg.signal_column]).reset_index(drop=True)
    tiered_costs = bool(cfg.liquidity_tiers and cfg.liquidity_column and cfg.liquidity_column in data.columns)
    if tiered_costs:
        data = _assign_liquidity_tier_rates(data, cfg)
    timestamps = list(data["timestamp"].drop_duplicates())
    ppy = cfg.periods_per_year if cfg.periods_per_year else infer_periods_per_year(timestamps)
    positions = {}
    ranking_rows, position_rows, trade_rows, pnl_rows, violation_rows = [], [], [], [], []
    equity = cfg.initial_equity
    peak_equity = cfg.initial_equity
    trailing_returns = []
    prev_prices, prev_positions = {}, {}
    for period, ts in enumerate(timestamps):
        next_ts = timestamps[period + 1] if period + 1 < len(timestamps) else pd.NaT
        vol_scale = 1.0
        if cfg.vol_target_annual > 0:
            if len(trailing_returns) >= cfg.vol_lookback:
                realized = float(np.std(trailing_returns[-cfg.vol_lookback:], ddof=1))
                target_daily = cfg.vol_target_annual / (ppy ** 0.5)
                if realized > 0:
                    vol_scale = min(1.0, target_daily / realized)
            else:
                vol_scale = cfg.vol_warmup_scale
        cross = data[data.timestamp == ts].copy()
        cross = cross.dropna(subset=[cfg.signal_column]).sort_values([cfg.signal_column, "asset"], ascending=[False, True])
        if tiered_costs:
            volumes_all = pd.to_numeric(cross[cfg.liquidity_column], errors="coerce")
            invalid_mask = ~np.isfinite(volumes_all) | (volumes_all <= 0)
            for asset in sorted(cross.loc[invalid_mask, "asset"].tolist()):
                violation_rows.append({"timestamp": ts, "type": "invalid_liquidity", "asset": asset, "last_price": float(cross.loc[cross.asset == asset, "price"].iloc[0]), "value": 0.0, "limit": 0.0})
            if invalid_mask.any():
                cross = cross[~invalid_mask]
        if prev_positions:
            missing_held = sorted(set(prev_positions) - set(cross.asset))
            if missing_held and cfg.delist_mode == "error":
                raise ValueError(f"Missing price for held positions at {ts}: {missing_held}")
            if missing_held and cfg.delist_mode == "forced_exit":
                for asset in missing_held:
                    violation_rows.append({"timestamp": ts, "type": "forced_exit", "asset": asset, "last_price": prev_prices.get(asset), "value": 0.0, "limit": 0.0})
        if len(cross) < cfg.n_long + cfg.n_short:
            raise ValueError(f"Not enough assets at {ts}")
        longs = cross.head(cfg.n_long)
        shorts = cross.tail(cfg.n_short)
        carry = {a: w for a, w in prev_positions.items() if a in set(cross.asset)} if prev_positions else {}
        if period % cfg.rebalance_every and carry:
            positions = carry.copy()
        elif len(longs) and len(shorts) and float(longs[cfg.signal_column].iloc[-1] - shorts[cfg.signal_column].iloc[0]) < cfg.min_signal_gap and carry:
            positions = carry.copy()
        else:
            w = cfg.gross_exposure / 2 * vol_scale
            # Sorted construction: dict insertion order feeds every downstream
            # sum/loop, so unordered construction would make results depend on
            # per-process hash randomization (PYTHONHASHSEED) at ULP level.
            positions = {a: w / cfg.n_long for a in sorted(set(longs.asset))}
            positions.update({a: -w / cfg.n_short for a in sorted(set(shorts.asset))})
        # Same reason: one canonical order for every cross-position aggregate.
        traded = sorted(set(positions) | set(prev_positions))
        long_assets = {a for a, v in positions.items() if v > 0}
        short_assets = {a for a, v in positions.items() if v < 0}
        overlap = long_assets & short_assets
        if overlap:
            raise AssertionError(f"Long/short ranking overlap: {sorted(overlap)}")
        ranking_rows.append({"timestamp": ts, "long_assets": ",".join(sorted(long_assets)), "short_assets": ",".join(sorted(short_assets)), "long_count": len(long_assets), "short_count": len(short_assets), "rank_overlap": len(overlap)})
        current_prices = dict(zip(cross.asset, cross.price))
        for asset, weight in positions.items():
            position_rows.append({"signal_timestamp": ts, "execution_timestamp": next_ts, "timestamp": ts, "asset": asset, "weight": weight, "price": current_prices[asset]})
        turnover_notional = sum(abs(positions.get(a, 0) - prev_positions.get(a, 0)) for a in traded) * equity
        if cfg.liquidity_tiers and cfg.liquidity_column and cfg.liquidity_column in cross:
            volumes = dict(zip(cross.asset, cross[cfg.liquidity_column]))
            for asset in traded:
                change_notional = abs(positions.get(asset, 0) - prev_positions.get(asset, 0)) * equity
                if asset in volumes and (not np.isfinite(volumes[asset]) or volumes[asset] <= 0):
                    raise ValueError(f"Invalid liquidity value for {asset} at {ts}")
                if asset in volumes and change_notional > float(volumes[asset]) * cfg.max_volume_participation:
                    violation_rows.append({"timestamp": ts, "type": "capacity_limit", "asset": asset, "notional": change_notional, "max_notional": float(volumes[asset]) * cfg.max_volume_participation})
        if prev_positions:
            price_pnl = sum(prev_positions.get(a, 0) * equity * (current_prices[a] / prev_prices[a] - 1) for a in prev_positions if a in current_prices and a in prev_prices)
            funding_rates = dict(zip(cross.asset, cross["funding_rate"] if "funding_rate" in cross else [0.0] * len(cross)))
            sign = -1 if cfg.funding_positive_paid_by_long else 1
            funding = sum(sign * prev_positions.get(a, 0) * equity * funding_rates.get(a, 0) for a in prev_positions)
        else:
            price_pnl = funding = 0.0
        if tiered_costs:
            fee_rates = dict(zip(cross.asset, cross["_fee_rate"]))
            slip_rates = dict(zip(cross.asset, cross["_slip_rate"]))
            fee = sum(abs(positions.get(a, 0) - prev_positions.get(a, 0)) * equity * fee_rates.get(a, cfg.fee_rate) for a in traded) * cfg.cost_multiplier
            slippage = sum(abs(positions.get(a, 0) - prev_positions.get(a, 0)) * equity * slip_rates.get(a, cfg.slippage_rate) for a in traded) * cfg.cost_multiplier
        else:
            fee = turnover_notional * cfg.fee_rate * cfg.cost_multiplier
            slippage = turnover_notional * cfg.slippage_rate * cfg.cost_multiplier
        total = price_pnl + funding - fee - slippage
        equity += total
        peak_equity = max(peak_equity, equity)
        daily_return = total / (equity - total)
        trailing_returns.append(daily_return)
        drawdown = equity / peak_equity - 1
        if daily_return < -cfg.daily_loss_limit:
            violation_rows.append({"timestamp": ts, "type": "daily_loss_limit", "value": daily_return, "limit": -cfg.daily_loss_limit})
        if drawdown < -cfg.max_drawdown_limit:
            violation_rows.append({"timestamp": ts, "type": "max_drawdown_limit", "value": drawdown, "limit": -cfg.max_drawdown_limit})
        if cfg.enforce_risk_limits and violation_rows and violation_rows[-1]["timestamp"] == ts:
            raise RuntimeError(f"Risk limit breached at {ts}: {violation_rows[-1]['type']}")
        pnl_rows.append({"timestamp": ts, "price_pnl": price_pnl, "funding_pnl": funding, "fee_cost": fee, "slippage_cost": slippage, "total_pnl": total, "equity": equity, "return": total / (equity - total) if equity != total else 0, "turnover": turnover_notional / (equity - total) if equity != total else 0})
        for asset in traded:
            change = positions.get(asset, 0) - prev_positions.get(asset, 0)
            if change:
                trade_rows.append({"timestamp": ts, "asset": asset, "weight_change": change, "notional": abs(change) * (equity - total)})
        prev_positions, prev_prices = positions.copy(), current_prices.copy()
    ranking = pd.DataFrame(ranking_rows)
    position_frame = pd.DataFrame(position_rows)
    exposure = position_frame.groupby("timestamp")["weight"].agg(long_exposure=lambda x: x[x > 0].sum(), short_exposure=lambda x: abs(x[x < 0].sum()), gross_exposure=lambda x: abs(x).sum(), net_exposure="sum").reset_index()
    turnover = pd.DataFrame(pnl_rows)
    pnl = pd.DataFrame(pnl_rows)
    active_pnl = pnl.iloc[1:]
    returns = active_pnl["return"]
    equity_curve = pnl["equity"]
    equity_full = pd.concat([pd.Series([cfg.initial_equity]), equity_curve], ignore_index=True)
    drawdown = (equity_full / equity_full.cummax() - 1).iloc[1:]
    summary = {"initial_equity": cfg.initial_equity, "final_equity": float(equity_curve.iloc[-1]), "total_return": float(equity_curve.iloc[-1] / cfg.initial_equity - 1), "periods": len(active_pnl), "periods_per_year": float(ppy), "annualized_return": float((equity_curve.iloc[-1] / cfg.initial_equity) ** (ppy / max(len(active_pnl), 1)) - 1), "annualized_volatility": float(returns.std(ddof=1) * (ppy ** 0.5)) if len(returns) > 1 else 0.0, "sharpe": float(returns.mean() / returns.std(ddof=1) * (ppy ** 0.5)) if len(returns) > 1 and returns.std(ddof=1) else 0.0, "max_drawdown": float(drawdown.min()), "winning_periods": int((returns > 0).sum()), "losing_periods": int((returns < 0).sum()), "total_fees": float(pnl.fee_cost.sum()), "total_slippage": float(pnl.slippage_cost.sum()), "average_turnover": float(active_pnl.turnover.mean()) if len(active_pnl) else 0.0}
    violations = pd.DataFrame(violation_rows)
    capacity = violations[violations["type"] == "capacity_limit"].copy() if not violations.empty and "type" in violations else pd.DataFrame()
    summary["capacity_violations"] = int(len(capacity))
    funding_warnings = funding_gaps[funding_gaps["reason"] == "active"].copy() if not funding_gaps.empty else funding_gaps.copy()
    summary["funding_warnings"] = int(len(funding_warnings))
    return {"ranking": ranking, "exposure": exposure, "positions": position_frame, "turnover": turnover[["timestamp", "turnover"]], "trades": pd.DataFrame(trade_rows), "pnl": pnl, "violations": violations, "capacity": capacity, "funding_gaps": funding_gaps, "funding_warnings": funding_warnings, "metrics": summary}


def write_reports(result, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(Path(output_dir) / f"{name}.csv", index=False)
    summary = result.get("metrics", {})
    summary["latest"] = {name: frame.tail(1).to_dict("records") for name, frame in result.items() if isinstance(frame, pd.DataFrame) and not frame.empty}
    atomic_write_json(Path(output_dir) / "summary.json", summary)
