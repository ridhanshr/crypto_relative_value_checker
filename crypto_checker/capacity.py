"""Capacity model: does the strategy survive its own AUM?

Re-runs the backtest at several starting equities
(10k / 100k / 1M / 10M USD by default) and reports, per level, net
performance plus how hard the strategy pushes against available volume:

* ``capacity_violations``: trades whose notional exceeds
  ``max_volume_participation`` of that bar's volume (fail-closed flag).
* ``max_participation_ratio``: worst trade notional / bar volume.
* ``headroom_multiple``: max_volume_participation / max_ratio. Values < 1
  mean the strategy ALREADY breaches at that AUM; values > 1 say how many
  times over the AUM could grow before the first breach (linearity is NOT
  assumed -- every level is actually re-run).

Costs are proportional, so net returns are identical across AUM until
participation binds; the curve answers "at which AUM does this stop being
sensible" instead of pretending fills are infinite. Requires a liquidity
column (quote_volume); without one the function returns
``status: no_liquidity_data`` rather than inventing volume.
"""

from dataclasses import replace
import pandas as pd
import numpy as np

from .core import CheckerConfig, check_strategy
from .assets import canonical_asset
from .schema import SCHEMA_VERSION
from .io import atomic_write_json
import warnings


def square_root_impact(order_size, adv, daily_volatility, impact_coefficient=1.0):
    """Market impact in basis points: coef * vol * sqrt(order/adv) * 1e4.

    impact_coefficient is UNCALIBRATED by default (1.0): treat absolute bps
    as order-of-magnitude, cross-AUM comparisons as structural. Calibrate
    against real execution data before quoting bps as fact.
    """
    order_size, adv, daily_volatility = float(order_size), float(adv), float(daily_volatility)
    if adv <= 0 or order_size <= 0 or daily_volatility < 0:
        return 0.0
    return float(impact_coefficient) * daily_volatility * (order_size / adv) ** 0.5 * 10000.0


def linear_impact(order_size, adv, daily_volatility, impact_coefficient=1.0):
    order_size, adv, daily_volatility = float(order_size), float(adv), float(daily_volatility)
    if adv <= 0 or order_size <= 0 or daily_volatility < 0:
        return 0.0
    return float(impact_coefficient) * daily_volatility * (order_size / adv) * 10000.0


def compute_adv(volume_history, lookback_days=90, exclude_outlier_days=True):
    """Trailing ADV over the last `lookback_days` prints.

    Outlier exclusion drops days > mean + 3*std (listing/delisting/news
    spikes) so ADV does not overstate real capacity. Empty history raises
    (no data = no estimate, never a silent zero).
    """
    series = pd.to_numeric(pd.Series(volume_history).dropna(), errors="coerce").dropna()
    series = series[np.isfinite(series) & (series > 0)].tail(int(lookback_days))
    if series.empty:
        raise ValueError("ADV needs at least one positive volume print")
    if exclude_outlier_days and len(series) >= 3:
        mean, std = float(series.mean()), float(series.std(ddof=1))
        if std > 0:
            series = series[series <= mean + 3 * std]
    if series.empty:
        raise ValueError("ADV empty after outlier exclusion")
    return float(series.mean())

DEFAULT_AUM_LEVELS = (10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0)


def capacity_impact_overlay(trades, data, liquidity_column, lookback_days=90,
                            exclude_outlier_days=True, impact_coefficient=1.0,
                            impact_model="sqrt", min_headroom=1.5, return_vol_window=30):
    """Per-trade sqrt (or deprecated linear) impact overlay.

    For every trade: trailing ADV + trailing return vol of that asset up to
    the trade bar -> participation, impact_bps, headroom =
    min_headroom / participation. Overall status = worst trade
    (SAFE >= min_headroom, MARGINAL >= 1, else BREACH). Assets whose lookback
    contained zero-volume prints are listed for manual review (delisting
    artifacts inflate headroom illusions).
    """
    if impact_model == "linear":
        warnings.warn("capacity_impact_model='linear' is deprecated: linear impact UNDERESTIMATES large-size impact; results are likely optimistic")
        impact_fn = linear_impact
    else:
        impact_fn = square_root_impact
    vols = data[["timestamp", "asset", liquidity_column, "price"]].copy()
    vols["timestamp"] = pd.to_datetime(vols["timestamp"], utc=True)
    vols["asset"] = vols["asset"].astype(str).str.upper().map(canonical_asset)
    vols = vols.sort_values(["asset", "timestamp"])
    vols["_ret"] = vols.groupby("asset")["price"].pct_change()
    rows = []
    anomaly_assets = set()
    for _, trade in trades.iterrows():
        asset, ts = str(trade["asset"]).upper(), pd.to_datetime(trade["timestamp"], utc=True)
        hist = vols[(vols["asset"] == asset) & (vols["timestamp"] <= ts)].tail(int(lookback_days))
        if hist.empty:
            continue
        if (pd.to_numeric(hist[liquidity_column], errors="coerce").fillna(0) <= 0).any():
            anomaly_assets.add(asset)
        try:
            adv = compute_adv(hist[liquidity_column], lookback_days, exclude_outlier_days)
        except ValueError:
            continue
        vol_window = pd.to_numeric(hist["_ret"], errors="coerce").dropna().tail(int(return_vol_window))
        daily_vol = float(vol_window.std(ddof=1)) if len(vol_window) >= 2 else 0.0
        notional = float(trade["notional"])
        participation = notional / adv if adv > 0 else float("inf")
        impact_bps = impact_fn(notional, adv, daily_vol, impact_coefficient)
        headroom = float(min_headroom) / participation if participation > 0 else float("inf")
        rows.append({"timestamp": str(ts), "asset": asset, "notional": notional, "adv": adv,
                     "participation_rate": float(participation), "impact_bps": float(impact_bps),
                     "headroom": float(headroom)})
    if not rows:
        return {"status": "NO_TRADES", "max_impact_bps": 0.0, "max_participation_rate": 0.0,
                "adv_anomaly_assets": sorted(anomaly_assets), "trades": []}
    worst = max(rows, key=lambda r: r["participation_rate"])
    status = "SAFE" if worst["headroom"] >= min_headroom else ("MARGINAL" if worst["headroom"] >= 1.0 else "BREACH")
    return {"status": status, "max_impact_bps": float(max(r["impact_bps"] for r in rows)),
            "max_participation_rate": float(worst["participation_rate"]),
            "adv_anomaly_assets": sorted(anomaly_assets), "trades": rows}


def capacity_curve(data, base_config=None, aum_levels=None, signal_column=None, output_dir=None):
    base = base_config or CheckerConfig()
    if signal_column:
        base = replace(base, signal_column=signal_column)
    levels = tuple(aum_levels or DEFAULT_AUM_LEVELS)
    liq_col = base.liquidity_column
    if not liq_col or liq_col not in data.columns:
        return {"schema_version": SCHEMA_VERSION, "status": "no_liquidity_data", "liquidity_column": liq_col, "note": "capacity needs a volume column; refusing to invent liquidity"}
    volume = data[["timestamp", "asset", liq_col]].copy()
    volume["timestamp"] = pd.to_datetime(volume["timestamp"], utc=True)
    # Trades carry canonical names (core maps MATIC->POL etc.); map volume
    # the same way or migrated assets would silently miss participation.
    volume["asset"] = volume["asset"].astype(str).str.upper().map(canonical_asset)
    rows = []
    for aum in levels:
        result = check_strategy(data, replace(base, initial_equity=float(aum), enforce_risk_limits=False))
        metrics = result["metrics"]
        trades = result["trades"]
        limit = float(base.max_volume_participation)
        if trades.empty:
            max_ratio, breached_notional, breach_trades = 0.0, 0.0, 0
        else:
            merged = trades.merge(volume, on=["timestamp", "asset"], how="left")
            vol = pd.to_numeric(merged[liq_col], errors="coerce")
            ratio = (pd.to_numeric(merged["notional"], errors="coerce") / vol).where(vol > 0, float("nan"))
            breached = ratio > limit
            max_ratio = float(ratio.max(skipna=True)) if ratio.notna().any() else 0.0
            breached_notional = float(merged.loc[breached.fillna(False), "notional"].sum())
            breach_trades = int(breached.fillna(False).sum())
        # v2 sqrt-impact overlay (informational): the participation hard gate
        # above is untouched; this models impact shape per trade instead.
        overlay = capacity_impact_overlay(
            trades, data, liq_col, lookback_days=base.capacity_adv_lookback_days,
            exclude_outlier_days=base.capacity_adv_exclude_outlier_days,
            impact_model=base.capacity_impact_model, min_headroom=base.capacity_min_headroom)
        rows.append({
            "aum": float(aum),
            "total_return": float(metrics["total_return"]),
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "average_turnover": float(metrics["average_turnover"]),
            "capacity_violations": int(metrics["capacity_violations"]),
            "breach_trades": breach_trades,
            "breached_notional": float(breached_notional),
            "max_participation_ratio": float(max_ratio),
            "headroom_multiple": float(limit / max_ratio) if max_ratio > 0 else float("inf"),
            # Sensible == no breached trade by our own participation math
            # (independent of whether the core run had tiered costs enabled).
            "sensible": bool(breach_trades == 0),
            "sqrt_impact": overlay,
        })
    report = {"schema_version": SCHEMA_VERSION, "status": "ok", "liquidity_column": liq_col, "max_volume_participation": limit, "levels": rows}
    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        atomic_write_json(Path(output_dir, "capacity_curve.json"), report)
    return report
