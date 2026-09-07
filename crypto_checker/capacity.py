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

from .core import CheckerConfig, check_strategy
from .assets import canonical_asset
from .schema import SCHEMA_VERSION
from .io import atomic_write_json

DEFAULT_AUM_LEVELS = (10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0)


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
        })
    report = {"schema_version": SCHEMA_VERSION, "status": "ok", "liquidity_column": liq_col, "max_volume_participation": limit, "levels": rows}
    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        atomic_write_json(Path(output_dir, "capacity_curve.json"), report)
    return report
