from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .core import CheckerConfig, check_strategy, infer_periods_per_year
from .reality_check import reality_check
from .schema import SCHEMA_VERSION
from .io import atomic_write_json


COST_STRESS_TIERS = ((0.0004, 0.0005), (0.0008, 0.001), (0.0012, 0.002), (0.0008, 0.002), (0.0015, 0.004))

SURVIVORSHIP_NOTE = "Universe berasal dari simbol yang tersedia saat ini; tanpa verifikasi point-in-time membership independen, hasil IC/return berpotensi bias survivorship yang belum terukur."


def _metrics(result):
    return result["metrics"]


def _segment_metrics(full_pnl, full_violations, dates, initial_equity):
    seg = full_pnl[full_pnl.timestamp.isin(dates)]
    if seg.empty:
        raise ValueError("Empty validation segment")
    first_idx = int(seg.index[0])
    equity_start = float(full_pnl.iloc[first_idx - 1].equity) if first_idx > 0 else float(initial_equity)
    equity_end = float(seg.equity.iloc[-1])
    ppy = infer_periods_per_year(dates)
    returns = seg["return"].astype(float)
    std = returns.std(ddof=1)
    curve = pd.concat([pd.Series([equity_start]), seg.equity.reset_index(drop=True)], ignore_index=True)
    drawdown = (curve / curve.cummax() - 1).iloc[1:]
    if not full_violations.empty and "timestamp" in full_violations:
        seg_violations = full_violations[full_violations.timestamp.isin(dates)]
    else:
        seg_violations = pd.DataFrame()
    avg_turnover = float(seg.turnover.mean())
    return {"initial_equity": equity_start, "final_equity": equity_end, "total_return": equity_end / equity_start - 1, "periods": int(len(seg)), "periods_per_year": float(ppy), "annualized_return": float((equity_end / equity_start) ** (ppy / max(len(seg), 1)) - 1), "annualized_volatility": float(std * ppy ** 0.5) if len(returns) > 1 and std else 0.0, "sharpe": float(returns.mean() / std * ppy ** 0.5) if len(returns) > 1 and std else 0.0, "max_drawdown": float(drawdown.min()), "winning_periods": int((returns > 0).sum()), "losing_periods": int((returns < 0).sum()), "average_turnover": avg_turnover, "turnover_flag": "OVER" if avg_turnover > 0.15 else "OK", "turnover_note": "average_turnover is fraction of equity turned over per period; academic survival threshold is ~50%/month one-sided (~0.023/day); OVER means turnover alone can dominate net returns", "risk_violations": int(len(seg_violations)), "capacity_violations": int((seg_violations["type"] == "capacity_limit").sum()) if not seg_violations.empty and "type" in seg_violations else 0, "forced_exits": int((seg_violations["type"] == "forced_exit").sum()) if not seg_violations.empty and "type" in seg_violations else 0}


def benchmark_equal_weight(data):
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["price"] = pd.to_numeric(frame["price"])
    returns = frame.sort_values(["asset", "timestamp"]).groupby("asset")["price"].pct_change()
    daily = returns.groupby(frame.loc[returns.index, "timestamp"]).mean().dropna()
    equity = (1 + daily).cumprod()
    return {"total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0, "periods": int(len(equity))}


def benchmark_suite(data):
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["asset", "timestamp"])
    returns = frame.groupby("asset")["price"].pct_change()
    daily = returns.groupby(frame.loc[returns.index, "timestamp"]).mean().dropna()

    def compound(series):
        series = series.dropna()
        return float((1 + series).prod() - 1) if len(series) else 0.0

    result = {"equal_weight": benchmark_equal_weight(frame)}
    for asset in ("BTCUSDT", "ETHUSDT"):
        series = frame[frame.asset == asset].set_index("timestamp")["price"].pct_change()
        result[asset.lower()] = {"total_return": compound(series), "periods": int(series.dropna().size)}
    # NOTE: a previous version clipped negative days here (daily.clip(lower=0)),
    # which fabricates returns no long-only portfolio can achieve. A true
    # long-only equal-weight basket is exactly `equal_weight` above.
    result["long_only_equal_weight"] = {"total_return": compound(daily), "periods": int(daily.size)}
    if "btcusdt" in result:
        result["btc_buy_hold"] = result["btcusdt"]
    if "ethusdt" in result:
        result["eth_buy_hold"] = result["ethusdt"]
    result["market_neutral_random_reference"] = {"total_return": 0.0, "periods": int(daily.size), "note": "zero-return theoretical neutral reference"}
    return result


def regime_labels(data):
    btc = data[data.asset == "BTCUSDT"].sort_values("timestamp").drop_duplicates("timestamp").copy()
    btc["ret"] = btc["price"].pct_change()
    btc["trend"] = btc["ret"].rolling(30).sum()
    btc["vol"] = btc["ret"].rolling(30).std()
    # Expanding quantile: the hi/lo-vol cutoff at time t uses only data
    # available up to t. A full-sample quantile would leak future volatility
    # regimes into past labels.
    vol_cutoff = btc["vol"].expanding().quantile(0.66)
    trend = np.select([btc["trend"] > 0.10, btc["trend"] < -0.10], ["bull", "bear"], default="sideways")
    vol_state = np.where(btc["vol"] > vol_cutoff, "hi_vol", "lo_vol")
    btc["regime"] = [f"{t}_{v}" for t, v in zip(trend, vol_state)]
    return btc[["timestamp", "regime"]]


def regime_performance(data, pnl):
    labels = regime_labels(data)
    active = pnl.iloc[1:][["timestamp", "return"]]
    joined = active.merge(labels, on="timestamp", how="left").dropna(subset=["regime"])
    if joined.empty:
        return []
    stats = joined.groupby("regime").agg(days=("return", "size"), mean_daily_return=("return", "mean"), positive_days=("return", lambda x: int((x > 0).sum())), total_return=("return", lambda x: float((1 + x).prod() - 1))).reset_index()
    return stats.to_dict("records")


def run_validation(data, config=None, train_ratio=0.6, validation_ratio=0.2, output_dir=None):
    cfg = config or CheckerConfig()
    research_cfg = replace(cfg, enforce_risk_limits=False)
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    dates = sorted(frame["timestamp"].drop_duplicates())
    if len(dates) < 30 or not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("Insufficient data or invalid split ratios")
    first = int(len(dates) * train_ratio)
    second = int(len(dates) * (train_ratio + validation_ratio))
    if first < 1 or second <= first or second >= len(dates):
        raise ValueError("Each validation split must contain at least one timestamp")
    slice_dates = {"train": dates[:first], "validation": dates[first:second], "out_of_sample": dates[second:]}
    if cfg.signal_column in frame.columns:
        frame = frame.dropna(subset=[cfg.signal_column])
        slice_dates = {name: [d for d in ds if d in set(frame.timestamp)] for name, ds in slice_dates.items()}
    if any(not ds for ds in slice_dates.values()):
        raise ValueError("Validation split became empty after signal warm-up filtering")
    full_run = check_strategy(frame, research_cfg)
    results = {name: _segment_metrics(full_run["pnl"], full_run["violations"], ds, cfg.initial_equity) for name, ds in slice_dates.items()}
    violations = {name: metrics["risk_violations"] for name, metrics in results.items()}
    stress = {}
    oos_frame = frame[frame.timestamp.isin(slice_dates["out_of_sample"])]
    tiered = bool(research_cfg.liquidity_tiers and research_cfg.liquidity_column)
    stress_tiers = []
    for fee, slippage in COST_STRESS_TIERS:
        if tiered:
            multiplier = max(fee / 0.0004, slippage / 0.0005)
            stress_tiers.append((f"cost_x_{multiplier:.2f}", {"cost_multiplier": multiplier}))
        else:
            stress_tiers.append((f"fee_{fee}_slippage_{slippage}", {"fee_rate": fee, "slippage_rate": slippage}))
    for name, overrides in stress_tiers:
        stress[name] = _metrics(check_strategy(oos_frame, replace(research_cfg, **overrides)))
    result = {"schema_version": SCHEMA_VERSION, "splits": {name: {"start": str(min(ds)), "end": str(max(ds)), "periods": len(ds)} for name, ds in slice_dates.items()}, "performance": results, "risk_violations": violations, "cost_stress": stress, "benchmarks": benchmark_suite(oos_frame), "regimes": regime_performance(frame, full_run["pnl"]), "reality_check": reality_check([]), "continuous_equity": True, "survivorship_note": SURVIVORSHIP_NOTE, "gates": {"train_positive": results["train"]["total_return"] > 0, "validation_positive": results["validation"]["total_return"] > 0, "oos_positive": results["out_of_sample"]["total_return"] > 0, "oos_sharpe_above_one": results["out_of_sample"]["sharpe"] > 1, "oos_drawdown_above_minus_20pct": results["out_of_sample"]["max_drawdown"] > -0.20, "stress_positive": all(x["total_return"] > 0 for x in stress.values()), "no_risk_violations": all(value == 0 for value in violations.values()), "no_capacity_violations": all(metrics["capacity_violations"] == 0 for metrics in results.values())}}
    result["deployable"] = all(result["gates"].values())
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        atomic_write_json(Path(output_dir, "validation.json"), result)
    return result
