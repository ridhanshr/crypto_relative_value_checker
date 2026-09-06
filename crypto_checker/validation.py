from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .core import CheckerConfig, check_strategy


COST_STRESS_TIERS = ((0.0004, 0.0005), (0.0008, 0.001), (0.0012, 0.002), (0.0008, 0.002), (0.0015, 0.004))


def _metrics(result):
    return result["metrics"]


def benchmark_equal_weight(data):
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["price"] = pd.to_numeric(frame["price"])
    returns = frame.sort_values(["asset", "timestamp"]).groupby("asset")["price"].pct_change()
    daily = returns.groupby(frame.loc[returns.index, "timestamp"]).mean().dropna()
    equity = (1 + daily).cumprod()
    return {"total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0, "periods": int(len(equity))}


def regime_labels(data):
    btc = data[data.asset == "BTCUSDT"].sort_values("timestamp").drop_duplicates("timestamp").copy()
    btc["ret"] = btc["price"].pct_change()
    btc["trend"] = btc["ret"].rolling(30).sum()
    btc["vol"] = btc["ret"].rolling(30).std()
    vol_cutoff = btc["vol"].quantile(0.66)
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
    pieces = {"train": frame[frame.timestamp.isin(dates[:first])], "validation": frame[frame.timestamp.isin(dates[first:second])], "out_of_sample": frame[frame.timestamp.isin(dates[second:])]}
    if cfg.signal_column in frame.columns:
        pieces = {name: part.dropna(subset=[cfg.signal_column]) for name, part in pieces.items()}
        frame = frame.dropna(subset=[cfg.signal_column])
    if any(part.empty for part in pieces.values()):
        raise ValueError("Validation split became empty after signal warm-up filtering")
    results = {name: _metrics(check_strategy(part, research_cfg)) for name, part in pieces.items()}
    stress = {}
    for fee, slippage in COST_STRESS_TIERS:
        stress[f"fee_{fee}_slippage_{slippage}"] = _metrics(check_strategy(pieces["out_of_sample"], replace(research_cfg, fee_rate=fee, slippage_rate=slippage)))
    violations = {name: int(len(check_strategy(part, research_cfg)["violations"])) for name, part in pieces.items()}
    full_run = check_strategy(frame, research_cfg)
    result = {"splits": {name: {"start": str(part.timestamp.min()), "end": str(part.timestamp.max()), "periods": int(part.timestamp.nunique())} for name, part in pieces.items()}, "performance": results, "risk_violations": violations, "cost_stress": stress, "benchmark_equal_weight": benchmark_equal_weight(pieces["out_of_sample"]), "regimes": regime_performance(frame, full_run["pnl"]), "gates": {"train_positive": results["train"]["total_return"] > 0, "validation_positive": results["validation"]["total_return"] > 0, "oos_positive": results["out_of_sample"]["total_return"] > 0, "oos_sharpe_above_one": results["out_of_sample"]["sharpe"] > 1, "oos_drawdown_above_minus_20pct": results["out_of_sample"]["max_drawdown"] > -0.20, "stress_positive": all(x["total_return"] > 0 for x in stress.values()), "no_risk_violations": all(value == 0 for value in violations.values())}}
    result["deployable"] = all(result["gates"].values())
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "validation.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
