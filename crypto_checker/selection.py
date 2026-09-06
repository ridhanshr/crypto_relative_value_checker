from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from .core import CheckerConfig, check_strategy, infer_periods_per_year
from .signals import available_candidates, build_signals


N_SIDES_GRID = (2, 3, 4)


def _active_returns(result):
    pnl = result["pnl"]
    if len(pnl) < 2:
        return pd.Series(dtype=float)
    return pnl["return"].iloc[1:].reset_index(drop=True)


def _summarize(returns, periods_per_year=365.0):
    if len(returns) < 2:
        return {"total_return": float((1 + returns).prod() - 1) if len(returns) else 0.0, "periods": int(len(returns)), "periods_per_year": float(periods_per_year), "sharpe": 0.0, "max_drawdown": 0.0}
    std = returns.std(ddof=1)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {"total_return": float(equity.iloc[-1] - 1), "periods": int(len(returns)), "periods_per_year": float(periods_per_year), "annualized_volatility": float(std * periods_per_year ** 0.5), "sharpe": float(returns.mean() / std * periods_per_year ** 0.5) if std else 0.0, "max_drawdown": float(drawdown.min())}


def _slice(frame, cand, part):
    return part.dropna(subset=[cand])


def _run(frame, cand, cfg, part):
    return _active_returns(check_strategy(_slice(frame, cand, part), replace(cfg, signal_column=cand, enforce_risk_limits=False)))


def walk_forward(data, base_config=None, min_train_days=365, test_days=120, output_dir=None, fee_stress_pairs=None, candidates=None, n_sides_grid=None, vol_target_annual=0.0, vol_target_grid=None):
    base = base_config or CheckerConfig()
    vt_grid = vol_target_grid or (vol_target_annual,)
    cfg = replace(base, vol_target_annual=vt_grid[0], enforce_risk_limits=False)
    frame = build_signals(data)
    frame = frame.sort_values(["timestamp", "asset"]).reset_index(drop=True)
    candidates = candidates or available_candidates(frame)
    n_sides_grid = n_sides_grid or N_SIDES_GRID
    if not candidates:
        raise ValueError("No signal candidates available")
    dates = sorted(frame["timestamp"].drop_duplicates())
    ppy = infer_periods_per_year(dates)
    periods_per_day = ppy / 365.0
    min_train_periods = max(1, int(round(min_train_days * periods_per_day)))
    test_periods = max(1, int(round(test_days * periods_per_day)))
    if len(dates) < min_train_periods + test_periods:
        raise ValueError("Insufficient data for walk-forward with the requested windows")
    folds = []
    cursor = min_train_periods
    folds_data = []
    while cursor + test_periods <= len(dates):
        train_dates = dates[:cursor]
        test_dates = dates[cursor:cursor + test_periods]
        folds_data.append((train_dates, test_dates))
        cursor += test_periods
    fee_stress_pairs = fee_stress_pairs or [(0.0008, 0.001), (0.0015, 0.002)]
    tiered = bool(cfg.liquidity_tiers and cfg.liquidity_column)
    stress_variants = []
    for pair in fee_stress_pairs:
        if tiered:
            multiplier = max(pair[0] / 0.0004, pair[1] / 0.0005)
            stress_variants.append((f"cost_x_{multiplier:.2f}", {"cost_multiplier": multiplier}))
        else:
            stress_variants.append((f"fee_{pair[0]}_slippage_{pair[1]}", {"fee_rate": pair[0], "slippage_rate": pair[1]}))
    segments = {"base": [], "stress": {name: [] for name, _ in stress_variants}}
    choices = []
    for index, (train_dates, test_dates) in enumerate(folds_data):
        train = frame[frame.timestamp.isin(train_dates)]
        test = frame[frame.timestamp.isin(test_dates)]
        trials = []
        for cand in candidates:
            warm = train.dropna(subset=[cand])
            for n in n_sides_grid:
                for vt in vt_grid:
                    try:
                        returns = _run(frame, cand, replace(cfg, n_long=n, n_short=n, vol_target_annual=vt), train)
                    except (ValueError, AssertionError):
                        continue
                    if len(returns) < 60:
                        continue
                    std = returns.std(ddof=1)
                    sharpe = float(returns.mean() / std) if std else -9e9
                    trials.append({"signal": cand, "n_sides": n, "vol_target": vt, "train_sharpe": sharpe, "train_return": float((1 + returns).prod() - 1)})
        if not trials:
            raise ValueError(f"No viable trial in fold {index}")
        best = max(trials, key=lambda t: t["train_sharpe"])
        test_returns = _run(frame, best["signal"], replace(cfg, n_long=best["n_sides"], n_short=best["n_sides"], vol_target_annual=best["vol_target"]), test)
        segments["base"].append(test_returns)
        for name, overrides in stress_variants:
            stressed = _run(frame, best["signal"], replace(cfg, n_long=best["n_sides"], n_short=best["n_sides"], vol_target_annual=best["vol_target"], **overrides), test)
            segments["stress"][name].append(stressed)
        choices.append({**best, "fold": index, "train_end": str(train_dates[-1]), "test_start": str(test_dates[0]), "test_end": str(test_dates[-1]), "test_return": float((1 + test_returns).prod() - 1)})
    base_returns = pd.concat(segments["base"], ignore_index=True)
    stress_summary = {name: _summarize(pd.concat(parts, ignore_index=True), ppy) for name, parts in segments["stress"].items()}
    folds_profitable = int(sum(1 for c in choices if c["test_return"] > 0))
    result = {"candidates": candidates, "n_folds": len(folds_data), "folds": choices, "periods_per_year": float(ppy), "walk_forward": _summarize(base_returns, ppy), "cost_stress": stress_summary, "folds_profitable": folds_profitable, "state_policy": "fresh_deployment_per_fold; selection-only, live decisions use continuous-equity validation", "gates": {"wf_positive": _summarize(base_returns, ppy)["total_return"] > 0, "wf_sharpe_above_one": _summarize(base_returns, ppy)["sharpe"] > 1, "wf_drawdown_above_minus_20pct": _summarize(base_returns, ppy)["max_drawdown"] > -0.20, "stress_positive": all(x["total_return"] > 0 for x in stress_summary.values()), "majority_folds_profitable": folds_profitable > len(folds_data) / 2}}
    result["deployable"] = all(result["gates"].values())
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "walk_forward.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
