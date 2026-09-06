from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from .core import CheckerConfig, check_strategy
from .signals import available_candidates, build_signals


N_SIDES_GRID = (2, 3, 4)


def _active_returns(result):
    pnl = result["pnl"]
    if len(pnl) < 2:
        return pd.Series(dtype=float)
    return pnl["return"].iloc[1:].reset_index(drop=True)


def _summarize(returns):
    if len(returns) < 2:
        return {"total_return": float((1 + returns).prod() - 1) if len(returns) else 0.0, "periods": int(len(returns)), "sharpe": 0.0, "max_drawdown": 0.0}
    std = returns.std(ddof=1)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {"total_return": float(equity.iloc[-1] - 1), "periods": int(len(returns)), "annualized_volatility": float(std * 365 ** 0.5), "sharpe": float(returns.mean() / std * 365 ** 0.5) if std else 0.0, "max_drawdown": float(drawdown.min())}


def _slice(frame, cand, part):
    return part.dropna(subset=[cand])


def _run(frame, cand, cfg, part):
    return _active_returns(check_strategy(_slice(frame, cand, part), replace(cfg, signal_column=cand, enforce_risk_limits=False)))


def walk_forward(data, base_config=None, min_train_days=365, test_days=120, output_dir=None, fee_stress_pairs=None, candidates=None, n_sides_grid=None, vol_target_annual=0.0, vol_target_grid=None):
    base = base_config or CheckerConfig()
    vt_grid = vol_target_grid or (vol_target_annual,)
    cfg = replace(base, vol_target_annual=vt_grid[0])
    frame = build_signals(data)
    frame = frame.sort_values(["timestamp", "asset"]).reset_index(drop=True)
    candidates = candidates or available_candidates(frame)
    n_sides_grid = n_sides_grid or N_SIDES_GRID
    if not candidates:
        raise ValueError("No signal candidates available")
    dates = sorted(frame["timestamp"].drop_duplicates())
    if len(dates) < min_train_days + test_days:
        raise ValueError("Insufficient data for walk-forward with the requested windows")
    folds = []
    cursor = min_train_days
    folds_data = []
    while cursor + test_days <= len(dates):
        train_dates = dates[:cursor]
        test_dates = dates[cursor:cursor + test_days]
        folds_data.append((train_dates, test_dates))
        cursor += test_days
    fee_stress_pairs = fee_stress_pairs or [(0.0008, 0.001), (0.0015, 0.002)]
    segments = {"base": [], "stress": {pair: [] for pair in fee_stress_pairs}}
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
        for pair in fee_stress_pairs:
            stressed = _run(frame, best["signal"], replace(cfg, n_long=best["n_sides"], n_short=best["n_sides"], vol_target_annual=best["vol_target"], fee_rate=pair[0], slippage_rate=pair[1]), test)
            segments["stress"][pair].append(stressed)
        choices.append({**best, "fold": index, "train_end": str(train_dates[-1]), "test_start": str(test_dates[0]), "test_end": str(test_dates[-1]), "test_return": float((1 + test_returns).prod() - 1)})
    base_returns = pd.concat(segments["base"], ignore_index=True)
    stress_summary = {f"fee_{pair[0]}_slippage_{pair[1]}": _summarize(pd.concat(parts, ignore_index=True)) for pair, parts in segments["stress"].items()}
    folds_profitable = int(sum(1 for c in choices if c["test_return"] > 0))
    result = {"candidates": candidates, "n_folds": len(folds_data), "folds": choices, "walk_forward": _summarize(base_returns), "cost_stress": stress_summary, "folds_profitable": folds_profitable, "gates": {"wf_positive": _summarize(base_returns)["total_return"] > 0, "wf_sharpe_above_one": _summarize(base_returns)["sharpe"] > 1, "wf_drawdown_above_minus_20pct": _summarize(base_returns)["max_drawdown"] > -0.20, "stress_positive": all(x["total_return"] > 0 for x in stress_summary.values()), "majority_folds_profitable": folds_profitable > len(folds_data) / 2}}
    result["deployable"] = all(result["gates"].values())
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "walk_forward.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
