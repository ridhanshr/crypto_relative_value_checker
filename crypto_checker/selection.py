from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from .core import CheckerConfig, check_strategy, infer_periods_per_year
from .signals import available_candidates, build_signals
from .reality_check import deflated_sharpe_ratio
from .schema import SCHEMA_VERSION
from .io import atomic_write_json
from .trial_registry import TrialRegistry, ExclusionReason
from .ensemble_check import pre_ensemble_report


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
    return _run_indexed(frame, cand, cfg, part).reset_index(drop=True)


def _run_indexed(frame, cand, cfg, part):
    """Same as _run but indexed by bar timestamp (for ensemble averaging)."""
    result = check_strategy(_slice(frame, cand, part), replace(cfg, signal_column=cand, enforce_risk_limits=False))
    pnl = result["pnl"]
    if len(pnl) < 2:
        return pd.Series(dtype=float)
    stamps = pd.to_datetime(pnl["timestamp"].iloc[1:], utc=True)
    return pd.Series(pnl["return"].iloc[1:].to_numpy(), index=stamps)


def walk_forward(data, base_config=None, min_train_days=365, test_days=120, output_dir=None, fee_stress_pairs=None, candidates=None, n_sides_grid=None, vol_target_annual=0.0, vol_target_grid=None, ensemble_top_k=0):
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
    all_trials = []
    registry = TrialRegistry()
    registry_effective_by_fold = []
    ensemble_segments = []
    ensemble_members = []
    for index, (train_dates, test_dates) in enumerate(folds_data):
        train = frame[frame.timestamp.isin(train_dates)]
        test = frame[frame.timestamp.isin(test_dates)]
        trials = []
        fold_series = {}
        for cand in candidates:
            warm = train.dropna(subset=[cand])
            for n in n_sides_grid:
                for vt in vt_grid:
                    trial_id = f"{cand}|n{n}|vt{vt}|fold{index}"
                    params = {"signal": cand, "n_sides": n, "vol_target": vt, "fold": index}
                    try:
                        indexed = _run_indexed(frame, cand, replace(cfg, n_long=n, n_short=n, vol_target_annual=vt), train)
                    except (ValueError, AssertionError):
                        # Logged, not skipped silently: failed attempts count
                        # in raw_n_trials with a structured reason.
                        registry.log_trial(trial_id, params, None, ExclusionReason.DATA_ERROR)
                        continue
                    if len(indexed) < 60:
                        registry.log_trial(trial_id, params, None, ExclusionReason.INSUFFICIENT_SAMPLE)
                        continue
                    returns = indexed.reset_index(drop=True)
                    std = returns.std(ddof=1)
                    sharpe = float(returns.mean() / std) if std else -9e9
                    registry.log_trial(trial_id, params, sharpe, returns=indexed)
                    fold_series[trial_id] = indexed
                    trials.append({"signal": cand, "n_sides": n, "vol_target": vt, "train_sharpe": sharpe, "train_return": float((1 + returns).prod() - 1)})
        if not trials:
            raise ValueError(f"No viable trial in fold {index}")
        all_trials.extend(trials)
        registry_effective_by_fold.append(registry.effective_n_trials(0.9, fold_series))
        best = max(trials, key=lambda t: t["train_sharpe"])
        test_returns = _run(frame, best["signal"], replace(cfg, n_long=best["n_sides"], n_short=best["n_sides"], vol_target_annual=best["vol_target"]), test)
        segments["base"].append(test_returns)
        for name, overrides in stress_variants:
            stressed = _run(frame, best["signal"], replace(cfg, n_long=best["n_sides"], n_short=best["n_sides"], vol_target_annual=best["vol_target"], **overrides), test)
            segments["stress"][name].append(stressed)
        choices.append({**best, "fold": index, "train_end": str(train_dates[-1]), "test_start": str(test_dates[0]), "test_end": str(test_dates[-1]), "test_return": float((1 + test_returns).prod() - 1)})
        # Portfolio of signals: equal-weight ensemble of the top-k DISTINCT
        # signals by train Sharpe (each run at its own best n_sides/vol
        # target). Averaging member return series == running each member at
        # 1/k capital. Diagnostic only: membership is itself selected, so
        # the ensemble is reported alongside -- never instead of -- the
        # winner-takes-all path, and is NOT part of the deployable verdict.
        if ensemble_top_k and ensemble_top_k > 1:
            per_signal = {}
            for trial in trials:
                prev = per_signal.get(trial["signal"])
                if prev is None or trial["train_sharpe"] > prev["train_sharpe"]:
                    per_signal[trial["signal"]] = trial
            top = sorted(per_signal.values(), key=lambda t: t["train_sharpe"], reverse=True)[:ensemble_top_k]
            member_series = []
            for member in top:
                try:
                    series = _run_indexed(frame, member["signal"], replace(cfg, n_long=member["n_sides"], n_short=member["n_sides"], vol_target_annual=member["vol_target"]), test)
                except (ValueError, AssertionError):
                    continue
                if len(series):
                    member_series.append(series.rename(member["signal"]))
            if member_series:
                aligned = pd.concat(member_series, axis=1, join="inner")
                if not aligned.empty:
                    pre = pre_ensemble_report({str(c): aligned[c] for c in aligned.columns}, cfg)
                    ensemble_members.append({"fold": index, "members": [str(c) for c in aligned.columns],
                                             "pre_check": pre["recommendation"], "max_pairwise_corr": pre["max_pairwise_corr"]})
                    if pre["recommendation"] == "BLOCK_ENSEMBLE":
                        continue
                    ensemble_segments.append(aligned.mean(axis=1))
    base_returns = pd.concat(segments["base"], ignore_index=True)
    stress_summary = {name: _summarize(pd.concat(parts, ignore_index=True), ppy) for name, parts in segments["stress"].items()}
    folds_profitable = int(sum(1 for c in choices if c["test_return"] > 0))
    result = {"schema_version": SCHEMA_VERSION, "candidates": candidates, "n_folds": len(folds_data), "folds": choices, "periods_per_year": float(ppy), "walk_forward": _summarize(base_returns, ppy), "cost_stress": stress_summary, "folds_profitable": folds_profitable, "state_policy": "fresh_deployment_per_fold; selection-only, live decisions use continuous-equity validation", "gates": {"wf_positive": _summarize(base_returns, ppy)["total_return"] > 0, "wf_sharpe_above_one": _summarize(base_returns, ppy)["sharpe"] > 1, "wf_drawdown_above_minus_20pct": _summarize(base_returns, ppy)["max_drawdown"] > -0.20, "stress_positive": all(x["total_return"] > 0 for x in stress_summary.values()), "majority_folds_profitable": folds_profitable > len(folds_data) / 2}}
    result["deployable"] = all(result["gates"].values())
    # Deflated Sharpe Ratio over EFFECTIVE (correlation-clustered) trials:
    # raw attempts count the search breadth, valid trials feed the variance,
    # and clustering merges near-duplicate configs (e.g. same signal at
    # different n_sides) so the null bar is not inflated by grid redundancy.
    # Filter the -9e9 sentinel used for zero-variance trials: it is a
    # "do not pick me" marker, not a Sharpe estimate.
    trial_sharpes_ann = [float(t["train_sharpe"]) * (ppy ** 0.5) for t in all_trials if float(t["train_sharpe"]) > -1e9]
    # Global clustering across folds (same configs recur on overlapping
    # windows): summing per-fold counts would double-count correlated trials.
    # Per-fold values are kept as diagnostics only.
    effective_n = int(registry.effective_n_trials(0.9))
    assert effective_n <= registry.raw_n_trials(), "effective trials cannot exceed raw attempts"
    result["trial_registry"] = {"raw_n_trials": registry.raw_n_trials(), "valid_n_trials": len(registry.valid_trials()),
                                "effective_n_trials": effective_n, "effective_by_fold_diagnostic": [int(x) for x in registry_effective_by_fold]}
    oos = pd.Series(base_returns).dropna().astype(float)
    if len(oos) >= 2:
        result["data_mining"] = {
            "n_trials": effective_n,
            "raw_n_trials": registry.raw_n_trials(),
            "n_trials_used": effective_n,
            "skewness": float(oos.skew()) if len(oos) >= 3 else 0.0,
            "kurtosis": float(oos.kurt() + 3) if len(oos) >= 4 else 3.0,
            **deflated_sharpe_ratio(
                observed_sr=result["walk_forward"]["sharpe"],
                n_trials=effective_n,
                skew=float(oos.skew()) if len(oos) >= 3 else 0.0,
                kurtosis=float(oos.kurt() + 3) if len(oos) >= 4 else 3.0,
                n_obs=len(oos),
                trials_variance=float(pd.Series(trial_sharpes_ann).var(ddof=1)) if len(trial_sharpes_ann) >= 2 else 0.0,
            ),
        }
    else:
        result["data_mining"] = {"n_trials": effective_n, "raw_n_trials": registry.raw_n_trials(), "note": "insufficient_oos_observations"}
    if ensemble_segments:
        ens_returns = pd.concat(ensemble_segments, ignore_index=True)
        ens_summary = _summarize(ens_returns, ppy)
        result["ensemble"] = {"top_k": int(ensemble_top_k), "members_per_fold": ensemble_members, "ensemble": ens_summary, "gates": {"ens_positive": ens_summary["total_return"] > 0, "ens_sharpe_above_one": ens_summary["sharpe"] > 1, "ens_drawdown_above_minus_20pct": ens_summary["max_drawdown"] > -0.20}, "note": "diagnostic_only; ensemble membership is selected per fold, so this is not an unbiased estimate and never enters the deployable verdict"}
    if ensemble_members:
        recs = [m.get("pre_check", "PROCEED") for m in ensemble_members]
        overall = "BLOCK_ENSEMBLE" if "BLOCK_ENSEMBLE" in recs else ("DROP_REDUNDANT" if "DROP_REDUNDANT" in recs else "PROCEED")
        result["ensemble_pre_check"] = {"per_fold": ensemble_members, "recommendation": overall,
                                        "note": "pre-assembly correlation screen; BLOCK folds are not aggregated"}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        atomic_write_json(Path(output_dir, "walk_forward.json"), result)
    return result
