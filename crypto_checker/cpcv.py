"""Combinatorial Purged Cross-Validation (Lopez de Prado 2018, adapted).

Walk-forward linear biasa punya dua kelemahan: (a) train/test bisa bocor
melewati batas (fitur rolling 14-hari memakai bar di seberang batas;
autocorrelation residual), (b) hasil agregat menyembunyikan ketergantungan
regime. CPCV menjawab keduanya: semua kombinasi blok test (bukan satu
urutan), purge + embargo di setiap split, dan ringkasan per-regime.

strategy_fn interface (dipakai run_cpcv dan test):
  strategy_fn(train_frame, test_frame) -> {
      "test_returns": pd.Series (indexed by test timestamps),
      "train_equity": pd.Series (normalized, first value = start equity),
      "test_equity":  pd.Series (normalized, first value = start equity),
  }
Deterministic: combinations + array_split orders are fixed; no RNG here.
"""

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    fold_id: int
    train_idx: pd.Index
    test_idx: pd.Index
    sharpe_oos: float
    test_return: float
    equity_curve_train: pd.Series
    equity_curve_oos: pd.Series
    regime_label: str


def generate_cpcv_splits(labels, n_groups, n_test_groups, embargo_pct, purge_pct):
    """All C(n_groups, n_test_groups) train/test splits with purge+embargo.

    ``labels``: sorted unique timestamps (pd.Index). Returns list of
    (train_idx, test_idx) label Index pairs. purge_len/embargo_len are
    rounded fractions of N (min 1 bar when the fraction is positive).
    """
    stamps = pd.Index(labels).sort_values().unique()
    n = len(stamps)
    if n_groups < 2 or not 1 <= n_test_groups < n_groups:
        raise ValueError("need 2 <= n_groups and 1 <= n_test_groups < n_groups")
    if n < n_groups:
        raise ValueError("fewer timestamps than groups")
    pos = np.arange(n)
    groups = [g for g in np.array_split(pos, n_groups) if len(g)]
    purge_len = max(1, int(round(purge_pct * n))) if purge_pct > 0 else 0
    embargo_len = max(1, int(round(embargo_pct * n))) if embargo_pct > 0 else 0
    splits = []
    for combo in combinations(range(len(groups)), n_test_groups):
        test_pos = np.sort(np.concatenate([groups[g] for g in combo]))
        test_set = set(test_pos.tolist())
        train = [p for p in pos if p not in test_set]
        if purge_len:
            train = [p for p in train if min(abs(p - t) for t in test_pos) > purge_len]
        if embargo_len:
            train = [p for p in train if not any(t < p <= t + embargo_len for t in test_pos)]
        splits.append((stamps[sorted(train)], stamps[sorted(test_pos.tolist())]))
    return splits


def label_regime(returns, method="realized_vol_percentile", window=30):
    """Per-bar regime labels. realized_vol_percentile -> high_vol/low_vol/mid_vol."""
    if method != "realized_vol_percentile":
        raise NotImplementedError(f"regime method {method!r} not implemented (use realized_vol_percentile)")
    series = pd.Series(returns).astype(float)
    vol = series.rolling(window, min_periods=window).std()
    rank = vol.expanding(min_periods=window).rank(pct=True)
    labels = pd.Series("mid_vol", index=series.index)
    labels[rank >= 2 / 3] = "high_vol"
    labels[rank <= 1 / 3] = "low_vol"
    return labels


def run_cpcv(data, strategy_fn, config, timestamp_column="timestamp", periods_per_year=365.0):
    """Run strategy_fn over every CPCV split. Returns list[FoldResult]."""
    frame = data.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    splits = generate_cpcv_splits(frame[timestamp_column].drop_duplicates().sort_values(),
                                  config.cpcv_n_groups, config.cpcv_n_test_groups,
                                  config.cpcv_embargo_pct, config.cpcv_purge_pct)
    vol = None
    if config.regime_label_source == "realized_vol_percentile":
        ref = frame.sort_values(timestamp_column).groupby(timestamp_column)["price"].last() if "price" in frame.columns else None
        rets = ref.pct_change() if ref is not None else pd.Series(dtype=float)
        vol = label_regime(rets, method="realized_vol_percentile")
    results = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train = frame[frame[timestamp_column].isin(train_idx)]
        test = frame[frame[timestamp_column].isin(test_idx)]
        if train.empty or test.empty:
            continue
        out = strategy_fn(train, test)
        test_returns = pd.Series(out["test_returns"]).dropna().astype(float)
        std = test_returns.std(ddof=1)
        sharpe = float(test_returns.mean() / std * periods_per_year ** 0.5) if std else 0.0
        test_equity = pd.Series(out["test_equity"]).astype(float)
        test_return = float(test_equity.iloc[-1] / test_equity.iloc[0] - 1) if len(test_equity) and test_equity.iloc[0] else 0.0
        if vol is not None:
            regime = vol.reindex(pd.to_datetime(test_idx, utc=True)).dropna()
            regime_label = str(regime.mode().iloc[0]) if len(regime) else "unknown"
        else:
            regime_label = "unknown"
        results.append(FoldResult(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx,
                                  sharpe_oos=sharpe, test_return=test_return,
                                  equity_curve_train=pd.Series(out["train_equity"]).astype(float),
                                  equity_curve_oos=test_equity, regime_label=regime_label))
    return results


def _max_drawdown(equity):
    curve = pd.Series(equity).astype(float)
    if not len(curve):
        return 0.0
    return float((curve / curve.cummax() - 1).min())


def summarize_cpcv(results):
    """Single source of truth for equity-derived CPCV stats (gate only reads)."""
    if not results:
        return {"n_folds": 0, "pct_profitable_folds": 0.0, "sharpe_mean": 0.0,
                "max_dd_walk_forward": 0.0, "max_dd_oos": 0.0,
                "sharpe_by_regime": {}, "regime_concentration_flag": False}
    sharpes = [r.sharpe_oos for r in results]
    profits = {r.fold_id: r.test_return for r in results}
    by_regime = {}
    for r in results:
        by_regime.setdefault(r.regime_label, []).append(r.sharpe_oos)
    profit_by_regime = {}
    for r in results:
        if r.test_return > 0:
            profit_by_regime[r.regime_label] = profit_by_regime.get(r.regime_label, 0.0) + r.test_return
    total_profit = sum(profit_by_regime.values())
    concentration = (max(profit_by_regime.values()) / total_profit > 0.8) if total_profit > 0 else False
    return {
        "n_folds": len(results),
        "pct_profitable_folds": float(sum(1 for r in results if r.test_return > 0) / len(results)),
        "sharpe_mean": float(sum(sharpes) / len(sharpes)),
        "max_dd_walk_forward": float(min(_max_drawdown(r.equity_curve_train) for r in results)),
        "max_dd_oos": float(min(_max_drawdown(r.equity_curve_oos) for r in results)),
        "sharpe_by_regime": {k: float(sum(v) / len(v)) for k, v in by_regime.items()},
        "regime_concentration_flag": bool(concentration),
        "_debug_profit_by_regime": {k: float(v) for k, v in profit_by_regime.items()},
    }
