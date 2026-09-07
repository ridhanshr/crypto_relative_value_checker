"""Audit #1: apakah max DD CPCV mengandung artefak lompatan gap?

Test paths gabungan 2 grup yang BELUM TENTU adjacent -> check_strategy
menghitung return melintasi gap multi-bulan sebagai SATU bar. Skrip ini
merekonstruksi equity per fold dengan return bar-gap di-nol-kan dan
membandingkan DD dengan/tanpa gap-jump.

Output: per-fold DD asli vs DD-tanpa-gap + identifikasi fold penentu max.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from crypto_checker.signals import build_signals

FOLDS = Path("reports/cpcv_lowvol/fold_results.jsonl")

data = build_signals(pd.read_csv("data/midcap_2y_daily_2_funded.csv"))
data = data.dropna(subset=["low_vol_14"]).reset_index(drop=True)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)


def actual_stamps(idx):
    # Mirror the engine EXACTLY: the t+1 shift runs WITHIN the slice, so each
    # asset's first slice bar has no executable signal; a timestamp survives
    # iff at least one asset has a signal there.
    want = set(pd.to_datetime(pd.Index(idx), utc=True))
    sl = data[data["timestamp"].isin(want)].copy()
    sl["_exec"] = sl.groupby("asset")["low_vol_14"].transform(lambda s: s.shift(1))
    return pd.DatetimeIndex(sorted(set(sl.dropna(subset=["_exec"])["timestamp"])))


def max_dd(equity):
    curve = pd.Series(equity, dtype=float)
    return float((curve / curve.cummax() - 1).min())


def dd_without_gap_jumps(idx, equity):
    ts = actual_stamps(idx)
    eq = pd.Series(equity, dtype=float)
    assert len(ts) == len(eq), (len(ts), len(eq))
    gaps = ts.to_series().diff().dt.total_seconds() / 86400.0
    median_step = float(gaps.median())
    rets = eq.pct_change().fillna(0.0)
    mask_gap = gaps > median_step * 1.5
    rets[mask_gap.values] = 0.0
    rebuilt = (1 + rets).cumprod() * eq.iloc[0]
    return max_dd(rebuilt), int(mask_gap.sum()), median_step


rows = []
for line in FOLDS.read_text().splitlines():
    d = json.loads(line)
    dd_test, n_gap_test, step = dd_without_gap_jumps(d["test_idx"], d["equity_curve_oos"])
    dd_train, n_gap_train, _ = dd_without_gap_jumps(d["train_idx"], d["equity_curve_train"])
    rows.append({"fold": d["fold_id"], "regime": d["regime_label"],
                 "dd_test_raw": max_dd(d["equity_curve_oos"]), "dd_test_nogap": dd_test,
                 "gap_bars_test": n_gap_test, "dd_train_raw": max_dd(d["equity_curve_train"]),
                 "dd_train_nogap": dd_train, "gap_bars_train": n_gap_train})
frame = pd.DataFrame(rows)
frame["dd_test_gap_contrib"] = frame["dd_test_raw"] - frame["dd_test_nogap"]
frame["dd_train_gap_contrib"] = frame["dd_train_raw"] - frame["dd_train_nogap"]
print("median step days:", step, flush=True)
print(frame.sort_values("dd_test_raw").head(8).to_string(), flush=True)
print("---", flush=True)
print(frame.sort_values("dd_train_raw").head(8).to_string(), flush=True)
worst_test = frame.loc[frame["dd_test_raw"].idxmin()]
worst_train = frame.loc[frame["dd_train_raw"].idxmin()]
print("WORST test fold:", worst_test.to_dict(), flush=True)
print("WORST train fold:", worst_train.to_dict(), flush=True)
print("max_dd_oos raw=%.4f nogap-worst=%.4f" % (frame["dd_test_raw"].min(), frame["dd_test_nogap"].min()), flush=True)
print("max_dd_train raw=%.4f nogap-worst=%.4f" % (frame["dd_train_raw"].min(), frame["dd_train_nogap"].min()), flush=True)
print("AUDIT_DONE", flush=True)
