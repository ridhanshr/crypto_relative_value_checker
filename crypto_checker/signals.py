import numpy as np
import pandas as pd


SIGNAL_CANDIDATES = ("reversal_1", "momentum_7", "momentum_14", "momentum_30", "vol_adj_momentum_14", "vol_adj_momentum_30", "carry", "resid_reversal_14", "resid_reversal_30", "low_vol_14", "low_vol_30", "funding_surprise", "vol_adj_carry", "carry_mom_z", "carry_lowvol_z")


def _cross_sectional_z(frame, column):
    return frame.groupby("timestamp")[column].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else 0.0)


def _residual_reversal(frame, k):
    wide = frame.pivot(index="timestamp", columns="asset", values="asset_return")
    btc = wide["BTCUSDT"]
    out = pd.DataFrame(index=wide.index)
    btc_k = btc.rolling(k, min_periods=k).sum()
    for asset in wide.columns:
        if asset == "BTCUSDT":
            neutral = pd.Series(0.0, index=wide.index)
            neutral.iloc[:k - 1] = np.nan
            out[asset] = neutral
            continue
        series = wide[asset]
        beta = series.rolling(k, min_periods=k).cov(btc) / btc.rolling(k, min_periods=k).var()
        resid = series.rolling(k, min_periods=k).sum() - beta * btc_k
        out[asset] = -resid
    stacked = out.rename_axis(columns="asset").stack().rename(f"resid_reversal_{k}").reset_index()
    return frame.merge(stacked, on=["timestamp", "asset"], how="left")


def build_signals(data):
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    if "asset_return" not in frame.columns:
        frame["asset_return"] = frame.groupby("asset")["price"].pct_change()
    for k in (7, 14, 30):
        frame[f"momentum_{k}"] = frame.groupby("asset")["asset_return"].transform(lambda s, k=k: s.shift(1).rolling(k, min_periods=k).sum())
        vol = frame.groupby("asset")["asset_return"].transform(lambda s, k=k: s.shift(1).rolling(k, min_periods=k).std())
        frame[f"vol_adj_momentum_{k}"] = frame[f"momentum_{k}"] / vol.replace(0, np.nan)
        frame[f"low_vol_{k}"] = -vol
    frame["reversal_1"] = -frame["asset_return"]
    if "funding_rate" in frame.columns:
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
        frame["carry"] = -frame["funding_rate"]
        mean_funding = frame.groupby("asset")["funding_rate"].transform(lambda s: s.shift(1).rolling(14, min_periods=14).mean())
        frame["funding_surprise"] = -(frame["funding_rate"] - mean_funding)
        vol_14 = frame.groupby("asset")["asset_return"].transform(lambda s: s.shift(1).rolling(14, min_periods=14).std())
        frame["vol_adj_carry"] = frame["carry"] / vol_14.replace(0, np.nan)
    for k in (14, 30):
        if "BTCUSDT" in set(frame.asset) and frame.timestamp.nunique() > k:
            frame = _residual_reversal(frame, k)
    if "vol_adj_carry" in frame.columns and "vol_adj_momentum_14" in frame.columns:
        frame["carry_mom_z"] = _cross_sectional_z(frame, "vol_adj_carry") + _cross_sectional_z(frame, "vol_adj_momentum_14")
    if "vol_adj_carry" in frame.columns and "low_vol_14" in frame.columns:
        frame["carry_lowvol_z"] = _cross_sectional_z(frame, "vol_adj_carry") + _cross_sectional_z(frame, "low_vol_14")
    return frame


def available_candidates(frame):
    return [name for name in SIGNAL_CANDIDATES if name in frame.columns and frame[name].notna().sum() > 0]


def warmup_days(signal_name):
    if signal_name in ("reversal_1", "carry"):
        return 1
    if signal_name == "funding_surprise":
        return 15
    if signal_name == "vol_adj_carry":
        return 15
    if signal_name == "carry_mom_z":
        return 31
    if signal_name.startswith("resid_reversal_"):
        return int(signal_name.rsplit("_", 1)[-1])
    if signal_name.startswith("low_vol_"):
        return int(signal_name.rsplit("_", 1)[-1])
    if signal_name.startswith(("momentum_", "vol_adj_momentum_")):
        return int(signal_name.rsplit("_", 1)[-1])
    raise ValueError(f"Unknown signal: {signal_name}")
