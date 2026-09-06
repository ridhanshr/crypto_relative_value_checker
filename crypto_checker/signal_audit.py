from pathlib import Path
import pandas as pd


def audit_signals(data, output_dir=None, column="signal"):
    required = {"timestamp", "asset", "price", column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["price"] = pd.to_numeric(frame["price"], errors="raise")
    frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    frame["price_return"] = frame.groupby("asset")["price"].pct_change()
    frame["signal_change"] = frame.groupby("asset")[column].diff()
    frame["signal_is_forward_fill"] = frame["signal_change"].eq(0) & frame.groupby("asset")[column].shift().notna()
    stats = frame.groupby("asset").agg(rows=(column, "size"), unique_signal=(column, "nunique"), signal_std=(column, "std"), signal_min=(column, "min"), signal_max=(column, "max"), zero_changes=("signal_change", lambda x: int(x.eq(0).sum())), forward_fill_rows=("signal_is_forward_fill", "sum"), cumulative_signal_check=(column, lambda x: int(x.abs().max() > 10)))
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path / "signal_audit.csv", index=False)
        stats.reset_index().to_csv(path / "signal_audit_stats.csv", index=False)
    if (stats.unique_signal <= 1).any() or (stats.forward_fill_rows > 0).any() or (stats.cumulative_signal_check > 0).any():
        raise ValueError(f"Signal audit failed on column {column}: constant, forward-filled, or cumulative signal detected")
    return frame, stats
