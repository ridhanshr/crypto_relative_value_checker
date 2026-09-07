"""Dataset repair primitives: dirty panel -> verifiable clean panel.

Three fail-closed operations, each returning (frame, log) so every dropped
or relabeled row is accounted for in a manifest -- never silent:

* ``relabel_migration_history``: pre-cutover rows carrying the TARGET ticker
  (GAL history labeled GUSDT, OMNI history labeled NOMUSDT) are restored to
  the source ticker, so official ratio machinery applies. Cutoffs are
  explicit arguments (first real print of the target, cross-checked against
  the historical listing manifest), never guessed here.
* ``drop_stale_rows``: frozen prints with no volume (halt artifacts the
  exchange keeps publishing) are removed. Precedent: volume<=0 is not a
  trade and must not anchor returns.
* ``assert_no_residual_duplicates``: anything still duplicated on
  (timestamp, asset) after the above is a hard refusal with samples --
  canonicalize_assets() would otherwise silently aggregate (summing
  funding twice).

``repair_panel`` chains the three and returns (repaired_frame, log) for a
one-call dirty -> clean path used by scripts and tests.
"""

import pandas as pd


def _upper_asset(frame):
    return frame["asset"].astype(str).str.upper()


def relabel_migration_history(data, cutovers):
    """Relabel pre-cutover target-ticker rows to their source ticker.

    ``cutovers``: mapping target symbol -> (source symbol, cutover instant);
    rows with timestamp < cutover are renamed to source. Returns (frame, log).
    """
    frame = data.copy()
    frame["_ts"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    total = 0
    applied = []
    for target, (source, cutover) in cutovers.items():
        moment = pd.Timestamp(cutover, tz="UTC")
        mask = (_upper_asset(frame) == str(target).upper()) & (frame["_ts"] < moment)
        count = int(mask.sum())
        frame.loc[mask, "asset"] = str(source).upper()
        total += count
        applied.append({"target": str(target).upper(), "source": str(source).upper(), "cutover": str(moment), "rows": count})
    frame = frame.drop(columns=["_ts"])
    return frame.reset_index(drop=True), {"relabel_migration_history": {"rows": total, "applied": applied}}


def drop_stale_rows(data):
    """Drop frozen prints with no trade behind them (volume<=0 / NaN)."""
    frame = data.copy()
    vol = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(float("nan"), index=frame.index)
    stale = vol.isna() | ~(vol > 0)
    dropped_assets = _upper_asset(frame.loc[stale]).value_counts().to_dict() if stale.any() else {}
    frame = frame.loc[~stale].reset_index(drop=True)
    return frame, {"drop_stale_rows": {"rows": int(stale.sum()), "assets": dropped_assets}}


def assert_no_residual_duplicates(data):
    """Hard refusal on surviving (timestamp, asset) duplicates with samples."""
    dup = data.duplicated(["timestamp", "asset"], keep=False)
    if dup.any():
        sample = data.loc[dup, [c for c in ("timestamp", "asset", "price", "volume") if c in data.columns]].head(20).to_dict("records")
        raise ValueError(f"REPAIR_REFUSED: {int(dup.sum())} residual (timestamp, asset) duplicates: {sample}")
    return data


def repair_panel(data, cutovers, drop_stale=True):
    """Dirty -> clean: relabel, drop stale, refuse on residual duplicates."""
    log = {"input_rows": int(len(data))}
    frame, relabel_log = relabel_migration_history(data, cutovers)
    log.update(relabel_log)
    if drop_stale:
        frame, stale_log = drop_stale_rows(frame)
        log.update(stale_log)
    frame = assert_no_residual_duplicates(frame)
    log["output_rows"] = int(len(frame))
    return frame.reset_index(drop=True), log
