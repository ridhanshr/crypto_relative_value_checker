"""Asset lifecycle engine: listed_at / delisted_at / migration segments.

Single source of truth for "which canonical assets were tradable at time t".

A lifecycle is a table with one row per (canonical asset, source symbol)
segment::

    canonical | symbol | listed_at | delisted_at | event | source

* ``listed_at``: first tradable instant (inclusive).
* ``delisted_at``: first NON-tradable instant (exclusive). ``None``/``NaT``
  means still tradable at the end of the observed data.
* ``event``: migration_source | migration_target | listing | delisting |
  active.
* ``source``: "official" (exchange announcement effective date from
  ``MIGRATION_METADATA``) or "inferred_from_data" (first/last seen in the
  dataset -- a placeholder that MUST be replaced with the official listing
  date before any survivorship-free claim).

``build_lifecycle`` derives segments from a dataset (descriptive manifest).
``audit_universe_compliance`` checks any dataset against any lifecycle --
most usefully against a hand-maintained OFFICIAL manifest -- and flags rows
trading outside their segment. ``active_assets`` answers the point-in-time
membership question the backtest relies on (``core`` ranks only assets
present at bar t, which is lifecycle membership derived from data).
"""

import pandas as pd

from .assets import MIGRATION_MAP, MIGRATION_METADATA, canonical_asset, _base_token

OFFICIAL = "official"
INFERRED = "inferred_from_data"

LIFECYCLE_COLUMNS = ["canonical", "symbol", "listed_at", "delisted_at", "event", "source"]


def _migration_effective(source_base):
    meta = MIGRATION_METADATA.get(source_base, {})
    eff = meta.get("effective_date")
    return pd.Timestamp(eff, tz="UTC") if eff is not None else None


def build_lifecycle(data, delist_buffer_days=7):
    """Derive lifecycle segments from observed (timestamp, asset) coverage.

    Migration boundaries come from official effective dates; listing dates
    and delisting dates are inferred from first/last seen prints and MUST be
    treated as placeholders (see module docstring).
    """
    frame = data.copy()
    if "asset" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("Lifecycle needs 'asset' and 'timestamp' columns")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if "source_asset" in frame.columns:
        frame["source_asset"] = frame["source_asset"].astype(str).str.upper()
    else:
        frame["source_asset"] = frame["asset"].astype(str).str.upper()
    frame["canonical"] = frame["source_asset"].map(canonical_asset)
    data_end = frame["timestamp"].max()
    buffer = pd.Timedelta(days=delist_buffer_days)
    rows = []
    for canonical, group in frame.groupby("canonical"):
        for source, seg in group.groupby("source_asset"):
            first_seen = seg["timestamp"].min()
            last_seen = seg["timestamp"].max()
            step = seg["timestamp"].drop_duplicates().sort_values().diff().median()
            if pd.isna(step) or step <= pd.Timedelta(0):
                step = pd.Timedelta(days=1)
            source_base = _base_token(source)
            effective = _migration_effective(source_base)
            if source_base in MIGRATION_MAP and effective is not None:
                rows.append({
                    "canonical": canonical,
                    "symbol": source,
                    "listed_at": first_seen,
                    "delisted_at": effective,
                    "event": "migration_source",
                    "source": f"delisted:{OFFICIAL};listed:{INFERRED}",
                })
            else:
                target_of = [old for old, new in MIGRATION_MAP.items() if new == _base_token(source)]
                if target_of:
                    eff = _migration_effective(target_of[0])
                    listed, listed_src = (eff, OFFICIAL) if eff is not None else (first_seen, INFERRED)
                    event = "migration_target"
                else:
                    listed, listed_src = first_seen, INFERRED
                    event = "listing"
                if last_seen < data_end - buffer:
                    rows.append({
                        "canonical": canonical,
                        "symbol": source,
                        "listed_at": listed,
                        "delisted_at": last_seen + step,
                        "event": "delisting" if event == "listing" else event,
                        "source": f"listed:{listed_src};delisted:{INFERRED}",
                    })
                else:
                    rows.append({
                        "canonical": canonical,
                        "symbol": source,
                        "listed_at": listed,
                        "delisted_at": None,
                        "event": event if event != "listing" else "active",
                        "source": f"listed:{listed_src};delisted:still_tradable",
                    })
    lifecycle = pd.DataFrame(rows, columns=LIFECYCLE_COLUMNS)
    return lifecycle.sort_values(["canonical", "listed_at"]).reset_index(drop=True)


def _normalize_lifecycle(lifecycle):
    """Coerce lifecycle timestamps to tz-aware UTC (fail-closed on garbage).

    Hand-maintained manifests often carry tz-naive columns (especially
    ``delisted_at`` full of NaT); without normalization every comparison
    against tz-aware bar timestamps explodes.
    """
    frame = pd.DataFrame(lifecycle, columns=LIFECYCLE_COLUMNS).copy()
    for col in ("listed_at", "delisted_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    if frame["listed_at"].isna().any():
        raise ValueError("Lifecycle manifest has unparseable listed_at values")
    return frame


def active_assets(lifecycle, ts):
    """Canonical assets tradable at instant ``ts`` (point-in-time universe)."""
    moment = pd.Timestamp(ts, tz="UTC")
    frame = _normalize_lifecycle(lifecycle)
    live = frame[(frame["listed_at"] <= moment) & (frame["delisted_at"].isna() | (moment < frame["delisted_at"]))]
    return sorted(live["canonical"].unique().tolist())


def write_lifecycle_manifest(lifecycle, path):
    frame = pd.DataFrame(lifecycle, columns=LIFECYCLE_COLUMNS)
    frame.to_csv(path, index=False)
    return str(path)


def load_lifecycle_manifest(path):
    frame = pd.read_csv(path)
    missing = [c for c in LIFECYCLE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Lifecycle manifest missing columns: {missing}")
    for col in ("listed_at", "delisted_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    return frame[LIFECYCLE_COLUMNS]


def audit_universe_compliance(data, lifecycle):
    """Flag (timestamp, asset) rows trading outside their lifecycle segment.

    Returns DataFrame [timestamp, asset, reason] with reason in
    ("trading_before_listing", "trading_after_delisting",
    "no_lifecycle_segment"). Empty means fully compliant.
    """
    cols = ["timestamp", "asset", "reason"]
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "source_asset" in frame.columns:
        frame["source_asset"] = frame["source_asset"].astype(str).str.upper()
    else:
        frame["source_asset"] = frame["asset"].astype(str).str.upper()
    frame["canonical"] = frame["source_asset"].map(canonical_asset)
    lifecycle = _normalize_lifecycle(lifecycle)
    rows = []
    for _, row in frame.iterrows():
        ts, canonical = row["timestamp"], row["canonical"]
        segs = lifecycle[lifecycle["canonical"] == canonical]
        if segs.empty:
            rows.append({"timestamp": ts, "asset": row["asset"] if "asset" in frame.columns else canonical, "reason": "no_lifecycle_segment"})
            continue
        live = segs[(segs["listed_at"] <= ts) & (segs["delisted_at"].isna() | (ts < segs["delisted_at"]))]
        if live.empty:
            if (segs["listed_at"] > ts).any():
                reason = "trading_before_listing"
            else:
                reason = "trading_after_delisting"
            rows.append({"timestamp": ts, "asset": row["asset"] if "asset" in frame.columns else canonical, "reason": reason})
    return pd.DataFrame(rows, columns=cols)
