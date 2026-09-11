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


def active_assets(lifecycle, ts):
    """Canonical assets tradable at instant ``ts`` (point-in-time universe)."""
    moment = pd.Timestamp(ts, tz="UTC")
    frame = _normalize_lifecycle(lifecycle)
    live = frame[(frame["listed_at"] <= moment) & (frame["delisted_at"].isna() | (moment < frame["delisted_at"]))]
    return sorted(live["canonical"].unique().tolist())


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

    Matching is SYMBOL-level, not canonical-level: a row is compliant iff a
    segment exists for its exact source symbol covering the timestamp. This
    is what catches mislabeled migration history (GAL prices labeled GUSDT
    before G lists) -- canonical-level matching would wave it through.
    ``source_asset`` (kept by canonicalize_assets) is honored when present,
    else the asset column itself is the source.

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
        ts, canonical, source = row["timestamp"], row["canonical"], row["source_asset"]
        segs = lifecycle[lifecycle["canonical"] == canonical]
        asset_label = row["asset"] if "asset" in frame.columns else canonical
        if segs.empty:
            rows.append({"timestamp": ts, "asset": asset_label, "reason": "no_lifecycle_segment"})
            continue
        segs_sym = segs[segs["symbol"].astype(str).str.upper() == source]
        pool = segs_sym if not segs_sym.empty else segs
        live = pool[(pool["listed_at"] <= ts) & (pool["delisted_at"].isna() | (ts < pool["delisted_at"]))]
        if live.empty:
            if (pool["listed_at"] > ts).any():
                reason = "trading_before_listing"
            else:
                reason = "trading_after_delisting"
            rows.append({"timestamp": ts, "asset": asset_label, "reason": reason})
    return pd.DataFrame(rows, columns=cols)


def classify_timestamp_gaps(data, expected_frequency="D", adjacency_days=7):
    """Classify contiguous blocks of missing asset-periods.

    Returns DataFrame [asset, gap_start, gap_end, gap_days, reason] with
    reason in:
      - "migration_halt": the missing block contains or adjoins (within
        ``adjacency_days``) the official effective date of that canonical
        token's migration -- a documented exchange halt, safe for research
        and futures alike (positions are force-exited, never carried).
      - "unexplained_interior": a hole with no migration event nearby --
        genuine missing data. The ONLY gap class that must fail a futures
        run.

    Listing/delisting edges never appear here by construction (the expected
    range runs first-seen -> last-seen per asset); they are covered by
    lifecycle segments, not by this classifier.
    """
    cols = ["asset", "gap_start", "gap_end", "gap_days", "reason"]
    frame = data.copy()
    if "asset" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("Gap classification needs 'asset' and 'timestamp' columns")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["canonical"] = frame["asset"].astype(str).str.upper().map(canonical_asset)
    effective = {}
    for old, meta in MIGRATION_METADATA.items():
        eff = meta.get("effective_date")
        if eff is not None:
            effective[meta["canonical"]] = pd.Timestamp(eff, tz="UTC")
    adjacency = pd.Timedelta(days=adjacency_days)
    rows = []
    try:
        offset = pd.tseries.frequencies.to_offset(expected_frequency)
        step = pd.Timedelta(offset.nanos, unit="ns")
    except Exception:
        step = pd.Timedelta(days=1)
    for canonical, group in frame.groupby("canonical"):
        dates = group["timestamp"].drop_duplicates().sort_values()
        if len(dates) < 2:
            continue
        expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=expected_frequency, tz="UTC")
        missing = expected.difference(dates)
        if missing.empty:
            continue
        # Group consecutive missing periods into blocks.
        blocks = []
        start = prev = missing[0]
        for ts in missing[1:]:
            if ts - prev <= step * 1.5:
                prev = ts
                continue
            blocks.append((start, prev))
            start = prev = ts
        blocks.append((start, prev))
        eff = effective.get(_base_token(canonical))
        for start, end in blocks:
            days = int(round((end - start) / step)) + 1
            if eff is not None and (start - adjacency) <= eff <= (end + adjacency):
                reason = "migration_halt"
            else:
                reason = "unexplained_interior"
            rows.append({"asset": canonical, "gap_start": start, "gap_end": end, "gap_days": days, "reason": reason})
    return pd.DataFrame(rows, columns=cols)


def lifecycle_from_listing_manifest(manifest):
    """Adapt a historical listing manifest into lifecycle segments.

    Input columns: symbol, listed_at, delisted_at (None/NaT = still live).
    Symbols are canonicalized (GALUSDT + GUSDT -> one GUSDT canonical with
    two segments), so the manifest plugs straight into
    ``audit_universe_compliance`` and ``active_assets``. Segment sources are
    labeled ``listing_manifest:<source>`` to preserve the inferred/official
    distinction of the manifest itself.
    """
    frame = pd.DataFrame(manifest).copy()
    missing = [c for c in ("symbol", "listed_at") if c not in frame.columns]
    if missing:
        raise ValueError(f"Listing manifest missing columns: {missing}")
    if "delisted_at" not in frame.columns:
        frame["delisted_at"] = pd.NaT
    if "source" not in frame.columns:
        frame["source"] = "unknown"
    frame["listed_at"] = pd.to_datetime(frame["listed_at"], utc=True, errors="coerce")
    frame["delisted_at"] = pd.to_datetime(frame["delisted_at"], utc=True, errors="coerce")
    if frame["listed_at"].isna().any():
        raise ValueError("Listing manifest has unparseable listed_at values")
    frame["canonical"] = frame["symbol"].astype(str).str.upper().map(canonical_asset)
    rows = []
    for _, row in frame.iterrows():
        alive = pd.isna(row["delisted_at"])
        rows.append({
            "canonical": row["canonical"],
            "symbol": str(row["symbol"]).upper(),
            "listed_at": row["listed_at"],
            "delisted_at": None if alive else row["delisted_at"],
            "event": "active" if alive else "delisting",
            "source": f"listing_manifest:{row['source']}",
        })
    return pd.DataFrame(rows, columns=LIFECYCLE_COLUMNS).sort_values(["canonical", "listed_at"]).reset_index(drop=True)


def measure_survivorship_gap(data, manifest):
    """Measure survivorship bias EXPLICITLY as a number, not a note.

    Compares the dataset's canonical universe against every manifest symbol
    whose [listed_at, delisted_at] overlaps the dataset window:

    * ``missing_dead``: tradable in-window but dead/absent from the data --
      the bias carriers (their crashes never enter the cross-section).
    * ``missing_alive``: tradable in-window, still alive, but not in the
      dataset -- a coverage choice, not bias (e.g. midcap scope).
    * ``coverage_ratio``: dataset assets / manifest assets in-window.

    A cross-sectional strategy picking "top volume" in-window while its
    dataset silently drops the dead is flattered by exactly the names in
    ``missing_dead``.
    """
    frame = pd.DataFrame(manifest).copy()
    frame["listed_at"] = pd.to_datetime(frame["listed_at"], utc=True, errors="coerce")
    frame["delisted_at"] = pd.to_datetime(frame["delisted_at"], utc=True, errors="coerce")
    data_ts = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    start, end = data_ts.min(), data_ts.max()
    have = set(data["asset"].astype(str).str.upper().map(canonical_asset).unique().tolist())
    in_window = frame[(frame["listed_at"] <= end) & (frame["delisted_at"].isna() | (frame["delisted_at"] >= start))].copy()
    in_window["canonical"] = in_window["symbol"].astype(str).str.upper().map(canonical_asset)
    missing = in_window[~in_window["canonical"].isin(have)].copy()
    dead = missing[missing["delisted_at"].notna()].copy()
    alive = missing[missing["delisted_at"].isna()].copy()
    dead["overlap_days"] = (
        dead["delisted_at"].clip(upper=end) - dead["listed_at"].clip(lower=start)
    ).dt.total_seconds().div(86400).clip(lower=0).add(1)
    universe = sorted(in_window["canonical"].unique().tolist())
    return {
        "window_start": str(start),
        "window_end": str(end),
        "n_manifest_in_window": int(len(universe)),
        "n_dataset": int(len(have)),
        "coverage_ratio": float(len(set(have) & set(universe)) / max(len(universe), 1)),
        "n_missing_dead": int(dead["canonical"].nunique()),
        "missing_dead_asset_days": int(dead["overlap_days"].sum()) if not dead.empty else 0,
        "missing_dead": sorted(dead["canonical"].unique().tolist()),
        "n_missing_alive": int(alive["canonical"].nunique()),
        "missing_alive": sorted(alive["canonical"].unique().tolist()),
        "note": "missing_dead carried crashes the dataset never sees; cross-sectional returns measured without them are flattered by construction",
    }
