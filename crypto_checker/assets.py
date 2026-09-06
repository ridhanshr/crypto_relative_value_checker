import numpy as np
import pandas as pd


MIGRATION_MAP = {
    "GAL": "G",
    "OMNI": "NOM",
    "MATIC": "POL",
    "NANO": "XNO",
    "VEN": "VET",
    "BCC": "BCH",
    "ANTOLD": "ANT",
}

MIGRATION_METADATA = {
    "GAL": {"canonical": "G", "ratio": "1 GAL = 60 G", "reason": "Galxe to Gravity", "effective_date": "2024-07-19T08:00:00Z", "source_url": "https://www.binance.com/en/support/announcement/detail/6df074e0264648448c260b1430240e2f", "announcement_id": "6df074e0264648448c260b1430240e2f"},
    "OMNI": {"canonical": "NOM", "ratio": "1 OMNI = 75 NOM", "reason": "Omni to Nomina", "effective_date": "2025-10-01T08:00:00Z", "source_url": "https://www.binance.com/en-IN/support/announcement/detail/13d0a79f79ef4406963dcd7f9d1de325", "announcement_id": "13d0a79f79ef4406963dcd7f9d1de325"},
    "MATIC": {"canonical": "POL", "ratio": "1 MATIC = 1 POL", "reason": "MATIC to POL", "effective_date": "2024-09-13T10:00:00Z", "source_url": "https://www.binance.com/de/support/announcement/detail/6a6de383727f4659a3050f7982e1620f", "announcement_id": "6a6de383727f4659a3050f7982e1620f"},
    "NANO": {"canonical": "XNO", "ratio": "1 NANO = 1 XNO", "reason": "Nano ticker migration", "effective_date": "2022-01-28T04:00:00Z", "source_url": "https://www.binance.com/en-TR/support/announcement/detail/3dc8f6de281f4781a246a1658a21cb80", "announcement_id": "3dc8f6de281f4781a246a1658a21cb80"},
    "VEN": {"canonical": "VET", "ratio": "1 VEN = 100 VET", "reason": "VeChain ticker migration", "effective_date": "2018-07-25T04:00:00Z", "source_url": "https://www.binance.com/en-NG/support/announcement/detail/360007439672", "announcement_id": "360007439672"},
    "BCC": {"canonical": "BCH", "reason": "Bitcoin Cash ticker migration"},
    "ANTOLD": {"canonical": "ANT", "reason": "Aragon legacy ticker migration"},
}

PRICE_ADJUSTMENT_FACTORS = {"GAL": 60.0, "OMNI": 75.0, "MATIC": 1.0, "NANO": 1.0, "VEN": 100.0, "BCC": None, "ANTOLD": None}
MIGRATION_EFFECTIVE_DATES = {key: value.get("effective_date") for key, value in MIGRATION_METADATA.items()}


def canonical_asset(symbol):
    value = str(symbol).upper()
    quote = ""
    for suffix in ("USDT", "USDC", "BUSD", "BTC", "ETH"):
        if value.endswith(suffix):
            value, quote = value[:-len(suffix)], suffix
            break
    value = MIGRATION_MAP.get(value, value)
    return value + quote


def migration_manifest():
    return [{"source_token": old, "canonical_token": new, "price_adjustment_factor": PRICE_ADJUSTMENT_FACTORS[old], **MIGRATION_METADATA[old]} for old, new in MIGRATION_MAP.items()]


# Canonical tokens (without quote suffix) that went through a migration,
# mapped to their official effective date. Used to tell migration-related
# funding gaps apart from genuinely missing funding on active assets.
MIGRATION_WINDOW_DAYS = 7

_MIGRATION_CANONICAL_EFFECTIVE = {}
for _old, _new in MIGRATION_MAP.items():
    _eff = MIGRATION_METADATA.get(_old, {}).get("effective_date")
    if _eff is not None:
        _MIGRATION_CANONICAL_EFFECTIVE[_new] = pd.Timestamp(_eff, tz="UTC")


def _base_token(symbol):
    value = str(symbol).upper()
    for suffix in ("USDT", "USDC", "BUSD", "BTC", "ETH"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def classify_funding_gaps(data):
    """Classify every (timestamp, asset) row with missing funding.

    Returns a DataFrame with columns timestamp, asset, reason, where reason
    is one of:
      - "not_listed": before the asset's first observed funding print.
      - "delisted": after the asset's last observed funding print.
      - "migration": within +/-MIGRATION_WINDOW_DAYS of the official
        effective date of a migrated canonical token.
      - "active": funding exists both before and after this row, i.e. the
        asset was tradable but funding is missing -> investigate.

    An empty DataFrame (with the same columns) is returned when there is no
    funding column or no missing funding.
    """
    cols = ["timestamp", "asset", "reason"]
    empty = pd.DataFrame(columns=cols)
    if "funding_rate" not in data.columns:
        return empty
    if "asset" not in data.columns or "timestamp" not in data.columns:
        # Let canonicalize_assets/_validate raise the proper schema error.
        return empty
    frame = data.copy()
    # Map to canonical names WITHOUT aggregating: canonicalize_assets()
    # would sum duplicate funding rows (turning all-NaN groups into 0.0),
    # which would hide gaps. Mapping alone is idempotent.
    frame["asset"] = frame["asset"].astype(str).map(canonical_asset)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    rates = pd.to_numeric(frame["funding_rate"], errors="coerce")
    missing = frame[rates.isna()].copy()
    if missing.empty:
        return empty
    have = frame[rates.notna()].copy()
    first_seen = have.groupby("asset")["timestamp"].min().to_dict() if not have.empty else {}
    last_seen = have.groupby("asset")["timestamp"].max().to_dict() if not have.empty else {}
    if "price" in frame.columns:
        price_ok = pd.to_numeric(frame["price"], errors="coerce")
        price_ok = price_ok.notna() & np.isfinite(price_ok) & (price_ok > 0)
        price_ok = price_ok.reindex(missing.index, fill_value=False)
    else:
        price_ok = pd.Series(False, index=missing.index)
    vol_cols = [c for c in ("volume", "quote_volume") if c in frame.columns]
    if vol_cols:
        vol = pd.Series(np.nan, index=frame.index)
        for col in vol_cols:
            col_vol = pd.to_numeric(frame[col], errors="coerce")
            vol = vol.fillna(col_vol)
        vol_ok = vol.notna() & np.isfinite(vol) & (vol > 0)
        vol_ok = vol_ok.reindex(missing.index, fill_value=False)
    else:
        # No volume column at all: cannot assess activity from volume,
        # so price alone decides (documented limitation).
        vol_ok = pd.Series(True, index=missing.index)
    rows = []
    window = pd.Timedelta(days=MIGRATION_WINDOW_DAYS)
    no_coverage_reference = not first_seen
    for idx, row in missing.iterrows():
        ts, asset = row["timestamp"], row["asset"]
        if pd.isna(ts):
            reason = "active" if bool(price_ok.get(idx, False)) else "delisted"
            rows.append({"timestamp": row["timestamp"] if "timestamp" in row else ts, "asset": asset, "reason": reason})
            continue
        base = _base_token(asset)
        eff = _MIGRATION_CANONICAL_EFFECTIVE.get(base)
        if eff is not None and abs(ts - eff) <= window:
            reason = "migration"
        elif no_coverage_reference:
            # No funding prints exist at all: cannot establish listing
            # coverage, so anything tradable must be investigated.
            reason = "active" if bool(price_ok.get(idx, False)) else "delisted"
        elif asset not in first_seen:
            reason = "not_listed"
        elif ts < first_seen[asset]:
            reason = "not_listed"
        elif ts > last_seen[asset]:
            reason = "delisted"
        elif not bool(price_ok.get(idx, False)) or not bool(vol_ok.get(idx, False)):
            reason = "delisted"
        else:
            reason = "active"
        rows.append({"timestamp": ts, "asset": asset, "reason": reason})
    return pd.DataFrame(rows, columns=cols)


def apply_price_adjustments(data, factors=None):
    frame = data.copy()
    factors = factors or PRICE_ADJUSTMENT_FACTORS
    frame["price_adjustment_factor"] = 1.0
    for old in MIGRATION_MAP:
        mask = frame["source_asset"].str.removesuffix("USDT") == old
        factor = factors.get(old)
        effective = factors.get(f"{old}:effective_date", MIGRATION_EFFECTIVE_DATES.get(old)) if isinstance(factors, dict) else MIGRATION_EFFECTIVE_DATES.get(old)
        if factor is not None and effective is not None:
            effective = pd.Timestamp(effective, tz="UTC")
            mask = mask & (pd.to_datetime(frame["timestamp"], utc=True) < effective)
            frame.loc[mask, "price"] = frame.loc[mask, "price"] / float(factor)
            frame.loc[mask, "price_adjustment_factor"] = float(factor)
    return frame


def audit_migration_discontinuities(data, tolerance=0.20):
    frame = data.sort_values(["asset", "timestamp"]).copy()
    rows = []
    for asset, group in frame.groupby("asset"):
        sources = group["source_asset"].drop_duplicates().tolist()
        for source in sources[1:]:
            current = group[group["source_asset"] == source]
            previous = group[group.index < current.index[0]]["price"]
            if not current.empty and not previous.empty:
                jump = float(current.iloc[0].price / previous.iloc[-1] - 1)
                if abs(jump) > tolerance:
                    rows.append({"asset": asset, "source_asset": source, "timestamp": current.iloc[0].timestamp, "price_jump": jump, "status": "REQUIRES_OFFICIAL_FACTOR"})
    return pd.DataFrame(rows, columns=["asset", "source_asset", "timestamp", "price_jump", "status"])


def audit_migration_collisions(data):
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["source_asset"] = frame["asset"].astype(str).str.upper()
    frame["asset"] = frame["source_asset"].map(canonical_asset)
    grouped = frame.groupby(["timestamp", "asset"])["source_asset"].unique()
    rows = [{"timestamp": ts, "asset": asset, "source_assets": ",".join(sorted(sources)), "status": "MIGRATION_COLLISION"} for (ts, asset), sources in grouped.items() if len(sources) > 1]
    return pd.DataFrame(rows, columns=["timestamp", "asset", "source_assets", "status"])


def canonicalize_assets(data):
    frame = data.copy()
    if "asset" not in frame.columns:
        raise ValueError("Missing columns: ['asset']")
    frame["source_asset"] = frame["asset"].astype(str).str.upper()
    frame["asset"] = frame["source_asset"].map(canonical_asset)
    frame = apply_price_adjustments(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    duplicate_keys = ["timestamp", "asset"]
    if frame.duplicated(duplicate_keys).any():
        numeric = [c for c in ("open", "high", "low", "price", "volume", "quote_volume", "funding_rate") if c in frame]
        aggregations = {c: "last" for c in numeric}
        if "funding_rate" in aggregations:
            aggregations["funding_rate"] = "sum"
        keep = [c for c in frame.columns if c not in numeric]
        frame = frame.groupby(duplicate_keys, as_index=False).agg({**aggregations, **{c: "last" for c in keep if c not in duplicate_keys}})
    frame = frame.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    frame["asset_return"] = frame.groupby("asset")["price"].pct_change()
    if "signal" not in frame.columns:
        frame["signal"] = frame["asset_return"]
    return frame


def audit_asset_continuity(data, expected_frequency="D"):
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    gaps = []
    for asset, group in frame.groupby("asset"):
        dates = group["timestamp"].drop_duplicates().sort_values()
        if len(dates) < 2:
            continue
        expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=expected_frequency, tz="UTC")
        missing = expected.difference(dates)
        gaps.extend({"asset": asset, "timestamp": ts} for ts in missing)
    return pd.DataFrame(gaps, columns=["asset", "timestamp"])
