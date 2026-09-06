from pathlib import Path
import json
import numpy as np
import pandas as pd
from .assets import canonicalize_assets, audit_migration_discontinuities


def validate_dataset(data, require_funding=False, require_liquidity=False, min_assets=2, min_periods=2, expected_frequency=None):
    errors = []
    warnings = []
    required = {"timestamp", "asset", "price", "signal"}
    missing = sorted(required - set(data.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    frame = data.copy()
    if frame.empty:
        return {"valid": False, "errors": ["Dataset is empty"], "warnings": []}
    try:
        raw_ts = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        raw_asset = frame["asset"].astype(str).str.upper()
        if pd.DataFrame({"timestamp": raw_ts, "asset": raw_asset}).duplicated().any():
            errors.append("Duplicate (timestamp, asset) rows")
    except Exception as exc:
        errors.append(f"Invalid timestamp: {exc}")
    if require_funding:
        if "funding_rate" not in frame.columns:
            errors.append("Funding rate column required but missing")
        else:
            raw_funding = pd.to_numeric(frame["funding_rate"], errors="coerce")
            if raw_funding.isna().any() or not np.isfinite(raw_funding).all():
                errors.append("Funding rate contains NaN, non-numeric, or infinite values")
    try:
        frame = canonicalize_assets(frame)
    except Exception as exc:
        errors.append(f"Asset normalization failed: {exc}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    migration_gaps = audit_migration_discontinuities(frame)
    if not migration_gaps.empty:
        errors.append(f"Migration price discontinuities require official factors: {len(migration_gaps)}")
        warnings.extend(migration_gaps.astype(str).to_dict("records"))
    if frame["timestamp"].duplicated().any():
        warnings.append("Some timestamps contain multiple assets; this is expected for panel data")
    if frame.duplicated(["timestamp", "asset"]).any():
        errors.append("Duplicate (timestamp, asset) rows")
    if not np.isfinite(frame["price"]).all() or (frame["price"] <= 0).any():
        errors.append("Price contains non-finite or non-positive values")
    signal = pd.to_numeric(frame["signal"], errors="coerce")
    if signal.isna().any() or not np.isfinite(signal).all():
        errors.append("Signal contains NaN, non-numeric, or infinite values")
    if require_funding:
        if "funding_rate" not in frame.columns:
            errors.append("Funding rate column required but missing")
        else:
            funding = pd.to_numeric(frame["funding_rate"], errors="coerce")
            if funding.isna().any() or not np.isfinite(funding).all():
                errors.append("Funding rate contains NaN, non-numeric, or infinite values")
    if require_liquidity and "quote_volume" not in frame.columns:
        errors.append("quote_volume required for liquidity-aware costs")
    assets = int(frame["asset"].nunique())
    periods = int(frame["timestamp"].nunique())
    if assets < min_assets:
        errors.append(f"Asset count {assets} below minimum {min_assets}")
    if periods < min_periods:
        errors.append(f"Period count {periods} below minimum {min_periods}")
    counts = frame.groupby("asset")["timestamp"].nunique()
    if len(counts) and counts.min() != counts.max():
        warnings.append(f"Unequal asset coverage: min={int(counts.min())}, max={int(counts.max())}")
    if expected_frequency:
        gaps = []
        for asset, group in frame.groupby("asset"):
            dates = group["timestamp"].drop_duplicates().sort_values()
            expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=expected_frequency, tz="UTC")
            gaps.append(int(len(expected.difference(dates))))
        if sum(gaps):
            errors.append(f"Timestamp gaps detected: {sum(gaps)} missing asset-periods")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "rows": int(len(frame)), "assets": assets, "periods": periods, "start": str(frame["timestamp"].min()), "end": str(frame["timestamp"].max()), "funding_present": "funding_rate" in frame.columns, "liquidity_present": "quote_volume" in frame.columns}


def write_preflight(result, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
