from pathlib import Path
import json
import numpy as np
import pandas as pd
from .assets import canonicalize_assets, audit_migration_discontinuities, audit_migration_collisions, classify_funding_gaps
from .lifecycle import build_lifecycle, classify_timestamp_gaps, measure_survivorship_gap, INFERRED
from .schema import SCHEMA_VERSION
from .io import atomic_write_json


def validate_dataset(data, require_funding=False, require_liquidity=False, min_assets=2, min_periods=2, expected_frequency=None, allow_gaps=False, listing_manifest=None, tolerated_gap_days=1):
    errors = []
    warnings = []
    required = {"timestamp", "asset", "price", "signal"}
    missing = sorted(required - set(data.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")
        return {"valid": False, "schema_version": SCHEMA_VERSION, "errors": errors, "warnings": warnings}
    frame = data.copy()
    if frame.empty:
        return {"valid": False, "schema_version": SCHEMA_VERSION, "errors": ["Dataset is empty"], "warnings": []}
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
            if not np.isfinite(raw_funding.dropna()).all():
                errors.append("Funding rate contains non-numeric or infinite values")
    try:
        frame = canonicalize_assets(frame)
    except Exception as exc:
        errors.append(f"Asset normalization failed: {exc}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    collisions = audit_migration_collisions(data)
    if not collisions.empty:
        errors.append(f"Migration source collision detected: {len(collisions)} timestamp/asset rows")
        warnings.extend(collisions.astype(str).to_dict("records"))
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
    # Funding gaps are classified on raw data with canonical mapping applied
    # inside the classifier (no aggregation), so duplicate collapsing in
    # canonicalize_assets() cannot mask them. Missing funding never blocks
    # ranking validity: not-listed/delisted/migration gaps are ignored
    # (funding treated as 0.0 post-ranking); gaps on otherwise active rows
    # only warn for investigation.
    funding_gaps = classify_funding_gaps(data)
    if require_funding and "funding_rate" in frame.columns:
        funding = pd.to_numeric(frame["funding_rate"], errors="coerce")
        if not np.isfinite(funding.dropna()).all():
            errors.append("Funding rate contains non-numeric or infinite values")
        for reason in ("not_listed", "delisted", "migration"):
            count = int((funding_gaps["reason"] == reason).sum()) if not funding_gaps.empty else 0
            if count:
                warnings.append(f"Ignored {count} missing-funding rows ({reason}); funding treated as 0.0 post-ranking")
        active_gaps = funding_gaps[funding_gaps["reason"] == "active"] if not funding_gaps.empty else funding_gaps
        if not active_gaps.empty:
            sample = active_gaps[["timestamp", "asset"]].astype(str).head(10).to_dict("records")
            warnings.append(f"FUNDING_GAP_ACTIVE: {len(active_gaps)} rows have price+volume but no funding (treated as 0.0); investigate: {sample}")
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
    statuses = frame.groupby("asset").agg(first_seen=("timestamp", "min"), last_seen=("timestamp", "max"), periods=("timestamp", "nunique")).reset_index()
    end = frame["timestamp"].max()
    try:
        buffer = pd.Timedelta(7 * pd.tseries.frequencies.to_offset(expected_frequency).nanos, unit="ns") if expected_frequency else pd.Timedelta(days=7)
    except Exception:
        buffer = pd.Timedelta(days=7)
    suspects = statuses[pd.to_datetime(statuses.last_seen, utc=True) < end - buffer].copy()
    if not suspects.empty:
        warnings.append(f"Delisting/rebrand suspects: {sorted(suspects.asset.tolist())}; their last trading day must be treated as forced exit, not silently dropped")
    statuses["status"] = np.where(statuses.periods == periods, "active_full_period", "partial_history_or_delisted")
    if expected_frequency:
        gaps = []
        gap_classification = {}
        for asset, group in frame.groupby("asset"):
            dates = group["timestamp"].drop_duplicates().sort_values()
            expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=expected_frequency, tz="UTC")
            gaps.append(int(len(expected.difference(dates))))
        # Gap triage: documented exchange halts (migration_halt) are a
        # market fact -- always warning, never error. Only genuinely
        # unexplained interior holes can fail a run (or warn in tolerant
        # mode). This resolves the old gap-tolerant-vs-futures tension:
        # halts are not "tolerated gaps", they are documented non-trading.
        gap_table = classify_timestamp_gaps(frame, expected_frequency=expected_frequency)
        halted = gap_table[gap_table["reason"] == "migration_halt"] if not gap_table.empty else gap_table
        unexplained = gap_table[gap_table["reason"] == "unexplained_interior"] if not gap_table.empty else gap_table
        gap_classification = {
            "migration_halt_days": int(halted["gap_days"].sum()) if not halted.empty else 0,
            "unexplained_interior_days": int(unexplained["gap_days"].sum()) if not unexplained.empty else 0,
            "migration_halt_assets": sorted(halted["asset"].unique().tolist()) if not halted.empty else [],
            "unexplained_assets": sorted(unexplained["asset"].unique().tolist()) if not unexplained.empty else [],
        }
        if not halted.empty:
            halt_detail = "; ".join(f"{r.asset}:{r.gap_start.date()}->{r.gap_end.date()}" for r in halted.itertuples())
            warnings.append(f"Documented exchange halt(s): {int(halted['gap_days'].sum())} non-trading asset-periods ({halt_detail}); positions are force-exited across halts, never carried")
        if not unexplained.empty:
            tiny = unexplained[unexplained["gap_days"] <= tolerated_gap_days]
            big = unexplained[unexplained["gap_days"] > tolerated_gap_days]
            # Policy leniency: single-day holes (default) are API hiccups with
            # bounded impact -- the backtest force-exits across them. They
            # warn, never fail. Only multi-day unexplained holes can fail a
            # run (or warn in tolerant mode). Taxonomy is unchanged: both
            # stay "unexplained_interior" in the classifier.
            if not tiny.empty:
                tiny_detail = "; ".join(f"{r.asset}:{r.gap_start.date()}" for r in tiny.itertuples())
                warnings.append(f"Tolerated {int(tiny['gap_days'].sum())} single-day gap(s) ({tiny_detail}); treated as data hiccups, forced-exit applies if held")
            if not big.empty:
                gap_detail = "; ".join(f"{r.asset}:{r.gap_days}" for r in big.itertuples())
                message = f"Unexplained interior gaps: {int(big['gap_days'].sum())} missing asset-periods ({gap_detail[:200]})"
                (errors if not allow_gaps else warnings).append(message)
                if allow_gaps:
                    warnings.append("Gap-tolerant exploratory mode: unexplained gaps trigger forced exits in backtest; NOT valid for futures deployment")
        elif sum(gaps) and halted.empty:
            message = f"Timestamp gaps detected: {sum(gaps)} missing asset-periods"
            (errors if not allow_gaps else warnings).append(message)
    if funding_gaps.empty:
        funding_gap_summary = {}
    else:
        funding_gap_summary = {reason: int((funding_gaps["reason"] == reason).sum()) for reason in ("not_listed", "delisted", "migration", "active")}
    membership = frame.groupby(["timestamp", "asset"]).size().reset_index(name="rows")
    membership["timestamp"] = membership["timestamp"].astype(str)
    # Asset lifecycle engine: listed_at/delisted_at segments are the
    # point-in-time universe authority. Inferred listing dates are
    # placeholders -- replace with official listing dates before any
    # survivorship-free claim.
    lifecycle = build_lifecycle(frame)
    inferred_listings = sorted(lifecycle[lifecycle["source"].str.contains(INFERRED, na=False)]["canonical"].unique().tolist())
    if inferred_listings:
        warnings.append(f"Lifecycle listing dates inferred from data (NOT official) for: {inferred_listings}; replace with exchange listing dates before survivorship-free claims")
    survivorship_gap = {}
    if listing_manifest is not None:
        manifest = listing_manifest
        if isinstance(manifest, (str, Path)):
            import pandas as _pd
            manifest = _pd.read_csv(manifest)
        survivorship_gap = measure_survivorship_gap(frame, manifest)
        if survivorship_gap["n_missing_dead"]:
            sample = ",".join(survivorship_gap["missing_dead"][:15])
            warnings.append(f"SURVIVORSHIP_GAP: {survivorship_gap['n_missing_dead']} dead-in-window assets absent from dataset (e.g. {sample}); cross-sectional returns exclude their crashes (coverage {survivorship_gap['coverage_ratio']:.2f})")
    return {"valid": not errors, "schema_version": SCHEMA_VERSION, "errors": errors, "warnings": warnings, "rows": int(len(frame)), "assets": assets, "periods": periods, "start": str(frame["timestamp"].min()), "end": str(frame["timestamp"].max()), "funding_present": "funding_rate" in frame.columns, "liquidity_present": "quote_volume" in frame.columns, "funding_gap_summary": funding_gap_summary, "gap_classification": gap_classification if expected_frequency else {}, "survivorship_gap": survivorship_gap, "universe_membership": membership.to_dict("records"), "asset_status": statuses.astype(str).to_dict("records"), "delisting_suspects": suspects.astype(str).to_dict("records"), "lifecycle_manifest": lifecycle.astype(str).to_dict("records"), "lifecycle_segments": int(len(lifecycle)), "survivorship_note": "Universe berasal dari simbol yang tersedia saat ini; tanpa verifikasi point-in-time membership independen, hasil IC/return berpotensi bias survivorship yang belum terukur."}


def write_preflight(result, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / "preflight.json", result)
