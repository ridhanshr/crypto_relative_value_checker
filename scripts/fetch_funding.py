"""Download daily funding rates for a klines CSV and merge them in.

- Monthly Binance Vision fundingRate archives per raw symbol; fapi API
  fallback per missing month.
- Rows canonicalized (OMNI->NOM, GAL->G, MATIC->POL) with migration
  cutover by official effective date (old symbol rows only before it,
  new symbol rows only on/after it) to avoid double counting.
- Aggregated to daily sums per (date, canonical asset) regardless of
  settlement count (8h/4h/1h intervals all occur); per-asset max
  settlements/day plus max |daily rate| go to the frequency manifest.
- Merges into klines on (date, asset); rows without funding stay NaN
  (preflight --require-funding will catch them loudly).

Usage:
    python scripts/fetch_funding.py --input data/midcap_2y_daily_repaired.csv \
        --output data/midcap_2y_daily_funded.csv
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crypto_checker.assets import canonical_asset, MIGRATION_METADATA
from crypto_checker.binance_vision import FUNDING_MONTHLY_URL, FUNDING_API_URL

LEGACY_SYMBOLS = ["MATICUSDT", "GALUSDT", "OMNIUSDT"]


def build_funding_frequency_report(funding):
    """Aggregate funding events to daily sums + audit manifest.

    funding: event-level DataFrame with timestamp/asset/funding_rate.
    Returns (freq_report, daily) where freq_report is indexed by asset with
    max_settlements_per_day plus max_abs_daily_rate(_date), and daily holds
    per-(date, asset) sums. Sub-8h intervals are aggregated, never rejected.
    """
    frame = funding.copy()
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("D")
    freq_report = (
        frame.groupby("asset")["date"]
        .apply(lambda s: s.value_counts().max())
        .rename("max_settlements_per_day")
        .to_frame()
    )

    def _extreme_idx(s):
        s = s.dropna()
        return s.abs().idxmax() if len(s) else np.nan

    daily = frame.groupby(["date", "asset"], as_index=False)["funding_rate"].sum()
    # Extreme-value audit: max |daily aggregated rate| per asset and when it
    # occurred. Large prints (e.g. API3USDT -10% on 2025-08-19 from hourly
    # settlements incl. a -2% cap print) are genuine Binance data, but they
    # dominate PnL attribution and must be auditable from the manifest alone.
    extreme_idx = daily.groupby("asset")["funding_rate"].apply(_extreme_idx).dropna()
    if len(extreme_idx):
        extreme = daily.loc[extreme_idx.values, ["asset", "date", "funding_rate"]].rename(
            columns={"date": "max_abs_daily_rate_date", "funding_rate": "max_abs_daily_rate"})
        freq_report = freq_report.merge(extreme.set_index("asset"), left_index=True, right_index=True, how="left")
    else:
        freq_report["max_abs_daily_rate_date"] = pd.NaT
        freq_report["max_abs_daily_rate"] = np.nan
    return freq_report, daily


def fetch_funding_archive(url):
    with urlopen(url, timeout=30) as response:
        archive = ZipFile(BytesIO(response.read()))
        raw = archive.read(archive.namelist()[0])
    funding_frame = pd.read_csv(BytesIO(raw))
    funding_frame.columns = [str(c).strip() for c in funding_frame.columns]
    normalized = {c.lower().replace("_", ""): c for c in funding_frame.columns}
    time_col = next((normalized[k] for k in ("calctime", "fundingtime", "fundingtimestamp") if k in normalized), None)
    rate_col = next((normalized[k] for k in ("lastfundingrate", "fundingrate", "funding") if k in normalized), None)
    if time_col is None or rate_col is None:
        raise ValueError(f"Unrecognized funding columns: {list(funding_frame.columns)}")
    return pd.DataFrame({
        "timestamp": pd.to_datetime(funding_frame[time_col], unit="ms", utc=True),
        "funding_rate": funding_frame[rate_col].astype(float),
    })


def fetch_funding_api(symbol, range_start, range_end):
    rows = []
    cursor = int(range_start.timestamp() * 1000)
    end_ms = int(range_end.timestamp() * 1000)
    while cursor < end_ms:
        params = urlencode({"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
        with urlopen(FUNDING_API_URL + "?" + params, timeout=30) as response:
            batch = json.loads(response.read())
        if not batch:
            break
        for row in batch:
            rows.append({"timestamp": pd.to_datetime(row["fundingTime"], unit="ms", utc=True), "funding_rate": float(row["fundingRate"])})
        latest = max(int(r["fundingTime"]) for r in batch)
        if latest < cursor:
            raise RuntimeError(f"Funding pagination stalled for {symbol}")
        cursor = latest + 1
    return pd.DataFrame(rows)


def fetch_month(symbol, stamp):
    url = FUNDING_MONTHLY_URL.format(symbol=symbol, date=stamp)
    try:
        frame = fetch_funding_archive(url)
        return ("archive", frame, None)
    except Exception as archive_exc:
        try:
            year, month = int(stamp[:4]), int(stamp[5:7])
            start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            end = min(start + pd.offsets.MonthEnd(0), pd.Timestamp.now(tz="UTC"))
            frame = fetch_funding_api(symbol, start, end)
            if frame.empty:
                return ("missing", None, f"archive 404 + API empty ({archive_exc})")
            return ("api", frame, None)
        except Exception as api_exc:
            return ("missing", None, f"archive 404 + API failed ({api_exc})")


def main():
    parser = argparse.ArgumentParser(description="Download funding and merge into klines CSV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    klines = pd.read_csv(args.input, parse_dates=["timestamp"])
    assets = sorted(klines.asset.unique().tolist())
    raw_symbols = sorted(set(assets) | set(LEGACY_SYMBOLS))
    # Legacy symbols whose canonical form is absent need no fetch, but fetching
    # them is harmless; restrict months to reduce noise later via cutover.
    start_ts = pd.Timestamp("2024-01-01", tz="UTC")
    end_ts = pd.Timestamp("2026-08-31", tz="UTC")
    months = pd.date_range(start_ts.replace(day=1), end_ts, freq="MS").strftime("%Y-%m").tolist()

    jobs = [(s, m) for s in raw_symbols for m in months]
    frames, missing = [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        future_map = {pool.submit(fetch_month, s, m): (s, m) for s, m in jobs}
        for future in as_completed(future_map):
            symbol, stamp = future_map[future]
            try:
                kind, frame, err = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                missing.append({"symbol": symbol, "month": stamp, "reason": str(exc)})
                continue
            if kind == "missing":
                missing.append({"symbol": symbol, "month": stamp, "reason": err})
                continue
            frame["source"] = symbol
            frame["month"] = stamp
            frames.append(frame)
    print(f"fetched {len(frames)} month-files, missing {len(missing)}", flush=True)

    funding = pd.concat(frames, ignore_index=True)
    funding["asset"] = funding["source"].map(canonical_asset)

    # Migration cutover: legacy rows only before effective date, new rows only on/after.
    quote_suffixes = ("USDT", "USDC", "BUSD", "BTC", "ETH")
    from crypto_checker.assets import MIGRATION_MAP
    reverse = {}
    for old, new in MIGRATION_MAP.items():
        for q in quote_suffixes:
            reverse.setdefault(new + q, []).append(old + q)
    keep_masks = []
    dropped_overlap = 0
    for asset, group in funding.groupby("asset"):
        raws = sorted(group.source.unique().tolist())
        if len(raws) <= 1:
            keep_masks.append(group)
            continue
        # More than one raw symbol feeding one canonical asset: split by effective date.
        base = asset
        for q in quote_suffixes:
            if base.endswith(q):
                base = base[: -len(q)]
                break
        olds = [o for o, n in MIGRATION_MAP.items() if n == base]
        if not olds:
            keep_masks.append(group)
            continue
        eff = pd.Timestamp((MIGRATION_METADATA.get(olds[0]) or {}).get("effective_date"), tz="UTC")
        legacy_syms = {o + q for o in olds for q in quote_suffixes}
        g = group.copy()
        g["date"] = g["timestamp"].dt.floor("D")
        legacy_keep = g.source.isin(legacy_syms) & (g["date"] < eff.floor("D"))
        new_keep = ~g.source.isin(legacy_syms) & (g["date"] >= eff.floor("D"))
        dropped_overlap += int(((g.source.isin(legacy_syms)) & (g["date"] >= eff.floor("D"))).sum())
        dropped_overlap += int(((~g.source.isin(legacy_syms)) & (g["date"] < eff.floor("D"))).sum())
        keep_masks.append(g[legacy_keep | new_keep].drop(columns=["date"]))
    funding = pd.concat(keep_masks, ignore_index=True)
    print(f"overlap rows excluded by cutover: {dropped_overlap}", flush=True)

    # Exact-duplicate funding prints (same ts/asset/rate from both archives).
    before = len(funding)
    funding = funding.drop_duplicates(subset=["timestamp", "asset", "funding_rate"])
    print(f"exact-duplicate funding rows removed: {before - len(funding)}", flush=True)
    conflicts = funding.duplicated(["timestamp", "asset"], keep=False)
    if conflicts.any():
        bad = funding[conflicts].sort_values(["asset", "timestamp"])
        bad.to_csv(out.parent / (out.stem + "_funding_conflicts.csv"), index=False)
        raise RuntimeError(f"{int(conflicts.sum())} conflicting funding prints remain; manifest saved; refusing to guess")

    funding["date"] = funding["timestamp"].dt.floor("D")
    if not (funding["timestamp"] < funding["date"] + pd.Timedelta(days=1)).all():
        raise RuntimeError("Funding event settled after candle close; refusing lookahead merge")
    freq_report, daily = build_funding_frequency_report(funding)
    print("max settlements/day per asset (values >3 mean sub-8h funding interval, aggregated by sum):", flush=True)
    print(freq_report[freq_report["max_settlements_per_day"] > 3].to_string(), flush=True)
    freq_report.to_csv(out.parent / (out.stem + "_funding_frequency.csv"))

    if "funding_rate" in klines.columns:
        print("dropping pre-existing funding_rate column; refetching uniformly", flush=True)
        klines = klines.drop(columns=["funding_rate"])
    klines["date"] = klines["timestamp"].dt.floor("D")
    merged = klines.merge(daily, on=["date", "asset"], how="left")
    merged = merged.drop(columns=["date"])
    missing_rows = merged[merged.funding_rate.isna()][["timestamp", "asset"]].copy()
    print(f"klines rows: {len(merged)}, missing funding rows: {len(missing_rows)}", flush=True)
    cov = merged.groupby("asset")["funding_rate"].apply(lambda s: f"{int(s.notna().sum())}/{len(s)}").to_dict()
    for asset, stat in sorted(cov.items()):
        have, total = stat.split("/")
        if have != total:
            print(f"  GAP {asset}: funding {stat}", flush=True)

    merged.to_csv(out, index=False)
    pd.DataFrame(missing, columns=["symbol", "month", "reason"]).to_csv(out.parent / (out.stem + "_missing_funding.csv"), index=False)
    missing_rows.to_csv(out.parent / (out.stem + "_missing_funding_rows.csv"), index=False)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
