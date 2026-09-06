"""Repair a partial klines CSV into a clean canonical panel.

Repairs (all logged to a manifest, nothing silent):
1. Drop zero-volume rows (stale carry-forward filler after delisting/halt).
2. Apply official migration price factors to legacy rows dated BEFORE the
   official effective date (GAL/60, OMNI/75; MATIC factor 1.0 is a no-op).
3. Drop precomputed asset_return/signal so the pipeline recomputes them
   on adjusted prices.
4. Assert no (timestamp, asset) duplicates remain.

Usage:
    python scripts/repair_midcap.py --input data/midcap_2y_daily_partial.csv \
        --output data/midcap_2y_daily_repaired.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crypto_checker.assets import (
    MIGRATION_MAP,
    MIGRATION_METADATA,
    PRICE_ADJUSTMENT_FACTORS,
)


def repair_klines(df):
    """Apply all documented repairs; returns (repaired_frame, manifest)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    manifest = {"input_rows": int(len(df)), "actions": []}

    # 1. Drop stale zero-volume rows.
    stale = df[df.volume == 0]
    manifest["actions"].append({
        "action": "drop_zero_volume_rows",
        "rows_dropped": int(len(stale)),
        "by_asset": {k: int(v) for k, v in stale.groupby("asset").size().items()},
        "reason": "zero volume with flat carried-forward price is not a real market print",
    })
    df = df[df.volume > 0].copy()

    # 2. Official migration price factors on legacy rows only.
    # Canonical asset names are already merged (e.g. GAL klines live under GUSDT).
    # Cutover is by CALENDAR DATE: a daily candle dated on/after the effective
    # date belongs to the new series even if its 00:00 UTC open precedes the
    # intraday swap time. Using the raw intraday timestamp would misclassify
    # the new token's first daily candle as legacy (seen live: NOM 2025-10-01
    # $0.03975 wrongly divided by 75 into $0.00053, fabricating a +72x jump).
    quote_suffixes = ("USDT", "USDC", "BUSD", "BTC", "ETH")
    for old, new in MIGRATION_MAP.items():
        factor = PRICE_ADJUSTMENT_FACTORS.get(old)
        effective = (MIGRATION_METADATA.get(old) or {}).get("effective_date")
        if factor is None or effective is None:
            manifest["actions"].append({
                "action": "skip_price_adjustment",
                "legacy_token": old,
                "reason": "no official factor/effective-date pair configured; refusing to guess",
            })
            continue
        cutoff = pd.Timestamp(effective, tz="UTC").floor("D")
        new_symbols = {new + s for s in quote_suffixes}
        mask = df["asset"].isin(new_symbols) & (df["timestamp"].dt.floor("D") < cutoff)
        n = int(mask.sum())
        if n:
            for col in ("open", "high", "low", "price"):
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / float(factor)
        manifest["actions"].append({
            "action": "apply_official_migration_factor",
            "legacy_token": old,
            "canonical_token": new,
            "ratio": (MIGRATION_METADATA.get(old) or {}).get("ratio"),
            "effective_date": effective,
            "cutoff_rule": "candle date < effective calendar date",
            "rows_adjusted": n,
            "rule": "rows dated before effective date are legacy-scale prices; divided by factor",
        })

    # 3. Drop precomputed return/signal columns (stale after price adjustment).
    for col in ("asset_return", "signal"):
        if col in df.columns:
            df = df.drop(columns=[col])
            manifest["actions"].append({"action": f"drop_stale_column_{col}", "reason": "recomputed downstream on adjusted prices"})

    # 4. No duplicates may remain.
    dupes = int(df.duplicated(["timestamp", "asset"]).sum())
    manifest["remaining_duplicates"] = dupes
    if dupes:
        manifest["unresolved_dupes"] = df[df.duplicated(["timestamp", "asset"], keep=False)].sort_values(["asset", "timestamp"]).astype(str).to_dict("records")
        raise RuntimeError(f"{dupes} duplicate (timestamp, asset) rows remain; refusing to proceed")

    # 5. Seam diagnostics at each migration effective date.
    seams = []
    for old, new in MIGRATION_MAP.items():
        effective = (MIGRATION_METADATA.get(old) or {}).get("effective_date")
        if effective is None:
            continue
        new_symbols = {new + s for s in quote_suffixes}
        sub = df[df["asset"].isin(new_symbols)].sort_values("timestamp")
        if sub.empty:
            continue
        cutoff = pd.Timestamp(effective, tz="UTC").floor("D")
        before = sub[sub.timestamp.dt.floor("D") < cutoff].tail(1)
        after = sub[sub.timestamp.dt.floor("D") >= cutoff].head(1)
        if not before.empty and not after.empty:
            jump = float(after.iloc[0].price / before.iloc[0].price - 1)
            seams.append({"canonical": sorted(new_symbols & set(sub.asset.unique())), "last_legacy": str(before.iloc[0].timestamp), "first_new": str(after.iloc[0].timestamp), "seam_return": jump})
    manifest["migration_seams"] = seams

    df = df.sort_values(["timestamp", "asset"]).reset_index(drop=True)
    manifest["output_rows"] = int(len(df))
    manifest["assets"] = sorted(df.asset.unique().tolist())
    return df, manifest


def main():
    parser = argparse.ArgumentParser(description="Repair partial klines into a clean canonical panel")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df, manifest = repair_klines(df)
    df.to_csv(out, index=False)
    (out.parent / (out.stem + "_manifest.json")).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "assets"}, indent=2, default=str), flush=True)
    print(f"Wrote {out} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
