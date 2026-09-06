"""Uji mode ketat: preflight TANPA toleransi gap (allow_gaps=False).

Menjawab: apa yang terjadi bila --allow-gaps tidak dipakai.
Plus klasifikasi tiap gap: tepi (listing/delisting, ekspektabel) vs
interior (lubang tak terjelaskan) vs dalam window migrasi resmi.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from crypto_checker.preflight import validate_dataset
from crypto_checker.assets import (
    canonicalize_assets,
    MIGRATION_METADATA,
    MIGRATION_MAP,
    _base_token,
)

data = pd.read_csv("data/midcap_2y_daily_2_funded.csv")
print("dataset:", len(data), "baris,", data["asset"].nunique(), "aset", flush=True)

print("=== 1. preflight KETAT (allow_gaps=False) ===", flush=True)
strict = validate_dataset(
    data.assign(signal=1.0),
    require_funding=True,
    require_liquidity=True,
    min_assets=2,
    min_periods=2,
    expected_frequency="D",
    allow_gaps=False,
)
print("valid:", strict["valid"], flush=True)
for e in strict["errors"]:
    print("ERROR:", str(e)[:300], flush=True)

print("=== 2. klasifikasi gap per aset ===", flush=True)
frame = canonicalize_assets(data.assign(signal=1.0))
eff_dates = {}
for old, meta in MIGRATION_METADATA.items():
    if meta.get("effective_date"):
        eff_dates[meta["canonical"]] = pd.Timestamp(meta["effective_date"], tz="UTC")

total_interior = 0
rows = []
for asset, group in frame.groupby("asset"):
    dates = group["timestamp"].drop_duplicates().sort_values()
    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D", tz="UTC")
    missing = expected.difference(dates)
    n_edge = 0  # terpotong tepi = listing/delisting, ekspektabel by design
    interior = []
    eff = eff_dates.get(_base_token(asset))
    near_migration = 0
    for ts in missing:
        if eff is not None and abs(ts - eff) <= pd.Timedelta(days=7):
            near_migration += 1
        else:
            interior.append(ts)
    total_interior += len(interior)
    if len(missing):
        rows.append((asset, len(dates), str(dates.iloc[0].date()), str(dates.iloc[-1].date()), len(missing), near_migration, len(interior)))

rows.sort(key=lambda r: -r[4])
print("aset | baris | first | last | gap_total | gap_migrasi | gap_interior_tak_terjelaskan", flush=True)
for r in rows:
    print(r[0], r[1], r[2], r[3], r[4], r[5], r[6], flush=True)
print("TOTAL gap interior tak terjelaskan:", total_interior, flush=True)

print("=== 3. konsekuensi pipeline ===", flush=True)
print("analyze_midcap.py: preflight invalid -> SystemExit(PREFLIGHT_FAILED).", flush=True)
print("Artinya tanpa --allow-gaps, run BERHENTI di gerbang, tidak ada backtest sama sekali.", flush=True)
print("DONE", flush=True)
