"""Verifikasi generik dataset pasca-repair: preflight ketat + survivorship.

Usage:
    python scripts/verify_dataset.py --input data/midcap_2y_daily_2_repaired.csv [--manifest data/historical_listing_manifest.csv]

Target: strict valid True, errors kosong.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from crypto_checker.preflight import validate_dataset, write_preflight

parser = argparse.ArgumentParser(description="Strict-verify a (repaired) dataset")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="reports/verify")
parser.add_argument("--manifest", default="data/historical_listing_manifest.csv")
parser.add_argument("--tolerated-gap-days", type=int, default=1)
args = parser.parse_args()

data = pd.read_csv(args.input)
print(f"dataset: {len(data)} baris, {data['asset'].nunique()} aset", flush=True)
manifest = args.manifest if Path(args.manifest).exists() else None
if manifest is None:
    print("listing manifest not found; skipping survivorship", flush=True)

check = validate_dataset(
    data.assign(signal=1.0) if "signal" not in data.columns else data,
    require_funding=True,
    require_liquidity=True,
    min_assets=2,
    min_periods=2,
    expected_frequency="D",
    allow_gaps=False,
    listing_manifest=manifest,
    tolerated_gap_days=args.tolerated_gap_days,
)
write_preflight(check, args.output)
print("strict valid:", check["valid"], flush=True)
print("errors:", check["errors"], flush=True)
print("gap_classification:", check.get("gap_classification"), flush=True)
if check.get("survivorship_gap"):
    sg = check["survivorship_gap"]
    print(f"survivorship: coverage={sg['coverage_ratio']:.4f} missing_dead={sg['n_missing_dead']}", flush=True)
if not check["valid"]:
    raise SystemExit("VERIFY_FAILED")
print("VERIFY_OK", flush=True)
