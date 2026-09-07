"""Regenerate a benchmark dataset from scratch (requires network).

Datasets are research artifacts, NOT source code: only their identity
(SHA256SUMS + README manifest) lives in git. This script replays the
documented pipeline and verifies the output hash matches.

Usage:
    python benchmark/47_asset/regenerate.py [--execute]

Default (no flag): prints the exact pipeline steps (dry run).
--execute: runs them. Needs Binance Vision + fapi reachability.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SHA256 = "53e2db604567de92fbcbf10525a0b36707d67da9c5aae050c03a5df74a65d12d"
OUTPUT = Path("data/midcap_2y_daily_2_funded.csv")

STEPS = [
    # 1. klines + funding download (see scripts/download_midcap.py --help and scripts/fetch_funding.py --help
    #    for the exact sector list, date range 2024-01-01..2026-08-31, and cutover conventions),
    # 2. stale-row repair (scripts/repair_midcap.py: drop zero-volume prints, apply GAL/60 + OMNI/75),
    # 3. funding merge (scripts/fetch_funding.py: monthly archives + fapi fallback, daily-sum aggregation).
    "python scripts/download_midcap.py --start 2024-01-01 --end 2026-08-31 --interval 1d --sectors <see README> --include-mega",
    "python scripts/repair_midcap.py --input <partial> --output <repaired>",
    "python scripts/fetch_funding.py --input <repaired> --output data/midcap_2y_daily_2_funded.csv",
]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Regenerate the 47-asset benchmark dataset")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("DRY RUN -- re-run with --execute. Pipeline:", flush=True)
        for step in STEPS:
            print("  " + step, flush=True)
        print(f"Expected sha256: {EXPECTED_SHA256}  {OUTPUT}", flush=True)
        return
    for step in STEPS:
        print("RUN:", step, flush=True)
        subprocess.run(step, shell=True, cwd=ROOT, check=True)
    actual = sha256_of(OUTPUT)
    print(f"sha256: {actual}", flush=True)
    if actual != EXPECTED_SHA256:
        raise SystemExit(f"HASH MISMATCH: expected {EXPECTED_SHA256}")
    print("REGENERATE_OK", flush=True)


if __name__ == "__main__":
    main()
