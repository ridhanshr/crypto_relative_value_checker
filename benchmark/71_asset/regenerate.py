"""Regenerate the 71-asset benchmark dataset from scratch (requires network).

Identity (SHA256SUMS + README manifest) lives in git; the bytes do not.
Replays: raw download -> scripts/repair_71.py -> hash verification.

Usage:
    python benchmark/71_asset/regenerate.py [--execute]
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SHA256 = "8b07528ec9d3756526d810c62933fa4f20f7ab0466c12196e5189031719365c5"
OUTPUT = Path("data/midcap_2y_daily_2_repaired.csv")

STEPS = [
    "python scripts/download_midcap.py --start 2024-01-02 --end 2026-08-31 --interval 1d --output data/midcap_2y_daily_2.csv",
    "python scripts/repair_71.py",
]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Regenerate the 71-asset benchmark dataset")
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
