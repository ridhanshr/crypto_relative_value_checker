"""Final recheck via the OFFICIAL v1.1 entry point (dogfoods api.validate_csv).

Runs both benchmark datasets end-to-end and prints the envelope verdicts.
Outputs: reports/recheck_final/{47,71}/decision.json (+ preflight, capacity).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from crypto_checker.api import validate_csv

DATASETS = {
    "47": "data/midcap_2y_daily_2_funded.csv",
    "71": "data/midcap_2y_daily_2_repaired.csv",
}

for tag, path in DATASETS.items():
    print("=" * 30, tag, "=" * 30, flush=True)
    env = validate_csv(
        path, output_dir=f"reports/recheck_final/{tag}",
        n_long=3, n_short=3, min_train_days=365, test_days=120,
        ensemble_top_k=3, listing_manifest="data/historical_listing_manifest.csv",
    )
    print("status:", env["status"], "| decision:", env["decision"], "| deployable:", env["deployable"], flush=True)
    print("metrics:", json.dumps(env["metrics"], indent=1), flush=True)
    print("capacity:", json.dumps(env["capacity"]), flush=True)
    print("data_quality:", json.dumps(env["data_quality"]), flush=True)
    print("gates:", json.dumps(env["gates"], indent=1), flush=True)
print("RECHECK_FINAL_DONE", flush=True)
