"""L2 determinism proof: same input -> byte-identical normalized output.

Runs deployment_decision TWICE on the funded 47-asset dataset and asserts:
  * normalized JSON SHA256 identical (timestamps/dates are data-derived,
    so no wall-clock normalization is needed -- and this script would catch
    it if that ever changed),
  * numeric metrics identical within 1e-9.

Usage: python scripts/determinism_check.py [--input ...] [--out ...]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from crypto_checker.decision import deployment_decision

parser = argparse.ArgumentParser(description="Prove run-to-run determinism")
parser.add_argument("--input", default="data/midcap_2y_daily_2_funded.csv")
parser.add_argument("--out", default="reports/determinism")
parser.add_argument("--tol", type=float, default=1e-9)
args = parser.parse_args()

KWARGS = dict(min_train_days=365, test_days=120, n_sides_grid=(3,), vol_target_annual=0.20,
              rebalance_every=5, fee_rate=0.0004, slippage_rate=0.0005,
              require_funding=True, delist_mode="forced_exit", ensemble_top_k=3)


def normalized_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def metrics_of(decision):
    wf = decision["walk_forward"]["walk_forward"]
    dm = decision.get("data_mining") or {}
    return {"wf_total_return": wf["total_return"], "wf_sharpe": wf["sharpe"], "wf_dd": wf["max_drawdown"],
            "dsr": dm.get("dsr"), "deployable": decision["deployable"]}


data = pd.read_csv(args.input)
out = Path(args.out)
hashes, metrics = [], []
for i in (1, 2):
    print(f"run {i}...", flush=True)
    decision = deployment_decision(data, output_dir=str(out / f"run{i}"), **KWARGS)
    hashes.append(normalized_hash(decision))
    metrics.append(metrics_of(decision))

print("hash run1:", hashes[0], flush=True)
print("hash run2:", hashes[1], flush=True)
assert hashes[0] == hashes[1], "NON-DETERMINISTIC OUTPUT"
a, b = metrics
for key in a:
    va, vb = a[key], b[key]
    if isinstance(va, bool) or va is None:
        assert va == vb, key
    else:
        assert abs(va - vb) <= args.tol, (key, va, vb)
print("metrics identical within", args.tol, flush=True)
print("DETERMINISM_OK", flush=True)
