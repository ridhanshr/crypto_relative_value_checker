"""Lanjutan recheck: diagnosa preflight errors + capacity curve (data mentah -> build_signals dulu)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from crypto_checker.preflight import validate_dataset
from crypto_checker.capacity import capacity_curve
from crypto_checker.core import CheckerConfig
from crypto_checker.signals import build_signals

OUT = Path("reports/recheck_v2")

data = pd.read_csv("data/midcap_2y_daily_2_funded.csv")
check = validate_dataset(data, require_funding=True, require_liquidity=True, min_assets=2, min_periods=2, expected_frequency="D", allow_gaps=True)
print("preflight valid:", check["valid"])
print("ERRORS:")
for e in check["errors"]:
    print(" -", e)
print("WARNINGS:", len(check["warnings"]))
for w in check["warnings"][:10]:
    print(" -", str(w)[:220])

decision = json.loads((OUT / "decision" / "deployment_decision.json").read_text())
best_signal = decision["best_signal_walk_forward"]
best_n = decision["best_n_sides"]
print("best:", best_signal, "n:", best_n)

built = build_signals(data)
print("capacity curve...", flush=True)
cap = capacity_curve(
    built,
    base_config=CheckerConfig(
        n_long=best_n, n_short=best_n, fee_rate=0.0004, slippage_rate=0.0005,
        signal_column=best_signal, vol_target_annual=0.20, rebalance_every=5,
        require_funding=True, delist_mode="forced_exit",
        liquidity_column="quote_volume",
        liquidity_tiers=((0.5, 0.0004, 0.0005), (0.0, 0.0015, 0.004)),
    ),
    output_dir=str(OUT / "capacity"),
)
print("capacity levels:", json.dumps(cap.get("levels"), indent=1), flush=True)

wf = decision["walk_forward"]
print("WF summary:", json.dumps(wf["walk_forward"], indent=1))
print("WF folds:")
for f in wf["folds"]:
    print("  fold", f["fold"], f["signal"], "n=", f["n_sides"], "test_ret=", round(f["test_return"], 4))
if wf.get("ensemble"):
    print("ENSEMBLE:", json.dumps(wf["ensemble"]["ensemble"], indent=1))
    print("members:", json.dumps(wf["ensemble"]["members_per_fold"], indent=1))
print("DONE", flush=True)
