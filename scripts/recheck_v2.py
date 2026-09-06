"""Recheck penuh dataset funded dengan engine terbaru.

Menjalankan: preflight (+lifecycle) -> deployment_decision (walk-forward +
ensemble top-3 + DSR + validasi continuous) -> capacity curve 10k-10M.
Semua output ke reports/recheck_v2/.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crypto_checker.preflight import validate_dataset, write_preflight
from crypto_checker.decision import deployment_decision
from crypto_checker.capacity import capacity_curve
from crypto_checker.core import CheckerConfig
from crypto_checker.lifecycle import write_lifecycle_manifest, build_lifecycle
from crypto_checker.assets import canonicalize_assets

OUT = Path("reports/recheck_v2")
OUT.mkdir(parents=True, exist_ok=True)

print("loading data...", flush=True)
data = pd.read_csv("data/midcap_2y_daily_2_funded.csv")
print("rows:", len(data), "assets:", data["asset"].nunique(), flush=True)

print("preflight...", flush=True)
check = validate_dataset(
    data,
    require_funding=True,
    require_liquidity=True,
    min_assets=2,
    min_periods=2,
    expected_frequency="D",
    allow_gaps=True,
)
write_preflight(check, OUT)
print("preflight valid:", check["valid"], "warnings:", len(check["warnings"]), flush=True)

print("lifecycle manifest...", flush=True)
frame = canonicalize_assets(data)
life = build_lifecycle(frame)
write_lifecycle_manifest(life, OUT / "lifecycle_manifest.csv")
print("segments:", len(life), flush=True)

print("deployment decision (walk-forward + ensemble + DSR)...", flush=True)
decision = deployment_decision(
    data,
    output_dir=str(OUT / "decision"),
    min_train_days=365,
    test_days=120,
    n_sides_grid=(3,),
    vol_target_annual=0.20,
    rebalance_every=5,
    fee_rate=0.0004,
    slippage_rate=0.0005,
    require_funding=True,
    delist_mode="forced_exit",
    ensemble_top_k=3,
)
print("deployable:", decision["deployable"], flush=True)
print("gates:", json.dumps(decision["gates"], indent=1), flush=True)
print("data_mining:", json.dumps(decision.get("data_mining"), indent=1), flush=True)

print("capacity curve...", flush=True)
best_signal = decision["best_signal_walk_forward"]
best_n = decision["best_n_sides"]
cap = capacity_curve(
    data,
    base_config=CheckerConfig(
        n_long=best_n,
        n_short=best_n,
        fee_rate=0.0004,
        slippage_rate=0.0005,
        signal_column=best_signal,
        vol_target_annual=0.20,
        rebalance_every=5,
        require_funding=True,
        delist_mode="forced_exit",
        liquidity_column="quote_volume",
        liquidity_tiers=((0.5, 0.0004, 0.0005), (0.0, 0.0015, 0.004)),
    ),
    output_dir=str(OUT / "capacity"),
)
print("capacity:", json.dumps(cap.get("levels"), indent=1), flush=True)
print("DONE", flush=True)
