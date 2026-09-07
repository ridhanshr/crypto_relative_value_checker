"""Effective-N DSR for vol_adj_momentum_30 on 71-asset repaired data.

Runs walk_forward (registry logs every attempt) and prints trial counts +
effective-N DSR. Same rigor as the low_vol_14 confirmation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from crypto_checker.core import CheckerConfig
from crypto_checker.selection import walk_forward

data = pd.read_csv("data/midcap_2y_daily_2_repaired.csv")
print("dataset:", len(data), "aset:", data["asset"].nunique(), flush=True)
result = walk_forward(
    data,
    base_config=CheckerConfig(n_long=3, n_short=3, fee_rate=0.0004, slippage_rate=0.0005,
                              vol_target_annual=0.20, rebalance_every=5, require_funding=True,
                              delist_mode="forced_exit", liquidity_column="quote_volume",
                              liquidity_tiers=((0.5, 0.0004, 0.0005), (0.0, 0.0015, 0.004))),
    min_train_days=365, test_days=120, output_dir="reports/wf_mom30_71",
)
print("trial_registry:", json.dumps(result["trial_registry"], indent=1), flush=True)
print("data_mining:", json.dumps(result["data_mining"], indent=1), flush=True)
print("WF:", json.dumps(result["walk_forward"], indent=1), flush=True)
print("folds:", [(f["signal"], round(f["test_return"], 4)) for f in result["folds"]], flush=True)
print("WF71_DONE", flush=True)
