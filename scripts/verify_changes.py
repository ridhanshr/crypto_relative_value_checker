"""Verifikasi perubahan di dataset yang sama (funded 47-aset).

1. Preflight KETAT (allow_gaps=False) -> harus valid (dulu PREFLIGHT_FAILED).
2. gap_classification -> semua migration_halt, nol unexplained.
3. listing_manifest -> survivorship_gap terisi + warning SURVIVORSHIP_GAP.
4. Smoke backtest low_vol_14 end-to-end (bukti pipeline jalan pasca-gerbang).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from crypto_checker.preflight import validate_dataset
from crypto_checker.signals import build_signals
from crypto_checker.core import CheckerConfig, check_strategy

data = pd.read_csv("data/midcap_2y_daily_2_funded.csv")
print("dataset:", len(data), "baris,", data["asset"].nunique(), "aset", flush=True)

check = validate_dataset(
    data.assign(signal=1.0),
    require_funding=True,
    require_liquidity=True,
    min_assets=2,
    min_periods=2,
    expected_frequency="D",
    allow_gaps=False,
    listing_manifest="data/historical_listing_manifest.csv",
)
print("1. strict valid:", check["valid"], "| errors:", check["errors"], flush=True)
print("2. gap_classification:", check["gap_classification"], flush=True)
sg = check["survivorship_gap"]
print("3. survivorship: coverage =", round(sg["coverage_ratio"], 4), "| missing_dead =", sg["n_missing_dead"], flush=True)
print("   warnings SURVIVORSHIP_GAP:", sum("SURVIVORSHIP_GAP" in w for w in check["warnings"]), flush=True)
print("   warnings halt:", sum("Documented exchange halt" in w for w in check["warnings"]), flush=True)

assert check["valid"], "STRICT PREFLIGHT STILL FAILS"
assert check["gap_classification"]["unexplained_interior_days"] == 0
assert sg["n_missing_dead"] > 0

built = build_signals(data)
res = check_strategy(
    built,
    CheckerConfig(n_long=3, n_short=3, fee_rate=0.0004, slippage_rate=0.0005, signal_column="low_vol_14",
                  vol_target_annual=0.20, rebalance_every=5, require_funding=True,
                  delist_mode="forced_exit", liquidity_column="quote_volume",
                  liquidity_tiers=((0.5, 0.0004, 0.0005), (0.0, 0.0015, 0.004))),
)
m = res["metrics"]
print("4. smoke low_vol_14: return =", round(m["total_return"], 4), "| sharpe =", round(m["sharpe"], 4),
      "| max_dd =", round(m["max_drawdown"], 4), "| cap_viol =", m["capacity_violations"], flush=True)
print("VERIFY_OK", flush=True)
