"""Exposure sweep for risk scaling (M11B remediation, step 3).

Freezes the working setup (low_vol family, rebalance_every=10,
min_signal_gap=0.002, vol_target 0.15, spread slippage, forced exit) and
varies ONLY gross exposure: 1.0x / 0.75x / 0.5x / 0.35x of the 2.0 baseline.
Per level: full-sample backtest (low_vol_14) with PnL decomposition
(price / funding / fee / slippage), walk-forward, and static validation
(train/val/OOS) for the gate check.

Usage:
    python scripts/sweep_exposure.py \
        --input data/midcap_2y_daily_2_funded.csv \
        --spread-csv reports/midcap_analysis_v5/spread_check.csv \
        --output reports/exposure_sweep
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_checker.assets import canonicalize_assets
from crypto_checker.signals import build_signals
from crypto_checker.core import CheckerConfig, check_strategy
from crypto_checker.selection import walk_forward
from crypto_checker.validation import run_validation

parser = argparse.ArgumentParser(description="Sweep gross exposure with frozen setup")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="reports/exposure_sweep")
parser.add_argument("--spread-csv", default="")
args = parser.parse_args()

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(args.input)
data = canonicalize_assets(raw)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
data["signal"] = data["asset_return"]
data = data.dropna(subset=["signal"]).reset_index(drop=True)
built = build_signals(data)
frame = built.dropna(subset=["low_vol_14"]).reset_index(drop=True)

tiers = ((0.75, 0.0005, 0.0005), (0.5, 0.0005, 0.001),
         (0.25, 0.0005, 0.002), (0.0, 0.0005, 0.004))
slippage_mode = "spread" if args.spread_csv else "tier"

results = {}
for scale, gross in (("1.00x", 2.0), ("0.75x", 1.5), ("0.50x", 1.0), ("0.35x", 0.7)):
    cfg = CheckerConfig(
        n_long=3, n_short=3, fee_rate=0.0005, slippage_rate=0.0005,
        gross_exposure=gross, rebalance_every=10, min_signal_gap=0.002,
        vol_target_annual=0.15, liquidity_column="quote_volume",
        liquidity_tiers=tiers, slippage_mode=slippage_mode,
        spread_csv=args.spread_csv, require_funding=True,
        delist_mode="forced_exit", signal_column="low_vol_14",
    )
    full = check_strategy(frame, cfg)
    m = full["metrics"]
    pnl = full["pnl"]
    decomp = {
        "gross_exposure": gross,
        "total_return": float(m["total_return"]),
        "sharpe": float(m["sharpe"]),
        "max_drawdown": float(m["max_drawdown"]),
        "avg_turnover": float(m["average_turnover"]),
        "price_pnl": float(pnl["price_pnl"].sum()),
        "funding_pnl": float(pnl["funding_pnl"].sum()),
        "fee": float(pnl["fee_cost"].sum()),
        "slippage": float(pnl["slippage_cost"].sum()),
    }
    wf = walk_forward(
        data, base_config=cfg, candidates=("low_vol_14", "low_vol_30"),
        n_sides_grid=(3,), min_train_days=365, test_days=120,
        output_dir=out / f"wf_{scale.replace('.', '_')}", vol_target_annual=0.15,
    )
    val = run_validation(frame, cfg, output_dir=out / f"validation_{scale.replace('.', '_')}")
    oos = val["performance"]["out_of_sample"]
    decomp["wf"] = wf["walk_forward"]
    decomp["wf_gates"] = wf["gates"]
    decomp["oos"] = {k: oos[k] for k in ("total_return", "sharpe", "max_drawdown", "average_turnover")}
    decomp["oos_gates"] = val["gates"]
    decomp["deployable_oos"] = bool(val["deployable"])
    results[scale] = decomp
    print(f"[{scale}] gross={gross} full_ret={decomp['total_return']:.3f} "
          f"sharpe={decomp['sharpe']:.2f} dd={decomp['max_drawdown']:.3f} "
          f"wf_ret={wf['walk_forward']['total_return']:.3f} "
          f"wf_sharpe={wf['walk_forward']['sharpe']:.2f} "
          f"oos_ret={oos['total_return']:.3f} oos_sharpe={oos['sharpe']:.2f}",
          flush=True)

(out / "sweep_summary.json").write_text(json.dumps(results, indent=2, default=str))
print(f"Wrote {out / 'sweep_summary.json'}", flush=True)
