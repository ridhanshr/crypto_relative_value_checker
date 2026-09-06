import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_checker.assets import canonicalize_assets
from crypto_checker.preflight import validate_dataset, write_preflight
from crypto_checker.signals import build_signals
from crypto_checker.core import CheckerConfig, check_strategy, write_reports
from crypto_checker.selection import walk_forward
from crypto_checker.spread_check import check_order_book_spread


parser = argparse.ArgumentParser(description="Validate and analyze canonical midcap panel")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="reports/midcap_analysis")
parser.add_argument("--require-funding", action="store_true")
parser.add_argument("--interval", default="1d", choices=["1d", "4h", "1h"])
args = parser.parse_args()

out = Path(args.output)
raw = pd.read_csv(args.input)
data = canonicalize_assets(raw)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
data["signal"] = data["asset_return"]
data = data.dropna(subset=["signal"]).reset_index(drop=True)
counts = data.groupby("asset")["timestamp"].nunique()
full_count = int(counts.max())
valid_assets = counts[counts == full_count].index
data = data[data.asset.isin(valid_assets)].copy()
data.to_csv(out.parent / f"{out.name}_canonical.csv", index=False)

frequency = {"1d": "D", "4h": "4h", "1h": "h"}[args.interval]
preflight = validate_dataset(data, require_funding=args.require_funding, require_liquidity=True, min_assets=6, min_periods=30, expected_frequency=frequency)
write_preflight(preflight, out)
if not preflight["valid"]:
    print(json.dumps(preflight, indent=2))
    raise SystemExit("PREFLIGHT_FAILED")

symbols = sorted(data.asset.unique())
check_order_book_spread(symbols, output=out / "spread_check.csv")
built = build_signals(data)
tiers = ((0.75, 0.0004, 0.0005), (0.5, 0.0008, 0.001), (0.25, 0.0012, 0.002), (0.0, 0.0015, 0.004))
base = dict(n_long=3, n_short=3, fee_rate=0.0004, slippage_rate=0.0005, rebalance_every=5, vol_target_annual=0.15, liquidity_column="quote_volume", liquidity_tiers=tiers)
for signal in ("low_vol_14", "low_vol_30", "reversal_1", "resid_reversal_14", "resid_reversal_30"):
    frame = built.dropna(subset=[signal]).reset_index(drop=True)
    result = check_strategy(frame, CheckerConfig(signal_column=signal, **base))
    write_reports(result, out / signal)

wf = walk_forward(data, base_config=CheckerConfig(**base), candidates=("low_vol_14", "low_vol_30", "reversal_1", "resid_reversal_14", "resid_reversal_30"), n_sides_grid=(3,), min_train_days=365, test_days=120, output_dir=out / "walk_forward", vol_target_annual=0.15)
(out / "analysis_summary.json").write_text(json.dumps({"mode": "futures" if "funding_rate" in data.columns else "price_only", "assets": symbols, "preflight": preflight, "walk_forward": wf}, indent=2, default=str), encoding="utf-8")
print(json.dumps({"mode": "futures" if "funding_rate" in data.columns else "price_only", "assets": len(symbols), "walk_forward_deployable": wf["deployable"]}, indent=2))
