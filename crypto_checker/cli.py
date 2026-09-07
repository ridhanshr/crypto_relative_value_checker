import argparse
import pandas as pd
from .core import CheckerConfig, check_strategy, write_reports
from .signal_audit import audit_signals
from .validation import run_validation
from .signals import build_signals, available_candidates
from .selection import walk_forward
from .preflight import validate_dataset, write_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit cross-sectional crypto strategy")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports")
    parser.add_argument("--n-long", type=int, default=3)
    parser.add_argument("--n-short", type=int, default=3)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--signal-lookback", type=int, default=1)
    parser.add_argument("--min-signal-gap", type=float, default=0.0)
    parser.add_argument("--signal", default="momentum_30", help="Signal column to rank on; 'signal' keeps the raw input column")
    parser.add_argument("--enforce-risk-limits", action="store_true", help="Hard-stop the run when a risk limit is breached (production mode)")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--research", action="store_true", help="Run the full factor research pipeline and deployment decision")
    parser.add_argument("--liquidity-column", default="", help="Column for liquidity-tiered costs, e.g. quote_volume")
    parser.add_argument("--cost-preset", default="liquid", choices=["liquid", "midcap"], help="liquid=flat 5+5bps taker; midcap=tiered fee 5bps + slippage 5/10/20/40bps by trailing volume percentile")
    parser.add_argument("--slippage-mode", default="tier", choices=["tier", "spread"], help="tier=fixed tier floors; spread=max(measured full spread, tier floor) from --spread-csv")
    parser.add_argument("--spread-csv", default="", help="Path to spread_check.csv (asset,spread_bps); required when --slippage-mode=spread")
    _run(parser.parse_args(argv))


def _run(args):
    liquidity_tiers = ()
    if args.cost_preset == "midcap":
        liquidity_tiers = ((0.75, 0.0005, 0.0005), (0.5, 0.0005, 0.001), (0.25, 0.0005, 0.002), (0.0, 0.0005, 0.004))

    raw = pd.read_csv(args.input)
    preflight = validate_dataset(raw, require_funding=True, require_liquidity=args.cost_preset == "midcap", min_assets=max(args.n_long + args.n_short, 2), min_periods=30, expected_frequency="D")
    write_preflight(preflight, args.output)
    if not preflight["valid"]:
        raise SystemExit(f"PREFLIGHT_FAILED {preflight['errors']}")
    if args.research:
        from .decision import deployment_decision
        decision = deployment_decision(raw, output_dir=args.output)
        print(f"RESEARCH_DEPLOYABLE {decision['deployable']} best_signal={decision['best_signal_walk_forward']} gates={decision['gates']}")
        raise SystemExit(0)
    data = build_signals(raw) if args.signal != "signal" else raw
    data = data.dropna(subset=[args.signal]).reset_index(drop=True)
    config = CheckerConfig(n_long=args.n_long, n_short=args.n_short, fee_rate=args.fee_rate, slippage_rate=args.slippage_rate, signal_lookback=args.signal_lookback, min_signal_gap=args.min_signal_gap, signal_column=args.signal, enforce_risk_limits=args.enforce_risk_limits, liquidity_column=args.liquidity_column, liquidity_tiers=liquidity_tiers, slippage_mode=args.slippage_mode, spread_csv=args.spread_csv, require_funding=True)
    audit_signals(data, args.output, column=args.signal)
    validation = run_validation(data, config, output_dir=args.output)
    print(f"DEPLOYABLE {validation['deployable']} gates={validation['gates']}")
    if not args.skip_walk_forward:
        wf = walk_forward(raw, base_config=CheckerConfig(n_long=args.n_long, n_short=args.n_short, fee_rate=args.fee_rate, slippage_rate=args.slippage_rate), output_dir=args.output)
        print(f"WALK_FORWARD {wf['deployable']} wf={wf['walk_forward']} gates={wf['gates']}")
    result = check_strategy(data, config)
    write_reports(result, args.output)
    print(f"Reports written to {args.output}")


if __name__ == "__main__":
    main()
