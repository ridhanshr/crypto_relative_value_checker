from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from .core import CheckerConfig, check_strategy
from .validation import run_validation
from .selection import walk_forward


REGIME_MIN_PROFIT = 0.5


# Deployment verdict uses ONLY these hard gates. Everything else in
# "gates" is informational context. If any hard gate fails,
# deployable is False and evaluation stops there.
HARD_GATES = ("wf_positive", "oos_positive", "no_risk_violations", "no_capacity_violations")


def evaluate_deployable(gates):
    """Pure deployability verdict from a gates mapping.

    Missing hard-gate keys count as failed (fail-closed).
    """
    return all(bool(gates.get(name, False)) for name in HARD_GATES)


def deployment_decision(data, output_dir="reports/research", min_train_days=365, test_days=120, candidates=None, n_sides_grid=(3,), vol_target_annual=0.20, rebalance_every=5, fee_rate=0.0004, slippage_rate=0.0005, require_funding=False, spread_check_passed=False, delist_mode="error"):
    from .signals import build_signals, available_candidates
    built = build_signals(data)
    all_candidates = available_candidates(built)
    wf_candidates = candidates or all_candidates

    ic_pass = []
    ic_table = []
    from .research import rank_ic_analysis
    ic = rank_ic_analysis(built, all_candidates)
    for _, row in ic.iterrows():
        ic_table.append(row.to_dict())
        if not pd.isna(row["ic_tstat"]) and abs(row["ic_tstat"]) >= 2.0:
            ic_pass.append(row["signal"])

    wf = walk_forward(data, base_config=CheckerConfig(fee_rate=fee_rate, slippage_rate=slippage_rate, rebalance_every=rebalance_every, require_funding=require_funding, delist_mode=delist_mode), min_train_days=min_train_days, test_days=test_days, candidates=wf_candidates, n_sides_grid=n_sides_grid, vol_target_annual=vol_target_annual, output_dir=output_dir)
    best_signal = pd.Series([f["signal"] for f in wf["folds"]]).mode().iloc[0]
    best_n = int(pd.Series([f["n_sides"] for f in wf["folds"]]).mode().iloc[0])
    config = CheckerConfig(n_long=best_n, n_short=best_n, fee_rate=fee_rate, slippage_rate=slippage_rate, signal_column=best_signal, vol_target_annual=vol_target_annual, rebalance_every=rebalance_every, require_funding=require_funding, delist_mode=delist_mode)
    validation = run_validation(built.dropna(subset=[best_signal]).reset_index(drop=True), config, output_dir=output_dir)

    regime_rows = validation["regimes"]
    profitable_regimes = sum(1 for r in regime_rows if r["total_return"] > 0)
    gates = {
        "ic_signal_found": bool(ic_pass),
        "wf_positive": wf["gates"]["wf_positive"],
        "wf_sharpe_above_one": wf["gates"]["wf_sharpe_above_one"],
        "wf_drawdown_gate": wf["gates"]["wf_drawdown_above_minus_20pct"],
        "wf_stress_positive": wf["gates"]["stress_positive"],
        "majority_folds_profitable": wf["gates"]["majority_folds_profitable"],
        "static_train_positive": validation["gates"]["train_positive"],
        "static_validation_positive": validation["gates"]["validation_positive"],
        "static_oos_positive": validation["gates"]["oos_positive"],
        "static_oos_sharpe_above_one": validation["gates"]["oos_sharpe_above_one"],
        "static_oos_drawdown_gate": validation["gates"]["oos_drawdown_above_minus_20pct"],
        "static_stress_positive": validation["gates"]["stress_positive"],
        "no_risk_violations": validation["gates"]["no_risk_violations"],
        "no_capacity_violations": validation["gates"].get("no_capacity_violations", False),
        "oos_positive": validation["gates"]["oos_positive"],
        "spread_check_passed": bool(spread_check_passed),
        "majority_regimes_profitable": bool(profitable_regimes >= len(regime_rows) * REGIME_MIN_PROFIT),
    }
    decision = {
        "best_signal_walk_forward": str(best_signal),
        "best_n_sides": best_n,
        "config": {"n_long": best_n, "n_short": best_n, "fee_rate": fee_rate, "slippage_rate": slippage_rate, "vol_target_annual": vol_target_annual, "rebalance_every": rebalance_every},
        "ic_pass_signals": ic_pass,
        "ic_table": ic_table,
        "walk_forward": wf,
        "validation_best_signal": {k: v for k, v in validation.items() if k != "regimes"},
        "regimes": regime_rows,
        "profitable_regimes": profitable_regimes,
        "total_regimes": len(regime_rows),
        "gates": gates,
        "hard_gates": list(HARD_GATES),
    }
    decision["deployable"] = evaluate_deployable(gates)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "deployment_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    return decision
