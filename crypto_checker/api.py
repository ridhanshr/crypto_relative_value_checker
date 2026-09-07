"""Official entry point for Quantara integration: validate_csv().

Quantara calls ONE function and reads ONE file. Never core.py, selection.py,
or any internal directly -- those are implementation details and may change
without notice. This module is the contract boundary:

* input:  CSV dataset path + options (all have production defaults),
* output: (a) reports/<run>/decision.json -- ALWAYS written, atomically,
            in every state including failures;
          (b) the same envelope as a dict (for in-process consumers).
* states: SUCCESS + APPROVED/REJECTED (exit 0),
          FAILED_VALIDATION (exit 2, decision.json still written),
          CHECKER_ERROR (exit 1, decision.json still attempted).

metrics/capacity/risk/data_quality population follows the locked table:
  SUCCESS          -> metrics/risk/capacity/data_quality populated, decision set.
  FAILED_VALIDATION-> metrics/capacity null, decision null, errors populated.
  CHECKER_ERROR    -> metrics/capacity null, decision null, errors populated.
"""

import traceback
from pathlib import Path

import pandas as pd

from .core import CheckerConfig
from .decision import deployment_decision
from .preflight import validate_dataset, write_preflight
from .signals import build_signals
from .capacity import capacity_curve, DEFAULT_AUM_LEVELS
from .schema import SCHEMA_VERSION, DEPLOYABLE_MEANING
from .io import atomic_write_json
from .gate import evaluate_gate, GateStatus

EXIT_OK = 0
EXIT_CHECKER_ERROR = 1
EXIT_FAILED_VALIDATION = 2

STATUS_FOR_EXIT = {"SUCCESS": EXIT_OK, "FAILED_VALIDATION": EXIT_FAILED_VALIDATION, "CHECKER_ERROR": EXIT_CHECKER_ERROR}


def _envelope(status, decision, deployable, gates, metrics, capacity, risk, data_quality, warnings, errors, artifacts,
              review_required=False, gate_decision=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": decision,
        "deployable": bool(deployable),
        "deployable_meaning": DEPLOYABLE_MEANING,
        "review_required": bool(review_required),
        "gate_decision": gate_decision if gate_decision is not None else {},
        "gates": gates,
        "metrics": metrics,
        "capacity": capacity,
        "risk": risk,
        "data_quality": data_quality,
        "warnings": list(warnings),
        "errors": list(errors),
        "artifacts": dict(artifacts),
    }


def _checker_error_envelope(outdir, warnings, errors, artifacts, data_quality=None):
    return _envelope("CHECKER_ERROR", None, False, {}, None, None, {}, data_quality or {},
                     warnings, errors, artifacts)


def validate_csv(input_path, output_dir="reports/validation_run", n_long=3, n_short=3,
                 fee_rate=0.0004, slippage_rate=0.0005, vol_target_annual=0.20,
                 rebalance_every=5, min_train_days=365, test_days=120,
                 ensemble_top_k=3, expected_frequency="D", tolerated_gap_days=1,
                 listing_manifest=None, capacity_aums=None, candidates=None,
                 strategy_id="strategy", cpcv_mode="search"):
    """Run the full validation pipeline on a CSV dataset.

    Returns the v1.1 result envelope (dict) and always writes
    <output_dir>/decision.json atomically, even on failure paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {"decision": str(out / "decision.json")}
    warnings: list = []
    try:
        data = pd.read_csv(input_path)
    except Exception as exc:
        envelope = _checker_error_envelope(out, [], [f"CHECKER_ERROR: cannot read input: {exc}"], artifacts)
        atomic_write_json(out / "decision.json", envelope)
        return envelope

    liquidity_col = "quote_volume" if "quote_volume" in data.columns else ""
    require_funding = "funding_rate" in data.columns
    check = validate_dataset(
        data.assign(signal=1.0) if "signal" not in data.columns else data,
        require_funding=require_funding, require_liquidity=bool(liquidity_col),
        min_assets=max(n_long + n_short, 2), min_periods=30,
        expected_frequency=expected_frequency, allow_gaps=False,
        listing_manifest=listing_manifest, tolerated_gap_days=tolerated_gap_days,
    )
    write_preflight(check, out)
    artifacts["preflight"] = str(out / "preflight.json")
    warnings.extend(check.get("warnings", []))
    gap = check.get("gap_classification", {}) or {}
    funding_gaps = check.get("funding_gap_summary", {}) or {}
    surv = check.get("survivorship_gap", {}) or {}
    data_quality = {
        "migration_halt_days": int(gap.get("migration_halt_days", 0)),
        "unexplained_interior_days": int(gap.get("unexplained_interior_days", 0)),
        "funding_active_gaps": int(funding_gaps.get("active", 0)),
        "survivorship_missing_dead": int(surv.get("n_missing_dead", 0)),
    }
    if not check.get("valid", False):
        envelope = _envelope("FAILED_VALIDATION", None, False, {}, None, None, {}, data_quality,
                             warnings, [str(e) for e in check.get("errors", [])], artifacts)
        atomic_write_json(out / "decision.json", envelope)
        return envelope

    try:
        decision = deployment_decision(
            data, output_dir=str(out), min_train_days=min_train_days, test_days=test_days,
            candidates=candidates, n_sides_grid=(n_long,), vol_target_annual=vol_target_annual, rebalance_every=rebalance_every,
            fee_rate=fee_rate, slippage_rate=slippage_rate, require_funding=require_funding,
            delist_mode="forced_exit", ensemble_top_k=ensemble_top_k,
        )
        artifacts["deployment_decision"] = str(out / "deployment_decision.json")
        artifacts["walk_forward"] = str(out / "walk_forward.json")
        artifacts["validation"] = str(out / "validation.json")
        wf_summary = decision["walk_forward"]["walk_forward"]
        oos = decision["validation_best_signal"]["performance"]["out_of_sample"]
        metrics = {
            "wf_total_return": float(wf_summary["total_return"]),
            "wf_sharpe": float(wf_summary["sharpe"]),
            "wf_drawdown": float(wf_summary["max_drawdown"]),
            "oos_total_return": float(oos["total_return"]),
            "oos_sharpe": float(oos["sharpe"]),
            "oos_drawdown": float(oos["max_drawdown"]),
            "dsr": float((decision.get("data_mining") or {}).get("dsr")) if (decision.get("data_mining") or {}).get("dsr") is not None else None,
            "turnover_daily": float(oos["average_turnover"]),
        }
        seg_risk = decision["validation_best_signal"].get("risk_violations", {}) or {}
        seg_perf = decision["validation_best_signal"].get("performance", {}) or {}
        risk = {
            "risk_violations": int(sum(int(v) for v in seg_risk.values())),
            "capacity_violations": int(sum(int(m.get("capacity_violations", 0)) for m in seg_perf.values())),
            "oos_turnover_flag": str(oos.get("turnover_flag", "UNKNOWN")),
        }
        best_signal = decision["best_signal_walk_forward"]
        best_n = int(decision["best_n_sides"])
        if liquidity_col:
            cap = capacity_curve(
                build_signals(data),
                base_config=CheckerConfig(n_long=best_n, n_short=best_n, fee_rate=fee_rate,
                                          slippage_rate=slippage_rate, signal_column=best_signal,
                                          vol_target_annual=vol_target_annual, rebalance_every=rebalance_every,
                                          require_funding=require_funding, delist_mode="forced_exit",
                                          liquidity_column=liquidity_col,
                                          liquidity_tiers=((0.5, 0.0004, 0.0005), (0.0, 0.0015, 0.004))),
                aum_levels=capacity_aums,
                output_dir=str(out / "capacity"),
            )
            artifacts["capacity"] = str(out / "capacity" / "capacity_curve.json")
            sensible = [lvl["aum"] for lvl in cap.get("levels", []) if lvl.get("sensible")]
            capacity = {"max_sensible": float(max(sensible)) if sensible else None, "status": cap.get("status", "ok")}
        else:
            capacity = {"max_sensible": None, "status": "no_liquidity_data"}
            warnings.append("Capacity not computed: no volume column; max_sensible is null, not zero")
        verdict = "APPROVED" if decision["deployable"] else "REJECTED"
        envelope = dict(decision)
        envelope.update(_envelope("SUCCESS", verdict, decision["deployable"], decision["gates"],
                                  metrics, capacity, risk, data_quality, warnings, [], artifacts))
        # v2 automated gate (search mode: summary from validation segments;
        # final_validation: full CPCV budget on the best config, once).
        gate_cfg = CheckerConfig(n_long=best_n, n_short=best_n, fee_rate=fee_rate,
                                 slippage_rate=slippage_rate, vol_target_annual=vol_target_annual,
                                 rebalance_every=rebalance_every)
        dm = decision.get("data_mining") or {}
        dsr_result = {"dsr": dm.get("dsr", 0.0), "sharpe_null_bar": dm.get("expected_sharpe_null", 0.0),
                      "observed_sharpe": float(wf_summary["sharpe"]),
                      "n_trials_used": dm.get("n_trials_used", dm.get("n_trials", 0)),
                      "skewness": dm.get("skewness", 0.0), "kurtosis": dm.get("kurtosis", 3.0)}
        perf = decision["validation_best_signal"].get("performance", {}) or {}
        oos_perf = perf.get("out_of_sample", {}) or {}
        train_perf = perf.get("train", {}) or {}
        if cpcv_mode == "final_validation":
            from .cpcv import run_cpcv, summarize_cpcv
            from .core import check_strategy
            built = build_signals(data)
            best_cfg = CheckerConfig(n_long=best_n, n_short=best_n, fee_rate=fee_rate,
                                     slippage_rate=slippage_rate, signal_column=best_signal,
                                     vol_target_annual=vol_target_annual, rebalance_every=rebalance_every,
                                     require_funding=require_funding, delist_mode="forced_exit")

            def _strategy(train_frame, test_frame):
                tr = check_strategy(train_frame, best_cfg)
                te = check_strategy(test_frame, best_cfg)
                te_pnl = te["pnl"]
                rets = pd.Series(te_pnl["return"].iloc[1:].to_numpy(),
                                 index=pd.to_datetime(te_pnl["timestamp"].iloc[1:], utc=True))
                return {"test_returns": rets,
                        "train_equity": pd.Series(tr["pnl"]["equity"].to_numpy(), dtype=float),
                        "test_equity": pd.Series(te_pnl["equity"].to_numpy(), dtype=float)}

            cpcv_summary = summarize_cpcv(run_cpcv(built, _strategy, gate_cfg))
        else:
            cpcv_summary = {"n_folds": 0, "pct_profitable_folds": 0.0,
                            "sharpe_mean": float(oos_perf.get("sharpe", 0.0)),
                            "max_dd_walk_forward": float(train_perf.get("max_drawdown", 0.0)),
                            "max_dd_oos": float(oos_perf.get("max_drawdown", 0.0)),
                            "sharpe_by_regime": {}, "regime_concentration_flag": False,
                            "provenance": "validation_segments_search_mode"}
        ens_pre = (decision.get("walk_forward") or {}).get("ensemble_pre_check")
        cap_gate = {str(lvl["aum"]): {"status": lvl.get("sqrt_impact", {}).get("status", "UNKNOWN")}
                    for lvl in cap.get("levels", [])} if liquidity_col else {}
        dq_gate = {"halt_count": int(gap.get("halt_blocks", 0)),
                   "halt_unexplained_count": int(gap.get("unexplained_blocks", 0)),
                   "halt_tolerated_count": int(gap.get("tolerated_blocks", 0)),
                   "dead_asset_days": None, "dead_asset_count": int(surv.get("n_missing_dead", 0)),
                   "universe_size": int(check.get("assets", 0))}
        gate = evaluate_gate(strategy_id, dsr_result, cpcv_summary, ens_pre, cap_gate,
                             gate_cfg, data_quality_report=dq_gate,
                             data_as_of=str(pd.to_datetime(data["timestamp"], utc=True).max().date()))
        gate_block = {"status": gate.status.value, "reasons": gate.reasons,
                      "reviewed_by": gate.reviewed_by, "review_decision": gate.review_decision,
                      "review_timestamp": gate.review_timestamp}
        review_required = gate.status == GateStatus.FLAG_REVIEW
        if gate.status == GateStatus.REJECTED:
            verdict, deployable_final = "REJECTED", False
        elif gate.status == GateStatus.FLAG_REVIEW:
            verdict, deployable_final = None, False
        else:
            verdict, deployable_final = "APPROVED", bool(decision["deployable"])
        envelope.update({"decision": verdict, "deployable": deployable_final,
                         "review_required": bool(review_required), "gate_decision": gate_block})
        atomic_write_json(out / "decision.json", envelope)
        return envelope
    except Exception as exc:
        trace = traceback.format_exc(limit=5)
        envelope = _checker_error_envelope(out, warnings, [f"CHECKER_ERROR: {type(exc).__name__}: {exc}", trace], artifacts, data_quality)
        atomic_write_json(out / "decision.json", envelope)
        return envelope
