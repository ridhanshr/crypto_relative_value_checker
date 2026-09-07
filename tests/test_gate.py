"""Fase 5 tests: automated gate incl. known-REJECTED fixtures (§10d)."""

import json
import pytest

from crypto_checker.core import CheckerConfig
from crypto_checker.gate import evaluate_gate, promote_after_review, GateStatus

CFG = CheckerConfig()
FIXTURES = json.load(open("tests/fixtures/known_rejected_cases.json"))


def _case(sid):
    for case in FIXTURES["cases"]:
        if case["strategy_id"] == sid:
            return case
    raise KeyError(sid)


def _gate_inputs(case, **over):
    dsr = dict(case["dsr_result"])
    cpcv = dict(case["cpcv_summary"])
    cpcv.update(over.get("cpcv", {}))
    dsr.update(over.get("dsr", {}))
    cap = {t: {"status": "BREACH"} for t in case.get("capacity_breach_tiers", [])}
    return dsr, cpcv, cap


def test_gate_rejects_known_low_vol_14_47asset_case():
    case = _case("low_vol_14_47asset")
    dsr, cpcv, cap = _gate_inputs(case)
    decision = evaluate_gate(case["strategy_id"], dsr, cpcv, None, cap, CFG, data_as_of="2026-08-31")
    assert decision.status == GateStatus.REJECTED
    assert any("DSR" in r for r in decision.reasons)
    assert decision.data_as_of == "2026-08-31"


def test_gate_rejects_known_vol_adj_momentum_30_71asset_case():
    case = _case("vol_adj_momentum_30_71asset")
    dsr, cpcv, cap = _gate_inputs(case)
    decision = evaluate_gate(case["strategy_id"], dsr, cpcv, None, cap, CFG, data_as_of="2026-08-31")
    assert decision.status == GateStatus.REJECTED
    # Fixture locked: a future engine must never promote this input.
    assert case["expected_status"] == "REJECTED"


def test_rejected_when_dsr_below_reject_threshold():
    dsr = {"dsr": 0.2}
    cpcv = {"sharpe_mean": 1.5, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}
    d = evaluate_gate("s", dsr, cpcv, None, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.REJECTED


def test_flag_review_when_dsr_in_gray_zone():
    dsr = {"dsr": 0.7}
    cpcv = {"sharpe_mean": 1.5, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}
    d = evaluate_gate("s", dsr, cpcv, None, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.FLAG_REVIEW
    assert any("abu-abu" in r for r in d.reasons)


def test_approved_when_dsr_above_high_confidence_threshold():
    dsr = {"dsr": 0.97}
    cpcv = {"sharpe_mean": 1.5, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}
    d = evaluate_gate("s", dsr, cpcv, None, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.APPROVED_CANDIDATE and d.reasons == []


def test_rejected_when_oos_sharpe_negative():
    d = evaluate_gate("s", {"dsr": 0.99}, {"sharpe_mean": -0.1, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}, None, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.REJECTED


def test_rejected_when_dd_thresholds_exceeded():
    d = evaluate_gate("s", {"dsr": 0.99}, {"sharpe_mean": 1.0, "max_dd_walk_forward": -0.30, "max_dd_oos": -0.05}, None, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.REJECTED and any("walk-forward" in r for r in d.reasons)
    d = evaluate_gate("s", {"dsr": 0.99}, {"sharpe_mean": 1.0, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.30}, None, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.REJECTED and any("OOS" in r for r in d.reasons)


def test_flag_review_when_ensemble_correlated_but_dsr_ok():
    pre = {"max_pairwise_corr": 0.88, "recommendation": "BLOCK_ENSEMBLE"}
    d = evaluate_gate("s", {"dsr": 0.99}, {"sharpe_mean": 1.0, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}, pre, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.FLAG_REVIEW


def test_capacity_breach_recorded_without_forcing_reject():
    cap = {"1000000": {"status": "BREACH", "headroom": 0.02}}
    d = evaluate_gate("s", {"dsr": 0.99}, {"sharpe_mean": 1.0, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}, None, cap, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.APPROVED_CANDIDATE
    assert any("1000000" in r for r in d.reasons)


def test_flag_review_requires_reviewed_by_before_promotion():
    d = evaluate_gate("s", {"dsr": 0.7}, {"sharpe_mean": 1.0, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}, None, {}, CFG, data_as_of="2024-01-01")
    assert d.status == GateStatus.FLAG_REVIEW
    with pytest.raises(ValueError, match="required"):
        promote_after_review(d, None, None, None)
    assert promote_after_review(d, "human", "REJECT", "2026-09-08") == "REJECTED"
    assert d.reviewed_by == "human" and d.review_decision == "REJECT"


def test_data_quality_forces_flag_review_but_never_reject_alone():
    dq = {"halt_unexplained_count": 3}
    d = evaluate_gate("s", {"dsr": 0.99}, {"sharpe_mean": 1.0, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}, None, {}, CFG,
                      data_quality_report=dq, data_as_of="2024-01-01")
    assert d.status == GateStatus.FLAG_REVIEW
    assert any("Unexplained halts" in r for r in d.reasons)


def test_data_quality_does_not_downgrade_existing_rejected_status():
    dq = {"halt_unexplained_count": 3}
    d = evaluate_gate("s", {"dsr": 0.1}, {"sharpe_mean": 1.0, "max_dd_walk_forward": -0.05, "max_dd_oos": -0.05}, None, {}, CFG,
                      data_quality_report=dq, data_as_of="2024-01-01")
    assert d.status == GateStatus.REJECTED
