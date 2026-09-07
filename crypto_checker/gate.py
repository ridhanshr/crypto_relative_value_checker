"""Automated gate decision: consistent kill/keep/review across strategies.

Pure function of its inputs (no wall-clock inside): data_as_of is an
explicit deterministic argument (last input date); generated_at defaults to
data_as_of so artifacts stay byte-deterministic, and callers that need an
audit wall-clock pass it explicitly (it is then excluded from determinism
comparisons by construction, per contract).
"""

from dataclasses import dataclass, field
from enum import Enum


class GateStatus(Enum):
    APPROVED_CANDIDATE = "APPROVED_CANDIDATE"
    REJECTED = "REJECTED"
    FLAG_REVIEW = "FLAG_REVIEW"


@dataclass
class GateDecision:
    strategy_id: str
    status: GateStatus
    reasons: list[str]
    dsr_result: dict
    cpcv_summary: dict
    ensemble_pre_check: dict | None
    capacity_report: dict
    data_as_of: str
    generated_at: str
    reviewed_by: str | None = None
    review_decision: str | None = None
    review_timestamp: str | None = None


def evaluate_gate(strategy_id, dsr_result, cpcv_summary, ensemble_pre_check,
                  capacity_report, config, data_quality_report=None,
                  data_as_of="", generated_at=None):
    reasons = []
    status = GateStatus.APPROVED_CANDIDATE

    dsr = float(dsr_result["dsr"])
    if dsr < config.dsr_reject_threshold:
        reasons.append(f"DSR {dsr:.3f} < reject threshold {config.dsr_reject_threshold}")
        status = GateStatus.REJECTED
    elif dsr < config.dsr_high_confidence_threshold:
        reasons.append(f"DSR {dsr:.3f} berada di zona abu-abu ({config.dsr_reject_threshold}-{config.dsr_high_confidence_threshold}) — perlu review manual")
        status = GateStatus.FLAG_REVIEW

    if config.gate_auto_reject_on_oos_sharpe_negative and float(cpcv_summary["sharpe_mean"]) < 0:
        reasons.append(f"OOS Sharpe mean negatif: {float(cpcv_summary['sharpe_mean']):.3f}")
        status = GateStatus.REJECTED

    if config.gate_auto_reject_on_dd:
        dd_wf = cpcv_summary.get("max_dd_walk_forward")
        dd_oos = cpcv_summary.get("max_dd_oos")
        if dd_wf is not None and dd_wf < config.dd_max_threshold_wf:
            reasons.append(f"DD walk-forward {dd_wf:.2%} melewati threshold {config.dd_max_threshold_wf:.2%}")
            status = GateStatus.REJECTED
        if dd_oos is not None and dd_oos < config.dd_max_threshold_oos:
            reasons.append(f"DD OOS {dd_oos:.2%} melewati threshold {config.dd_max_threshold_oos:.2%}")
            status = GateStatus.REJECTED

    if cpcv_summary.get("regime_concentration_flag"):
        reasons.append("Profit terkonsentrasi pada satu regime market")
        if status != GateStatus.REJECTED:
            status = GateStatus.FLAG_REVIEW

    if config.gate_flag_review_on_ensemble_corr and ensemble_pre_check and ensemble_pre_check.get("max_pairwise_corr", 0) > config.ensemble_max_pairwise_corr:
        reasons.append(f"Korelasi antar-member ensemble {ensemble_pre_check['max_pairwise_corr']:.2f} > threshold")
        if status != GateStatus.REJECTED:
            status = GateStatus.FLAG_REVIEW

    for tier, report in (capacity_report or {}).items():
        if isinstance(report, dict) and report.get("status") == "BREACH":
            reasons.append(f"Capacity breach di AUM tier {tier}")
            # Breach records an AUM constraint, never a kill by itself.

    if data_quality_report is not None:
        unexplained = int(data_quality_report.get("halt_unexplained_count", 0))
        if unexplained > config.data_quality_max_unexplained_halts:
            reasons.append(f"Unexplained halts {unexplained} > threshold {config.data_quality_max_unexplained_halts}")
            # Data-quality issues force at minimum FLAG_REVIEW, never
            # escalate an existing REJECTED and never reject alone.
            if status != GateStatus.REJECTED and status != GateStatus.FLAG_REVIEW:
                status = GateStatus.FLAG_REVIEW

    return GateDecision(
        strategy_id=strategy_id, status=status, reasons=reasons,
        dsr_result=dsr_result, cpcv_summary=cpcv_summary,
        ensemble_pre_check=ensemble_pre_check, capacity_report=capacity_report,
        data_as_of=data_as_of, generated_at=generated_at if generated_at is not None else data_as_of,
    )


def promote_after_review(decision, reviewed_by, review_decision, review_timestamp):
    """Manual review step: the ONLY path from FLAG_REVIEW to a final verdict.

    Refuses to promote while review fields are missing (fail-closed).
    Returns "APPROVED" or "REJECTED" for the envelope decision field.
    """
    if decision.status != GateStatus.FLAG_REVIEW:
        raise ValueError("Only FLAG_REVIEW decisions go through manual review")
    if not reviewed_by or review_decision not in ("APPROVE", "REJECT") or not review_timestamp:
        raise ValueError("reviewed_by/review_decision/review_timestamp are all required")
    decision.reviewed_by = reviewed_by
    decision.review_decision = review_decision
    decision.review_timestamp = review_timestamp
    return "APPROVED" if review_decision == "APPROVE" else "REJECTED"
