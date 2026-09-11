import pytest

from crypto_checker.strategy_state import StrategyState, StrategyStateMachine, state_from_validation


def test_research_rejected_and_reopen():
    machine = StrategyStateMachine("s")
    machine.transition(StrategyState.REJECTED, "checker", "2026-01-01", "failed gate")
    machine.transition(StrategyState.RESEARCH, "researcher", "2026-01-02", "new hypothesis")
    assert machine.state is StrategyState.RESEARCH
    assert len(machine.events) == 2


def test_flag_review_requires_explicit_human_promotion():
    machine = StrategyStateMachine("s")
    machine.transition(StrategyState.FLAG_REVIEW, "checker", "2026-01-01", "review required")
    with pytest.raises(ValueError):
        machine.transition(StrategyState.APPROVED_PAPER, "reviewer", "2026-01-02", "skip candidate")
    machine.transition(StrategyState.APPROVED_CANDIDATE, "reviewer", "2026-01-02", "manual approval")
    assert machine.state is StrategyState.APPROVED_CANDIDATE


def test_live_requires_paper():
    machine = StrategyStateMachine("s")
    with pytest.raises(ValueError):
        machine.transition(StrategyState.APPROVED_LIVE, "operator", "2026-01-01", "skip paper")
    machine.transition(StrategyState.APPROVED_CANDIDATE, "reviewer", "2026-01-01", "candidate approved")
    machine.transition(StrategyState.APPROVED_PAPER, "operator", "2026-01-02", "paper gate passed")
    machine.transition(StrategyState.APPROVED_LIVE, "operator", "2026-02-02", "live gate passed")
    assert machine.state is StrategyState.APPROVED_LIVE


def test_validation_mapping_never_grants_live():
    rejected = state_from_validation({"status": "SUCCESS", "decision": "REJECTED", "review_required": False}, "s", "2026-01-01")
    assert rejected.state is StrategyState.REJECTED
    review = state_from_validation({"status": "SUCCESS", "decision": None, "review_required": True}, "s", "2026-01-01")
    assert review.state is StrategyState.FLAG_REVIEW
    approved = state_from_validation({"status": "SUCCESS", "decision": "APPROVED", "review_required": False}, "s", "2026-01-01")
    assert approved.state is StrategyState.APPROVED_CANDIDATE
    assert approved.state is not StrategyState.APPROVED_LIVE
