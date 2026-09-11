"""Fail-closed strategy promotion state machine for Quantara integration.

Checker owns validation verdicts and review completion. Quantara owns actual
paper/live operations after consuming this auditable state. No transition to
APPROVED_PAPER or APPROVED_LIVE is possible without explicit actor, decision,
and timestamp fields.
"""

from dataclasses import dataclass, asdict
from enum import Enum


class StrategyState(Enum):
    RESEARCH = "RESEARCH"
    REJECTED = "REJECTED"
    FLAG_REVIEW = "FLAG_REVIEW"
    APPROVED_CANDIDATE = "APPROVED_CANDIDATE"
    APPROVED_PAPER = "APPROVED_PAPER"
    APPROVED_LIVE = "APPROVED_LIVE"


@dataclass(frozen=True)
class StateEvent:
    from_state: str
    to_state: str
    actor: str
    timestamp: str
    reason: str


@dataclass
class StrategyStateMachine:
    strategy_id: str
    state: StrategyState = StrategyState.RESEARCH
    events: list[StateEvent] | None = None

    def __post_init__(self):
        self.events = list(self.events or [])

    def transition(self, target, actor, timestamp, reason):
        target = target if isinstance(target, StrategyState) else StrategyState(target)
        if not actor or not timestamp or not reason:
            raise ValueError("state transition requires actor, timestamp, and reason")
        allowed = {
            StrategyState.RESEARCH: {StrategyState.REJECTED, StrategyState.FLAG_REVIEW, StrategyState.APPROVED_CANDIDATE},
            StrategyState.FLAG_REVIEW: {StrategyState.REJECTED, StrategyState.APPROVED_CANDIDATE},
            StrategyState.APPROVED_CANDIDATE: {StrategyState.REJECTED, StrategyState.APPROVED_PAPER},
            StrategyState.APPROVED_PAPER: {StrategyState.REJECTED, StrategyState.APPROVED_LIVE},
            StrategyState.REJECTED: {StrategyState.RESEARCH},
            StrategyState.APPROVED_LIVE: {StrategyState.REJECTED},
        }
        if target not in allowed[self.state]:
            raise ValueError(f"invalid strategy transition {self.state.value} -> {target.value}")
        if self.state == StrategyState.FLAG_REVIEW and target == StrategyState.APPROVED_CANDIDATE and not actor:
            raise ValueError("FLAG_REVIEW promotion requires human actor")
        event = StateEvent(self.state.value, target.value, str(actor), str(timestamp), str(reason))
        self.events.append(event)
        self.state = target
        return self

    def to_dict(self):
        return {"strategy_id": self.strategy_id, "state": self.state.value,
                "events": [asdict(event) for event in self.events]}


def state_from_validation(envelope, strategy_id, data_as_of, actor="checker"):
    """Map checker result to initial operational state; never grants live."""
    machine = StrategyStateMachine(strategy_id)
    status = envelope.get("status")
    decision = envelope.get("decision")
    review_required = bool(envelope.get("review_required"))
    if status == "CHECKER_ERROR" or status == "FAILED_VALIDATION":
        target = StrategyState.REJECTED
        reason = f"checker status {status}"
    elif review_required:
        target = StrategyState.FLAG_REVIEW
        reason = "manual review required by gate"
    elif decision == "APPROVED":
        target = StrategyState.APPROVED_CANDIDATE
        reason = "checker validation approved candidate"
    else:
        target = StrategyState.REJECTED
        reason = "checker validation rejected strategy"
    return machine.transition(target, actor, data_as_of, reason)
