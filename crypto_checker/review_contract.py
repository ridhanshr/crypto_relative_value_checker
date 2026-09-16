"""Validated review payload contract owned by Quantara persistence layer."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from .io import atomic_write_json


@dataclass(frozen=True)
class ReviewUpdate:
    strategy_id: str
    expected_state: str
    reviewed_by: str
    review_decision: str
    review_timestamp: str
    source_run_id: str

    def to_dict(self):
        return asdict(self)


def validate_review_update(payload):
    required = ("strategy_id", "expected_state", "reviewed_by", "review_decision", "review_timestamp", "source_run_id")
    if not isinstance(payload, dict) or any(not payload.get(key) for key in required):
        raise ValueError("review payload requires strategy_id, expected_state, reviewed_by, review_decision, review_timestamp, source_run_id")
    if payload["expected_state"] != "FLAG_REVIEW":
        raise ValueError("review expected_state must be FLAG_REVIEW")
    if payload["review_decision"] not in ("APPROVE", "REJECT"):
        raise ValueError("review_decision must be APPROVE or REJECT")
    return ReviewUpdate(**{key: str(payload[key]) for key in required})


class ReviewStore:
    """Small file adapter for tests/local Quantara integration.

    Quantara production should replace this adapter with a transactional DB
    repository using the same optimistic conflict and append-only contract.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.history_path = self.path.with_name(self.path.stem + ".history.jsonl")

    def read(self):
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def apply(self, payload, current_state, actor_validator=None):
        update = validate_review_update(payload)
        if current_state != update.expected_state:
            raise ValueError(f"review conflict: expected {update.expected_state}, current {current_state}")
        if actor_validator is not None and not actor_validator(update.reviewed_by):
            raise ValueError("reviewed_by failed actor identity validation")
        record = update.to_dict()
        record["from_state"] = current_state
        record["to_state"] = "APPROVED_CANDIDATE" if update.review_decision == "APPROVE" else "REJECTED"
        prior = self.read()
        if prior is not None and prior.get("source_run_id") == update.source_run_id:
            raise ValueError("duplicate review source_run_id")
        atomic_write_json(self.path, record)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return record
