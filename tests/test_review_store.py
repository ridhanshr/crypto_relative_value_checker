import pytest
from crypto_checker.review_contract import ReviewStore


def payload(run="run-1"):
    return {"strategy_id": "s", "expected_state": "FLAG_REVIEW", "reviewed_by": "user:1",
            "review_decision": "APPROVE", "review_timestamp": "2026-09-16T00:00:00Z", "source_run_id": run}


def test_review_store_conflict_actor_and_history(tmp_path):
    store = ReviewStore(tmp_path / "review.json")
    record = store.apply(payload(), "FLAG_REVIEW", actor_validator=lambda actor: actor.startswith("user:"))
    assert record["to_state"] == "APPROVED_CANDIDATE"
    assert len((tmp_path / "review.history.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="conflict"):
        store.apply(payload("run-2"), "REJECTED", actor_validator=lambda _: True)
    with pytest.raises(ValueError, match="duplicate"):
        store.apply(payload(), "FLAG_REVIEW", actor_validator=lambda _: True)
