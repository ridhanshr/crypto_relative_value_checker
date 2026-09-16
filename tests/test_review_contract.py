import pytest

from crypto_checker.review_contract import validate_review_update


def test_review_contract_validates_quantara_payload():
    update = validate_review_update({
        "strategy_id": "s", "expected_state": "FLAG_REVIEW",
        "reviewed_by": "user:42", "review_decision": "APPROVE",
        "review_timestamp": "2026-09-16T00:00:00Z", "source_run_id": "run-1",
    })
    assert update.reviewed_by == "user:42"


@pytest.mark.parametrize("field", ["reviewed_by", "source_run_id"])
def test_review_contract_rejects_missing_identity(field):
    payload = {
        "strategy_id": "s", "expected_state": "FLAG_REVIEW",
        "reviewed_by": "user:42", "review_decision": "APPROVE",
        "review_timestamp": "2026-09-16T00:00:00Z", "source_run_id": "run-1",
    }
    payload[field] = ""
    with pytest.raises(ValueError):
        validate_review_update(payload)
