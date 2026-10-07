import pytest
from scripts.check_behavioral_readiness import REQUIRED, readiness


def evidence():
    return {'status': 'complete', 'seed_count': 20,
            'gates': {**dict.fromkeys(REQUIRED, True), 'progression_ready': True}}


def test_partial_or_missing_evidence_cannot_unlock_navigation():
    for change in ({'status': 'partial'}, {'seed_count': 19}, {'gates': {}}):
        with pytest.raises(ValueError):
            readiness({**evidence(), **change})


def test_a_failed_gate_blocks_even_when_others_pass():
    data = evidence()
    data['gates']['useful_response'] = False
    data['gates']['progression_ready'] = False
    assert readiness(data) == ['useful_response']
    data['gates']['progression_ready'] = True
    with pytest.raises(ValueError):
        readiness(data)
    assert readiness(evidence()) == []
