from api_client import build_request
from scenarios import SCENARIOS


def test_at_least_four_scenarios():
    assert len(SCENARIOS) >= 4


def test_every_scenario_builds_a_valid_request():
    for name, scenario in SCENARIOS.items():
        payload = build_request(scenario["state"], scenario["questions"])
        assert payload["state"].strip(), name
        types = {q["type"] for q in payload["questions"].values()}
        assert types == {"noul", "choice", "score"}, name


def test_spec_scenarios_present():
    support = SCENARIOS["Customer support triage"]
    assert "25 employees" in support["state"]
    by_key = {q.key: q for q in support["questions"]}
    assert list(by_key["department"].criteria) == ["billing", "technical", "sales"]
    assert by_key["severity"].criteria == ["low", "medium", "high", "critical"]

    smart = {q.key: q for q in SCENARIOS["Smart home"]["questions"]}
    assert list(smart["action"].criteria) == [
        "maintain temperature", "reduce cooling", "activate away mode"]
