import pytest

from api_client import Question
from compare_backends import answers_agree, compare, top_answer
from config import get_settings

ENV_KEYS = (
    "DECIDER_BACKEND", "DECIDER_MODEL", "DECIDER_DEVICE", "DECIDER_ENDPOINT",
    "DECIDER_API_MODEL", "DECIDER_API_DEVICE", "DECIDER_API_ENDPOINT", "DECIDER_TIMEOUT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_backend_presets(monkeypatch):
    assert get_settings("strands").endpoint == "http://127.0.0.1:8000/v1/systemone"
    local = get_settings("opendecider")
    assert local.endpoint == "http://127.0.0.1:8001/v1/systemone"
    assert local.model == "manjunathshiva/opendecider-nano"
    remote = get_settings("lmstudio")
    assert remote.endpoint == "http://127.0.0.1:8001/v1/systemone"
    assert remote.device == "remote"
    assert remote.backend == "lmstudio"
    monkeypatch.setenv("DECIDER_BACKEND", "lmstudio")
    assert get_settings().backend == "lmstudio"


def test_overrides_do_not_leak_between_backends(monkeypatch):
    monkeypatch.setenv("DECIDER_ENDPOINT", "http://localhost:9100/v1/systemone")
    monkeypatch.setenv("DECIDER_API_ENDPOINT", "http://localhost:9200/v1/systemone")
    monkeypatch.setenv("DECIDER_API_MODEL", "custom/model")
    assert get_settings("strands").endpoint == "http://localhost:9100/v1/systemone"
    other = get_settings("opendecider")
    assert other.endpoint == "http://localhost:9200/v1/systemone"
    assert other.model == "custom/model"
    assert other.health_url == "http://localhost:9200/health"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown DECIDER_BACKEND"):
        get_settings("nope")


def test_top_answer_and_agreement():
    assert top_answer({"type": "noul", "noul": 0.8}) == ("noul", True)
    assert top_answer({"type": "choice", "choice": "billing"}) == ("choice", "billing")
    assert top_answer({"type": "score", "score": 1.2}) == ("score", 1)
    assert top_answer({"type": "score", "score": 1.2, "level": 2}) == ("score", 2)

    strands = {"type": "score", "score": 1.98, "probabilities": {}, "legend": {}}
    open_decider = {"type": "score", "score": 1.0, "level": 2}
    assert answers_agree(strands, open_decider)
    assert not answers_agree({"type": "noul", "noul": 0.8},
                             {"type": "noul", "noul": 0.4})
    assert not answers_agree({"type": "choice", "choice": "billing"},
                             {"type": "choice", "choice": "sales"})


class FakeClient:
    def __init__(self, answers=None, error=None):
        self.answers = answers or {}
        self.error = error

    def decide(self, state, questions):
        if self.error:
            raise self.error
        from api_client import DecisionResult
        return DecisionResult({"answers": self.answers}, 12.0)


def test_compare_collects_agreement_and_errors():
    scenario = {
        "state": "charged twice",
        "questions": [Question("urgent", "noul", "Urgent?", None)],
    }
    from api_client import DeciderConnectionError

    clients = {
        "strands": FakeClient({"urgent": {"type": "noul", "noul": 0.9}}),
        "opendecider": FakeClient({"urgent": {"type": "noul", "noul": 0.7}}),
        "lmstudio": FakeClient(error=DeciderConnectionError("offline")),
    }
    report = compare(clients, {"demo": scenario})
    row = report["scenarios"][0]
    assert row["agreement"]["urgent"] is None  # one backend errored
    assert "error" in row["results"]["lmstudio"]
    assert row["results"]["strands"]["latency_ms"] == 12.0

    clients["lmstudio"] = FakeClient({"urgent": {"type": "noul", "noul": 0.1}})
    report = compare(clients, {"demo": scenario})
    assert report["scenarios"][0]["agreement"]["urgent"] is False
