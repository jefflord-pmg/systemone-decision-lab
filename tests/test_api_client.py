import pytest
import requests

from api_client import (
    DeciderClient,
    DeciderConnectionError,
    DeciderHTTPError,
    DeciderResponseError,
    DeciderTimeoutError,
    Question,
    QuestionError,
    build_request,
    parse_levels,
    parse_options,
    slugify,
    validate_response,
)
from config import Settings, get_settings

ENDPOINT = "http://127.0.0.1:8000/v1/systemone"
HEALTH = "http://127.0.0.1:8000/health"

QUESTIONS = [
    Question("is_urgent", "noul", "Is this urgent?"),
    Question("department", "choice", "Which department?", {"billing": "money", "sales": None}),
    Question("severity", "score", "How severe?", ["low", "medium", "high"]),
]

# Shaped like the server's SystemOneResponse; the numbers are arbitrary test fixtures.
VALID_RESPONSE = {
    "model": "test-model",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.5},
        "department": {"type": "choice", "choice": "billing", "confidence": 0.2,
                       "probabilities": {"billing": 0.6, "sales": 0.4}},
        "severity": {"type": "score", "score": 1.0, "confidence": 0.3,
                     "legend": {"0": "low", "1": "medium", "2": "high"},
                     "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25}},
    },
    "usage": {"input_tokens": 10, "output_tokens": 3},
    "latency_ms": 12.5,
}


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", raise_json=False):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.reason = "reason"
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._json


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def _respond(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.exc:
            raise self.exc
        return self.response

    def get(self, url, **kwargs):
        return self._respond("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._respond("POST", url, **kwargs)


def client(session):
    return DeciderClient(ENDPOINT, HEALTH, timeout=5, session=session)


def test_build_request_shape():
    payload = build_request("state text", QUESTIONS)
    assert payload == {
        "state": "state text",
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "department": {"type": "choice", "instructions": "Which department?",
                           "criteria": {"billing": "money", "sales": None}},
            "severity": {"type": "score", "instructions": "How severe?",
                         "criteria": ["low", "medium", "high"]},
        },
    }


@pytest.mark.parametrize("question", [
    Question("q", "choice", "Pick", {"only": None}),
    Question("q", "score", "Rate", ["one"]),
    Question("q", "score", "Rate", [str(i) for i in range(11)]),
    Question("q", "score", "Rate", ["a", "a"]),
    Question("q", "noul", "   "),
    Question("q", "other", "Huh?"),
])
def test_invalid_questions_rejected(question):
    with pytest.raises(QuestionError):
        build_request("s", [question])


def test_duplicate_keys_and_empty_rejected():
    with pytest.raises(QuestionError):
        build_request("s", [])
    with pytest.raises(QuestionError):
        build_request("s", [Question("a", "noul", "x"), Question("a", "noul", "y")])


def test_parsers():
    assert parse_options("billing: money\n\nsales\n tech : bugs ") == {
        "billing": "money", "sales": None, "tech": "bugs"}
    assert parse_levels("low, medium\nhigh") == ["low", "medium", "high"]
    assert slugify("Is this urgent?") == "is_this_urgent"
    assert slugify("???") == "question"


def test_decide_success_posts_one_request():
    session = FakeSession(FakeResponse(json_data=VALID_RESPONSE))
    result = client(session).decide("state", QUESTIONS)
    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", ENDPOINT)
    assert set(kwargs["json"]["questions"]) == {"is_urgent", "department", "severity"}
    assert kwargs["timeout"] == 5
    assert result.data == VALID_RESPONSE
    assert result.server_latency_ms == 12.5
    assert result.round_trip_ms >= 0


def test_decide_connection_error():
    session = FakeSession(exc=requests.ConnectionError("refused"))
    with pytest.raises(DeciderConnectionError):
        client(session).decide("state", QUESTIONS)


def test_decide_timeout():
    session = FakeSession(exc=requests.Timeout("slow"))
    with pytest.raises(DeciderTimeoutError):
        client(session).decide("state", QUESTIONS)


def test_decide_http_error_includes_detail():
    session = FakeSession(FakeResponse(422, {"detail": [{"msg": "too long"}]}))
    with pytest.raises(DeciderHTTPError) as err:
        client(session).decide("state", QUESTIONS)
    assert err.value.status_code == 422
    assert "too long" in err.value.detail


def test_decide_invalid_json():
    session = FakeSession(FakeResponse(raise_json=True))
    with pytest.raises(DeciderResponseError):
        client(session).decide("state", QUESTIONS)


def test_validate_response_rejects_missing_or_malformed_answers():
    with pytest.raises(DeciderResponseError):
        validate_response({"answers": {}}, ["is_urgent"])
    with pytest.raises(DeciderResponseError):
        validate_response({"answers": {"is_urgent": {"type": "noul"}}}, ["is_urgent"])
    with pytest.raises(DeciderResponseError):
        validate_response(["not", "a", "dict"], ["is_urgent"])
    assert validate_response(VALID_RESPONSE, list(VALID_RESPONSE["answers"]))


def test_health_online():
    info = {"status": "ok", "model": "m", "device": "cpu"}
    status = client(FakeSession(FakeResponse(json_data=info))).health()
    assert status.online and status.info == info and status.error is None


@pytest.mark.parametrize("session", [
    FakeSession(exc=requests.ConnectionError()),
    FakeSession(exc=requests.Timeout()),
    FakeSession(FakeResponse(503, {"detail": "loading"})),
    FakeSession(FakeResponse(raise_json=True)),
])
def test_health_offline(session):
    status = client(session).health()
    assert not status.online and status.error


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("DECIDER_MODEL", "some/model")
    monkeypatch.setenv("DECIDER_DEVICE", "cuda")
    monkeypatch.setenv("DECIDER_ENDPOINT", "http://localhost:9000/v1/systemone")
    monkeypatch.setenv("DECIDER_TIMEOUT", "30")
    settings = get_settings()
    assert settings == Settings("some/model", "cuda", "http://localhost:9000/v1/systemone", 30.0)
    assert settings.health_url == "http://localhost:9000/health"


def test_live_server_answers_all_types():
    """Real request to the running model server; skipped when it is not up."""
    settings = get_settings()
    live = DeciderClient(settings.endpoint, settings.health_url, timeout=settings.timeout)
    if not live.health().online:
        pytest.skip(f"model server not running at {settings.health_url}")
    result = live.decide("My payouts have failed for three days.", QUESTIONS)
    answers = result.data["answers"]
    assert 0.0 <= answers["is_urgent"]["noul"] <= 1.0
    assert answers["department"]["choice"] in {"billing", "sales"}
    assert sum(answers["department"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-2)
    assert 0.0 <= answers["severity"]["score"] <= 2.0
