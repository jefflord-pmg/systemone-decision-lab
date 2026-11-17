"""HTTP client for the strands-decider SystemOne server.

Talks to the model only over HTTP: POST /v1/systemone and GET /health.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

QUESTION_TYPES = ("noul", "choice", "score")
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MIN_CHOICE_OPTIONS = 2


class DeciderError(Exception):
    """Base error for anything that stops a decision from being returned."""


class DeciderConnectionError(DeciderError):
    pass


class DeciderTimeoutError(DeciderError):
    pass


class DeciderHTTPError(DeciderError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class DeciderResponseError(DeciderError):
    """The server answered, but not with a usable SystemOne response."""


class QuestionError(ValueError):
    """A question cannot be sent as defined."""


@dataclass
class Question:
    """One question in a request.

    criteria is None for noul, a dict of option -> description (or None) for choice,
    and an ascending list of levels for score.
    """

    key: str
    type: str
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None

    def validate(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise QuestionError(f"{self.key}: unknown type {self.type!r}")
        if not self.instructions.strip():
            raise QuestionError(f"{self.key}: the question text is empty")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < MIN_CHOICE_OPTIONS:
                raise QuestionError(f"{self.key}: a choice needs at least 2 options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not (
                MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS
            ):
                raise QuestionError(f"{self.key}: a score needs between 2 and 10 levels")
            if len(set(self.criteria)) != len(self.criteria):
                raise QuestionError(f"{self.key}: score levels must be unique")

    def to_payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions.strip()}
        if self.type == "choice":
            body["criteria"] = dict(self.criteria)  # type: ignore[arg-type]
        elif self.type == "score":
            body["criteria"] = list(self.criteria)  # type: ignore[arg-type]
        return body


def slugify(text: str, fallback: str = "question") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40].strip("_")
    return slug or fallback


def parse_options(text: str) -> dict[str, str | None]:
    """Parse one option per line, as `label` or `label: description`."""
    options: dict[str, str | None] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        label, _, description = line.partition(":")
        label = label.strip()
        if label:
            options[label] = description.strip() or None
    return options


def parse_levels(text: str) -> list[str]:
    """Parse ordered levels, one per line or comma-separated, lowest first."""
    parts = re.split(r"[\n,]", text)
    return [p.strip() for p in parts if p.strip()]


def build_request(state: str, questions: list[Question]) -> dict[str, Any]:
    if not questions:
        raise QuestionError("Add at least one question")
    keys = [q.key for q in questions]
    if len(set(keys)) != len(keys):
        raise QuestionError("Question keys must be unique")
    for q in questions:
        q.validate()
    return {"state": state, "questions": {q.key: q.to_payload() for q in questions}}


def validate_response(data: Any, question_keys: list[str]) -> dict[str, Any]:
    """Check the response has an answer of the right shape for every question."""
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise DeciderResponseError("Response has no 'answers' object")
    answers = data["answers"]
    for key in question_keys:
        answer = answers.get(key)
        if not isinstance(answer, dict):
            raise DeciderResponseError(f"No answer for question {key!r}")
        kind = answer.get("type")
        if kind == "noul":
            required = {"noul": (int, float)}
        elif kind == "choice":
            required = {"choice": str, "probabilities": dict}
        elif kind == "score":
            required = {"score": (int, float), "probabilities": dict, "legend": dict}
        else:
            raise DeciderResponseError(f"Answer {key!r} has unknown type {kind!r}")
        for name, expected in required.items():
            if not isinstance(answer.get(name), expected):
                raise DeciderResponseError(f"Answer {key!r} is missing a valid {name!r}")
    return data


@dataclass
class HealthStatus:
    online: bool
    latency_ms: float | None = None
    info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DecisionResult:
    data: dict[str, Any]
    round_trip_ms: float

    @property
    def server_latency_ms(self) -> float | None:
        value = self.data.get("latency_ms")
        return float(value) if isinstance(value, (int, float)) else None


class DeciderClient:
    def __init__(self, endpoint: str, health_url: str, timeout: float = 180.0,
                 session: requests.Session | None = None):
        self.endpoint = endpoint
        self.health_url = health_url
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self, timeout: float = 3.0) -> HealthStatus:
        started = time.perf_counter()
        try:
            response = self.session.get(self.health_url, timeout=timeout)
        except requests.ConnectionError:
            return HealthStatus(False, error="Server not reachable (offline, or still loading the model)")
        except requests.Timeout:
            return HealthStatus(False, error="Health check timed out (server busy or starting)")
        except requests.RequestException as exc:
            return HealthStatus(False, error=str(exc))
        latency = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return HealthStatus(False, latency, error=f"Health check returned HTTP {response.status_code}")
        try:
            info = response.json()
        except ValueError:
            return HealthStatus(False, latency, error="Health check returned invalid JSON")
        if not isinstance(info, dict):
            return HealthStatus(False, latency, error="Health check returned unexpected JSON")
        return HealthStatus(info.get("status") == "ok", latency, info=info,
                            error=None if info.get("status") == "ok" else "Server reports not ready")

    def decide(self, state: str, questions: list[Question]) -> DecisionResult:
        payload = build_request(state, questions)
        started = time.perf_counter()
        try:
            response = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
        except requests.Timeout as exc:
            raise DeciderTimeoutError(
                f"No answer within {self.timeout:.0f} s. CPU inference can be slow; "
                "raise DECIDER_TIMEOUT if needed."
            ) from exc
        except requests.ConnectionError as exc:
            raise DeciderConnectionError(
                f"Cannot reach {self.endpoint}. Start it with ./start-model.sh and wait for "
                "it to finish loading."
            ) from exc
        except requests.RequestException as exc:
            raise DeciderError(str(exc)) from exc
        round_trip = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            raise DeciderHTTPError(response.status_code, _error_detail(response))
        try:
            data = response.json()
        except ValueError as exc:
            raise DeciderResponseError("Server returned invalid JSON") from exc
        validate_response(data, [q.key for q in questions])
        return DecisionResult(data, round_trip)


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500] or response.reason or "no detail"
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):
        return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return str(detail)
