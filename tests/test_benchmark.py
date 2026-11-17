import threading
import time

import pytest

from api_client import DeciderConnectionError, DecisionResult, Question, QuestionError
from benchmark import MAX_CONCURRENCY, percentile, run_benchmark, summarize

QUESTIONS = [Question("is_urgent", "noul", "Is this urgent?")]


class FakeClient:
    """Stands in for DeciderClient; shares call bookkeeping across instances."""

    def __init__(self, shared, fail_on=(), delay=0.0):
        self.shared = shared
        self.fail_on = set(fail_on)
        self.delay = delay
        self.session = self  # run_benchmark closes client.session
        self.closed = False

    def close(self):
        self.closed = True

    def decide(self, state, questions):
        with self.shared["lock"]:
            n = self.shared["calls"]
            self.shared["calls"] += 1
            self.shared["active"] += 1
            self.shared["peak"] = max(self.shared["peak"], self.shared["active"])
        try:
            time.sleep(self.delay)
            if n in self.fail_on:
                raise DeciderConnectionError(f"call {n} failed")
            return DecisionResult({"answers": {}, "latency_ms": 10.0 + n, "n": n}, 20.0 + n)
        finally:
            with self.shared["lock"]:
                self.shared["active"] -= 1


def factory(**kwargs):
    shared = {"lock": threading.Lock(), "calls": 0, "active": 0, "peak": 0, "clients": []}

    def make():
        client = FakeClient(shared, **kwargs)
        shared["clients"].append(client)
        return client

    return make, shared


def test_percentile_and_summary():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    stats = summarize([10.0, 20.0, 30.0])
    assert (stats.count, stats.min, stats.avg, stats.p50, stats.max) == (3, 10, 20, 20, 30)
    assert summarize([]) is None


def test_single_request():
    make, shared = factory()
    bench = run_benchmark(make, "state", QUESTIONS)
    assert shared["calls"] == 1 and bench.succeeded == 1 and bench.failed == 0
    assert bench.last.data["n"] == 0
    assert bench.round_trip_stats().avg == 20.0
    assert bench.server_stats().avg == 10.0
    assert bench.overhead_stats().avg == 10.0


def test_repeats_and_concurrency_limit():
    make, shared = factory(delay=0.02)
    bench = run_benchmark(make, "state", QUESTIONS, repeats=12, concurrency=4)
    assert shared["calls"] == 12 and bench.succeeded == 12
    assert 1 < shared["peak"] <= 4
    assert len(shared["clients"]) <= 4
    assert all(c.closed for c in shared["clients"])
    assert sorted(s.index for s in bench.samples) == list(range(12))


def test_concurrency_capped_at_repeats_and_maximum():
    make, _ = factory()
    assert run_benchmark(make, "s", QUESTIONS, repeats=3, concurrency=10).concurrency == 3
    make, _ = factory()
    bench = run_benchmark(make, "s", QUESTIONS, repeats=50, concurrency=1000)
    assert bench.concurrency == MAX_CONCURRENCY


def test_partial_failures_are_recorded_and_run_continues():
    make, _ = factory(fail_on={1, 3})
    bench = run_benchmark(make, "state", QUESTIONS, repeats=5, concurrency=1)
    assert bench.succeeded == 3 and bench.failed == 2
    assert "failed" in bench.first_error
    assert bench.last.data["n"] == 4
    assert bench.round_trip_stats().count == 3


def test_all_failures_leave_no_last_result():
    make, _ = factory(fail_on=set(range(3)))
    bench = run_benchmark(make, "state", QUESTIONS, repeats=3)
    assert bench.last is None and bench.failed == 3
    assert bench.round_trip_stats() is None


def test_discard_excludes_first_requests_from_stats():
    make, _ = factory()
    bench = run_benchmark(make, "state", QUESTIONS, repeats=4, concurrency=1, discard=2)
    stats = bench.round_trip_stats()
    assert stats.count == 2 and stats.min == 22.0 and stats.max == 23.0
    assert bench.succeeded == 4  # still counted toward throughput


def test_discard_never_removes_every_request():
    make, _ = factory()
    bench = run_benchmark(make, "state", QUESTIONS, repeats=3, discard=10)
    assert bench.discard == 2 and bench.round_trip_stats().count == 1


def test_invalid_questions_fail_before_sending():
    make, shared = factory()
    with pytest.raises(QuestionError):
        run_benchmark(make, "state", [Question("q", "noul", "  ")], repeats=5)
    assert shared["calls"] == 0
