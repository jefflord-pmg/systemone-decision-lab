"""Repeated and concurrent decision requests, with latency statistics.

The server handles each request on its own thread but does not batch separate requests
together, so concurrency shows how one model shares the device between callers.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from api_client import DeciderClient, DeciderError, DecisionResult, Question, build_request

MAX_REPEATS = 200
MAX_CONCURRENCY = 16


@dataclass
class Sample:
    index: int  # order the request was submitted
    started_ms: float  # relative to the start of the run
    ended_ms: float
    round_trip_ms: float | None = None
    server_ms: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class LatencyStats:
    count: int
    min: float
    avg: float
    p50: float
    p95: float
    max: float


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(values: list[float]) -> LatencyStats | None:
    if not values:
        return None
    return LatencyStats(len(values), min(values), sum(values) / len(values),
                        percentile(values, 50), percentile(values, 95), max(values))


@dataclass
class BenchmarkResult:
    samples: list[Sample]  # in completion order
    wall_ms: float
    repeats: int
    concurrency: int
    discard: int
    last: DecisionResult | None  # the most recently completed successful response

    @property
    def succeeded(self) -> int:
        return sum(s.ok for s in self.samples)

    @property
    def failed(self) -> int:
        return len(self.samples) - self.succeeded

    @property
    def first_error(self) -> str | None:
        return next((s.error for s in self.samples if s.error), None)

    @property
    def measured(self) -> list[Sample]:
        """Successful samples, excluding the first `discard` requests submitted."""
        return [s for s in self.samples if s.ok and s.index >= self.discard]

    @property
    def throughput(self) -> float:
        """Successful requests per second over the whole run, warm-up included."""
        return self.succeeded / (self.wall_ms / 1000) if self.wall_ms > 0 else 0.0

    def round_trip_stats(self) -> LatencyStats | None:
        return summarize([s.round_trip_ms for s in self.measured if s.round_trip_ms is not None])

    def server_stats(self) -> LatencyStats | None:
        return summarize([s.server_ms for s in self.measured if s.server_ms is not None])

    def overhead_stats(self) -> LatencyStats | None:
        """Round trip minus server time: waiting for a server thread, plus HTTP."""
        return summarize([s.round_trip_ms - s.server_ms for s in self.measured
                          if s.round_trip_ms is not None and s.server_ms is not None])


def run_benchmark(
    make_client: Callable[[], DeciderClient],
    state: str,
    questions: list[Question],
    repeats: int = 1,
    concurrency: int = 1,
    discard: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> BenchmarkResult:
    """Send `repeats` identical requests, at most `concurrency` in flight at once.

    Each worker thread gets its own client, since a requests.Session is not guaranteed to
    be thread-safe. Failed requests are recorded and the run continues. `on_progress` is
    called from the calling thread as (completed, total).
    """
    build_request(state, questions)  # invalid questions fail before anything is sent
    repeats = max(1, min(repeats, MAX_REPEATS))
    concurrency = max(1, min(concurrency, MAX_CONCURRENCY, repeats))
    discard = max(0, min(discard, repeats - 1))

    local = threading.local()
    clients: list[DeciderClient] = []
    clients_lock = threading.Lock()
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    def call(index: int) -> tuple[Sample, DecisionResult | None]:
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = make_client()
            with clients_lock:
                clients.append(client)
        sample = Sample(index, elapsed_ms(), 0.0)
        try:
            result = client.decide(state, questions)
        except DeciderError as exc:
            sample.ended_ms, sample.error = elapsed_ms(), str(exc)
            return sample, None
        sample.ended_ms = elapsed_ms()
        sample.round_trip_ms = result.round_trip_ms
        sample.server_ms = result.server_latency_ms
        return sample, result

    samples: list[Sample] = []
    last: DecisionResult | None = None
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(call, i) for i in range(repeats)]
            for done, future in enumerate(as_completed(futures), 1):
                sample, result = future.result()
                samples.append(sample)
                if result is not None:
                    last = result
                if on_progress:
                    on_progress(done, repeats)
    finally:
        for client in clients:
            client.session.close()
    return BenchmarkResult(samples, elapsed_ms(), repeats, concurrency, discard, last)
