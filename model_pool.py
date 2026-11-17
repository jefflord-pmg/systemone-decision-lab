"""Round-robin proxy for a pool of strands-decider servers.

`strands-decider serve` is single-process, so concurrent decisions share one
interpreter and one GPU. Running several server processes gives each one its own
GIL; this proxy keeps the dashboard's single endpoint unchanged and spreads
requests over the processes. Backends are tried in rotation, and a dead one is
skipped.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HEALTH_PATHS = ("/health", "/ping")
DECISION_PATHS = ("/v1/systemone", "/invocations")


def parse_backends(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _request(backend: str, method: str, path: str, body: bytes | None,
             timeout: float) -> tuple[int, bytes]:
    host, port = backend.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def backend_health(backend: str, timeout: float = 3.0) -> dict | None:
    """Return a backend's health payload, or None if it is not serving yet."""
    try:
        status, payload = _request(backend, "GET", "/health", None, timeout)
    except (OSError, http.client.HTTPException, ValueError):
        return None
    if status != 200:
        return None
    try:
        info = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    return info if isinstance(info, dict) and info.get("status") == "ok" else None


class Pool:
    def __init__(self, backends: list[str], timeout: float = 180.0):
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = list(backends)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._next = 0

    def _order(self) -> list[str]:
        with self._lock:
            start = self._next
            self._next = (start + 1) % len(self.backends)
        return self.backends[start:] + self.backends[:start]

    def forward(self, method: str, path: str, body: bytes | None) -> tuple[int, bytes]:
        """Try backends in rotation until one answers; 503 if none can."""
        errors = []
        for backend in self._order():
            try:
                return _request(backend, method, path, body, self.timeout)
            except (OSError, http.client.HTTPException, ValueError) as exc:
                errors.append(f"{backend}: {exc}")
        detail = {"detail": "no model server available: " + "; ".join(errors)}
        return 503, json.dumps(detail).encode()

    def health(self) -> dict | None:
        # Parallel checks keep the dashboard's short health timeout safe even
        # when one backend is stuck and the pool has several processes.
        with ThreadPoolExecutor(max_workers=len(self.backends)) as checks:
            for info in checks.map(lambda b: backend_health(b, 2.0), self.backends):
                if info is not None:
                    return info
        return None


class PoolHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 proxy; JSON responses keep Content-Length for keep-alive."""

    pool: Pool

    def log_message(self, fmt: str, *args) -> None:
        # The server processes already log; one line per proxy request is noise.
        pass

    def _send(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _forward(self, body: bytes | None = None) -> None:
        status, payload = self.pool.forward(self.command, self.path, body)
        self._send(status, payload)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in HEALTH_PATHS:
            info = self.pool.health()
            if info is None:
                self._send(503, b'{"status":"loading","detail":"model servers not ready"}')
                return
            info = dict(info)
            info["pool"] = {"backends": len(self.pool.backends)}
            self._send(200, json.dumps(info).encode())
        else:
            self._send(404, b'{"detail":"not found"}')

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in DECISION_PATHS:
            self._send(404, b'{"detail":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send(400, b'{"detail":"invalid Content-Length"}')
            return
        self._forward(self.rfile.read(length))


def wait_for_backends(backends: list[str], timeout: float, *, interval: float = 1.0,
                      log=print) -> bool:
    """Wait until every backend answers /health; return False on timeout."""
    deadline = time.monotonic() + timeout
    last_count = -1
    while True:
        ready = [b for b in backends if backend_health(b) is not None]
        if len(ready) == len(backends):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = [b for b in backends if b not in ready]
            log(f"Timed out waiting for model servers: {', '.join(missing)}")
            return False
        if len(ready) != last_count:
            log(f"Waiting for model servers: {len(ready)}/{len(backends)} ready")
            last_count = len(ready)
        time.sleep(min(interval, remaining))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    proxy = commands.add_parser("proxy", help="serve the pool endpoint")
    proxy.add_argument("--host", default=os.environ.get("DECIDER_HOST", "127.0.0.1"))
    proxy.add_argument("--port", type=int, default=int(os.environ.get("DECIDER_PORT", "8000")))
    proxy.add_argument("--backends", default=os.environ.get("DECIDER_POOL_BACKENDS", ""))
    proxy.add_argument("--timeout", type=float,
                       default=float(os.environ.get("DECIDER_TIMEOUT", "180")))

    wait = commands.add_parser("wait", help="wait for every backend to become healthy")
    wait.add_argument("--backends", default=os.environ.get("DECIDER_POOL_BACKENDS", ""))
    wait.add_argument("--timeout", type=float, default=1800.0)

    args = parser.parse_args(argv)
    backends = parse_backends(args.backends)
    if not backends:
        parser.error("no backends: pass --backends or set DECIDER_POOL_BACKENDS")

    if args.command == "wait":
        return 0 if wait_for_backends(backends, args.timeout) else 1

    pool = Pool(backends, timeout=args.timeout)
    PoolHandler.pool = pool
    server = ThreadingHTTPServer((args.host, args.port), PoolHandler)
    server.daemon_threads = True
    print(f"Pool proxy on http://{args.host}:{args.port} -> {', '.join(backends)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
