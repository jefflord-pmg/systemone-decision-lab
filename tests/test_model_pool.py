import http.client
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model_pool import Pool, PoolHandler, parse_backends, wait_for_backends


@contextmanager
def fake_backend(name, *, status="ok"):
    state = {"status": status}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200 if state["status"] == "ok" else 503,
                       {"status": state["status"], "model": name, "device": "cuda"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            self._send(200, {"from": name, "state": request["state"], "latency_ms": 12.5})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(address, payload):
    host, port = address.rsplit(":", 1)
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    try:
        conn.request("POST", "/v1/systemone", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def get(address, path):
    host, port = address.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_parse_backends_ignores_blanks():
    assert parse_backends(" 127.0.0.1:8001, ,127.0.0.1:8002, ") == [
        "127.0.0.1:8001", "127.0.0.1:8002"]


def test_pool_round_robin_and_passthrough():
    with fake_backend("a") as (a, _), fake_backend("b") as (b, _):
        pool = Pool([a, b])
        seen = []
        for i in range(3):
            status, payload = pool.forward("POST", "/v1/systemone",
                                           json.dumps({"state": f"s{i}"}).encode())
            assert status == 200
            answer = json.loads(payload)
            seen.append(answer["from"])
            assert answer["state"] == f"s{i}"
        assert seen == ["a", "b", "a"]


def test_pool_skips_a_dead_backend():
    with fake_backend("alive") as (alive, _):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed = f"127.0.0.1:{probe.getsockname()[1]}"
        pool = Pool([closed, alive])
        status, payload = pool.forward("POST", "/v1/systemone", b'{"state":"x"}')
        assert status == 200
        assert json.loads(payload)["from"] == "alive"


def test_pool_health_uses_first_healthy_backend():
    with fake_backend("down", status="loading") as (down, down_state), \
         fake_backend("up") as (up, _):
        pool = Pool([down, up])
        assert pool.health()["model"] == "up"
        down_state["status"] = "ok"
        assert pool.health()["model"] == "down"


def test_proxy_forwards_and_aggregates_health():
    with fake_backend("one") as (one, _), fake_backend("two") as (two, _):
        handler = type("BoundHandler", (PoolHandler,), {"pool": Pool([one, two])})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            address = f"127.0.0.1:{server.server_port}"
            assert post(address, {"state": "s"})[1]["from"] == "one"
            assert post(address, {"state": "s"})[1]["from"] == "two"
            status, info = get(address, "/health")
            assert status == 200
            assert info["status"] == "ok"
            assert info["pool"] == {"backends": 2}
            assert get(address, "/nope")[0] == 404
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_wait_for_backends_success_and_timeout():
    with fake_backend("ready") as (ready, _), fake_backend("loading", status="loading") as (loading, _):
        assert wait_for_backends([ready], 2.0, interval=0.01)
        assert not wait_for_backends([loading], 0.15, interval=0.02, log=lambda *_: None)
