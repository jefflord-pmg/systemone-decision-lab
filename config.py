"""Settings for the Streamlit app and the launch scripts, read from the environment.

The launch scripts source `.env`; this module also reads it directly so that
`streamlit run app.py` and `pytest` work without them.

`DECIDER_BACKEND` selects a preset: `strands` (the original local model server),
`opendecider` (an OpenDecider model served locally) or `lmstudio` (an OpenDecider
model served by LM Studio through `opendecider serve`). The legacy
`DECIDER_MODEL` / `DECIDER_DEVICE` / `DECIDER_ENDPOINT` variables apply to
`strands`; the other presets use `DECIDER_API_*` overrides so a `.env` tuned for
the local pool cannot accidentally point the dashboard at the wrong server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = Path(__file__).with_name(".env")

DEFAULT_MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
DEFAULT_DEVICE = "cpu"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/systemone"
DEFAULT_TIMEOUT = 180.0
DEFAULT_BACKEND = "strands"

# OpenDecider serves the same Jev-compatible /v1/systemone contract, on its own port.
OPENDECIDER_ENDPOINT = "http://127.0.0.1:8001/v1/systemone"

BACKENDS: dict[str, dict[str, str]] = {
    "strands": {"model": DEFAULT_MODEL, "device": DEFAULT_DEVICE, "endpoint": DEFAULT_ENDPOINT},
    "opendecider": {
        "model": "manjunathshiva/opendecider-nano",
        "device": "auto",
        "endpoint": OPENDECIDER_ENDPOINT,
    },
    "lmstudio": {
        "model": "opendecider-small (LM Studio)",
        "device": "remote",
        "endpoint": OPENDECIDER_ENDPOINT,
    },
}


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE lines into os.environ without overriding variables already set."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    model: str
    device: str
    endpoint: str
    timeout: float
    backend: str = "strands"

    @property
    def base_url(self) -> str:
        suffix = "/v1/systemone"
        url = self.endpoint.rstrip("/")
        return url[: -len(suffix)] if url.endswith(suffix) else url

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


def get_settings(backend: str | None = None) -> Settings:
    """Build settings for `backend`, or `DECIDER_BACKEND` when not given.

    Every preset can be overridden with DECIDER_API_ENDPOINT / DECIDER_API_MODEL /
    DECIDER_API_DEVICE. The strands preset additionally honours the original
    DECIDER_ENDPOINT / DECIDER_MODEL / DECIDER_DEVICE names.
    """
    load_env_file()
    name = (backend or os.environ.get("DECIDER_BACKEND", DEFAULT_BACKEND)).strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown DECIDER_BACKEND {name!r}; choose from {', '.join(BACKENDS)}"
        )
    preset = BACKENDS[name]
    api = (
        os.environ.get("DECIDER_API_MODEL", preset["model"]),
        os.environ.get("DECIDER_API_DEVICE", preset["device"]),
        os.environ.get("DECIDER_API_ENDPOINT", preset["endpoint"]),
    )
    if name == "strands":
        api = (
            os.environ.get("DECIDER_MODEL", api[0]),
            os.environ.get("DECIDER_DEVICE", api[1]),
            os.environ.get("DECIDER_ENDPOINT", api[2]),
        )
    return Settings(
        model=api[0],
        device=api[1],
        endpoint=api[2],
        timeout=float(os.environ.get("DECIDER_TIMEOUT", DEFAULT_TIMEOUT)),
        backend=name,
    )
