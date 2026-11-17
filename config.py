"""Settings for the Streamlit app and the launch scripts, read from the environment.

The launch scripts source `.env`; this module also reads it directly so that
`streamlit run app.py` and `pytest` work without them.
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

    @property
    def base_url(self) -> str:
        suffix = "/v1/systemone"
        url = self.endpoint.rstrip("/")
        return url[: -len(suffix)] if url.endswith(suffix) else url

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


def get_settings() -> Settings:
    load_env_file()
    return Settings(
        model=os.environ.get("DECIDER_MODEL", DEFAULT_MODEL),
        device=os.environ.get("DECIDER_DEVICE", DEFAULT_DEVICE),
        endpoint=os.environ.get("DECIDER_ENDPOINT", DEFAULT_ENDPOINT),
        timeout=float(os.environ.get("DECIDER_TIMEOUT", DEFAULT_TIMEOUT)),
    )
