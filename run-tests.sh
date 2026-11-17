#!/usr/bin/env bash
# Runs the test suite. The live-server test is skipped unless the model server is up.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/pytest ]; then
    echo "Virtual environment missing. Run ./setup.sh first." >&2
    exit 1
fi

exec .venv/bin/pytest -v "$@"
