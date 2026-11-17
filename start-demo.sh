#!/usr/bin/env bash
# Process B: the Streamlit dashboard. Talks to the model server over HTTP only.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/streamlit ]; then
    echo "Virtual environment missing. Run ./setup.sh first." >&2
    exit 1
fi

# Load .env without overriding variables already set in the environment.
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        [[ $key =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
        [ -z "${!key+x}" ] && export "$key=$value"
    done < .env
fi

exec .venv/bin/streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false "$@"
