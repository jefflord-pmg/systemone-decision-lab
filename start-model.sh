#!/usr/bin/env bash
# Process A: the Strands Decider model server.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/strands-decider ]; then
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

MODEL="${DECIDER_MODEL:-StrandsAgents/strands-decider-2B-qwen3.5-v1-2610}"
DEVICE="${DECIDER_DEVICE:-cpu}"
HOST="${DECIDER_HOST:-127.0.0.1}"
PORT="${DECIDER_PORT:-8000}"
MAX_BATCH="${DECIDER_MAX_BATCH:-8}"

.venv/bin/python hardware.py || true

# Never block CPU operation: if a GPU device is configured but unusable, fall back.
if [ "$DEVICE" = "cuda" ] && ! .venv/bin/python hardware.py --test-gpu; then
    echo "DECIDER_DEVICE=$DEVICE is not usable here; falling back to cpu."
    DEVICE=cpu
fi

ARGS=(serve "$MODEL" --device "$DEVICE" --host "$HOST" --port "$PORT" --max-batch "$MAX_BATCH")
if [ "${DECIDER_NO_WARMUP:-0}" = "1" ]; then
    ARGS+=(--no-warmup)
fi

echo
echo "Starting: strands-decider ${ARGS[*]}"
echo "The first run downloads the model (several GB). The port opens only after the"
echo "model has loaded and warmed up; on CPU this can take a few minutes."
echo
exec .venv/bin/strands-decider "${ARGS[@]}"
