#!/usr/bin/env bash
# Process A: the Strands Decider model server, optionally a pool of them.
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

# One entry point for every decision backend. OpenDecider has its own launcher.
if [ "${DECIDER_BACKEND:-strands}" != "strands" ]; then
    exec "$PWD/start-opendecider.sh" "${DECIDER_BACKEND}"
fi

MODEL="${DECIDER_MODEL:-StrandsAgents/strands-decider-2B-qwen3.5-v1-2610}"
DEVICE="${DECIDER_DEVICE:-cpu}"
HOST="${DECIDER_HOST:-127.0.0.1}"
PORT="${DECIDER_PORT:-8000}"
MAX_BATCH="${DECIDER_MAX_BATCH:-8}"
POOL_SIZE="${DECIDER_POOL_SIZE:-1}"

.venv/bin/python hardware.py || true

# Never block CPU operation: if a GPU device is configured but unusable, fall back.
if [ "$DEVICE" = "cuda" ] && ! .venv/bin/python hardware.py --test-gpu; then
    echo "DECIDER_DEVICE=$DEVICE is not usable here; falling back to cpu."
    DEVICE=cpu
fi

if ! [[ "$POOL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "DECIDER_POOL_SIZE must be a positive integer (got '$POOL_SIZE')." >&2
    exit 2
fi

ARGS=(serve "$MODEL" --device "$DEVICE" --host 127.0.0.1 --max-batch "$MAX_BATCH")
if [ "${DECIDER_NO_WARMUP:-0}" = "1" ]; then
    ARGS+=(--no-warmup)
fi

echo
echo "Starting: strands-decider ${ARGS[*]} --port ${PORT}"
echo "The first run downloads the model (several GB). The port opens only after the"
echo "model has loaded and warmed up; on CPU this can take a few minutes."
echo

if [ "$POOL_SIZE" -eq 1 ]; then
    exec .venv/bin/strands-decider "${ARGS[@]}" --port "$PORT"
fi

# Several server processes avoid one interpreter's GIL becoming the concurrency
# ceiling. Each one loads its own copy of the model into GPU memory.
PIDS=()
cleanup() {
    trap - EXIT INT TERM
    local pid
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

BACKENDS=()
for ((i = 0; i < POOL_SIZE; i++)); do
    backend_port=$((PORT + 1 + i))
    BACKENDS+=("127.0.0.1:$backend_port")
    echo "Starting model server $((i + 1))/$POOL_SIZE on port $backend_port"
    .venv/bin/strands-decider "${ARGS[@]}" --port "$backend_port" &
    PIDS+=("$!")
done

BACKEND_LIST=$(IFS=,; echo "${BACKENDS[*]}")

# Start the proxy first: the dashboard sees OFFLINE until the backends are ready,
# just as with a single server, but the port is already open.
.venv/bin/python model_pool.py proxy --host "$HOST" --port "$PORT" --backends "$BACKEND_LIST" &
PIDS+=("$!")

echo "Waiting for $POOL_SIZE model servers to load and warm up (up to 30 minutes)..."
.venv/bin/python model_pool.py wait --backends "$BACKEND_LIST" --timeout 1800
echo "All $POOL_SIZE model servers ready. Requests go to http://$HOST:$PORT"

wait "${PIDS[${#PIDS[@]}-1]}"
