#!/usr/bin/env bash
# Serve an OpenDecider model on the same /v1/systemone endpoint as strands-decider.
#   ./start-opendecider.sh opendecider   # local PyTorch model (default: nano)
#   ./start-opendecider.sh lmstudio      # opendecider-small loaded in LM Studio
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/opendecider ]; then
    echo "OpenDecider missing. Install it with: .venv/bin/pip install \"opendecider[serve]>=0.2.1\"" >&2
    exit 1
fi

# Load .env without overriding variables already set in the environment.
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        [[ $key =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
        [ -z "${!key+x}" ] && export "$key=$value"
    done < .env
fi

BACKEND="${1:-${DECIDER_BACKEND:-opendecider}}"
PORT="${DECIDER_API_PORT:-8001}"
TIMEOUT="${DECIDER_TIMEOUT:-180}"

ARGS=(serve --host 127.0.0.1 --port "$PORT" --request-timeout-s "$TIMEOUT")

case "$BACKEND" in
    opendecider)
        MODEL="${DECIDER_API_MODEL:-manjunathshiva/opendecider-nano}"
        DEVICE="${DECIDER_API_DEVICE:-auto}"
        ARGS+=(--model "$MODEL")
        # `auto` is the CLI's own default; passing it explicitly is rejected.
        if [ -n "$DEVICE" ] && [ "$DEVICE" != "auto" ]; then
            ARGS+=(--device "$DEVICE")
        fi
        echo "Serving OpenDecider model '$MODEL' on http://127.0.0.1:$PORT (device $DEVICE)"
        ;;
    lmstudio)
        MODEL_ID="${LMSTUDIO_MODEL:-opendecider-small}"
        REMOTE="${LMSTUDIO_URL:-http://10.0.0.177:1234/v1}"
        # The client does not run this model; it reads option probabilities from
        # the LM Studio server's top_logprobs. Remote plain HTTP must be allowed.
        export OPENDECIDER_MODEL="lmstudio:$MODEL_ID"
        export OPENDECIDER_REMOTE_URL="$REMOTE"
        export OPENDECIDER_REMOTE_ALLOW_HTTP=1
        echo "Serving '$MODEL_ID' from LM Studio at $REMOTE on http://127.0.0.1:$PORT"
        echo "Load the model in LM Studio first: lms load opendecider-small@q8_0 --identifier opendecider-small"
        ;;
    *)
        echo "Unknown backend '$BACKEND' (expected opendecider or lmstudio)." >&2
        exit 2
        ;;
esac

exec .venv/bin/opendecider "${ARGS[@]}"
