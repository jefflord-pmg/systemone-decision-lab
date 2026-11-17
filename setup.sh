#!/usr/bin/env bash
# Creates .venv with Python 3.12 and installs the model server and demo dependencies.
#
#   ./setup.sh          CPU-only PyTorch (default, works everywhere)
#   ./setup.sh --cuda   CUDA PyTorch for NVIDIA GPUs (Linux or WSL2), then a GPU check
#
# TORCH_INDEX_URL overrides where PyTorch comes from, e.g.
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup.sh --cuda
set -euo pipefail
cd "$(dirname "$0")"

MODE=cpu
case "${1:-}" in
    "") ;;
    --cuda) MODE=cuda ;;
    *) echo "Usage: ./setup.sh [--cuda]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- virtual environment
if command -v uv >/dev/null 2>&1; then
    # uv fetches Python 3.12 if needed and does not depend on the distro's venv package.
    uv venv --python 3.12 --seed --allow-existing .venv
elif command -v python3.12 >/dev/null 2>&1; then
    if ! python3.12 -m venv .venv; then
        echo "Creating the venv failed. On Ubuntu/Debian: sudo apt install python3.12-venv" >&2
        exit 1
    fi
else
    echo "Python 3.12 is required. Either install it, or install uv and re-run:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
echo "Using $(.venv/bin/python --version)"

PIP=(.venv/bin/python -m pip)
"${PIP[@]}" install --upgrade pip

# ---------------------------------------------------------------- PyTorch
if [ "$MODE" = "cuda" ]; then
    # Replace a CPU-only build left by an earlier ./setup.sh; pip would keep it otherwise.
    if .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.version.cuda is None else 1)" 2>/dev/null; then
        "${PIP[@]}" uninstall -y torch
    fi
    # The default PyPI wheels for Linux x86_64 are CUDA builds.
    if [ -n "${TORCH_INDEX_URL:-}" ]; then
        "${PIP[@]}" install torch --index-url "$TORCH_INDEX_URL"
    else
        "${PIP[@]}" install torch
    fi
    # Triton kernels for the Gated DeltaNet layers. Optional: without them transformers
    # uses a slower reference path. causal-conv1d is also optional and needs nvcc to build.
    "${PIP[@]}" install flash-linear-attention \
        || echo "flash-linear-attention failed to install; continuing without it (slower)."
else
    # CPU-only wheels avoid several GB of CUDA libraries.
    "${PIP[@]}" install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
fi

"${PIP[@]}" install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# ---------------------------------------------------------------- GPU check
if [ "$MODE" = "cuda" ]; then
    echo
    if .venv/bin/python hardware.py --test-gpu; then
        sed -i 's/^DECIDER_DEVICE=.*/DECIDER_DEVICE=cuda/' .env
        echo "GPU check passed: set DECIDER_DEVICE=cuda in .env."
        echo "If the model server misbehaves on the GPU, set DECIDER_DEVICE=cpu."
    else
        sed -i 's/^DECIDER_DEVICE=.*/DECIDER_DEVICE=cpu/' .env
        echo "GPU check failed: DECIDER_DEVICE stays cpu. See README 'GPU and other machines'."
    fi
fi

echo
echo "Setup complete."
echo "  Terminal 1: ./start-model.sh"
echo "  Terminal 2: ./start-demo.sh"
