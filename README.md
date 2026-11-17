# SystemOne Decision Lab

A local demo dashboard for [Strands Decider](https://github.com/strands-labs/strands-decider),
a small "system one" decision model. It shows the model's three question types:

| Type | Asks | Returns |
|---|---|---|
| `noul` | a yes/no question | P(yes) |
| `choice` | pick one of N options | the winning option, a probability for each option, a confidence |
| `score` | rate against ordered levels | an expected score, the distribution across levels, a confidence |

Everything runs on your machine. No AWS, cloud services or credentials are needed. The only
network access is the first-run model download from Hugging Face.

## Quick start

```bash
git clone https://github.com/jefflord-pmg/systemone-decision-lab.git
cd systemone-decision-lab
./setup.sh            # or ./setup.sh --cuda on a machine with an NVIDIA GPU (Linux or WSL2)
./start-model.sh      # terminal 1: wait for "Uvicorn running on http://127.0.0.1:8000"
./start-demo.sh       # terminal 2: then open http://localhost:8501
```

On Windows, run these inside WSL2; see [NVIDIA under WSL2](#nvidia-under-wsl2-windows).

## How it fits together

Two separate processes:

```
 Terminal 1                              Terminal 2
 ./start-model.sh                        ./start-demo.sh
 strands-decider serve  <── HTTP ──>     streamlit run app.py
 127.0.0.1:8000                          127.0.0.1:8501
   POST /v1/systemone                      api_client.py (requests)
   GET  /health
```

The Streamlit app never imports or loads the model; it only calls the server over HTTP.

## Requirements

- Linux or WSL2 on Windows
- Python 3.12. If [uv](https://docs.astral.sh/uv/) is installed, `setup.sh` uses it and it
  downloads Python 3.12 when needed. Otherwise it needs `python3.12` with the venv module.
- RAM: on CPU the 2B model loads in fp32, and this machine measured about 6.3 GB for the server.
  Leave around 10 GB free to avoid swapping. On a GPU the weights go to GPU memory instead.
- About 6 GB of disk for the model (cached in `~/.cache/huggingface`)
- For `--cuda`: a C compiler and Python headers (`sudo apt install build-essential python3-dev`)

## Setup

```bash
./setup.sh          # CPU only (works everywhere)
./setup.sh --cuda   # NVIDIA GPU, on Linux or WSL2
```

This creates `.venv` with Python 3.12, installs PyTorch, then `requirements.txt`
(strands-decider, Streamlit, requests, pytest), and copies `.env.example` to `.env`. With
`--cuda` it installs the CUDA build of PyTorch and runs a GPU check. If the check passes, it
sets `DECIDER_DEVICE=cuda` in `.env`; if it fails, the device stays `cpu`. See
[GPU and other machines](#gpu-and-other-machines).

## Run it (two terminals)

**Terminal 1: model server**

```bash
./start-model.sh
```

It prints hardware diagnostics, then starts `strands-decider serve`. The first run downloads
the model. The server opens its port only after the model has loaded **and warmed up**,
which takes a few minutes on CPU. Until then, the dashboard shows OFFLINE. It is ready when
the log shows `Uvicorn running on http://127.0.0.1:8000`. To check by hand:

```bash
curl -s localhost:8000/health
```

**Terminal 2: dashboard**

```bash
./start-demo.sh
```

Open <http://127.0.0.1:8501>. Pick a scenario in the sidebar (or "Custom (blank)"), edit the
situation and the questions, and press **Run decision**. All questions go to the server in one
POST.

To measure latency, set **Repeat** (total requests, up to 200), **Concurrency** (requests in
flight at once, up to 16) and **Discard first** (warm-up requests left out of the statistics)
before pressing **Run decision**. With more than one request, the results add a **Load test**
panel: success count, wall time, throughput, min/avg/p50/p95/max for round trip and server
latency, and a latency chart per request. The answers shown are from the last successful
response. The requests are sent by the dashboard's Python process, not the browser. See
[Concurrency](#concurrency).

## Configuration

Settings live in `.env` (see `.env.example`). Both scripts and the app read them.

| Variable | Default | Meaning |
|---|---|---|
| `DECIDER_MODEL` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | Hub id or local checkpoint. `StrandsAgents/strands-decider-2B-hobson-v19` also works. |
| `DECIDER_DEVICE` | `cpu` | `--device` for the server: `cpu`, `cuda`, `mps` |
| `DECIDER_HOST` / `DECIDER_PORT` | `127.0.0.1` / `8000` | Where the server listens |
| `DECIDER_ENDPOINT` | `http://127.0.0.1:8000/v1/systemone` | What the app calls; `/health` is derived from it |
| `DECIDER_MAX_BATCH` | `8` | Questions per forward pass. Warm-up covers every size up to this, so smaller means faster startup |
| `DECIDER_NO_WARMUP` | `0` | `1` skips warm-up: faster startup, slower first requests |
| `DECIDER_TIMEOUT` | `180` | Seconds the app waits for a decision |

## GPU and other machines

The dashboard does not care where the model runs: it only talks to the server over HTTP. The
GPU setup is entirely on the model server's side, and is controlled by two things: which
PyTorch build `setup.sh` installs, and `DECIDER_DEVICE` in `.env`. Both default to CPU.

Diagnostics (printed by `start-model.sh` and shown in the dashboard sidebar) report the OS
(including WSL), Python, the PyTorch build, CUDA and ROCm/HIP availability, and any NVIDIA or
AMD GPU found. `.venv/bin/python hardware.py --test-gpu` runs a small computation on the GPU
to check that it really works.

**Safety net.** If `DECIDER_DEVICE=cuda` but the GPU check fails, `start-model.sh` prints a
warning and starts on CPU instead. If the server starts on the GPU but misbehaves, set
`DECIDER_DEVICE=cpu` in `.env` and restart it.

The `/health` endpoint and the dashboard's status bar report the device the server is actually
using.

### NVIDIA on Linux

```bash
./setup.sh --cuda
./start-model.sh       # should report device CUDA in the dashboard
```

The default PyPI wheels of PyTorch for Linux x86_64 are CUDA builds, and need a reasonably
recent NVIDIA driver. If PyTorch cannot see the GPU, pick a wheel matching the driver from
<https://pytorch.org/get-started/locally/>, for example:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup.sh --cuda
```

`--cuda` also tries to install `flash-linear-attention`, fast Triton kernels for the model's
linear-attention layers; without it the model still works on a slower reference path.
`causal-conv1d` is another optional kernel. It needs the CUDA toolkit (`nvcc`) to build:
`.venv/bin/pip install causal-conv1d --no-build-isolation`.

**Triton needs a C compiler.** The first time a Triton kernel runs, Triton compiles a small C
helper, which needs a C compiler and the Python development headers (`Python.h`). Without
them, the server crashes during warm-up with
`RuntimeError: Failed to find C compiler. Please specify via CC environment variable`.
Minimal Ubuntu and WSL installs often have neither. Install them before `./setup.sh --cuda`:

```bash
sudo apt install build-essential python3-dev
```

`setup.sh --cuda` skips `flash-linear-attention` when these are missing and prints this
command. `hardware.py --test-gpu` runs a small Triton kernel, so the GPU check fails with the
same advice if `flash-linear-attention` is installed but cannot run; `start-model.sh` then
falls back to CPU instead of crashing. After installing the tools, restart the server; if
setup skipped the kernels, run `./setup.sh --cuda` again first. To use the GPU without the
fast kernels instead, run `.venv/bin/python -m pip uninstall -y flash-linear-attention`.

### NVIDIA under WSL2 (Windows)

WSL2 gets GPU access through the **Windows** NVIDIA driver.

1. On Windows, install a current NVIDIA driver for your GPU. Do **not** install an NVIDIA
   Linux display driver inside WSL; the Windows driver provides the GPU to WSL.
2. In the WSL terminal, `nvidia-smi` should list the GPU. If not, update the Windows driver
   and run `wsl --update` from PowerShell.
3. Clone into the Linux filesystem (for example `~/src/...`), **not** under `/mnt/c/...`. On
   `/mnt/c` file access is slow and the scripts can lose their executable bit.
4. Install uv (recommended, provides Python 3.12):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new terminal.
   Without uv on Ubuntu, install `python3.12` and `python3.12-venv` with apt.
5. Install a C compiler and the Python headers for Triton (see above):
   `sudo apt install build-essential python3-dev`.
6. Run `./setup.sh --cuda`, then the two-terminal workflow above.
7. Open <http://localhost:8501> in a Windows browser. WSL2 forwards localhost ports to Windows.

WSL2 limits its memory to a share of the host's RAM by default. On a GPU the model weights live
in GPU memory, so this rarely matters. On CPU, raise the limit if the server swaps: add
`memory=16GB` under `[wsl2]` in `%UserProfile%\.wslconfig`, then run `wsl --shutdown`.

### AMD

PyTorch has ROCm builds for supported AMD GPUs, and they also use the `cuda` device name, so
`torch.cuda.is_available()` alone does not prove an NVIDIA GPU. The diagnostics read the
build type (`torch.version.hip` or `torch.version.cuda`) instead. Strands Decider's documented
GPU path is CUDA; on ROCm, verify that a real request works before relying on it:

1. Check that the card is on AMD's ROCm support list for your ROCm version.
2. Run `./setup.sh` (CPU), then replace PyTorch with a ROCm build from
   <https://pytorch.org/get-started/locally/>, for example
   `.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm6.4`
   (pick the version matching the installed ROCm).
3. Run `.venv/bin/python hardware.py --test-gpu`. If it passes, set `DECIDER_DEVICE=cuda` in
   `.env`, start the server and send a decision with all three question types.

Older Polaris cards such as the RX 470/480/570/580 (gfx803) are not supported by current
ROCm releases, so they stay on CPU.

### What to expect

On CPU, a request with three questions took about 4–6 seconds on an 8-thread desktop, with
about 2.5 minutes of startup (download excluded). Upstream measured a median of about 115 ms
per question on an RTX 3090. Actual speed and memory depend on the GPU, the software
versions and the request size.

### Concurrency

Expect each request to get slower as you raise **Concurrency**. The cause is the model server,
not the dashboard: in the load test, "Round trip − server" (the time the dashboard and HTTP
add) stays at a few milliseconds.

- **No batching across requests.** `strands-decider serve` handles each request on its own
  thread, with no lock on CUDA, so concurrent requests all start at once. But each one runs
  its own forward passes. The server batches only the questions *within* one request
  (`DECIDER_MAX_BATCH`); it has no option to merge separate requests.
- **One process, one GPU.** The request threads share Python's interpreter lock and the GPU,
  so they mostly take turns.
- **Launch-bound workload.** A 2B model on a few hundred tokens runs many small GPU
  operations, so the overhead of launching each one dominates. WSL2 adds overhead per launch.

Throughput is the number to watch. Keep **Repeat** fixed (for example 32) and step
**Concurrency** through 1, 2, 4 and 8: while requests per second rise, the hardware is still
gaining from concurrency; once they flatten, extra concurrency only adds latency. As one
example, an RTX 3080 under WSL2 ran 6 requests with three questions each, 3 at a time, at
about 4.9 requests per second and about 590 ms average latency.

Ways to get more out of the hardware:

| Option | Effect | Caveat |
|---|---|---|
| Ask more questions per request | Questions in one request are batched and share the scenario text: the most efficient path | Depends on how the API is used |
| Install `causal-conv1d` | Faster every request (the server log warns it is using a slow fallback) | Needs the CUDA toolkit (`nvcc`) matching PyTorch's CUDA version; may need a slow source build |
| Run two server processes behind a load balancer | Avoids interpreter-lock contention | About 4–5 GB of GPU memory per copy, and both still share one GPU |
| Run on native Linux instead of WSL2 | Lower GPU launch overhead | Different environment |
| Batch across requests in the server | The real fix for concurrency | Needs changes to, or a wrapper around, `strands-decider`; requests with different scenarios cannot be merged simply |

## Tests

```bash
./run-tests.sh
```

The unit tests cover the API client (request building, validation, connection, timeout,
HTTP and invalid-response errors, health checks), the load-test statistics (repeats, the
concurrency limit, partial failures, discarded warm-up requests) and the scenarios, using
fakes instead of a real server.
`test_live_server_answers_all_types` sends a real request with all three question types when the
model server is running, and is skipped otherwise.

## Files

```
app.py               Streamlit dashboard
api_client.py        HTTP client, request building and response validation
benchmark.py         Repeated and concurrent requests, latency statistics
config.py            Settings from the environment / .env
hardware.py          Hardware diagnostics (also runnable: .venv/bin/python hardware.py)
scenarios/examples.py  The four built-in scenarios
tests/               pytest suite
setup.sh, start-model.sh, start-demo.sh, run-tests.sh
```

If you move or rename this folder, delete `.venv` and run `./setup.sh` again, because a
virtual environment records its absolute path.
