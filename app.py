"""SYSTEMONE DECISION LAB: a Streamlit dashboard for a local strands-decider server.

The model is reached only over HTTP through api_client; nothing here loads it.
"""

from __future__ import annotations

import json
import os
import uuid
from importlib.metadata import PackageNotFoundError, version as package_version

import altair as alt
import pandas as pd
import streamlit as st

import hardware
from api_client import (
    DeciderClient,
    Question,
    QuestionError,
    parse_levels,
    parse_options,
    slugify,
)
from benchmark import MAX_CONCURRENCY, MAX_REPEATS, BenchmarkResult, run_benchmark
from config import get_settings
from scenarios import SCENARIOS

CUSTOM = "Custom (blank)"
TYPE_LABELS = {"noul": "NOUL · yes/no", "choice": "CHOICE · pick one", "score": "SCORE · ordered scale"}
ACCENT = "#7C5CFF"
MUTED = "#3A3F55"


def _version(name: str) -> str:
    """Installed package version, or a short fallback when it is not installed."""
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "not installed"


st.set_page_config(page_title="SystemOne Decision Lab", page_icon="◆", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 5rem; max-width: 1400px;}
    /* Streamlit's branding and toolbar sat on top of the heading; keep the page clean. */
    footer, div[data-testid="stFooter"] {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
    .lab-title {font-size: 2.1rem; font-weight: 800; letter-spacing: .18em; margin: 0;
        line-height: 1.25; padding-bottom: .25rem; color: #B4A2FF;
        background: linear-gradient(90deg, #7C5CFF, #22D3EE); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;}
    .lab-sub {color: #9AA3B8; margin-top: -.2rem; margin-bottom: 1.2rem;}
    .pill {display: inline-block; padding: .15rem .65rem; border-radius: 999px;
        font-size: .8rem; font-weight: 700; letter-spacing: .05em;}
    .pill-on {background: rgba(34,197,94,.15); color: #4ADE80; border: 1px solid #22C55E;}
    .pill-off {background: rgba(239,68,68,.12); color: #F87171; border: 1px solid #EF4444;}
    .pill-type {background: rgba(124,92,255,.15); color: #B4A2FF; border: 1px solid #7C5CFF;}
    .big-answer {font-size: 1.9rem; font-weight: 800; line-height: 1.1; margin: .2rem 0 .4rem 0;}
    .caption {color: #9AA3B8; font-size: .85rem;}
    div[data-testid="stMetricValue"] {font-size: 1.15rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()


@st.cache_resource
def get_client() -> DeciderClient:
    return DeciderClient(settings.endpoint, settings.health_url, timeout=settings.timeout)


def new_client() -> DeciderClient:
    """A client with its own HTTP session, one per load-test thread."""
    return DeciderClient(settings.endpoint, settings.health_url, timeout=settings.timeout)


@st.cache_data(show_spinner=False)
def get_hardware(device: str) -> dict:
    return hardware.collect(device)


client = get_client()


# ---------------------------------------------------------------- session state

def _question_entry(q: Question) -> dict:
    return {"uid": uuid.uuid4().hex[:8], "key": q.key, "type": q.type,
            "instructions": q.instructions, "criteria": q.criteria}


def load_scenario() -> None:
    name = st.session_state.scenario
    if name == CUSTOM:
        st.session_state.state_text = ""
        st.session_state.questions = [_question_entry(Question("question_1", "noul", ""))]
    else:
        scenario = SCENARIOS[name]
        st.session_state.state_text = scenario["state"]
        st.session_state.questions = [_question_entry(q) for q in scenario["questions"]]
    st.session_state.result = None
    st.session_state.error = None


if "questions" not in st.session_state:
    st.session_state.scenario = next(iter(SCENARIOS))
    st.session_state.decisions = 0
    st.session_state.last_latency = None
    load_scenario()


def add_question(kind: str) -> None:
    n = len(st.session_state.questions) + 1
    criteria = {"choice": {}, "score": []}.get(kind)
    st.session_state.questions.append(_question_entry(Question(f"question_{n}", kind, "", criteria)))


def remove_question(uid: str) -> None:
    st.session_state.questions = [q for q in st.session_state.questions if q["uid"] != uid]


def _criteria_text(entry: dict, kind: str) -> str:
    criteria = entry["criteria"] if entry["type"] == kind else None
    if kind == "choice" and isinstance(criteria, dict):
        return "\n".join(f"{k}: {v}" if v else k for k, v in criteria.items())
    if kind == "score" and isinstance(criteria, list):
        return "\n".join(criteria)
    return ""


# ---------------------------------------------------------------- header + status

st.markdown('<p class="lab-title">SYSTEMONE DECISION LAB</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="lab-sub">Structured decisions from a small local model: '
    "<b>noul</b> (yes/no probability), <b>choice</b> (one of N) and <b>score</b> "
    "(ordered scale).</p>",
    unsafe_allow_html=True,
)


@st.fragment(run_every="5s")
def status_bar() -> None:
    health = client.health()
    info = health.info
    cols = st.columns([1.1, 2.4, 0.9, 2.2, 1.2, 1])
    with cols[0]:
        st.caption("Model status")
        pill = '<span class="pill pill-on">● ONLINE</span>' if health.online else \
            '<span class="pill pill-off">● OFFLINE</span>'
        st.markdown(pill, unsafe_allow_html=True)
    cols[1].metric("Model", info.get("model") or settings.model.split("/")[-1],
                   help=f"Configured: {settings.model}")
    cols[2].metric("Device", str(info.get("device") or settings.device).upper(),
                   help="Reported by /health when online; otherwise DECIDER_DEVICE.")
    cols[3].metric("Endpoint", settings.endpoint.replace("http://", ""))
    last = st.session_state.last_latency
    cols[4].metric("Last request", f"{last:,.0f} ms" if last is not None else "—",
                   help="Client round-trip time of the most recent decision.")
    cols[5].metric("Decisions", st.session_state.decisions, help="Requests made this session.")
    if not health.online:
        st.warning(
            f"{health.error}. Start the server with `./start-model.sh`. Its port opens only "
            "after the model has loaded and warmed up, which can take minutes on CPU.",
            icon="⏳",
        )


status_bar()


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Scenario")
    st.selectbox("Load a scenario", [*SCENARIOS, CUSTOM], key="scenario",
                 on_change=load_scenario, label_visibility="collapsed")
    st.caption("Demo inputs only. Answers come from the model, not from the scenario.")

    st.subheader("Configuration")
    config_rows = {
        "Backend": settings.backend,
        "Model": settings.model,
        "Device": settings.device,
        "Endpoint": settings.endpoint,
        "Timeout": f"{settings.timeout:.0f} s",
    }
    if settings.backend == "strands":
        config_rows["Server pool"] = f"{os.environ.get('DECIDER_POOL_SIZE', '1')} process(es)"
    elif settings.backend == "lmstudio":
        config_rows["LM Studio URL"] = os.environ.get("LMSTUDIO_URL", "—")
        config_rows["LM Studio model"] = os.environ.get("LMSTUDIO_MODEL", "—")
    st.markdown("\n".join(f"- **{k}:** `{v}`" for k, v in config_rows.items()))
    st.caption("From `.env`; live model and device are in the status bar above.")

    st.subheader("Software")
    software_rows = {
        "Streamlit": st.__version__,
        "strands-decider": _version("strands-decider"),
        "OpenDecider": _version("opendecider"),
    }
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in software_rows.items()))
    st.caption("Python, PyTorch and GPU details are under Hardware below.")

    st.subheader("Hardware")
    hw = get_hardware(settings.device)
    rows = {
        "Operating system": hw["os"],
        "Python": hw["python"],
        "PyTorch": f"{hw['torch_version'] or 'not installed'} ({hw['torch_build'] or 'n/a'})",
        "CUDA available": "yes" if hw["cuda_available"] else "no",
        "ROCm/HIP available": "yes" if hw["rocm_available"] else "no",
        "NVIDIA GPU": ", ".join(hw["nvidia_gpus"]) or "none found",
        "AMD GPU": ", ".join(hw["amd_gpus"]) or "none found",
        "Configured device": hw["configured_device"],
    }
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in rows.items()))
    if hw["amd_gpus"] and not hw["rocm_available"]:
        st.info("An AMD GPU is present, but this PyTorch build cannot use it.")
    if hw["nvidia_gpus"] and not hw["cuda_available"]:
        st.info("An NVIDIA GPU is present, but this PyTorch build cannot use it. "
                "Run `./setup.sh --cuda` to install a CUDA build.")
    st.caption("PyTorch details describe this app's environment; the device the model "
               "actually runs on is shown in the status bar.")


# ---------------------------------------------------------------- inputs

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("A · Scenario")
    st.text_area("Situation", key="state_text", height=180,
                 placeholder="Describe the situation the model should decide about…")

    st.subheader("C · Run")
    load = st.columns(3)
    load[0].number_input("Repeat", min_value=1, max_value=MAX_REPEATS, value=1, step=1,
                         key="repeats", help="Total number of requests to send.")
    load[1].number_input("Concurrency", min_value=1, max_value=MAX_CONCURRENCY, value=1, step=1,
                         key="concurrency",
                         help="Requests in flight at once. Capped at Repeat.")
    load[2].number_input("Discard first", min_value=0, max_value=MAX_REPEATS - 1, value=0, step=1,
                         key="discard",
                         help="Leave the first N requests out of the latency statistics "
                              "(warm-up). They still count toward throughput.")
    run = st.button("Run decision", type="primary", width="stretch")
    st.caption(f"Each request sends all questions in one POST to `{settings.endpoint}`. "
               f"Timeout {settings.timeout:.0f} s per request.")

with right:
    st.subheader("B · Decision builder")
    for i, entry in enumerate(st.session_state.questions):
        uid = entry["uid"]
        with st.container(border=True):
            top = st.columns([1.3, 1.2, 0.35])
            entry["type"] = top[0].selectbox(
                "Type", list(TYPE_LABELS), format_func=TYPE_LABELS.get,
                index=list(TYPE_LABELS).index(entry["type"]), key=f"type_{uid}")
            entry["key"] = top[1].text_input("Key", value=entry["key"], key=f"key_{uid}",
                                             help="Name of this question in the request/response.")
            top[2].button("✕", key=f"rm_{uid}", on_click=remove_question, args=(uid,),
                          help="Remove question")
            entry["instructions"] = st.text_input(
                "Question", value=entry["instructions"], key=f"q_{uid}",
                placeholder="e.g. Is this urgent?")
            kind = entry["type"]
            if kind == "choice":
                text = st.text_area(
                    "Options (one per line, optionally `label: description`)",
                    value=_criteria_text(entry, "choice"), key=f"opts_choice_{uid}", height=100,
                    placeholder="billing: Payment and money issues\ntechnical\nsales")
                entry["criteria"] = parse_options(text)
            elif kind == "score":
                text = st.text_area(
                    "Levels, lowest first (one per line or comma-separated, 2–10)",
                    value=_criteria_text(entry, "score"), key=f"opts_score_{uid}", height=100,
                    placeholder="low\nmedium\nhigh")
                entry["criteria"] = parse_levels(text)
            else:
                entry["criteria"] = None

    add = st.columns(3)
    for col, kind in zip(add, TYPE_LABELS):
        col.button(f"+ {kind.upper()}", key=f"add_{kind}", on_click=add_question, args=(kind,),
                   width="stretch")


# ---------------------------------------------------------------- run

if run:
    questions = [
        Question(slugify(e["key"], f"question_{i + 1}"), e["type"], e["instructions"],
                 e["criteria"])
        for i, e in enumerate(st.session_state.questions)
    ]
    st.session_state.error = None
    repeats = int(st.session_state.repeats)
    progress = st.progress(0.0, text=f"Sending {repeats} request(s)… "
                                     "(CPU inference can take a while)")

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"{done} of {total} requests finished")

    try:
        bench = run_benchmark(new_client, st.session_state.state_text, questions,
                              repeats=repeats,
                              concurrency=int(st.session_state.concurrency),
                              discard=int(st.session_state.discard),
                              on_progress=on_progress)
    except QuestionError as exc:
        st.session_state.error = str(exc)
        st.session_state.result = None
    else:
        st.session_state.decisions += len(bench.samples)
        last = bench.last
        if last is None:
            st.session_state.error = bench.first_error or "No request succeeded."
            st.session_state.result = None
        else:
            st.session_state.result = {"data": last.data, "round_trip_ms": last.round_trip_ms,
                                       "server_ms": last.server_latency_ms,
                                       "questions": questions, "bench": bench}
            st.session_state.last_latency = last.round_trip_ms
    st.rerun()


# ---------------------------------------------------------------- results

def _bar_chart(df: pd.DataFrame, highlight: str | None, ordered: bool) -> alt.Chart:
    sort = list(df["label"]) if ordered else "-x"
    color = alt.condition(alt.datum.label == highlight, alt.value(ACCENT), alt.value(MUTED)) \
        if highlight is not None else alt.value(ACCENT)
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=sort, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="probability",
                axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("probability:Q", format=".2%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(color=color)
    text = base.mark_text(align="left", dx=4, color="#C9CFDD").encode(
        text=alt.Text("probability:Q", format=".1%"))
    return (bars + text).properties(height=max(90, 34 * len(df)))


def render_answer(question: Question, answer: dict) -> None:
    kind = answer.get("type")
    with st.container(border=True):
        st.markdown(f'<span class="pill pill-type">{kind.upper()}</span> &nbsp;'
                    f"<b>{question.instructions}</b> <span class='caption'>· {question.key}</span>",
                    unsafe_allow_html=True)
        if kind == "noul":
            p = float(answer["noul"])
            c1, c2 = st.columns([1, 2])
            c1.markdown(f'<div class="big-answer">{"YES" if p >= 0.5 else "NO"}</div>'
                        f'<div class="caption">leaning, at the 50% line</div>',
                        unsafe_allow_html=True)
            c2.metric("P(yes)", f"{p:.1%}")
            c2.progress(min(max(p, 0.0), 1.0))
            st.caption("The API returns only P(yes) for noul questions; no separate confidence.")
        elif kind == "choice":
            probs = answer["probabilities"]
            c1, c2, c3 = st.columns([1.4, 1, 1])
            c1.markdown(f'<div class="caption">Selected answer</div>'
                        f'<div class="big-answer">{answer["choice"]}</div>',
                        unsafe_allow_html=True)
            c2.metric("Probability", f"{probs.get(answer['choice'], 0):.1%}")
            if "confidence" in answer:
                c3.metric("Confidence", f"{answer['confidence']:.3f}",
                          help="Reported by the API: 0 = even split, 1 = certain.")
            df = pd.DataFrame({"label": list(probs), "probability": list(probs.values())})
            st.altair_chart(_bar_chart(df, answer["choice"], ordered=False),
                            width="stretch")
        elif kind == "score":
            legend = answer["legend"]
            probs = answer["probabilities"]
            idx = sorted(legend, key=int)
            top = max(idx, key=lambda k: probs.get(k, 0))
            max_level = len(idx) - 1
            score = float(answer["score"])
            c1, c2, c3 = st.columns([1.4, 1, 1])
            c1.markdown(f'<div class="caption">Most likely level</div>'
                        f'<div class="big-answer">{legend[top]}</div>',
                        unsafe_allow_html=True)
            c2.metric("Expected score", f"{score:.2f} / {max_level}",
                      help="Expected value over level indices (0 = lowest), from the API.")
            if "confidence" in answer:
                c3.metric("Confidence", f"{answer['confidence']:.3f}",
                          help="Reported by the API: how tightly the mass clusters on the scale.")
            st.progress(min(max(score / max_level if max_level else 0, 0.0), 1.0),
                        text=f"{legend[idx[0]]} → {legend[idx[-1]]}")
            df = pd.DataFrame({"label": [f"{k}: {legend[k]}" for k in idx],
                               "probability": [probs.get(k, 0.0) for k in idx]})
            st.altair_chart(_bar_chart(df, f"{top}: {legend[top]}", ordered=True),
                            width="stretch")


def render_load_test(bench: BenchmarkResult) -> None:
    with st.container(border=True):
        st.markdown(f"**Load test** <span class='caption'>· {bench.repeats} requests, "
                    f"up to {bench.concurrency} at once</span>", unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("Succeeded", f"{bench.succeeded} / {bench.repeats}")
        m[1].metric("Wall time", f"{bench.wall_ms / 1000:,.2f} s",
                    help="From the first request sent to the last one finished.")
        m[2].metric("Throughput", f"{bench.throughput:,.2f} req/s",
                    help="Successful requests per second over the whole run, warm-up included.")
        m[3].metric("Discarded", bench.discard,
                    help="First requests left out of the latency statistics.")
        if bench.failed:
            st.warning(f"{bench.failed} request(s) failed. First error: {bench.first_error}",
                       icon="⚠️")

        rows = {"Round trip": bench.round_trip_stats(),
                "Server latency": bench.server_stats(),
                "Round trip − server": bench.overhead_stats()}
        table = pd.DataFrame(
            {name: [s.min, s.avg, s.p50, s.p95, s.max] for name, s in rows.items() if s},
            index=["min", "avg", "p50", "p95", "max"],
        ).T
        if table.empty:
            st.info("No requests left to measure after discarding the warm-up.")
        else:
            st.dataframe(table.style.format("{:,.0f} ms"), width="stretch")
            st.caption("Statistics cover the successful requests after the discarded ones. "
                       "Round trip − server is time spent waiting for a server thread, "
                       "plus HTTP.")

        points = [
            {"request": s.index + 1, "series": series, "ms": value,
             "discarded": s.index < bench.discard}
            for s in bench.samples if s.ok
            for series, value in (("Round trip", s.round_trip_ms), ("Server latency", s.server_ms))
            if value is not None
        ]
        if points:
            chart = alt.Chart(pd.DataFrame(points)).mark_line(point=True).encode(
                x=alt.X("request:Q", title="request # (order sent)",
                        axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("ms:Q", title="latency (ms)"),
                color=alt.Color("series:N", title=None,
                                scale=alt.Scale(range=[ACCENT, "#22D3EE"]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("request:Q"), alt.Tooltip("series:N"),
                         alt.Tooltip("ms:Q", format=",.0f"), alt.Tooltip("discarded:N")],
            ).properties(height=240)
            st.altair_chart(chart, width="stretch")


st.divider()
st.subheader("D · Results")

if st.session_state.error:
    st.error(st.session_state.error, icon="⚠️")

result = st.session_state.get("result")
if result:
    data = result["data"]
    usage = data.get("usage", {})
    bench = result.get("bench")
    if bench is not None and bench.repeats > 1:
        render_load_test(bench)
        st.markdown(f"**Last response** <span class='caption'>· the most recent successful "
                    f"response of {bench.repeats}</span>", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Round trip", f"{result['round_trip_ms']:,.0f} ms", help="Measured by this app.")
    m[1].metric("Server latency",
                f"{result['server_ms']:,.0f} ms" if result["server_ms"] is not None else "—",
                help="latency_ms reported by the server.")
    m[2].metric("Input tokens", usage.get("input_tokens", "—"))
    m[3].metric("Model", data.get("model", "—"))

    cols = st.columns(2)
    for i, q in enumerate(result["questions"]):
        with cols[i % 2]:
            render_answer(q, data["answers"][q.key])

    with st.expander("Raw JSON response"):
        st.code(json.dumps(data, indent=2), language="json")
elif not st.session_state.error:
    st.info("Build your questions and press **Run decision**. Results appear here.")
