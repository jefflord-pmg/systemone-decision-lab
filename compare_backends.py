"""Run the demo scenarios against two decision backends and compare their answers.

Both strands-decider and OpenDecider speak /v1/systemone, so the dashboard's
ordinary client works unchanged. This script just sends the same request to each
configured endpoint and reports the top answer per question side by side.

Usage:
    ./start-model.sh                 # strands pool on port 8000
    ./start-opendecider.sh           # OpenDecider nano on port 8001
    .venv/bin/python compare_backends.py --out comparison.json

    # or compare the local pool with LM Studio-backed OpenDecider:
    DECIDER_BACKEND=lmstudio ./start-opendecider.sh
    .venv/bin/python compare_backends.py --backends strands,lmstudio --out lmstudio.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from api_client import DeciderClient, DeciderError
from config import BACKENDS, get_settings
from scenarios import SCENARIOS


def top_answer(answer: dict[str, Any]) -> tuple[str, Any]:
    """Return (kind, comparable top answer) for one SystemOne answer object."""
    kind = answer.get("type")
    if kind == "noul":
        return kind, bool(float(answer.get("noul", 0.0)) >= 0.5)
    if kind == "choice":
        return kind, answer.get("choice")
    if kind == "score":
        # OpenDecider reports the most likely level; strands-decider reports the
        # expected score. Round the expected score to the nearest level so the
        # two can be compared. Keep the raw values in the JSON report.
        level = answer.get("level")
        if level is not None:
            return kind, level
        value = answer.get("score")
        return kind, int(round(float(value))) if isinstance(value, (int, float)) else value
    return kind or "unknown", None


def answers_agree(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when two answers have the same type and the same top value."""
    return top_answer(left) == top_answer(right)


def describe(answer: dict[str, Any]) -> str:
    kind, value = top_answer(answer)
    confidence = answer.get("confidence")
    suffix = f" (p={confidence:.3f})" if isinstance(confidence, (int, float)) else ""
    return f"{kind}={value}{suffix}"


def run_scenario(client: DeciderClient, scenario: dict[str, Any]) -> dict[str, Any]:
    result = client.decide(scenario["state"], scenario["questions"])
    return {
        "answers": result.data.get("answers", {}),
        "model": result.data.get("model"),
        "latency_ms": round(result.round_trip_ms, 1),
    }


def compare(clients: dict[str, DeciderClient], scenarios: dict[str, Any],
            selected: list[str] | None = None) -> dict[str, Any]:
    """Send every selected scenario to every backend; collect answers and agreement."""
    names = list(clients)
    chosen = [s for s in (selected or list(scenarios)) if s in scenarios]
    if not chosen:
        raise ValueError("No scenarios selected")
    report: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "backends": names,
        "scenarios": [],
    }
    for name in chosen:
        scenario = scenarios[name]
        row: dict[str, Any] = {
            "name": name,
            "state": scenario["state"],
            "questions": [q.key for q in scenario["questions"]],
            "results": {},
        }
        for backend, client in clients.items():
            try:
                row["results"][backend] = run_scenario(client, scenario)
            except DeciderError as exc:
                row["results"][backend] = {"error": str(exc)}
        row["agreement"] = {}
        for key in row["questions"]:
            answers = {
                backend: result.get("answers", {}).get(key)
                for backend, result in row["results"].items()
                if "error" not in result
                and isinstance(result.get("answers", {}).get(key), dict)
            }
            values = list(answers.values())
            row["agreement"][key] = (
                all(answers_agree(values[0], other) for other in values[1:])
                if len(answers) == len(names) and values else None
            )
        report["scenarios"].append(row)
    return report


def print_report(report: dict[str, Any]) -> None:
    names = report["backends"]
    for row in report["scenarios"]:
        print(f"\n=== {row['name']} ===")
        print("  " + " | ".join(f"{name:>22}" for name in names))
        for key in row["questions"]:
            cells = []
            for name in names:
                result = row["results"].get(name, {})
                if "error" in result:
                    cells.append("ERROR")
                else:
                    answer = result.get("answers", {}).get(key)
                    cells.append(describe(answer) if isinstance(answer, dict) else "missing")
            mark = row["agreement"][key]
            tag = "same" if mark else ("DIFFER" if mark is False else "n/a")
            print(f"  " + " | ".join(f"{cell:>22}" for cell in cells) + f"   [{tag}]  {key}")
        latencies = []
        for name in names:
            result = row["results"].get(name, {})
            latencies.append(f"{name}={result.get('latency_ms', 'n/a')}ms")
        print("  " + ", ".join(latencies))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", default=",".join(
            name for name in BACKENDS if name != "lmstudio"
        ),
        help="comma-separated backend names (default: strands,opendecider)",
    )
    parser.add_argument("--scenario", action="append", help="scenario name (repeatable)")
    parser.add_argument("--out", help="write the full report as JSON to this path")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.backends.split(",") if name.strip()]
    clients: dict[str, DeciderClient] = {}
    for name in names:
        try:
            settings = get_settings(name)
        except ValueError as exc:
            parser.error(str(exc))
        clients[name] = DeciderClient(settings.endpoint, settings.health_url, settings.timeout)
        print(f"{name}: {settings.endpoint}")

    report = compare(clients, SCENARIOS, args.scenario)
    print_report(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
