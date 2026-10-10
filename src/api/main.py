"""Read-only JSON API for EventPulse's latest structured risk outputs.

The API serves persisted artifacts; it does not load transformer models or
perform expensive ingestion as part of a GET request. Run the live cycle with
`python -m src.engine.run_live_pipeline` to refresh the artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException, Query


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

app = FastAPI(
    title="EventPulse Risk Signals API",
    description=(
        "Read-only API for structured financial-event signals, mock-index "
        "rebalancing, synthetic-bank stress results, and live source health."
    ),
    version="1.0.0",
)


def _path(filename: str) -> Path:
    return DATA_DIR / filename


def _read_json(filename: str, *, optional: bool = False) -> dict[str, Any] | None:
    target = _path(filename)
    if not target.exists():
        if optional:
            return None
        raise HTTPException(
            status_code=404,
            detail=f"{filename} is not available yet; run the EventPulse live cycle first.",
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {filename}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"{filename} must contain a JSON object.")
    return payload


def _read_jsonl(
    filename: str,
    *,
    optional: bool = False,
) -> list[dict[str, Any]]:
    target = _path(filename)
    if not target.exists():
        if optional:
            return []
        raise HTTPException(
            status_code=404,
            detail=f"{filename} is not available yet; run the EventPulse live cycle first.",
        )
    rows = []
    try:
        with target.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid JSON in {filename} at line {line_number}.",
                    ) from exc
                if not isinstance(row, dict):
                    raise HTTPException(
                        status_code=500,
                        detail=f"{filename} line {line_number} must be a JSON object.",
                    )
                rows.append(row)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {filename}: {exc}") from exc
    return rows


def _history_response(filename: str, key: str, limit: int) -> dict[str, Any]:
    rows = _read_jsonl(filename, optional=True)
    latest = list(reversed(rows[-limit:]))
    return {"count": len(latest), key: latest, "limit": limit}


@app.get("/api/v1/health", tags=["system"])
def health() -> dict[str, Any]:
    """Report API availability and which persisted artifacts exist."""
    status = _read_json("latest_cycle_status.json", optional=True)
    artifact_names = [
        "latest_cycle_status.json",
        "latest_events.jsonl",
        "events_history.jsonl",
        "latest_rebalance.json",
        "rebalance_history.jsonl",
        "latest_stress_results.jsonl",
        "stress_results_history.jsonl",
    ]
    return {
        "status": "ok",
        "service": "EventPulse Risk Signals API",
        "latest_cycle": status,
        "artifacts": {name: _path(name).exists() for name in artifact_names},
    }


@app.get("/api/v1/cycle/latest", tags=["system"])
def latest_cycle() -> dict[str, Any]:
    """Return the last live-cycle summary and per-source success/error."""
    return _read_json("latest_cycle_status.json") or {}


@app.get("/api/v1/events/latest", tags=["events"])
def latest_events() -> dict[str, Any]:
    """Return the structured event signals from the most recent completed cycle."""
    records = _read_jsonl("latest_events.jsonl")
    return {"count": len(records), "events": records}


@app.get("/api/v1/events/history", tags=["events"])
def event_history(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Return recent event history, newest first."""
    return _history_response("events_history.jsonl", "events", limit)


@app.get("/api/v1/module-a/latest", tags=["downstream"])
def latest_rebalance() -> dict[str, Any]:
    """Return the latest Module A weights, changes, turnover, and signals."""
    return _read_json("latest_rebalance.json") or {}


@app.get("/api/v1/module-a/history", tags=["downstream"])
def rebalance_history(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Return recent Module A rebalancing snapshots, newest first."""
    return _history_response("rebalance_history.jsonl", "rebalances", limit)


@app.get("/api/v1/module-b/latest", tags=["downstream"])
def latest_stress_results() -> dict[str, Any]:
    """Return stress tests triggered by the most recent cycle."""
    records = _read_jsonl("latest_stress_results.jsonl")
    return {"count": len(records), "stress_tests": records}


@app.get("/api/v1/module-b/history", tags=["downstream"])
def stress_history(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Return recent Module B stress-test results, newest first."""
    return _history_response("stress_results_history.jsonl", "stress_tests", limit)


@app.get("/api/v1/summary", tags=["system"])
def risk_summary() -> dict[str, Any]:
    """Return compact operational and signal-quality summary statistics."""
    events = _read_jsonl("latest_events.jsonl", optional=True)
    stress_tests = _read_jsonl("latest_stress_results.jsonl", optional=True)

    labels = Counter(
        str((event.get("event_type") or {}).get("label", "Unknown"))
        for event in events
    )
    sentiments = [
        float((event.get("sentiment") or {}).get("score", 0.0))
        for event in events
        if isinstance(event.get("sentiment"), dict)
    ]
    impacts = [
        float((event.get("impact") or {}).get("score", 0.0))
        for event in events
        if isinstance(event.get("impact"), dict)
    ]
    status = _read_json("latest_cycle_status.json", optional=True)

    return {
        "event_count": len(events),
        "event_type_counts": dict(labels),
        "mean_sentiment": mean(sentiments) if sentiments else None,
        "mean_impact": mean(impacts) if impacts else None,
        "stress_test_count": len(stress_tests),
        "last_cycle_completed_at": status.get("completed_at") if status else None,
        "last_cycle_exit_code": status.get("exit_code") if status else None,
    }
