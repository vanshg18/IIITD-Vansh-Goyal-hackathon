from __future__ import annotations

import json

from fastapi.testclient import TestClient

import src.api.main as api


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_health_reports_service_and_artifact_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    (tmp_path / "latest_cycle_status.json").write_text(
        json.dumps({
            "exit_code": 0,
            "completed_at": "2026-10-10T10:00:00+00:00",
            "sources": [{"source_name": "GDELT", "success": True, "articles_returned": 5}],
        }),
        encoding="utf-8",
    )

    response = TestClient(api.app).get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["latest_cycle"]["exit_code"] == 0
    assert payload["artifacts"]["latest_cycle_status.json"] is True
    assert payload["artifacts"]["latest_events.jsonl"] is False


def test_latest_events_history_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    events = [
        {
            "event_id": "event_1",
            "event_type": {"label": "Credit"},
            "sentiment": {"score": -0.5},
            "impact": {"score": 8.0},
        },
        {
            "event_id": "event_2",
            "event_type": {"label": "Earnings"},
            "sentiment": {"score": 0.6},
            "impact": {"score": 5.0},
        },
    ]
    write_jsonl(tmp_path / "latest_events.jsonl", events)
    write_jsonl(tmp_path / "events_history.jsonl", events)
    write_jsonl(tmp_path / "latest_stress_results.jsonl", [{"run_id": "stress_1"}])
    (tmp_path / "latest_cycle_status.json").write_text(
        json.dumps({"exit_code": 0, "completed_at": "2026-10-10T10:00:00+00:00"}),
        encoding="utf-8",
    )
    client = TestClient(api.app)

    latest = client.get("/api/v1/events/latest")
    history = client.get("/api/v1/events/history?limit=1")
    summary = client.get("/api/v1/summary")

    assert latest.status_code == 200
    assert [row["event_id"] for row in latest.json()["events"]] == ["event_1", "event_2"]
    assert history.json()["events"][0]["event_id"] == "event_2"
    assert summary.json()["event_count"] == 2
    assert summary.json()["event_type_counts"] == {"Credit": 1, "Earnings": 1}
    assert abs(summary.json()["mean_sentiment"] - 0.05) < 1e-9
    assert summary.json()["mean_impact"] == 6.5
    assert summary.json()["stress_test_count"] == 1


def test_downstream_endpoints_serve_persisted_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    rebalance = {"portfolio_id": "mock_index", "one_way_turnover": 0.03, "after_weights": {"AAPL": 0.1}}
    (tmp_path / "latest_rebalance.json").write_text(json.dumps(rebalance), encoding="utf-8")
    write_jsonl(tmp_path / "rebalance_history.jsonl", [rebalance])
    write_jsonl(tmp_path / "latest_stress_results.jsonl", [{"run_id": "stress_1", "net_loss": 1234.0}])
    write_jsonl(tmp_path / "stress_results_history.jsonl", [{"run_id": "stress_1", "net_loss": 1234.0}])
    client = TestClient(api.app)

    assert client.get("/api/v1/module-a/latest").json()["portfolio_id"] == "mock_index"
    assert client.get("/api/v1/module-a/history").json()["rebalances"][0]["one_way_turnover"] == 0.03
    assert client.get("/api/v1/module-b/latest").json()["stress_tests"][0]["net_loss"] == 1234.0
    assert client.get("/api/v1/module-b/history").json()["count"] == 1


def test_missing_latest_events_returns_clear_404(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    response = TestClient(api.app).get("/api/v1/events/latest")

    assert response.status_code == 404
    assert "run the EventPulse live cycle first" in response.json()["detail"]


def test_history_limit_is_bounded_by_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    response = TestClient(api.app).get("/api/v1/events/history?limit=1001")
    assert response.status_code == 422
