from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rally_health import main
from rally_health.api import routes
from rally_health.core.config import Settings, get_settings
from rally_health.domain.schemas import ObservationCreate, Provenance
from rally_health.repositories.database import SQLiteDatabase
from rally_health.services.metrics import MetricEngine
from rally_health.services.observations import ObservationService
from rally_health.services.workouts import WorkoutCatalog
from rally_health.services.schedule import ScheduleService
from rally_health.services.memory import MemoryService
from rally_health.services.insights import InsightService
from rally_health.services.predictive import PredictiveService


def configure_test(tmp_path: Path, monkeypatch):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    (normalized / "rally.json").write_text('{"metrics":[{"metric_code":"RS-04","name":"Resting HR","group":"Recovery","evidence_level":"A","user_explanation":"Definition","source":{"sheet":"Chi_so","row":2}}],"supplemental_rules":[]}', encoding="utf-8")
    (normalized / "workouts.json").write_text('{"female":[{"exercise_id":"FE_1","name":"Test","equipment":["Mat"],"life_stages":["Adult"],"cycle_phases":[],"review_status":"approved"}],"male":[{"exercise_id":"MA_1","name_vi":"Test male","equipment":[],"life_stages":[],"cycle_phases":[],"review_status":"approved"}]}', encoding="utf-8")
    (normalized / "mens_wellness_roadmap.json").write_text("[]", encoding="utf-8")
    settings = Settings(api_key="test-secret", environment="development", database_path=tmp_path / "test.sqlite3", normalized_data_dir=normalized, source_dir=tmp_path)
    db = SQLiteDatabase(tmp_path / "test.sqlite3")
    db.sync_catalog([
        ("metric", "RS-04", {"metric_code":"RS-04","name":"Resting HR","group":"Recovery","evidence_level":"A","user_explanation":"Definition","source":{"sheet":"Chi_so","row":2}}),
        ("workout_female", "FE_1", {"exercise_id":"FE_1","name":"Test","equipment":["Mat"],"life_stages":["Adult"],"cycle_phases":[],"review_status":"approved"}),
    ])
    client = TestClient(main.create_app(lambda: settings))
    return client


def auth():
    return {"X-API-Key":"test-secret", "X-User-ID":"test-user"}


def test_api_health_auth_catalog(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/catalog/metrics").status_code == 401
        assert client.get("/v1/catalog/metrics", headers=auth()).json()["count"] == 1


def test_rally_registry_distinguishes_documentation_from_executable_aggregation(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        catalog = client.get("/v1/catalog/metrics", headers=auth()).json()
        metric = catalog["items"][0]
        assert metric["formula_registry"]["status"] == "prototype_raw_aggregation_enabled_not_rally_formula"
        assert metric["formula_registry"]["formula_id"] == "raw_observation_median"
        assert catalog["registry_version"] == "1.0.0"


def test_metric_without_registered_formula_is_not_computed(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        payload = {"metric_code":"RS-04","value":60,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z","provenance":{"source_type":"manual"}}
        assert client.post("/v1/observations", headers=auth(), json=payload).status_code == 200
        # A catalog metric outside the enabled aggregation set remains non-computable.
        monkeypatch.setattr(routes.metrics, "catalog_provider", lambda: [{"metric_code":"AC-11","calculation_description":None,"source":{"sheet":"Chi_so","row":12}}])
        result = client.get("/v1/metrics/AC-11/result", headers=auth()).json()
        assert result["status"] == "REVIEW_REQUIRED"
        assert result["value"] is None


def test_baseline_rejects_invalid_timezone_and_accepts_local_timezone(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        local=client.get("/v1/metrics/RS-04/baseline",headers=auth(),params={"timezone":"America/Los_Angeles"})
        assert local.status_code==200
        assert local.json()["status"]=="INSUFFICIENT_DATA"
        invalid=client.get("/v1/metrics/RS-04/baseline",headers=auth(),params={"timezone":"Not/A_Zone"})
        assert invalid.status_code==200
        assert invalid.json()["status"]=="INVALID_INPUT"
        assert invalid.json()["error"]=="invalid_timezone"



def test_missing_metric_is_null_not_zero(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        response = client.get("/v1/metrics/RS-04/result", headers=auth())
        assert response.status_code == 200
        assert response.json()["value"] is None
        assert response.json()["status"] == "INSUFFICIENT_DATA"


def test_observation_ingestion_and_metric_result(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        body = {"metric_code":"RS-04","value":60,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z","provenance":{"source_type":"manual"}}
        assert client.post("/v1/observations", headers=auth(), json=body).status_code == 200
        result = client.get("/v1/metrics/RS-04/result", headers=auth()).json()
        assert result["value"] == 60
        assert result["source_observation_ids"]


def test_assistant_resolves_metric_name_and_returns_exact_observation_sources(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        body = {"metric_code":"RS-04","value":60,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z","provenance":{"source_type":"manual"}}
        created = client.post("/v1/observations", headers=auth(), json=body).json()
        response = client.post("/v1/assistant/turns", headers=auth(), json={"question":"What is my Resting HR?"})
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        assert result["facts"]["metrics"][0]["metric_code"] == "RS-04"
        assert result["facts"]["metrics"][0]["value"] == 60
        assert result["facts"]["metrics"][0]["source_ids"] == [created["observation_id"]]
        assert result["sources"] == []
        assert result["facts"]["metrics"][0]["baseline_value"] is None
        assert result["facts"]["metrics"][0]["baseline_status"] == "INSUFFICIENT_DATA"


def test_assistant_observation_change_requires_confirmation(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        routes.settings = routes.settings.model_copy(update={"chat_enabled": True})

        async def fake_respond(question, history, facts):
            return {"answer": "Mình có thể đề xuất cập nhật; hãy xác nhận trước khi lưu.", "cited_source_ids": [],
                    "follow_up_question": None, "intent": "observation_write",
                    "observation_proposal": {"metric_code":"RS-04", "value":61, "unit":"bpm", "observed_at":"2026-10-20T07:00:00Z"}}

        monkeypatch.setattr(routes.gpt4o_chat, "respond", fake_respond)
        response = client.post("/v1/assistant/turns", headers=auth(), json={"question":"Please add my resting heart rate 61 bpm from 2026-10-20T07:00:00Z"})
        assert response.status_code == 200
        proposal = response.json()["action_proposal"]
        assert proposal["status"] == "AWAITING_CONFIRMATION"
        assert client.get("/v1/metrics/RS-04/result", headers=auth()).json()["value"] is None
        saved = client.post(f"/v1/assistant/action-proposals/{proposal['action_id']}/confirm", headers=auth())
        assert saved.status_code == 200
        assert saved.json()["observation"]["user_confirmed"] is True
        assert client.get("/v1/metrics/RS-04/result", headers=auth()).json()["value"] == 61
        assert client.post(f"/v1/assistant/action-proposals/{proposal['action_id']}/confirm", headers=auth()).status_code == 409


def test_assistant_rejects_provider_citation_not_in_retrieved_facts(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        routes.settings = routes.settings.model_copy(update={"chat_enabled": True})

        async def fake_respond(question, history, facts):
            return {"answer": "unsupported", "cited_source_ids": ["observation-from-another-user"],
                    "follow_up_question": None, "intent": "question", "observation_proposal": None}

        monkeypatch.setattr(routes.gpt4o_chat, "respond", fake_respond)
        response = client.post("/v1/assistant/turns", headers=auth(), json={"question":"How was my sleep?"})
        assert response.status_code == 200
        assert response.json()["fallback_used"] is True
        assert response.json()["sources"] == []
        assert "không khả dụng" in response.json()["answer"]


def test_assistant_conversation_is_scoped_to_authenticated_user(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        created = client.post("/v1/assistant/turns", headers=auth(), json={"question":"Hello"}).json()
        conversation_id = created["conversation_id"]
        other_user_headers = {"X-API-Key":"test-secret", "X-User-ID":"other-user"}
        response = client.get(f"/v1/assistant/conversations/{conversation_id}", headers=other_user_headers)
        assert response.status_code == 404
        own = client.get(f"/v1/assistant/conversations/{conversation_id}", headers=auth())
        assert own.status_code == 200
        assert [item["role"] for item in own.json()["messages"]] == ["user", "assistant"]


def test_assistant_reports_insufficient_context_for_unmatched_question(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        response = client.post("/v1/assistant/turns", headers=auth(), json={"question":"How was my sleep?"})
        assert response.status_code == 200
        assert response.json()["status"] == "insufficient_context"
        assert response.json()["facts"]["metrics"] == []
        assert response.json()["sources"] == []


def test_observation_rejects_wrong_unit(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        body = {"metric_code":"RS-04","value":60,"unit":"kg","observed_at":"2026-10-20T07:00:00Z","provenance":{"source_type":"manual"}}
        assert client.post("/v1/observations", headers=auth(), json=body).status_code == 422


def test_workout_catalog_and_coach_draft(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        found = client.get("/v1/catalog/workouts?sex=female&life_stage=Adult", headers=auth())
        assert found.status_code == 200 and found.json()["count"] == 1
        payload = {"title":"Workout","sex_catalog":"female","exercise_ids":["FE_1"],"starts_at":"2026-10-24T08:00:00+00:00","ends_at":"2026-10-24T08:30:00+00:00","requested_by_user":True,"idempotency_key":"draft-workout-20261024"}
        proposal = client.post("/v1/schedule/events/proposals", headers=auth(), json=payload)
        assert proposal.status_code == 200 and proposal.json()["status"] == "AWAITING_CONFIRMATION"


def test_schedule_idempotency_and_conflict(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        payload = {"title":"Walk","sex_catalog":"male","exercise_ids":["MA_1"],"starts_at":"2026-10-24T08:00:00Z","ends_at":"2026-10-24T08:30:00Z","requested_by_user":True,"idempotency_key":"walk-20261024-0800"}
        direct = client.post("/v1/schedule/events", headers=auth(), json=payload)
        assert direct.status_code == 405
        proposal = client.post("/v1/schedule/events/proposals", headers=auth(), json=payload)
        assert proposal.status_code == 200 and proposal.json()["committed"] is False
        action_id = proposal.json()["action_id"]
        first = client.post(f"/v1/schedule/events/proposals/{action_id}/confirm", headers=auth())
        second = client.post(f"/v1/schedule/events/proposals/{action_id}/confirm", headers=auth())
        assert first.status_code == 200
        assert second.status_code == 409


def test_memory_confirmation_and_delete(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        blocked = client.post("/v1/memories", headers=auth(), json={"field":"goal","value":"fitness"})
        assert blocked.status_code == 405
        proposal = client.post("/v1/memories/proposals", headers=auth(), json={"field":"goal","value":"fitness"})
        assert proposal.status_code == 200
        created = client.post(f"/v1/memories/proposals/{proposal.json()['action_id']}/confirm", headers=auth())
        assert created.status_code == 200
        assert client.delete(f"/v1/memories/{created.json()['memory_id']}", headers=auth()).status_code == 200


def test_report_upload_does_not_claim_ocr(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        response = client.post("/v1/reports/extraction-jobs", headers=auth(), files={"file":("lab.png",b"\x89PNG\r\n\x1a\nimage","image/png")})
        assert response.status_code == 200
        assert response.json()["status"] == "RECEIVED_NOT_PROCESSED"
        job = client.get(f"/v1/reports/extraction-jobs/{response.json()['job_id']}", headers=auth())
        assert job.status_code == 200
        assert job.json()["candidates"] == []


def test_report_candidates_only_create_observations_after_confirmation(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        uploaded = client.post("/v1/reports/extraction-jobs", headers=auth(), files={"file":("lab.png",b"\x89PNG\r\n\x1a\nimage","image/png")})
        job_id = uploaded.json()["job_id"]
        candidate = {"metric_code":"RS-04","source_analyte":"RHR","value":62,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z","source_text":"Resting HR 62 bpm 2026-10-20","source_page":1}
        staged = client.post(f"/v1/reports/extraction-jobs/{job_id}/candidates", headers=auth(), json={"candidates":[candidate]})
        assert staged.status_code == 200
        assert client.get("/v1/metrics/RS-04/result", headers=auth()).json()["value"] is None
        confirmed = client.post(f"/v1/reports/extraction-jobs/{job_id}/confirm", headers=auth(), json={"candidates":[candidate]})
        assert confirmed.status_code == 200
        assert confirmed.json()["status"] == "CONFIRMED"
        assert client.get("/v1/metrics/RS-04/result", headers=auth()).json()["value"] == 62


def test_formula_review_is_proposal_not_enablement(tmp_path, monkeypatch):
    with configure_test(tmp_path, monkeypatch) as client:
        response = client.post("/v1/admin/metric-formula-reviews", headers=auth(), json={
            "metric_code":"RS-04", "formula_id":"rhr-test", "formula_version":"0.1.0",
            "definition":{"method":"example"}, "test_cases":[{"input":60,"expected":60}]
        })
        assert response.status_code == 200
        assert response.json()["status"] == "PENDING_REVIEW"
        assert response.json()["executable"] is False
        pending = client.get("/v1/admin/metric-formula-reviews?status=PENDING_REVIEW", headers=auth())
        assert pending.json()["count"] == 1
