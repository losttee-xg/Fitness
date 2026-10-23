from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rally_health import main
from rally_health.core.config import Settings
from rally_health.repositories.database import SQLiteDatabase


def make_client(tmp_path: Path):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    (normalized / "rally.json").write_text('{"metrics":[{"metric_code":"RS-01","name":"Heart Rate","group":"Recovery","evidence_level":"A","user_explanation":"Raw HR","source":{"sheet":"Chi_so","row":2}}],"supplemental_rules":[]}', encoding="utf-8")
    (normalized / "workouts.json").write_text('{"female":[],"male":[]}', encoding="utf-8")
    (normalized / "mens_wellness_roadmap.json").write_text("[]", encoding="utf-8")
    settings = Settings(api_key="test-secret", environment="development", database_path=tmp_path / "e2e.sqlite3",
                        normalized_data_dir=normalized, source_dir=tmp_path)
    return TestClient(main.create_app(lambda: settings))


def headers():
    return {"X-API-Key": "test-secret", "X-User-ID": "task1-user"}


def test_manual_batch_ingestion_is_idempotent_and_rejects_changed_replay(tmp_path):
    with make_client(tmp_path) as client:
        payload = {"batch_id":"manual-batch-0001", "provider":"manual", "observations":[
            {"metric_code":"RS-04","value":62,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z",
             "provenance":{"source_type":"manual","source_id":"manual-entry-1"}}
        ]}
        first=client.post("/v1/ingest/manual",headers=headers(),json=payload)
        assert first.status_code==200, first.text
        assert first.json()["accepted_count"]==1
        retry=client.post("/v1/ingest/manual",headers=headers(),json=payload)
        assert retry.status_code==200, retry.text
        assert retry.json()["replayed"] is True
        assert retry.json()["accepted_count"]==1
        changed={**payload,"observations":[{**payload["observations"][0],"value":63}]}
        conflict=client.post("/v1/ingest/manual",headers=headers(),json=changed)
        assert conflict.status_code==409
        assert conflict.json()["detail"]["code"]=="BATCH_ID_REUSED_WITH_DIFFERENT_PAYLOAD"


def test_report_confirmation_is_atomic_when_a_candidate_is_invalid(tmp_path):
    with make_client(tmp_path) as client:
        uploaded=client.post("/v1/reports/upload-intake",headers=headers(),files={"file":("lab.pdf",b"%PDF-test","application/pdf")})
        job_id=uploaded.json()["job_id"]
        candidates=[
            {"metric_code":"RS-01","source_analyte":"Heart Rate","value":62,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z",
             "source_text":"Heart rate 62 bpm on 2026-10-20","source_page":1},
            {"metric_code":"RS-01","source_analyte":"Heart Rate","value":500,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z",
             "source_text":"Heart rate 500 bpm on 2026-10-20","source_page":1},
        ]
        staged=client.post(f"/v1/reports/extraction-jobs/{job_id}/candidates",headers=headers(),json={"candidates":candidates})
        assert staged.status_code==200, staged.text
        failed=client.post(f"/v1/reports/extraction-jobs/{job_id}/confirm",headers=headers(),json={"candidates":staged.json()["candidates"]})
        assert failed.status_code==422
        with SQLiteDatabase(tmp_path / "e2e.sqlite3").connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM observations WHERE metric_code='RS-01'").fetchone()[0]==0
        job=client.get(f"/v1/reports/extraction-jobs/{job_id}",headers=headers()).json()
        assert job["status"]=="AWAITING_USER_CONFIRMATION"
        assert job["confirmed_observations"]==[]


def test_report_review_can_fetch_original_file_and_rejects_unverifiable_candidate(tmp_path):
    with make_client(tmp_path) as client:
        fake_png=b"\x89PNG\r\n\x1a\n"+b"fake-image"
        uploaded=client.post("/v1/reports/upload-intake",headers=headers(),files={"file":("review.png",fake_png,"image/png")})
        job_id=uploaded.json()["job_id"]
        file_response=client.get(f"/v1/reports/extraction-jobs/{job_id}/file",headers=headers())
        assert file_response.status_code==200
        assert file_response.content==fake_png
        candidate={"metric_code":"RS-01","value":64,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z",
                   "source_text":"HR maybe 64","source_analyte":"Heart Rate","source_page":1}
        staged=client.post(f"/v1/reports/extraction-jobs/{job_id}/candidates",headers=headers(),json={"candidates":[candidate]})
        assert staged.status_code==422
        assert staged.json()["detail"]["code"]=="REPORT_CANDIDATE_UNVERIFIABLE"


def test_apple_health_xml_export_maps_supported_types_and_skips_other_types(tmp_path):
    xml=b'''<?xml version="1.0" encoding="UTF-8"?>
    <HealthData>
      <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" sourceVersion="10" device="&lt;&lt;Apple Watch&gt;&gt;" unit="count/min" creationDate="2026-10-20T07:01:00Z" startDate="2026-10-20T07:00:00Z" endDate="2026-10-20T07:00:01Z" value="64"/>
      <Record type="HKQuantityTypeIdentifierRestingHeartRate" sourceName="Apple Watch" unit="count/min" startDate="2026-10-20T06:00:00Z" endDate="2026-10-20T06:00:01Z" value="55"/>
      <Record type="HKQuantityTypeIdentifierOxygenSaturation" sourceName="Apple Watch" unit="%" startDate="2026-10-20T06:00:00Z" endDate="2026-10-20T06:00:01Z" value="0.98"/>
      <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" startDate="2026-10-19T23:00:00Z" endDate="2026-10-20T00:00:00Z" value="HKCategoryValueSleepAnalysisAsleepCore"/>
      <Record type="HKQuantityTypeIdentifierWalkingStepCount" sourceName="Apple Watch" unit="count" startDate="2026-10-20T06:00:00Z" endDate="2026-10-20T07:00:00Z" value="250"/>
    </HealthData>'''
    with make_client(tmp_path) as client:
        response=client.post("/v1/ingest/apple-health-export",headers={**headers(),"X-Import-Batch-ID":"apple-import-0001"},
                             files={"file":("export.xml",xml,"application/xml")})
        assert response.status_code==200, response.text
        body=response.json()
        assert body["accepted_count"]==4
        assert body["parser"]["unsupported_type"]==1
        assert body["parser"]["parsed"]==4
        hr=client.get("/v1/metrics/RS-01/result",headers=headers()).json()
        assert hr["value"]==64
        assert hr["source_observation_ids"]
        sleep=client.get("/v1/metrics/SL-05/result",headers=headers()).json()
        assert sleep["value"] is None
        assert sleep["status"]=="NOT_COMPUTABLE"


def test_apple_health_xml_rejects_dtd_even_after_large_comment(tmp_path):
    from rally_health.integrations.apple_health_export import parse_apple_health_export

    xml=b"<HealthData><!--" + (b"x" * 5000) + b"--><!DOCTYPE HealthData [<!ENTITY x SYSTEM 'file:///private'>]></HealthData>"
    try:
        parse_apple_health_export(xml,"apple-import-0002")
    except ValueError as error:
        assert "DTD and entity" in str(error)
    else:
        raise AssertionError("DTD/entity declarations must be rejected")


def test_report_upload_rejects_mismatched_file_signature(tmp_path):
    with make_client(tmp_path) as client:
        response=client.post("/v1/reports/upload-intake",headers=headers(),files={"file":("fake.png",b"not-a-png","image/png")})
        assert response.status_code==415
        assert "content" in response.json()["detail"]


def test_report_upload_runs_gpt4o_ocr_and_stages_only_supported_candidates(tmp_path, monkeypatch):
    from rally_health.api import ingestion_routes

    with make_client(tmp_path) as client:
        settings = ingestion_routes.settings.model_copy(update={"ocr_enabled":True,"ocr_api_key":"test-ocr-key","ocr_model":"gpt-4o"})
        ingestion_routes.set_runtime(settings, SQLiteDatabase(tmp_path / "e2e.sqlite3"))
        calls = []

        async def fake_extract(runtime_settings, content, content_type, rally_metrics):
            calls.append((runtime_settings.ocr_model, content, content_type, [item["metric_code"] for item in rally_metrics]))
            return {"provider":"openai-compatible","model":"gpt-4o","usage":{"total_tokens":10},
                    "transcription":"Heart rate 64 bpm 2026-10-20",
                    "unreadable":["Hemoglobin: unit unclear"],"candidates":[
                        {"metric_code":"RS-01","source_analyte":"Heart Rate","value":64,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z",
                         "source_text":"Heart rate 64 bpm 2026-10-20","source_page":1}]}

        monkeypatch.setattr(ingestion_routes,"extract_report_candidates",fake_extract)
        fake_png=b"\x89PNG\r\n\x1a\n"+b"fake-image-bytes"
        response=client.post("/v1/reports/upload-intake",headers=headers(),files={"file":("report.png",fake_png,"image/png")})
        assert response.status_code==200, response.text
        report_dir=tmp_path/"report_uploads"/__import__("hashlib").sha256(b"task1-user").hexdigest()
        assert report_dir.exists()
        body=response.json()
        assert body["status"]=="AWAITING_USER_CONFIRMATION"
        assert body["model"]=="gpt-4o"
        assert body["candidates"][0]["value"]==64
        assert body["unreadable"]==["Hemoglobin: unit unclear"]
        assert body["transcription"]=="Heart rate 64 bpm 2026-10-20"
        assert calls[0][:3]==("gpt-4o",fake_png,"image/png")
        assert client.get("/v1/metrics/RS-01/result",headers=headers()).json()["value"] is None


def test_report_ocr_prompt_payload_uses_gpt4o_temperature_zero_and_vision(tmp_path, monkeypatch):
    import asyncio
    import json as json_module
    from types import SimpleNamespace
    from rally_health.integrations import report_ocr

    captured = {}

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"choices":[{"message":{"content":json_module.dumps({"transcription":"Heart rate 64 bpm 2026-10-20","candidates":[
                {"metric_code":"RS-01","source_analyte":"Heart Rate","value":64,"unit":"bpm","observed_at":"2026-10-20T07:00:00Z",
                 "source_text":"Heart rate 64 bpm 2026-10-20","source_page":1}],"unreadable":[]})}}],"usage":{"total_tokens":12}}

    class FakeClient:
        def __init__(self,timeout):
            captured["timeout"]=timeout
        async def __aenter__(self): return self
        async def __aexit__(self,*args): return None
        async def post(self,url,headers,json):
            captured.update({"url":url,"payload":json})
            return FakeResponse()

    monkeypatch.setattr(report_ocr.httpx,"AsyncClient",FakeClient)
    settings=SimpleNamespace(ocr_api_key="test",ocr_base_url="https://mock.local/v1",ocr_timeout_seconds=15)
    result=asyncio.run(report_ocr.extract_report_candidates(settings,b"image-data","image/png",[
        {"metric_code":"RS-01","name":"Heart Rate","group":"Recovery"}]))
    payload=captured["payload"]
    assert payload["model"]=="gpt-4o"
    assert payload["temperature"]==0
    assert payload["response_format"]["json_schema"]["strict"] is True
    user_content=payload["messages"][1]["content"]
    assert any(part["type"]=="image_url" and part["image_url"]["url"].startswith("data:image/png;base64,") for part in user_content)
    assert "Never infer" in payload["messages"][0]["content"]
    assert len(result["candidates"])==1
    assert result["model"]=="gpt-4o"
    assert result["transcription"]=="Heart rate 64 bpm 2026-10-20"


def test_ocr_candidate_requires_known_analyte_and_quote_evidence():
    from datetime import datetime, timezone
    from rally_health.domain.schemas import ReportCandidate
    from rally_health.integrations.report_ocr import _is_known_analyte, _quote_supports_candidate

    metric={"metric_code":"RS-01","name":"Heart Rate"}
    supported=ReportCandidate(metric_code="RS-01",source_analyte="Heart Rate",value=64,unit="bpm",
        observed_at=datetime(2026,10,20, tzinfo=timezone.utc),source_text="Heart rate 64 bpm 2026-10-20",source_page=1)
    hallucinated_value=supported.model_copy(update={"value":65})
    wrong_analyte=supported.model_copy(update={"source_analyte":"Hemoglobin"})
    wrong_date=supported.model_copy(update={"observed_at":datetime(2026,10,21,tzinfo=timezone.utc)})
    assert _is_known_analyte(supported.source_analyte,metric)
    assert _quote_supports_candidate(supported)
    assert not _quote_supports_candidate(hallucinated_value)
    assert not _is_known_analyte(wrong_analyte.source_analyte,metric)
    assert not _quote_supports_candidate(wrong_date)


def test_sleep_summary_api_returns_prototype_and_refuses_fake_confidence(tmp_path):
    with make_client(tmp_path) as client:
        observations=[
            {"metric_code":"SL-05","aggregation":"interval","value":60,"unit":"min","observed_at":"2026-10-20T23:00:00Z","end_at":"2026-10-21T00:00:00Z","provenance":{"source_type":"apple_health","source_id":"sleep-1"}},
            {"metric_code":"SL-07","aggregation":"interval","value":60,"unit":"min","observed_at":"2026-10-20T23:30:00Z","end_at":"2026-10-21T00:30:00Z","provenance":{"source_type":"apple_health","source_id":"sleep-2"}},
        ]
        batch=client.post("/v1/ingest/batches",headers=headers(),json={"batch_id":"sleep-summary-batch-1","provider":"apple_health","observations":observations})
        assert batch.status_code==200, batch.text
        result=client.get("/v1/metrics/SL-02/sleep-summary",headers=headers(),params={"night":"2026-10-20","timezone":"UTC"})
        assert result.status_code==200, result.text
        assert result.json()["tst_minutes"]==90
        assert result.json()["formula_reviewed"] is False
        assert result.json()["window_basis"]=="local_noon_to_noon"
        assert result.json()["confidence_level"]=="UNAVAILABLE"
def test_manual_questionnaire_profile_and_unconfigured_report_flow(tmp_path):
    with make_client(tmp_path) as client:
        profile={"user_id":"ignored-client-user","goals":["sleep"],"fitness_level":"beginner",
                 "equipment":[],"availability":{},"movement_limitations":[],"health_context":{},"consent_scopes":[]}
        saved=client.put("/v1/profiles/me",headers=headers(),json=profile)
        assert saved.status_code==200 and saved.json()["profile"]["user_id"]=="task1-user"
        questionnaire=client.post("/v1/questionnaires/submissions",headers=headers(),json={
            "instrument":"daily-sleep-self-report","version":"1","completed_at":"2026-10-20T08:00:00Z",
            "answers":{"sleep_quality":3,"felt_recovered":False}})
        assert questionnaire.status_code==200
        assert questionnaire.json()["status"]=="RECEIVED_NOT_SCORED"
        report=client.post("/v1/reports/upload-intake",headers=headers(),files={"file":("lab.pdf",b"%PDF-test","application/pdf")})
        assert report.status_code==200
        assert report.json()["status"]=="RECEIVED_NOT_PROCESSED"
        job=client.get(f"/v1/reports/extraction-jobs/{report.json()['job_id']}",headers=headers())
        assert job.status_code==200 and job.json()["candidates"]==[]
