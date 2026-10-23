from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rally_health.repositories.database import SQLiteDatabase
from rally_health.services.metrics import MetricEngine
from rally_health.services.observations import ObservationService
from rally_health.domain.schemas import ObservationCreate
from rally_health.domain.schemas import Provenance
from fastapi import HTTPException


def payload(metric_code="SL-02", value=2, unit="hours", observed_at="2026-10-20T07:00:00-04:00", source="manual", variant=None, source_id="sample-1"):
    return ObservationCreate(
        metric_code=metric_code, metric_variant=variant, value=value, unit=unit,
        observed_at=observed_at,
        provenance=Provenance(source_type=source, source_id=source_id),
    )


def test_ingestion_converts_units_and_normalizes_timestamp(tmp_path):
    service = ObservationService(SQLiteDatabase(tmp_path / "task1.sqlite3"))
    stored = service.add("u1", payload())
    assert stored.value == 120
    assert stored.unit == "min"
    assert stored.observed_at == datetime(2026, 10, 20, 11, 0, tzinfo=timezone.utc)


def test_ingestion_rejects_exact_duplicate(tmp_path):
    service = ObservationService(SQLiteDatabase(tmp_path / "task1.sqlite3"))
    first = payload()
    service.add("u1", first)
    duplicate = first.model_copy(update={"observation_id": "different-id"})
    try:
        service.add("u1", duplicate)
    except HTTPException as error:
        assert error.status_code == 409
        assert error.detail["code"] == "DUPLICATE_OBSERVATION"
    else:
        raise AssertionError("Expected duplicate observation to be rejected")


def test_hrv_requires_variant_to_keep_sdnn_and_rmssd_separate(tmp_path):
    service = ObservationService(SQLiteDatabase(tmp_path / "task1.sqlite3"))
    try:
        service.add("u1", payload("RS-05", 42, "ms", variant=None))
    except HTTPException as error:
        assert error.status_code == 422
        assert "hrv_variant_required" in error.detail["quality_flags"]
    else:
        raise AssertionError("Expected HRV variant to be required")
    sdnn = service.add("u1", payload("RS-05", 42, "ms", variant="SDNN", source_id="sdnn"))
    rmssd = service.add("u1", payload("RS-05", 42, "ms", variant="RMSSD", source_id="rmssd"))
    assert sdnn.metric_variant != rmssd.metric_variant


def test_conflicting_sources_are_flagged_and_not_silently_merged(tmp_path):
    service = ObservationService(SQLiteDatabase(tmp_path / "task1.sqlite3"))
    when = "2026-10-20T07:00:00Z"
    service.add("u1", payload("RS-04", 60, "bpm", when, "manual", source_id="manual-1"))
    second = service.add("u1", payload("RS-04", 75, "bpm", when, "wearable", source_id="watch-1"))
    assert "source_conflict_review_required" in second.quality_flags
    assert service.resolve_source_conflicts("u1", "RS-04")["status"] == "REVIEW_REQUIRED"


def test_baseline_uses_local_timezone_and_rejects_unknown_zone(tmp_path):
    service = ObservationService(SQLiteDatabase(tmp_path / "timezone.sqlite3"))
    first = payload("RS-04",60,"bpm","2026-10-20T16:00:00Z",source_id="tz-1").model_copy(update={"received_at":datetime(2026,10,22,tzinfo=timezone.utc)})
    second = payload("RS-04",62,"bpm","2026-10-21T01:00:00Z",source_id="tz-2").model_copy(update={"received_at":datetime(2026,10,22,tzinfo=timezone.utc)})
    service.add("u1",first)
    service.add("u1",second)
    engine=MetricEngine(service)
    local=engine.baseline("u1","RS-04",window_days=365,minimum_samples=2,minimum_valid_days=2,timezone_name="America/Los_Angeles")
    utc=engine.baseline("u1","RS-04",window_days=365,minimum_samples=2,minimum_valid_days=2,timezone_name="UTC")
    assert local["valid_days"]==1
    assert utc["valid_days"]==2
    assert local["timezone"]=="America/Los_Angeles"
    invalid=engine.baseline("u1","RS-04",timezone_name="not/a-zone")
    assert invalid["status"]=="INVALID_INPUT"
    assert invalid["error"]=="invalid_timezone"


def test_baseline_requires_enough_distinct_valid_days(tmp_path):
    service = ObservationService(SQLiteDatabase(tmp_path / "task1.sqlite3"))
    base = datetime.now(timezone.utc) - timedelta(days=10)
    for index in range(7):
        when = (base + timedelta(days=index // 2)).isoformat()
        service.add("u1", payload("RS-04", 60 + index, "bpm", when, "manual", source_id=f"rhr-{index}"))
    result = MetricEngine(service).baseline("u1", "RS-04", window_days=28, minimum_samples=7, minimum_valid_days=7)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["valid_samples"] == 7
    assert result["valid_days"] == 4
