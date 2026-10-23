from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from rally_health.domain.schemas import AggregationKind, ObservationOut
from rally_health.services.sleep import summarize_sleep_intervals


def make_interval(code: str, start: str, end: str) -> ObservationOut:
    return ObservationOut(metric_code=code,aggregation=AggregationKind.INTERVAL,value=1,unit="min",
        observed_at=datetime.fromisoformat(start.replace("Z","+00:00")),end_at=datetime.fromisoformat(end.replace("Z","+00:00")),
        provenance={"source_type":"apple_health"},user_id="sleep-user")


def test_sleep_summary_unions_overlapping_asleep_stages_without_double_counting():
    rows=[make_interval("SL-05","2026-10-20T23:00:00Z","2026-10-21T00:00:00Z"),
          make_interval("SL-07","2026-10-20T23:30:00Z","2026-10-21T00:30:00Z"),
          make_interval("SL-04","2026-10-21T00:30:00Z","2026-10-21T00:40:00Z")]
    result=summarize_sleep_intervals(rows,date(2026,10,20),"UTC")
    assert result["tst_minutes"]==90
    assert result["window_basis"]=="local_noon_to_noon"
    assert result["stage_minutes"]["SL-05"]==60
    assert result["stage_minutes"]["SL-07"]==60
    assert result["stage_minutes"]["SL-04"]==10
    assert result["formula_reviewed"] is False
    assert result["confidence_level"]=="UNAVAILABLE"


def test_sleep_summary_clips_to_local_calendar_night():
    rows=[make_interval("SL-05","2026-10-20T06:30:00Z","2026-10-20T08:00:00Z")]
    result=summarize_sleep_intervals(rows,date(2026,10,19),"America/Los_Angeles")
    assert result["tst_minutes"]==90


def test_sleep_summary_reports_missing_intervals_and_invalid_timezone():
    missing=summarize_sleep_intervals([],date(2026,10,20),"UTC")
    assert missing["status"]=="INSUFFICIENT_DATA"
    invalid=summarize_sleep_intervals([],date(2026,10,20),"Invalid/Zone")
    assert invalid["status"]=="INVALID_INPUT"
