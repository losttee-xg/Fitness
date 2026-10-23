from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..domain.schemas import AggregationKind, ObservationOut

SLEEP_STAGE_CODES = {"SL-02", "SL-04", "SL-05", "SL-06", "SL-07"}
ASLEEP_CODES = {"SL-02", "SL-05", "SL-06", "SL-07"}


def _merge_minutes(intervals: list[tuple[datetime, datetime]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort(key=lambda pair: pair[0])
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += (end-start).total_seconds()/60
            start, end = next_start, next_end
    return total + (end-start).total_seconds()/60


def summarize_sleep_intervals(observations: list[ObservationOut], night: date, timezone_name: str="UTC") -> dict:
    try:
        user_zone=ZoneInfo(timezone_name) if timezone_name != "UTC" else timezone.utc
    except (ZoneInfoNotFoundError,ValueError):
        return {"status":"INVALID_INPUT","date":night.isoformat(),"error":"invalid_timezone"}
    local_start=datetime.combine(night,time(12,0),tzinfo=user_zone)
    local_end=datetime.combine(night+timedelta(days=1),time(12,0),tzinfo=user_zone)
    if isinstance(user_zone,ZoneInfo):
        local_start=local_start.replace(fold=0)
        local_end=local_end.replace(fold=0)
    window_start=local_start.astimezone(timezone.utc)
    window_end=local_end.astimezone(timezone.utc)
    stage_intervals:dict[str,list[tuple[datetime,datetime]]]=defaultdict(list)
    for item in observations:
        if item.metric_code not in SLEEP_STAGE_CODES or item.aggregation != AggregationKind.INTERVAL or item.end_at is None:
            continue
        start=max(item.observed_at.astimezone(timezone.utc),window_start)
        end=min(item.end_at.astimezone(timezone.utc),window_end)
        if end>start:
            stage_intervals[item.metric_code].append((start,end))
    if not stage_intervals:
        return {"status":"INSUFFICIENT_DATA","date":night.isoformat(),"timezone":timezone_name,
                "window_basis":"local_noon_to_noon","tst_minutes":None,"stage_minutes":{},"confidence_level":"UNAVAILABLE"}
    stage_minutes={code:round(_merge_minutes(intervals),2) for code,intervals in stage_intervals.items()}
    asleep_intervals=[interval for code,intervals in stage_intervals.items() if code in ASLEEP_CODES for interval in intervals]
    tst=round(_merge_minutes(asleep_intervals),2)
    return {"status":"PROTOTYPE_READY","date":night.isoformat(),"timezone":timezone_name,
            "window_basis":"local_noon_to_noon","tst_minutes":tst,"stage_minutes":stage_minutes,"confidence_level":"UNAVAILABLE",
            "formula_id":"sleep_interval_union_prototype_v1","formula_reviewed":False,
            "limitations":["Intervals are unioned to prevent overlap double-counting.",
                           "Time in bed is not available unless a validated in-bed interval source is supplied.",
                           "This prototype is not a reviewed Rally sleep formula and is not clinically validated."]}
