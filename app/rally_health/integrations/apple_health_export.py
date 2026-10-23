from __future__ import annotations

from datetime import datetime, timezone
from xml.etree import ElementTree

from ..domain.schemas import AggregationKind, DataSource, ObservationCreate, Provenance


TYPE_MAP = {
    "HKQuantityTypeIdentifierHeartRate": ("RS-01", "bpm", AggregationKind.INSTANT, None),
    "HKQuantityTypeIdentifierRestingHeartRate": ("RS-04", "bpm", AggregationKind.DAILY_DERIVED, None),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": ("RS-05", "ms", AggregationKind.DAILY_DERIVED, "SDNN"),
    "HKQuantityTypeIdentifierOxygenSaturation": ("RS-10", "%", AggregationKind.DAILY_DERIVED, None),
    "HKQuantityTypeIdentifierRespiratoryRate": ("RS-06", "breaths/min", AggregationKind.DAILY_DERIVED, None),
    "HKQuantityTypeIdentifierStepCount": ("AC-09", "steps", AggregationKind.DAILY_DERIVED, None),
    "HKCategoryTypeIdentifierSleepAnalysis": ("SL-02", "min", AggregationKind.INTERVAL, None),
}

SLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleepCore": ("SL-05", "Core"),
    "HKCategoryValueSleepAnalysisAsleepDeep": ("SL-06", "Deep"),
    "HKCategoryValueSleepAnalysisAsleepREM": ("SL-07", "REM"),
    "HKCategoryValueSleepAnalysisAwake": ("SL-04", "Awake"),
    "HKCategoryValueSleepAnalysisInBed": (None, "InBed"),
    "HKCategoryValueSleepAnalysisAsleepUnspecified": ("SL-02", "AsleepUnspecified"),
    "HKCategoryValueSleepAnalysisAsleep": ("SL-02", "AsleepUnspecified"),
}


def _time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return None
    return value.astimezone(timezone.utc)


def parse_apple_health_export(xml_bytes: bytes, batch_id: str) -> tuple[list[ObservationCreate], dict[str, int]]:
    if len(xml_bytes) > 50 * 1024 * 1024:
        raise ValueError("Health export XML exceeds 50 MB")
    upper = xml_bytes.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("DTD and entity declarations are not accepted")
    root = ElementTree.fromstring(xml_bytes)
    parsed: list[ObservationCreate] = []
    report = {"parsed": 0, "unsupported_type": 0, "invalid_record": 0}
    for item in root.iter():
        if item.tag != "Record":
            continue
        source_type = item.attrib.get("type", "")
        spec = TYPE_MAP.get(source_type)
        if spec is None:
            report["unsupported_type"] += 1
            continue
        code, default_unit, aggregation, variant = spec
        start = _time(item.attrib.get("startDate"))
        end = _time(item.attrib.get("endDate"))
        if start is None:
            report["invalid_record"] += 1
            continue
        try:
            if aggregation == AggregationKind.INTERVAL:
                stage_mapping = SLEEP_VALUES.get(item.attrib.get("value", ""))
                if stage_mapping is None or end is None or end <= start:
                    report["unsupported_type" if stage_mapping is None else "invalid_record"] += 1
                    continue
                metric_for_stage, stage_name = stage_mapping
                if metric_for_stage is None:
                    report["unsupported_type"] += 1
                    continue
                code = metric_for_stage
                value = (end - start).total_seconds() / 60.0
                variant = None
                unit = "min"
            else:
                value = float(item.attrib["value"])
                unit = item.attrib.get("unit", default_unit)
                if code == "RS-10" and unit.casefold() in {"%", "percent"}:
                    value *= 100
                if code == "AC-09" and unit.casefold() in {"count", "counts"}:
                    unit = "steps"
            metadata = {node.attrib.get("key"): node.attrib.get("value") for node in item.findall("MetadataEntry")}
            source_name = item.attrib.get("sourceName")
            device = item.attrib.get("device", "")
            provider = "Apple Health"
            source_id = source_name or provider
            provenance = Provenance(
                source_type=DataSource.APPLE_HEALTH,
                provider=provider,
                source_id=source_id,
                sample_id=item.attrib.get("uuid"),
                device_id=item.attrib.get("deviceId"),
                device_model=device or None,
                firmware_version=metadata.get("HKDeviceFirmwareVersion"),
                measurement_context=metadata.get("HKMetadataKeyHeartRateMotionContext"),
                import_batch_id=batch_id,
            )
            parsed.append(ObservationCreate(
                metric_code=code, metric_variant=variant, aggregation=aggregation,
                value=value, unit=unit, observed_at=start, end_at=end,
                received_at=datetime.now(timezone.utc), provenance=provenance,
            ))
            report["parsed"] += 1
        except (KeyError, ValueError, TypeError):
            report["invalid_record"] += 1
    return parsed, report
