from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASE_TIME = datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)


def observation(case_id: str, metric: str, value: float, unit: str, minute: int = 0,
                source: str = "manual", source_id: str | None = None, variant: str | None = None,
                timestamp: str | None = None, end_at: str | None = None,
                quality_flags: list[str] | None = None, expected_status: str = "ACCEPTED",
                expected_value: float | None = None, expected_unit: str | None = None,
                expected_flags: list[str] | None = None) -> dict[str, Any]:
    observed = timestamp or (BASE_TIME + timedelta(minutes=minute)).isoformat()
    return {
        "case_id": case_id,
        "synthetic": True,
        "user_id": "synthetic-task1-test",
        "observation": {
            "metric_code": metric, "metric_variant": variant, "value": value, "unit": unit,
            "observed_at": observed, "end_at": end_at,
            "provenance": {"source_type": source, "source_id": source_id or f"synthetic:{case_id}"},
            "quality_flags": quality_flags or [], "user_confirmed": False,
        },
        "expected": {"status": expected_status, "value": expected_value,
                     "unit": expected_unit, "quality_flags": expected_flags or []},
        "purpose": "Engineering behavior test only; not physiological ground truth.",
    }


def build_cases() -> list[dict[str, Any]]:
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    return [
        observation("unit-hours-to-minutes", "SL-02", 7.5, "hours", expected_value=450, expected_unit="min"),
        observation("unit-pounds-to-kilograms", "HS-05", 150, "lb", expected_value=68.0388555, expected_unit="kg"),
        observation("timestamp-offset-normalization", "RS-04", 64, "bpm",
                    timestamp="2026-01-15T00:00:00-08:00", expected_value=64, expected_unit="bpm"),
        observation("invalid-unit", "RS-04", 64, "kg", expected_status="UNSUPPORTED_UNIT"),
        observation("out-of-range-heart-rate", "RS-04", 900, "bpm", expected_status="INVALID_OBSERVATION",
                    expected_flags=["value_out_of_range"]),
        observation("hrv-sdnn-explicit", "RS-05", 42, "ms", variant="SDNN", expected_value=42, expected_unit="ms"),
        observation("hrv-rmssd-explicit", "RS-05", 31, "ms", variant="RMSSD", expected_value=31, expected_unit="ms"),
        observation("hrv-variant-missing", "RS-05", 42, "ms", expected_status="INVALID_OBSERVATION",
                    expected_flags=["hrv_variant_required"]),
        observation("future-timestamp", "RS-04", 64, "bpm", timestamp=future,
                    expected_status="INVALID_OBSERVATION", expected_flags=["timestamp_in_future"]),
        observation("end-before-start", "RS-04", 64, "bpm", end_at="2026-01-15T07:00:00+00:00",
                    expected_status="INVALID_OBSERVATION", expected_flags=["end_before_start"]),
        observation("missing-value-null", "RS-04", 0, "bpm", quality_flags=["missing_value"],
                    expected_status="INVALID_OBSERVATION", expected_flags=["missing_value"]),
        observation("same-source-different-id-not-exact-dedup", "RS-04", 60, "bpm", source="wearable", source_id="watch-record-100"),
        observation("conflict-reference-manual", "RS-04", 60, "bpm", source="manual", source_id="manual-reference"),
        observation("conflict-reference-wearable", "RS-04", 76, "bpm", source="wearable", source_id="watch-reference",
                    expected_flags=["source_conflict_review_required"]),
        observation("accept-sleepaccel-relative-timestamp", "RS-01", 71, "bpm", timestamp="1970-01-01T00:00:00Z",
                    quality_flags=["relative_seconds_from_psg_start_not_utc", "synthetic_dataset_fixture"],
                    expected_flags=["relative_seconds_from_psg_start_not_utc", "synthetic_dataset_fixture"]),
        observation("research-bidsleep-source-provenance", "RS-01", 68, "bpm", source="wearable", source_id="bidsleep:Bidslab00:night1:hr:row4",
                    quality_flags=["research_dataset_not_user_sample"],
                    expected_flags=["research_dataset_not_user_sample"]),
        {
            "case_id": "duplicate-observation-pair", "synthetic": True,
            "purpose": "Submitting the second observation unchanged should return DUPLICATE_OBSERVATION.",
            "sequence": [
                observation("duplicate-first", "RS-04", 60, "bpm", source="wearable", source_id="duplicate:watch:1"),
                observation("duplicate-second", "RS-04", 60, "bpm", source="wearable", source_id="duplicate:watch:1"),
            ],
            "expected": {"first": "ACCEPTED", "second": "DUPLICATE_OBSERVATION"},
        },
        {
            "case_id": "baseline-distinct-day-coverage", "synthetic": True,
            "purpose": "Seven records across four distinct UTC days must not satisfy a seven-valid-day baseline.",
            "records": [{"metric_code": "RS-04", "value": 60 + i, "unit": "bpm",
                         "observed_at": (BASE_TIME - timedelta(days=9 - i // 2)).isoformat(),
                         "source_id": f"baseline:{i}"} for i in range(7)],
            "expected": {"status": "INSUFFICIENT_DATA", "valid_samples": 7, "valid_days": 4,
                         "required_valid_days": 7},
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic Task 1 edge-case fixtures.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    with args.output.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(cases), "output": str(args.output), "synthetic_only": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
