from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rally_health.domain.schemas import ObservationCreate


UNIMPLEMENTED_CASES = {
    "unit-alias-healthkit-count-per-min": ["count/min unit alias"],
    "unit-hrv-seconds-to-ms": ["seconds-to-milliseconds conversion for HRV"],
    "metric-unknown-code": ["stable UNSUPPORTED_METRIC error; current normalizer reports unsupported unit"],
    "timestamp-future-within-tolerance": ["clock_skew_suspected flag for tolerated future timestamp"],
    "timestamp-future-at-tolerance-boundary": ["clock_skew_suspected flag for tolerated future timestamp"],
    "timestamp-implausibly-old-without-flag": ["old timestamp plausibility rejection"],
    "timestamp-relative-with-flag": ["accepting relative research timestamps with non-UTC semantics"],
    "interval-zero-duration": ["zero-duration interval rejection"],
    "interval-missing-end": ["required end_at for interval aggregation"],
    "caller-flags-not-trusted": ["server-owned quality flags; current service rejects any incoming flags"],
    "hrv-variant-case-insensitive": ["case-insensitive HRV variant canonicalization"],
    "hrv-variant-unknown": ["distinct unknown-variant validation error"],
    "sync-late-over-24h": ["received_at storage and late-sync annotation"],
    "sync-exactly-24h-not-late": ["received_at storage and sync-latency policy"],
    "sync-normal-delay": ["received_at storage and sync-latency policy"],
    "sync-received-before-observed": ["received_at clock-skew assessment"],
    "sync-received-before-observed-within-tolerance": ["received_at clock-skew assessment"],
    "research-dataset-provenance": ["research_dataset source type; current enum has no such value", "acceptance of informational quality flags"],
    "source-id-reused-different-payload": ["source_id uniqueness/reuse protection across changed payloads"],
    "conflict-below-threshold": ["absolute 10 bpm conflict threshold; current code uses relative 2 percent threshold"],
    "conflict-outside-time-window": ["five-minute conflict window; current code compares exact timestamps only"],
    "conflict-at-window-boundary": ["five-minute conflict window; current code compares exact timestamps only"],
    "conflict-multi-device": ["provider/device-aware conflict grouping; current code compares source_type only"],
    "conflict-skipped-for-non-instant": ["aggregation-aware conflict suppression"],
    "baseline-day-basis-local-utc-plus-7-insufficient": ["local-timezone baseline day grouping; current code groups by UTC"],
}


def current_request(observation: dict) -> tuple[dict, list[str], str | None]:
    request = deepcopy(observation)
    dropped = []
    for field in ("synthetic", "received_at", "aggregation"):
        if field in request:
            dropped.append(field)
            request.pop(field)
    provenance = request.get("provenance", {})
    if isinstance(provenance.get("device"), dict):
        dropped.append("provenance.device (object metadata is not in current schema)")
        device = provenance.pop("device")
        if "device_id" not in provenance:
            vendor = device.get("vendor")
            model = device.get("model")
            if vendor or model:
                provenance["device_id"] = ":".join(str(x) for x in (vendor, model) if x)
    try:
        ObservationCreate.model_validate(request)
        validation = "SCHEMA_VALID"
    except Exception as error:
        validation = f"SCHEMA_REJECTED:{type(error).__name__}"
    return request, dropped, validation


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: fix_task1_edge_cases_v2.py INPUT OUTPUT")
    source, target = map(Path, sys.argv[1:])
    fixed = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        case = json.loads(line)
        case["fixture_version"] = "2.1.0"
        case["data_class"] = "synthetic_specification_only"
        case["clock_policy"] = {
            "fixture_clock": case.get("clock"),
            "freeze_required_for_current_runtime": case.get("kind") in {"ingest", "baseline"},
            "note": "The current API reads system time directly; the fixture clock is metadata until the test harness injects it.",
        }
        case["current_implementation"] = {
            "status": "NOT_EXECUTED_BY_THIS_FILE",
            "unsupported_or_dropped_features": UNIMPLEMENTED_CASES.get(case["case_id"], []),
        }
        current_steps = []
        if case.get("kind") == "ingest":
            for step in case.get("steps", []):
                req, dropped, validation = current_request(step["observation"])
                current_steps.append({
                    "request": req,
                    "schema_validation": validation,
                    "dropped_target_fields": dropped,
                    "target_expectation": deepcopy(step.get("expect")),
                })
        elif case.get("kind") == "baseline":
            current_records = []
            dropped_features = set(case["current_implementation"]["unsupported_or_dropped_features"])
            for record in case.get("records", []):
                req, dropped, validation = current_request(record)
                current_records.append({"request": req, "schema_validation": validation,
                                        "dropped_target_fields": dropped})
            if case.get("params", {}).get("day_basis") == "local":
                dropped_features.add("local timezone day-basis")
            case["current_implementation"]["unsupported_or_dropped_features"] = sorted(dropped_features)
            case["current_api_records"] = current_records
            case["target_expectation"] = deepcopy(case.get("expect"))
            if case.get("expect", {}).get("status") == "SUFFICIENT_DATA":
                case["target_expectation"]["status"] = "READY"
                case["target_expectation"]["note"] = "Current baseline API uses READY for sufficient data."
        case["current_api_steps"] = current_steps
        case["target_contract"] = "Original steps/records and expectations describe intended ingest semantics; current_api_* contains a schema-compatible projection where possible. Dropped features are explicitly listed."
        fixed.append(case)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for case in fixed:
            stream.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(fixed), "output": str(target), "version": "2.1.0"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
