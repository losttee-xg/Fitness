from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


NORMALIZED_FIELDS = [
    "dataset_id", "dataset_version", "participant_id", "source_file", "source_row",
    "source_metric", "metric_code", "metric_variant", "raw_value", "raw_unit",
    "value", "unit", "source_timestamp", "observed_at_utc", "quality_flags",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def make_record(dataset: str, participant: str, file: Path, row_number: int, source_metric: str,
                metric_code: str | None, raw_value: Any, raw_unit: str | None,
                value: float | None, unit: str | None, source_time: str | None,
                utc_time: str | None, variant: str | None = None, flags: list[str] | None = None) -> dict[str, Any]:
    return {
        "dataset_id": dataset, "dataset_version": "as-published", "participant_id": participant,
        "source_file": str(file), "source_row": row_number, "source_metric": source_metric,
        "metric_code": metric_code, "metric_variant": variant,
        "raw_value": raw_value, "raw_unit": raw_unit, "value": value, "unit": unit,
        "source_timestamp": source_time, "observed_at_utc": utc_time,
        "quality_flags": json.dumps(flags or [], ensure_ascii=False),
    }


def export_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def process_pamap2(root: Path) -> list[dict[str, Any]]:
    rows = []
    for file in sorted(root.rglob("*.dat")):
        if not file.is_file():
            continue
        participant = file.stem.split("_")[0]
        with file.open(encoding="utf-8", errors="replace") as stream:
            for row_number, line in enumerate(stream, start=1):
                fields = line.split()
                if len(fields) < 3:
                    continue
                timestamp = numeric(fields[0])
                hr = numeric(fields[2])
                if timestamp is None or hr is None or hr <= 0:
                    continue
                utc = None
                flags = ["relative_elapsed_seconds_not_wall_clock_utc"]
                rows.append(make_record("pamap2", participant, file, row_number, "heart_rate", "RS-01", hr,
                                        "bpm", hr, "bpm", str(timestamp), utc, flags=flags))
    return rows


def process_physionet_wearable(root: Path) -> list[dict[str, Any]]:
    rows = []
    for file in sorted(root.rglob("*.csv")):
        if not file.is_file() or file.name.casefold() in {"subject-info.csv", "data_dictionary.csv", "stress_level_v1.csv", "stress_level_v2.csv"}:
            continue
        try:
            with file.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
                reader = csv.DictReader(stream)
                names = {str(name).strip().casefold(): name for name in (reader.fieldnames or [])}
                time_col = next((names[k] for k in ("time", "timestamp", "datetime", "date_time") if k in names), None)
                value_col = next((names[k] for k in ("value", "hr", "heart_rate", "bvp", "ibi", "eda", "temperature", "acc_x") if k in names), None)
                if value_col is None:
                    continue
                participant = next((part for part in file.parts if part.upper().startswith(("S", "F")) and len(part) <= 5), file.parent.name)
                for row_number, row in enumerate(reader, start=2):
                    raw = row.get(value_col)
                    value = numeric(raw)
                    if value is None:
                        continue
                    key = value_col.casefold()
                    metric, unit = ("RS-01", "bpm") if key in {"hr", "heart_rate"} else (None, None)
                    timestamp = row.get(time_col) if time_col else None
                    parsed_utc = None
                    flags = []
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            if dt.tzinfo is not None:
                                parsed_utc = dt.astimezone(timezone.utc).isoformat()
                            else:
                                flags.append("timezone_missing")
                        except ValueError:
                            flags.append("timestamp_format_unparsed")
                    else:
                        flags.append("timestamp_column_not_found")
                    rows.append(make_record("physionet_wearable_device", participant, file, row_number,
                                            str(value_col), metric, raw, unit, value, unit, timestamp, parsed_utc, flags=flags))
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows


def process_mmash(root: Path) -> list[dict[str, Any]]:
    rows = []
    for file in sorted(root.rglob("*.csv")):
        participant = file.parent.name
        try:
            with file.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
                reader = csv.DictReader(stream)
                for row_number, row in enumerate(reader, start=2):
                    for column, raw in row.items():
                        value = numeric(raw)
                        if value is None:
                            continue
                        code = None
                        unit = None
                        key = (column or "").casefold()
                        if "hr" in key and "rr" not in key:
                            code, unit = "RS-01", "bpm"
                        rows.append(make_record("mmash", participant, file, row_number, column or "unknown",
                                                code, raw, unit, value, unit, None, None,
                                                flags=["clock_time_or_day_context_not_normalized"] if code else ["unmapped_dataset_field"]))
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows


def process_bundles(extracted: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    processors = {
        "pamap2": process_pamap2,
        "physionet_wearable": process_physionet_wearable,
        "mmash": process_mmash,
    }
    for name, processor in processors.items():
        root = extracted / name
        if not root.exists():
            results[name] = {"status": "not_extracted", "records": 0}
            continue
        rows = processor(root)
        file_out = output / f"{name}_observations.csv"
        count = export_csv(file_out, rows)
        results[name] = {"status": "normalized", "records": count, "file": file_out.name,
                         "sha256": sha256_file(file_out)}
    # PPG-DaLiA pickle payloads are intentionally not loaded because pickle can execute code.
    ppg_root = extracted / "ppg_dalia"
    results["ppg_dalia"] = {"status": "preserved_raw_not_parsed", "records": 0,
                             "reason": "Archive contains Python pickle data; use a vetted non-executing conversion path before parsing." if ppg_root.exists() else "not_extracted"}
    bidmc_root = extracted / "bidmc"
    bidmc_csv = next(iter(sorted(bidmc_root.rglob("numerics.csv"))), None) if bidmc_root.exists() else None
    results["bidmc"] = {"status": "preserved_raw_pending_schema_adapter", "records": 0,
                         "reason": "BIDMC numerics can be mapped after checking column headers and record/time semantics." if bidmc_csv else "not_extracted_or_csv_not_found"}
    manifest = {"purpose": "engineering test data only; not personal user records or clinical validation",
                "normalization_version": "1.0.0", "datasets": results}
    manifest_path = output / "normalized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(process_bundles(args.extracted, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
