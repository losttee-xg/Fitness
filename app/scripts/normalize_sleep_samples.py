from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_MAP = {0: "WAKE", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "UNKNOWN"}


def finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(dataset: str, version: str, participant: str, path: Path, row: int | None,
           source_metric: str, metric_code: str | None, raw_value: Any, raw_unit: str | None,
           value: Any, unit: str | None, source_time: Any, utc_time: str | None,
           variant: str | None = None, flags: list[str] | None = None) -> dict[str, Any]:
    return {"dataset_id": dataset, "dataset_version": version, "participant_id": participant,
            "source_file": str(path), "source_row": row, "source_metric": source_metric,
            "metric_code": metric_code, "metric_variant": variant, "raw_value": raw_value,
            "raw_unit": raw_unit, "value": value, "unit": unit,
            "source_timestamp": source_time, "observed_at_utc": utc_time,
            "quality_flags": flags or []}


def normalize_sleepaccel(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"dataset_id": "sleepaccel", "version": "1.0.0", "files": {}}
    for path in sorted(root.glob("*.txt")):
        participant = path.name.split("_")[0]
        kind = path.name.removeprefix(participant + "_").removesuffix(".txt")
        file_rows = 0
        malformed = 0
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.strip().replace(",", " ").split()
                if len(fields) < 2:
                    malformed += 1
                    continue
                seconds = finite_float(fields[0])
                val = finite_float(fields[1])
                if seconds is None or val is None:
                    malformed += 1
                    continue
                utc = None
                flags = ["relative_seconds_from_psg_start_not_utc"]
                if kind == "heartrate":
                    rows.append(record("sleepaccel", "1.0.0", participant, path, line_number,
                                        "heart_rate", "RS-01", fields[1], "bpm", val, "bpm", seconds, utc,
                                        flags=flags))
                elif kind == "labeled_sleep":
                    stage_code = int(val)
                    rows.append(record("sleepaccel", "1.0.0", participant, path, line_number,
                                        "psg_sleep_stage", None, fields[1], "stage_code",
                                        STAGE_MAP.get(stage_code, f"UNMAPPED_{stage_code}"), "stage",
                                        seconds, utc, flags=flags + ["reference_label_not_user_metric"]))
                elif kind == "steps":
                    rows.append(record("sleepaccel", "1.0.0", participant, path, line_number,
                                        "steps", None, fields[1], "steps", val, "steps", seconds, utc,
                                        flags=flags + ["interval_end_or_start_semantics_require_review"]))
                elif kind == "acceleration" and len(fields) >= 4:
                    axes = [finite_float(item) for item in fields[1:4]]
                    if any(axis is None for axis in axes):
                        malformed += 1
                        continue
                    rows.append(record("sleepaccel", "1.0.0", participant, path, line_number,
                                        "acceleration_xyz", None, fields[1:4], "g", axes, "g", seconds,
                                        utc, flags=flags))
                file_rows += 1
        summary["files"][path.name] = {"parsed_rows": file_rows, "malformed_rows": malformed,
                                       "bytes": path.stat().st_size, "sha256": sha256(path)}
    return rows, summary


def normalize_bidsleep_night(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"dataset_id": "bidsleep", "version": "1.0.1", "participant_id": root.parent.name,
                               "night": root.name, "files": {}, "notes": []}
    hr_file = root / "hr.csv"
    if hr_file.exists():
        with hr_file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            for row_number, item in enumerate(reader, start=1):
                if len(item) < 2:
                    continue
                timestamp = finite_float(item[0])
                heart_rate = finite_float(item[1])
                if timestamp is None or heart_rate is None:
                    continue
                dt = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                rows.append(record("bidsleep", "1.0.1", root.parent.name, hr_file, row_number,
                                   "heart_rate", "RS-01", item[1], "bpm",
                                   heart_rate, "bpm", timestamp, dt, flags=["research_dataset_not_user_sample"]))
        summary["files"][hr_file.name] = {"bytes": hr_file.stat().st_size, "sha256": sha256(hr_file),
                                          "parsed_records": sum(row["source_file"] == str(hr_file) for row in rows)}
    motion_file = root / "motion.csv"
    if motion_file.exists():
        with motion_file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            count = 0
            for row_number, item in enumerate(reader, start=2):
                timestamp = finite_float(item.get("Timestamp", ""))
                axes = [finite_float(item.get(axis, "")) for axis in ("x", "y", "z")]
                if timestamp is None or any(axis is None for axis in axes):
                    continue
                dt = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                rows.append(record("bidsleep", "1.0.1", root.parent.name, motion_file, row_number,
                                   "acceleration_xyz", None, axes, "g", axes, "g", timestamp, dt,
                                   flags=["research_dataset_not_user_sample"]))
                count += 1
            summary["files"][motion_file.name] = {"bytes": motion_file.stat().st_size,
                                                  "sha256": sha256(motion_file), "parsed_records": count}
    labels = root / "labels.mat"
    if labels.exists():
        header = labels.read_bytes()[:128]
        summary["files"][labels.name] = {"bytes": labels.stat().st_size, "sha256": sha256(labels),
                                         "mat_header": header[:20].decode("ascii", errors="replace"),
                                         "status": "preserved_unparsed_mat_file"}
        summary["notes"].append("MAT labels retained with hash; scipy is not installed, no MATLAB parser assumed.")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleepaccel", type=Path, required=True)
    parser.add_argument("--bidsleep-night", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sleep_rows, sleep_summary = normalize_sleepaccel(args.sleepaccel)
    bidsleep_rows, bidsleep_summary = normalize_bidsleep_night(args.bidsleep_night)
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "sleep_samples.jsonl"
    with output_file.open("w", encoding="utf-8") as stream:
        for row in sleep_rows + bidsleep_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"purpose": "engineering ingestion tests only; not user data or clinical validation",
                "records": len(sleep_rows) + len(bidsleep_rows), "output": output_file.name,
                "datasets": [sleep_summary, bidsleep_summary]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": manifest["records"], "output": str(output_file),
                      "sleepaccel_files": len(sleep_summary["files"]),
                      "bidsleep_files": len(bidsleep_summary["files"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
