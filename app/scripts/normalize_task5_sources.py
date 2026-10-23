from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.normalize_sleep_samples import normalize_sleepaccel


def normalize_sleepaccel_tree(root: Path, output: Path) -> dict[str, Any]:
    participants: dict[Path, list[Path]] = defaultdict(list)
    for path in root.rglob("*_heartrate.txt"):
        participants[path.parent].append(path)
    if not participants:
        raise FileNotFoundError(f"No *_heartrate.txt files found under {root}")

    rows: list[dict[str, Any]] = []
    summaries = []
    for folder in sorted(participants):
        subject_rows, summary = normalize_sleepaccel(folder)
        rows.extend(subject_rows)
        summaries.append({"participant_folder": str(folder), **summary})
    output.mkdir(parents=True, exist_ok=True)
    target = output / "sleepaccel_observations.jsonl"
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_id": "sleepaccel",
        "dataset_version": "1.0.0",
        "purpose": "Engineering normalization only; no clinical validation.",
        "participant_folders": len(summaries),
        "records": len(rows),
        "output": str(target),
        "summaries": summaries,
        "notes": [
            "PSG-relative elapsed seconds are not converted to calendar dates.",
            "PSG stage labels remain reference labels, not consumer wearable estimates.",
            "Do not use this one-night-per-person dataset as a multi-day forecast training set.",
        ],
    }
    (output / "sleepaccel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def normalize_wearable_sessions(root: Path, output: Path) -> dict[str, Any]:
    """Parse documented single-channel E4 CSVs while retaining session time semantics."""
    signal_map = {
        "BVP.csv": ("bvp", None, None),
        "HR.csv": ("heart_rate", "RS-01", "bpm"),
        "IBI.csv": ("ibi", None, "seconds"),
        "EDA.csv": ("eda", None, "microsiemens"),
        "TEMP.csv": ("skin_temperature", None, "celsius"),
    }
    rows: list[dict[str, Any]] = []
    files = []
    for path in sorted(root.rglob("*.csv")):
        kind = signal_map.get(path.name.upper())
        if kind is None:
            continue
        source_metric, metric_code, unit = kind
        session = path.parent.name
        participant = path.parent.parent.name if path.parent.parent != root else path.parent.name
        parsed = 0
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            raw = list(reader)
        if not raw:
            continue
        # E4 exports commonly store start time, sample rate on the first rows, then values.
        if path.name.upper() in {"BVP.csv", "HR.csv", "EDA.csv", "TEMP.csv"} and len(raw) >= 3:
            start = _float(raw[0][0] if raw[0] else None)
            rate = _float(raw[1][0] if raw[1] else None)
            if start is not None and rate is not None and rate > 0:
                for index, item in enumerate(raw[2:]):
                    value = _float(item[0] if item else None)
                    if value is None:
                        continue
                    rows.append(_record(path, participant, session, source_metric, metric_code, value, unit,
                                        start + index / rate, ["unix_seconds_source_time", "research_only"]))
                    parsed += 1
        elif path.name.upper() == "IBI.csv":
            for line, item in enumerate(raw[1:], start=2):
                if len(item) < 2:
                    continue
                offset, ibi = _float(item[0]), _float(item[1])
                if offset is None or ibi is None:
                    continue
                rows.append(_record(path, participant, session, source_metric, metric_code, ibi, unit,
                                    offset, ["session_relative_seconds_not_utc", "research_only"]))
                parsed += 1
        else:
            for line, item in enumerate(raw, start=1):
                offset = _float(item[0] if item else None)
                if offset is None:
                    continue
                rows.append(_record(path, participant, session, source_metric, metric_code, offset, unit,
                                    None, ["protocol_tag_or_unix_time_requires_review", "research_only"]))
                parsed += 1
        files.append({"path": str(path), "parsed_records": parsed})
    output.mkdir(parents=True, exist_ok=True)
    target = output / "wearable_session_observations.jsonl"
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_id": "physionet_wearable_device",
        "purpose": "Protocol/session engineering and exploratory work only; not consumer-health validation.",
        "records": len(rows), "output": str(target), "files": files,
        "notes": [
            "Only known single-channel E4 CSV layouts are parsed.",
            "Dataset-specific labels, stress score files, sampling rates and participant mappings require manual review.",
            "IBI is not converted into HRV here; approved artifact filtering and window semantics are required first.",
            "Research-session timestamps are not treated as user calendar dates.",
        ],
    }
    (output / "wearable_session_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _float(value: Any) -> float | None:
    try:
        import math
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _record(path: Path, participant: str, session: str, metric: str, code: str | None,
            value: float, unit: str | None, source_time: float | None, flags: list[str]) -> dict[str, Any]:
    return {"dataset_id": "physionet_wearable_device", "dataset_version": "1.0.1",
            "participant_id": participant, "session_id": session, "source_file": str(path),
            "source_metric": metric, "metric_code": code, "metric_variant": None,
            "raw_value": value, "raw_unit": unit, "value": value, "unit": unit,
            "source_timestamp": source_time, "observed_at_utc": None,
            "quality_flags": flags}


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize reviewed Task 5 research sources without inventing calendar dates.")
    parser.add_argument("--sleepaccel-root", type=Path)
    parser.add_argument("--wearable-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    if args.sleepaccel_root:
        results.append(normalize_sleepaccel_tree(args.sleepaccel_root, args.output / "sleepaccel"))
    if args.wearable_root:
        results.append(normalize_wearable_sessions(args.wearable_root, args.output / "wearable_device"))
    if not results:
        parser.error("Provide at least one source root")
    print(json.dumps({"sources": [{"dataset_id": item["dataset_id"], "records": item["records"], "output": item["output"]} for item in results]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
