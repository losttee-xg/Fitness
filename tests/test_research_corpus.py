from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_task1_synthetic_cases import build_cases
from scripts.normalize_sleep_samples import normalize_bidsleep_night, normalize_sleepaccel


def test_synthetic_corpus_is_explicitly_marked_and_covers_edge_cases():
    cases = build_cases()
    assert all(case.get("synthetic") is True for case in cases if "sequence" not in case and "records" not in case)
    ids = {case["case_id"] for case in cases}
    assert {"invalid-unit", "future-timestamp", "hrv-variant-missing", "duplicate-observation-pair",
            "conflict-reference-wearable", "baseline-distinct-day-coverage"} <= ids
    assert all("physiological ground truth" in case.get("purpose", "") or "sequence" in case or "records" in case for case in cases)


def test_sleepaccel_mapping_keeps_relative_time_and_provenance(tmp_path: Path):
    (tmp_path / "1066528_heartrate.txt").write_text("10 72\n20 74\n", encoding="utf-8")
    (tmp_path / "1066528_labeled_sleep.txt").write_text("0 0\n30 4\n", encoding="utf-8")
    rows, summary = normalize_sleepaccel(tmp_path)
    heart = next(row for row in rows if row["source_metric"] == "heart_rate")
    assert heart["metric_code"] == "RS-01"
    assert heart["unit"] == "bpm"
    assert heart["observed_at_utc"] is None
    assert "relative_seconds_from_psg_start_not_utc" in heart["quality_flags"]
    stage = next(row for row in rows if row["source_metric"] == "psg_sleep_stage")
    assert stage["value"] == "WAKE"
    assert stage["metric_code"] is None
    assert summary["files"]["1066528_heartrate.txt"]["parsed_rows"] == 2


def test_bidsleep_unheaded_hr_and_unix_time_are_normalized(tmp_path: Path):
    (tmp_path / "hr.csv").write_text("1638504593.0,69.0\n1638504598.0,68.0\n", encoding="utf-8")
    (tmp_path / "motion.csv").write_text("Timestamp,x,y,z\n1638504593.0,-.1,.2,.9\n", encoding="utf-8")
    (tmp_path / "labels.mat").write_bytes(b"MATLAB 5.0 MAT-file")
    rows, summary = normalize_bidsleep_night(tmp_path)
    heart = next(row for row in rows if row["metric_code"] == "RS-01")
    assert heart["value"] == 69
    assert heart["unit"] == "bpm"
    assert heart["observed_at_utc"].endswith("+00:00")
    motion = next(row for row in rows if row["source_metric"] == "acceleration_xyz")
    assert len(motion["value"]) == 3
    assert summary["files"]["labels.mat"]["status"] == "preserved_unparsed_mat_file"


def test_synthetic_jsonl_round_trips(tmp_path: Path):
    path = tmp_path / "task1_cases.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for case in build_cases():
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    decoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(decoded) >= 18
    assert all(item.get("synthetic") is True for item in decoded)
