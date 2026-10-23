from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


ACTIVITY_NAMES = {
    0: "other", 1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work",
    11: "car_driving", 12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry", 19: "house_cleaning",
    20: "playing_soccer", 24: "rope_jumping",
}

OBS_FIELDS = ["dataset_id", "dataset_version", "participant_id", "session_id", "date",
              "window_start_utc", "window_end_utc", "time_scope", "metric_code", "metric_variant",
              "value", "unit", "raw_value", "raw_unit", "source_metric", "source_file", "source_row_start", "source_row_end",
              "aggregation", "quality_flags", "synthetic", "clinical_validation"]
LABEL_FIELDS = ["dataset_id", "dataset_version", "participant_id", "session_id", "date",
                "label_name", "label_value", "label_unit", "label_source", "label_status",
                "source_file", "source_row", "quality_flags", "synthetic"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def observation(dataset: str, version: str, participant: str, session: str, day: str | None,
                metric: str, value: float, unit: str, source_metric: str, source_file: str,
                scope: str, start: str | None = None, end: str | None = None,
                row_start: int | None = None, row_end: int | None = None,
                aggregation: str = "source_value", flags: list[str] | None = None) -> dict[str, Any]:
    return {"dataset_id": dataset, "dataset_version": version, "participant_id": participant,
            "session_id": session, "date": day, "window_start_utc": start, "window_end_utc": end,
            "time_scope": scope, "metric_code": metric, "metric_variant": None, "value": value,
            "unit": unit, "raw_value": value, "raw_unit": unit,
            "source_metric": source_metric, "source_file": source_file,
            "source_row_start": row_start, "source_row_end": row_end, "aggregation": aggregation,
            "quality_flags": json.dumps(flags or [], ensure_ascii=False), "synthetic": False,
            "clinical_validation": False}


def label(dataset: str, version: str, participant: str, session: str, day: str | None,
          name: str, value: Any, unit: str | None, source: str, status: str,
          file: str, row: int | None = None, flags: list[str] | None = None) -> dict[str, Any]:
    return {"dataset_id": dataset, "dataset_version": version, "participant_id": participant,
            "session_id": session, "date": day, "label_name": name, "label_value": value,
            "label_unit": unit, "label_source": source, "label_status": status, "source_file": file,
            "source_row": row, "quality_flags": json.dumps(flags or [], ensure_ascii=False), "synthetic": False}


def rais_daily(root: Path, observations: list[dict[str, Any]], labels: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    path = root / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv"
    if not path.exists():
        inventory["datasets"].append({"dataset_id": "lifesnaps_rais", "status": "missing_daily_csv"})
        return
    file_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"nonmissing": 0, "invalid": 0})
    people: set[str] = set()
    days: set[str] = set()
    metric_map = {
        "rmssd": ("HRV_RMSSD", "ms", "RMSSD", "source_value"),
        "nremhr": ("HEART_RATE_NREM", "bpm", "NREM heart rate", "source_value"),
        "resting_hr": ("RHR", "bpm", "resting_hr", "source_value"),
        "minutesAsleep": ("SLEEP_MINUTES_ASLEEP", "min", "minutesAsleep", "source_value"),
        "sleep_efficiency": ("SLEEP_EFFICIENCY", "%", "sleep_efficiency", "source_value"),
        "minutesAwake": ("SLEEP_MINUTES_AWAKE", "min", "minutesAwake", "source_value"),
        "minutesToFallAsleep": ("SLEEP_LATENCY", "min", "minutesToFallAsleep", "source_value"),
        "minutesAfterWakeup": ("SLEEP_AFTER_WAKE", "min", "minutesAfterWakeup", "source_value"),
        "steps": ("STEPS", "steps", "steps", "source_value"),
        "calories": ("CALORIES_ESTIMATE", "kcal", "calories", "source_value"),
        "distance": ("DISTANCE", "source_unit_unverified", "distance", "source_value"),
        "spo2": ("SPO2", "%", "spo2", "source_value"),
        "full_sleep_breathing_rate": ("RESPIRATORY_RATE_SLEEP", "breaths/min", "full_sleep_breathing_rate", "source_value"),
        "stress_score": ("VENDOR_STRESS_SCORE", "vendor_score_unverified", "stress_score", "source_value"),
        "sleep_points_percentage": ("VENDOR_SLEEP_SCORE", "vendor_score_unverified", "sleep_points_percentage", "source_value"),
        "sleep_duration": ("SLEEP_DURATION", "ms_source_unit_unverified", "sleep_duration", "source_value"),
        "nightly_temperature": ("NIGHTLY_TEMPERATURE", "source_unit_unverified", "nightly_temperature", "source_value"),
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        daily_rows = 0
        for row_number, row in enumerate(reader, start=2):
            daily_rows += 1
            participant = str(row.get("id", "")).strip()
            day = str(row.get("date", ""))[:10]
            if not participant or not day:
                continue
            people.add(participant)
            days.add(day)
            for source_name, (metric, unit, source_metric, aggregation) in metric_map.items():
                raw = row.get(source_name)
                value = safe_float(raw)
                if value is None:
                    file_stats[source_name]["invalid"] += 1
                    continue
                flags = ["unit_or_vendor_semantics_require_review"] if "unverified" in unit or "source_unit" in unit else []
                if source_name == "sleep_duration":
                    flags.append("raw_value_appears_milliseconds_preserved_no_conversion")
                observations.append(observation("lifesnaps_rais", "zenodo_v4_csv", participant, "",
                                                day, metric, value, unit, source_metric, str(path),
                                                "participant_day", row_start=row_number, row_end=row_number,
                                                aggregation=aggregation, flags=flags))
                file_stats[source_name]["nonmissing"] += 1
            for source_name in ("TIRED", "RESTED/RELAXED", "TENSE/ANXIOUS", "HAPPY", "SAD", "ALERT", "NEUTRAL"):
                value = safe_float(row.get(source_name))
                if value is None:
                    continue
                labels.append(label("lifesnaps_rais", "zenodo_v4_csv", participant, "", day,
                                    f"SEMA_{source_name.replace('/', '_').replace(' ', '_')}", int(value),
                                    "binary_ema_flag", "SEMA daily Fitbit export", "self_report_flag_semantics_not_fatigue_scale",
                                    str(path), row_number, ["do_not_treat_as_clinical_ground_truth"]))
    inventory["datasets"].append({"dataset_id": "lifesnaps_rais", "version": "Zenodo record 7229547 / CSV export",
                                  "status": "harmonized_daily_csv", "source_file": str(path),
                                  "source_sha256": sha256(path), "rows": daily_rows,
                                  "participants": len(people), "calendar_dates": len(days),
                                  "date_min": min(days) if days else None, "date_max": max(days) if days else None,
                                  "metric_counts": dict(file_stats), "label_note": "SEMA binary flags are not validated fatigue/readiness scales.",
                                  "license_note": "Verify the Zenodo record license and usage conditions before commercial model training."})


def rais_surveys(root: Path, labels: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    survey_dir = root / "scored_surveys"
    survey_specs = {
        "stai.csv": ("STAI_STRESS_SCORE", "score_points", "stress_anxiety_questionnaire"),
        "panas.csv": ("PANAS", "score_points", "affect_questionnaire"),
    }
    survey_stats: dict[str, dict[str, Any]] = {}
    for filename, (label_name, unit, kind) in survey_specs.items():
        path = survey_dir / filename
        if not path.exists():
            survey_stats[filename] = {"status": "missing"}
            continue
        people: set[str] = set()
        count = 0
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row_number, row in enumerate(csv.DictReader(stream), start=2):
                participant = str(row.get("user_id", "")).strip()
                day = str(row.get("submitdate", "")).strip()[:10] or None
                if not participant or not day:
                    continue
                people.add(participant)
                if filename == "stai.csv":
                    value = safe_float(row.get("stai_stress"))
                    if value is not None:
                        labels.append(label("lifesnaps_rais", "zenodo_v4_scored_surveys", participant, "", day,
                                            label_name, value, unit, "scored_surveys/stai.csv",
                                            "questionnaire_score_scored_not_clinical_diagnosis", str(path), row_number,
                                            ["self_report", "do_not_infer_clinical_diagnosis"]))
                        category = str(row.get("stai_stress_category", "")).strip()
                        if category:
                            labels.append(label("lifesnaps_rais", "zenodo_v4_scored_surveys", participant, "", day,
                                                "STAI_STRESS_CATEGORY", category, None, "scored_surveys/stai.csv",
                                                "source_category_unverified", str(path), row_number,
                                                ["self_report", "category_thresholds_not_independently_verified"]))
                        count += 1
                else:
                    positive = safe_float(row.get("positive_affect_score"))
                    negative = safe_float(row.get("negative_affect_score"))
                    for metric, value in (("PANAS_POSITIVE_AFFECT", positive), ("PANAS_NEGATIVE_AFFECT", negative)):
                        if value is not None:
                            labels.append(label("lifesnaps_rais", "zenodo_v4_scored_surveys", participant, "", day,
                                                metric, value, unit, "scored_surveys/panas.csv",
                                                "questionnaire_score_self_report_not_clinical_diagnosis", str(path), row_number,
                                                ["self_report", "do_not_infer_clinical_diagnosis"]))
                    if positive is not None or negative is not None:
                        count += 1
        survey_stats[filename] = {"status": "included_as_survey_labels", "rows": count,
                                  "participants": len(people), "sha256": sha256(path)}
    inventory["datasets"].append({"dataset_id": "lifesnaps_rais_surveys", "status": "scored_questionnaires_as_labels",
                                  "source": str(survey_dir), "participants": len({
                                      row["participant_id"] for row in labels
                                      if row["dataset_id"] == "lifesnaps_rais" and row["dataset_version"] == "zenodo_v4_scored_surveys"
                                  }), "files": survey_stats,
                                  "limitations": ["STAI/PANAS scores are self-reported questionnaire labels, not clinical diagnoses.",
                                                  "Survey submit dates are retained as reported; temporal joins should use a prespecified window and avoid leakage."]})


def bidsleep_nights(root: Path, observations: list[dict[str, Any]], labels: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    subjects = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Bidslab")) if root.exists() else []
    count_subjects = 0
    count_nights = 0
    complete_motion = 0
    for subject in subjects:
        count_subjects += 1
        for night in sorted(p for p in subject.iterdir() if p.is_dir()):
            count_nights += 1
            session = night.name
            hr_path = night / "hr.csv"
            motion_path = night / "motion.csv"
            mat_path = night / "labels.mat"
            if hr_path.exists():
                values: list[tuple[float, float, int]] = []
                with hr_path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    for row_idx, row in enumerate(reader, start=1):
                        if len(row) < 2:
                            continue
                        ts, hr = safe_float(row[0]), safe_float(row[1])
                        if ts is not None and hr is not None and 25 <= hr <= 240:
                            values.append((ts, hr, row_idx))
                if values:
                    start_ts, end_ts = values[0][0], values[-1][0]
                    start = datetime.fromtimestamp(start_ts, timezone.utc).isoformat()
                    end = datetime.fromtimestamp(end_ts, timezone.utc).isoformat()
                    day = datetime.fromtimestamp(start_ts, timezone.utc).date().isoformat()
                    hrs = [v[1] for v in values]
                    flags = ["recording_night_scope_not_daily_user_metric", "consumer_watch_estimated_hr"]
                    for metric, value, unit, agg in [
                        ("HR_NIGHT_MEAN", statistics.mean(hrs), "bpm", "mean_over_recording"),
                        ("HR_NIGHT_MEDIAN", statistics.median(hrs), "bpm", "median_over_recording"),
                        ("HR_NIGHT_MIN", min(hrs), "bpm", "minimum_over_recording"),
                        ("HR_NIGHT_MAX", max(hrs), "bpm", "maximum_over_recording"),
                        ("HR_SAMPLE_COUNT", float(len(hrs)), "samples", "count"),
                        ("HR_RECORDING_DURATION", end_ts-start_ts, "seconds", "last_timestamp_minus_first"),
                    ]:
                        observations.append(observation("bidsleep", "1.0.1", subject.name, session, day,
                                                        metric, value, unit, "hr.csv", str(hr_path), "recording_night",
                                                        start, end, values[0][2], values[-1][2], agg, flags))
            if mat_path.exists():
                labels.append(label("bidsleep", "1.0.1", subject.name, session, None,
                                    "sleep_stage_reference", None, "30_second_epoch", "labels.mat",
                                    "available_but_unparsed", str(mat_path), None,
                                    ["no stage value emitted", "MATLAB file not parsed; do not use as label until verified"]))
            if motion_path.exists() and not (night / "motion.csv.aria2").exists():
                complete_motion += 1
    inventory["datasets"].append({"dataset_id": "bidsleep", "version": "1.0.1",
                                  "status": "hr_night_summaries_labels_unparsed_motion_partial",
                                  "root": str(root), "participants": count_subjects, "nights": count_nights,
                                  "complete_motion_nights": complete_motion,
                                  "hr_files_present": sum(1 for p in root.rglob("hr.csv")),
                                  "label_files_present": sum(1 for p in root.rglob("labels.mat")),
                                  "motion_files_present": sum(1 for p in root.rglob("motion.csv")),
                                  "motion_partial_controls": sum(1 for p in root.rglob("motion.csv.aria2")),
                                  "license": "ODC-By 1.0 database license; verify content/data rights and cite PhysioNet",
                                  "limitations": ["Sleep-stage labels are present but not parsed into label values.",
                                                  "motion.csv partial transfers are excluded from all metrics.",
                                                  "HR per night is not a clinical resting heart rate and is not HRV."]})


def pamap_segments(root: Path, observations: list[dict[str, Any]], labels: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    files = sorted(root.glob("subject*.dat")) if root.exists() else []
    participants: set[str] = set()
    segments = 0
    parsed_rows = 0
    for path in files:
        participant = path.stem.replace("subject", "subject")
        participants.add(participant)
        active_id: int | None = None
        start_time: float | None = None
        last_time: float | None = None
        hrs: list[float] = []
        acc_mags: list[float] = []
        row_start = 0

        def flush() -> None:
            nonlocal segments, active_id, start_time, last_time, hrs, acc_mags, row_start
            if active_id is None or start_time is None or last_time is None:
                return
            session = f"protocol_activity_{segments + 1:06d}"
            duration = max(0.0, last_time-start_time)
            observations.append(observation("pamap2", "UCI_231", participant, session, None,
                                            "ACTIVITY_DURATION", duration, "seconds", "timestamp", str(path),
                                            "relative_activity_segment", row_start=row_start,
                                            aggregation="segment_duration",
                                            flags=["relative_elapsed_time_not_utc", "research_protocol_not_free_living"]))
            if hrs:
                observations.append(observation("pamap2", "UCI_231", participant, session, None,
                                                "HR_ACTIVITY_MEAN", statistics.mean(hrs), "bpm", "heart_rate", str(path),
                                                "relative_activity_segment", row_start=row_start,
                                                aggregation="mean_over_segment",
                                                flags=["relative_elapsed_time_not_utc", "not_resting_heart_rate"]))
            if acc_mags:
                observations.append(observation("pamap2", "UCI_231", participant, session, None,
                                                "ACC_MAGNITUDE_MEAN", statistics.mean(acc_mags), "m/s2", "hand_acc_xyz", str(path),
                                                "relative_activity_segment", row_start=row_start,
                                                aggregation="mean_vector_magnitude_over_segment",
                                                flags=["relative_elapsed_time_not_utc", "hand_sensor_only"]))
            labels.append(label("pamap2", "UCI_231", participant, session, None, "activity_id",
                                active_id, "pamap_activity_code", "PAMAP2 Protocol activity ID",
                                "source_protocol_label", str(path), row_start,
                                ["activity_name_mapping_requires_source_readme_review"]))
            segments += 1
            active_id, start_time, last_time, hrs, acc_mags, row_start = None, None, None, [], [], 0

        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.split()
                if len(fields) < 6:
                    continue
                timestamp = safe_float(fields[0])
                activity_value = safe_float(fields[1])
                hr = safe_float(fields[2])
                if timestamp is None or activity_value is None:
                    continue
                parsed_rows += 1
                activity_id = int(activity_value)
                if active_id is None:
                    active_id, start_time, row_start = activity_id, timestamp, line_number
                elif activity_id != active_id:
                    flush()
                    active_id, start_time, row_start = activity_id, timestamp, line_number
                last_time = timestamp
                if hr is not None and hr > 0:
                    hrs.append(hr)
                axes = [safe_float(x) for x in fields[4:7]]
                if len(axes) == 3 and all(v is not None for v in axes):
                    acc_mags.append(math.sqrt(sum(v*v for v in axes if v is not None)))
        flush()
    inventory["datasets"].append({"dataset_id": "pamap2", "version": "UCI_231",
                                  "status": "protocol_segments_hr_and_hand_acc_summary",
                                  "root": str(root), "protocol_files": len(files),
                                  "participants": len(participants), "parsed_rows": parsed_rows,
                                  "activity_segments": segments,
                                  "license": "CC BY 4.0", "limitations": [
                                      "Activity label codes are retained as raw source labels; mapping names must be checked against UCI README.",
                                      "Current segment summaries use HR and hand acceleration only; device-specific IMU fields are not all included.",
                                      "Elapsed timestamps are not real calendar dates; cannot enter daily forecast panel."]})


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source-separated harmonized tables for all local research datasets.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {"schema_version": "task5-harmonized-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "purpose": "Research harmonization; not clinical validation or user records.", "datasets": []}

    rais = args.dataset_root / "rais_anonymized"
    rais_daily(rais, observations, labels, inventory)
    rais_surveys(rais, labels, inventory)
    bidsleep_nights(args.dataset_root / "bidsleep-dataset", observations, labels, inventory)
    pamap_segments(Path("data/research/task5/extracted/pamap2/PAMAP2_Dataset/Protocol"), observations, labels, inventory)

    ppg_root = Path("data/research/task5/extracted/ppg_dalia_metadata_only/PPG_FieldStudy")
    ppg_status = "archive_only_metadata_not_extracted"
    ppg_participants = 0
    if ppg_root.exists():
        ppg_participants = sum(1 for p in ppg_root.iterdir() if p.is_dir() and p.name.startswith("S"))
        ppg_status = "metadata_available_sensor_payload_not_loaded"
    inventory["datasets"].append({"dataset_id": "ppg_dalia", "status": ppg_status,
                                  "participants_metadata": ppg_participants,
                                  "archive": str(args.dataset_root / "ppg+dalia" / "data.zip"),
                                  "license": "CC BY 4.0 per UCI dataset record; retain attribution",
                                  "limitations": ["Sensor files are pickle/HDF5/nested ZIP; no sensor payload has been loaded.",
                                                  "Never deserialize pickle unless using a reviewed safe conversion process."]})

    # Summary statistics are reference metadata, not participant observations.
    datasheet = args.dataset_root / "Datasheet1.csv"
    if datasheet.exists():
        rows = list(csv.DictReader(datasheet.open(encoding="utf-8-sig", newline="")))
        inventory["datasets"].append({"dataset_id": "datasheet1_summary_only", "status": "summary_statistics_not_training_rows",
                                      "rows": len(rows), "categories": dict(__import__("collections").Counter(r.get("Category", "") for r in rows)),
                                      "source_file": str(datasheet),
                                      "limitations": ["No participant/date-level rows; do not add as training samples.",
                                                      "Target VAS semantics and source/license not identified in this folder."]})

    write_csv(args.output / "observations.csv", observations, OBS_FIELDS)
    write_csv(args.output / "labels.csv", labels, LABEL_FIELDS)
    inventory["output_counts"] = {"observations": len(observations), "labels": len(labels),
                                   "participants_by_dataset": {item["dataset_id"]: item.get("participants", item.get("participants_metadata"))
                                                                for item in inventory["datasets"]}}
    (args.output / "dataset_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **inventory["output_counts"],
                      "dataset_statuses": [{"dataset_id": d["dataset_id"], "status": d["status"]} for d in inventory["datasets"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
