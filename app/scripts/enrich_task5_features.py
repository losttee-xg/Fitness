from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


METRIC_CATALOG = {
    "HRV_RMSSD": {"family": "heart_rate_variability", "unit": "ms", "meaning": "Vendor-exported RMSSD value; source semantics and device method require confirmation.", "cautions": ["Do not equate with SDNN.", "Not clinical validation."]},
    "HEART_RATE_NREM": {"family": "heart_rate", "unit": "bpm", "meaning": "Heart rate during NREM sleep as provided by the source export.", "cautions": ["Not resting heart rate."]},
    "RHR": {"family": "heart_rate", "unit": "bpm", "meaning": "Source resting-heart-rate field.", "cautions": ["Vendor/device definition should be checked."]},
    "SLEEP_MINUTES_ASLEEP": {"family": "sleep", "unit": "min", "meaning": "Source minutes-asleep field.", "cautions": ["Not PSG-derived sleep staging."]},
    "SLEEP_EFFICIENCY": {"family": "sleep", "unit": "%", "meaning": "Source sleep-efficiency field.", "cautions": ["Vendor semantics should be checked."]},
    "SLEEP_MINUTES_AWAKE": {"family": "sleep", "unit": "min", "meaning": "Source minutes-awake field.", "cautions": []},
    "SLEEP_LATENCY": {"family": "sleep", "unit": "min", "meaning": "Source minutes-to-fall-asleep field.", "cautions": []},
    "SLEEP_AFTER_WAKE": {"family": "sleep", "unit": "min", "meaning": "Source minutes-after-wakeup field.", "cautions": []},
    "STEPS": {"family": "activity", "unit": "steps", "meaning": "Source daily step count.", "cautions": []},
    "CALORIES_ESTIMATE": {"family": "activity", "unit": "kcal", "meaning": "Source estimated calories.", "cautions": ["Estimate, not direct calorimetry."]},
    "DISTANCE": {"family": "activity", "unit": "source_unit_unverified", "meaning": "Source distance field.", "cautions": ["Unit must be verified before quantitative use."]},
    "SPO2": {"family": "oxygen_saturation", "unit": "%", "meaning": "Source oxygen saturation field.", "cautions": ["Consumer wearable measurement; not a diagnosis."]},
    "RESPIRATORY_RATE_SLEEP": {"family": "respiratory_rate", "unit": "breaths/min", "meaning": "Source full-sleep breathing-rate field.", "cautions": []},
    "VENDOR_STRESS_SCORE": {"family": "vendor_score", "unit": "vendor_score_unverified", "meaning": "Vendor-provided stress score.", "cautions": ["Scale and direction need verification; not a validated stress label."]},
    "VENDOR_SLEEP_SCORE": {"family": "vendor_score", "unit": "vendor_score_unverified", "meaning": "Vendor-provided sleep score percentage.", "cautions": ["Vendor semantics require verification."]},
    "SLEEP_DURATION": {"family": "sleep", "unit": "ms_source_unit_unverified", "meaning": "Source sleep-duration field retained in its original value/unit.", "cautions": ["Do not use until unit is confirmed."]},
    "NIGHTLY_TEMPERATURE": {"family": "temperature", "unit": "source_unit_unverified", "meaning": "Source nightly temperature field.", "cautions": ["Unit and reference baseline require verification."]},
}

TARGETS = {
    "STAI_STRESS_SCORE": {"units": {"score_points"}, "source_prefix": "questionnaire_score"},
    "PANAS_POSITIVE_AFFECT": {"units": {"score_points"}, "source_prefix": "questionnaire_score"},
    "PANAS_NEGATIVE_AFFECT": {"units": {"score_points"}, "source_prefix": "questionnaire_score"},
}
FEATURE_METRICS = tuple(metric for metric, spec in METRIC_CATALOG.items()
                       if "unverified" not in spec["unit"] and spec["family"] != "vendor_score")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_survey_features(observations_path: Path, labels_path: Path, output_path: Path,
                          lookback_days: int = 7, min_observed_days: int = 3) -> dict[str, Any]:
    if lookback_days < 1 or min_observed_days < 1 or min_observed_days > lookback_days:
        raise ValueError("Require lookback_days >= min_observed_days >= 1")

    series: dict[tuple[str, str], dict[date, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in load_csv(observations_path):
        if row.get("dataset_id") != "lifesnaps_rais" or row.get("synthetic", "false").casefold() == "true":
            continue
        metric = row.get("metric_code", "")
        if metric not in FEATURE_METRICS:
            continue
        try:
            day, value = date.fromisoformat(row["date"][:10]), float(row["value"])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(value):
            series[(row["participant_id"], metric)][day].append(value)

    output: list[dict[str, Any]] = []
    for row in load_csv(labels_path):
        target = row.get("label_name", "")
        spec = TARGETS.get(target)
        if row.get("dataset_id") != "lifesnaps_rais" or not spec:
            continue
        if row.get("synthetic", "false").casefold() == "true" or not row.get("label_status", "").startswith(spec["source_prefix"]):
            continue
        if row.get("label_unit") not in spec["units"]:
            continue
        try:
            target_day = date.fromisoformat(row["date"][:10])
            target_value = float(row["label_value"])
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isfinite(target_value):
            continue

        record: dict[str, Any] = {
            "dataset_id": "lifesnaps_rais", "participant_id": row["participant_id"],
            "target_date": target_day.isoformat(), "target_name": target,
            "target_value": target_value, "target_unit": row["label_unit"],
            "target_source": row["label_source"], "target_source_row": row["source_row"],
            "feature_window_start": (target_day - timedelta(days=lookback_days)).isoformat(),
            "feature_window_end_exclusive": target_day.isoformat(),
            "lookback_days": lookback_days,
            "temporal_rule": "feature_date < target_date; no same-day or future observations",
            "synthetic": False, "clinical_validation": False,
        }
        total_days = 0
        for metric in FEATURE_METRICS:
            prefix = metric.lower()
            available = series.get((row["participant_id"], metric), {})
            window = [value for day in sorted(available)
                      if target_day - timedelta(days=lookback_days) <= day < target_day
                      for value in available[day]]
            daily = [statistics.median(available[day]) for day in sorted(available)
                     if target_day - timedelta(days=lookback_days) <= day < target_day]
            count_days = len(daily)
            total_days += count_days
            record[f"{prefix}_n_days"] = count_days
            record[f"{prefix}_coverage"] = count_days / lookback_days
            record[f"{prefix}_mean"] = statistics.mean(window) if window else None
            record[f"{prefix}_std"] = statistics.pstdev(window) if len(window) > 1 else (0.0 if window else None)
            record[f"{prefix}_last"] = daily[-1] if daily else None
            record[f"{prefix}_delta_first_last"] = daily[-1] - daily[0] if len(daily) > 1 else None
        record["feature_days_total"] = total_days
        record["eligible_for_modeling"] = total_days >= min_observed_days
        record["eligibility_rule"] = f"at least {min_observed_days} source observations across metrics in prior {lookback_days} calendar days"
        output.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0]) if output else ["dataset_id", "participant_id", "target_date", "target_name", "target_value"]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return {"output": str(output_path), "rows": len(output),
            "eligible_rows": sum(bool(row["eligible_for_modeling"]) for row in output),
            "targets": dict(__import__("collections").Counter(row["target_name"] for row in output)),
            "participants": len({row["participant_id"] for row in output}),
            "lookback_days": lookback_days, "min_observed_days": min_observed_days,
            "rule": "strictly prior dates only; no LLM-generated values or labels"}


def write_catalog(path: Path) -> None:
    catalog = {"catalog_version": "1.0", "created_by": "deterministic_project_metadata; not LLM-generated",
               "metrics": METRIC_CATALOG,
               "feature_transformations": {
                   "window": "per participant, [target_date-lookback_days, target_date)",
                   "aggregations": ["mean", "population_std", "last_observed", "first_to_last_delta", "observed_day_count", "observed_day_coverage"],
                   "label_join": "labels remain separate in source; one output row per eligible questionnaire score",
                   "missing_data": "preserve missingness as blank feature values; no imputation",
                   "synthetic_data": "excluded"
               },
               "warning": "Descriptions are engineering metadata; review against source codebook before scientific interpretation."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create leakage-aware LifeSnaps features for survey research.")
    parser.add_argument("--observations", type=Path, default=Path("dataset/harmonized/observations.csv"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/harmonized/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/harmonized/enriched"))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--min-observed-days", type=int, default=3)
    args = parser.parse_args()
    report = build_survey_features(args.observations, args.labels,
                                   args.output_dir / "lifesnaps_survey_features.csv",
                                   args.lookback_days, args.min_observed_days)
    write_catalog(args.output_dir / "feature_catalog.json")
    (args.output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
