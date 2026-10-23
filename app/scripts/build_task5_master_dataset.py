from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_num(raw: Any) -> float | None:
    try:
        value = float(raw)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_activity_observations(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, date], dict[str, float]], dict[str, Any]]:
    path = root / "day_activity_features.csv"
    dictionary_path = root / "activity_feature_data_dictionary.csv"
    fields, data = read_csv(path)
    _, dictionary_rows = read_csv(dictionary_path)
    descriptions = {(r.get("Column Name") or "").strip().strip('"'): (r.get("Description") or "").strip().strip('"')
                    for r in dictionary_rows}
    observations: list[dict[str, Any]] = []
    wide: dict[tuple[str, date], dict[str, float]] = defaultdict(dict)
    for source_row, row in enumerate(data, start=2):
        participant = row.get("user_id", "").strip()
        try:
            day = date.fromisoformat(row.get("date", "")[:10])
        except ValueError:
            continue
        if not participant:
            continue
        for metric in fields:
            if metric in {"user_id", "date"}:
                continue
            raw = row.get(metric, "")
            value = safe_num(raw)
            if value is None:
                continue
            wide[(participant, day)][metric] = value
            observations.append({"dataset_id": "fatigue_remote_wearable_zenodo_8018238",
                                 "dataset_version": "zenodo_8018238",
                                 "participant_id": participant, "date": day.isoformat(),
                                 "metric_code": metric, "metric_description": descriptions.get(metric, ""),
                                 "value": value, "unit": "not_stated_or_unverified_in_dictionary",
                                 "raw_value": raw, "raw_column": metric,
                                 "source_file": path.name, "source_row": source_row,
                                 "synthetic": False})
    return observations, wide, {"rows": len(data), "observations_nonmissing": len(observations),
                                "participants": len({r.get("user_id", "") for r in data}),
                                "dates": len({r.get("date", "") for r in data}),
                                "source_sha256": sha256(path), "dictionary_sha256": sha256(dictionary_path),
                                "dictionary_columns": len(descriptions)}


def build_survey_labels(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = [
        ("daily_surveys_one_click_responses.csv", "daily_ema", "daily_survey_data_dictionary.csv"),
        ("week1_survey.csv", "weekly_survey_week1", None),
        ("week2_survey.csv", "weekly_survey_week2", None),
        ("week3_survey.csv", "weekly_survey_week3", None),
        ("week4_survey.csv", "weekly_survey_week4", None),
        ("baseline_survey.csv", "baseline_survey", None),
        ("screener_survey.csv", "screener_survey", "screener_data_dictionary.csv"),
    ]
    labels: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    daily_dict: dict[str, dict[str, str]] = {}
    for filename, domain, dictionary_name in specs:
        path = root / filename
        fields, rows = read_csv(path)
        if dictionary_name:
            _, dict_rows = read_csv(root / dictionary_name)
            for d in dict_rows:
                q = (d.get("question_label") or "").strip()
                if q:
                    daily_dict[q] = {"question": (d.get("question_text") or "").strip(),
                                     "answer_encoding": (d.get("answer_encoding") or "").strip(),
                                     "answer_text": (d.get("answer_text") or "").strip()}
        survey_ids: set[str] = set()
        fatigue_labels = 0
        for row_number, row in enumerate(rows, start=2):
            participant = row.get("user_id", "").strip()
            if not participant:
                continue
            survey_ids.add(participant)
            survey_dt_raw = row.get("survey_start_datetime", row.get("one_click_submitted_datetime", ""))
            survey_date = survey_dt_raw[:10] if len(survey_dt_raw) >= 10 else ""
            try:
                survey_day = date.fromisoformat(survey_date)
            except ValueError:
                survey_day = None
            for field in fields:
                if field in {"user_id", "day_occurrence", "one_click_submitted_datetime", "one_click_response",
                             "survey_start_datetime", "survey_end_datetime"}:
                    continue
                raw = (row.get(field) or "").strip()
                if not raw:
                    continue
                fatigue_like = any(token in field.casefold() for token in
                                   ("fatigue", "tired", "exhaust", "energy", "sleep", "rested", "recovery", "soreness"))
                if fatigue_like:
                    fatigue_labels += 1
                label_date = survey_day
                target_period_start = None
                target_period_end = None
                if domain == "daily_ema" and "yesterday" in field.casefold() and survey_day:
                    label_date = survey_day - timedelta(days=1)
                    target_period_start = label_date.isoformat()
                    target_period_end = label_date.isoformat()
                elif domain.startswith("weekly_survey") and survey_day:
                    target_period_start = (survey_day - timedelta(days=7)).isoformat()
                    target_period_end = (survey_day - timedelta(days=1)).isoformat()
                numeric = safe_num(raw)
                dictionary = daily_dict.get(field, {})
                labels.append({"dataset_id": "fatigue_remote_wearable_zenodo_8018238",
                               "dataset_version": "zenodo_8018238", "participant_id": participant,
                               "survey_domain": domain, "survey_date": survey_date,
                               "label_date": label_date.isoformat() if label_date else "",
                               "target_period_start": target_period_start or "",
                               "target_period_end": target_period_end or "",
                               "label_name": field, "question_text": dictionary.get("question", field),
                               "label_raw": raw, "label_numeric_source": numeric,
                               "answer_text": dictionary.get("answer_text", ""),
                               "answer_encoding": dictionary.get("answer_encoding", ""),
                               "label_semantics": "self_report_not_clinical_ground_truth",
                               "fatigue_related_field": fatigue_like,
                               "source_file": filename, "source_row": row_number,
                               "synthetic": False, "clinical_validation": False})
        summary[filename] = {"rows": len(rows), "participants": len(survey_ids),
                             "labels_nonempty": sum(1 for x in labels if x["source_file"] == filename),
                             "fatigue_related_nonempty": fatigue_labels, "sha256": sha256(path)}
    return labels, summary


def build_lifesnaps_surveys(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    survey_root = root.parent / "rais_anonymized" / "scored_surveys"
    specs = {
        "stai.csv": ("lifesnaps_scored_survey", {"stai_stress": "STAI_STRESS_SCORE", "stai_stress_category": "STAI_STRESS_CATEGORY"}),
        "panas.csv": ("lifesnaps_scored_survey", {"positive_affect_score": "PANAS_POSITIVE_AFFECT", "negative_affect_score": "PANAS_NEGATIVE_AFFECT"}),
        "breq.csv": ("lifesnaps_scored_survey", {}),
        "personality.csv": ("lifesnaps_scored_survey", {}),
        "ttm.csv": ("lifesnaps_scored_survey", {}),
    }
    labels: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for filename, (domain, explicit_mapping) in specs.items():
        path = survey_root / filename
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        people: set[str] = set()
        added = 0
        for row_num, row in enumerate(rows, start=2):
            participant = (row.get("user_id") or "").strip()
            try:
                day = date.fromisoformat((row.get("submitdate") or "")[:10])
            except ValueError:
                continue
            if not participant:
                continue
            people.add(participant)
            for field in fields:
                if field in {"", "user_id", "type", "submitdate"}:
                    continue
                raw = (row.get(field) or "").strip()
                if not raw:
                    continue
                mapped = explicit_mapping.get(field, field.upper())
                is_fatigue = any(word in field.casefold() for word in ("fatigue", "tired", "exhaust", "energy", "stress"))
                labels.append({"dataset_id": "lifesnaps_rais", "dataset_version": "zenodo_7229547_scored_surveys",
                               "participant_id": participant, "survey_domain": domain, "survey_date": day.isoformat(),
                               "label_date": day.isoformat(), "target_period_start": day.isoformat(),
                               "target_period_end": day.isoformat(), "label_name": mapped,
                               "question_text": field, "label_raw": raw, "label_numeric_source": safe_num(raw),
                               "answer_text": "", "answer_encoding": "",
                               "label_semantics": "self_report_scored_survey_not_clinical_ground_truth",
                               "fatigue_related_field": is_fatigue, "source_file": filename, "source_row": row_num,
                               "synthetic": False, "clinical_validation": False})
                added += 1
        summary[filename] = {"rows": len(rows), "participants": len(people), "nonempty_responses": added,
                             "sha256": sha256(path)}
    return labels, summary


def build_lifesnaps_daily(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, date], dict[str, float]], dict[str, Any]]:
    path = root.parent / "rais_anonymized" / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv"
    fields, rows = read_csv(path)
    metric_fields = [x for x in fields if x not in {"", "id", "date", "gender", "step_goal_label", "badgeType", "activityType"}]
    wide: dict[tuple[str, date], dict[str, float]] = defaultdict(dict)
    observations = []
    for row_num, row in enumerate(rows, start=2):
        participant = row.get("id", "").strip()
        try:
            day = date.fromisoformat(row.get("date", "")[:10])
        except ValueError:
            continue
        if not participant:
            continue
        for metric in metric_fields:
            value = safe_num(row.get(metric))
            if value is None:
                continue
            wide[(participant, day)][metric] = value
            observations.append({"dataset_id": "lifesnaps_rais", "dataset_version": "zenodo_7229547_csv",
                                 "participant_id": participant, "date": day.isoformat(), "metric_code": metric,
                                 "value": value, "unit": "source_unit_preserved_or_requires_codebook_review",
                                 "raw_value": row.get(metric, ""), "source_file": path.name,
                                 "source_row": row_num, "synthetic": False})
    return observations, wide, {"rows": len(rows), "observations_nonmissing": len(observations),
                                "participants": len({r.get("id", "") for r in rows}),
                                "source_sha256": sha256(path),
                                "limitations": ["Original field names retained; units and Fitbit-specific definitions require source review."]}


def build_fatigue_examples(root: Path, labels: list[dict[str, Any]], activity: dict[tuple[str, date], dict[str, float]],
                           lookback: int, min_sensor_features: int, out_path: Path) -> dict[str, Any]:
    target_names = {"tired_yesterday"}
    survey_by_domain = {"daily_ema"}
    output: list[dict[str, Any]] = []
    for label in labels:
        if label["survey_domain"] not in survey_by_domain or label["label_name"] not in target_names:
            continue
        try:
            target_day = date.fromisoformat(label["label_date"])
        except ValueError:
            continue
        person = label["participant_id"]
        history_dates = [target_day - timedelta(days=i) for i in range(lookback, 0, -1)]
        daily = [activity.get((person, day)) for day in history_dates]
        observed = [row for row in daily if row]
        if not observed:
            continue
        feature_names = sorted({key for row in observed for key in row})
        result = {"dataset_id": label["dataset_id"], "participant_id": person,
                  "target_name": label["label_name"], "target_category": label["label_raw"],
                  "target_date": target_day.isoformat(), "survey_date": label["survey_date"],
                  "target_period_start": label["target_period_start"], "target_period_end": label["target_period_end"],
                  "target_source": label["question_text"], "target_source_file": label["source_file"],
                  "target_source_row": label["source_row"], "feature_window_start": history_dates[0].isoformat(),
                  "feature_window_end_exclusive": target_day.isoformat(), "feature_window_relation": "strictly_precedes_daily_fatigue_target",
                  "lookback_days": lookback, "observed_days": len(observed), "synthetic": False,
                  "clinical_validation": False}
        for metric in feature_names:
            vals = [row[metric] for row in observed if metric in row]
            result[f"feature__{metric}__mean_prior_window"] = statistics.mean(vals)
            result[f"feature__{metric}__n_days"] = len(vals)
        result["observed_sensor_metrics"] = len(feature_names)
        result["eligible_for_modeling"] = len(feature_names) >= min_sensor_features
        output.append(result)
    fields = list(dict.fromkeys(key for row in output for key in row)) if output else ["dataset_id", "participant_id", "target_name", "target_category"]
    write_csv(out_path, output, fields)
    return {"rows": len(output), "eligible_rows": sum(row["eligible_for_modeling"] for row in output),
            "participants": len({row["participant_id"] for row in output}),
            "targets": dict(Counter(row["target_name"] for row in output)),
            "output": str(out_path), "feature_window_rule": "strictly prior 7-day window before target fatigue date"}


def wide_windows_to_labels(labels: list[dict[str, Any]], features_by_dataset: dict[str, dict[tuple[str, date], dict[str, float]]],
                           lookback: int, out_path: Path) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    for label in labels:
        dataset = label["dataset_id"]
        try:
            label_day = date.fromisoformat(label.get("label_date") or label["survey_date"])
        except ValueError:
            continue
        if label["survey_domain"] not in {"daily_ema", "weekly_survey_week1", "weekly_survey_week2",
                                          "weekly_survey_week3", "weekly_survey_week4", "lifesnaps_scored_survey"}:
            continue
        if label["survey_domain"] == "daily_ema" and not any(term in label["label_name"].casefold()
                                                              for term in ("fatigue", "tired", "exhaust")):
            continue
        if label["survey_domain"].startswith("weekly_survey") and not any(t in label["label_name"].casefold()
                                                                           for t in ("fatigue", "tired", "exhaust")):
            continue
        features = features_by_dataset.get(dataset, {})
        participant = label["participant_id"]
        if label["survey_domain"].startswith("weekly_survey"):
            try:
                end_exclusive = date.fromisoformat(label["target_period_start"])
            except ValueError:
                continue
            relation = "strictly_precedes_weekly_target_period"
        else:
            end_exclusive = label_day
            relation = ("strictly_precedes_daily_fatigue_target" if label["survey_domain"] == "daily_ema"
                        else "strictly_precedes_survey_date_association_only")
        window_days = [end_exclusive - timedelta(days=i) for i in range(lookback, 0, -1)]
        day_rows = [features.get((participant, day)) for day in window_days]
        observed = [r for r in day_rows if r]
        if not observed:
            continue
        feature_names = sorted({key for r in observed for key in r})
        row: dict[str, Any] = {
            "dataset_id": dataset, "participant_id": participant, "survey_domain": label["survey_domain"],
            "survey_date": label["survey_date"], "target_date": label["label_date"],
            "target_name": label["label_name"], "target_category": label["label_raw"],
            "target_numeric_source": label["label_numeric_source"], "target_period_start": label["target_period_start"],
            "target_period_end": label["target_period_end"], "label_semantics": label["label_semantics"],
            "source_file": label["source_file"], "source_row": label["source_row"],
            "feature_window_start": window_days[0].isoformat(), "feature_window_end_exclusive": end_exclusive.isoformat(),
            "lookback_days": lookback, "observed_days": len(observed), "synthetic": False,
            "feature_window_relation": relation,
        }
        for feature in feature_names:
            vals = [r[feature] for r in observed if feature in r]
            row[f"feature__{feature}__mean_prior_window"] = statistics.mean(vals) if vals else None
            row[f"feature__{feature}__n_days"] = len(vals)
        row["eligible_for_modeling"] = len(feature_names) > 0 and len(observed) >= min(2, lookback)
        output.append(row)
    fields = list(dict.fromkeys(k for row in output for k in row)) if output else ["dataset_id", "participant_id", "target_name", "target_category"]
    write_csv(out_path, output, fields)
    return {"rows": len(output), "eligible_rows": sum(bool(r["eligible_for_modeling"]) for r in output),
            "participants": len({r["participant_id"] for r in output}),
            "targets": dict(Counter(r["target_name"] for r in output)), "output": str(out_path),
            "leakage_rule": "feature window ends strictly before label date; raw original target category retained"}


def build_recovery_proxy(features: dict[tuple[str, date], dict[str, float]], dataset_id: str,
                         out_path: Path, min_baseline_days: int = 7) -> dict[str, Any]:
    """Engineering-only relative wellness composite; deliberately not a target label."""
    metric_candidates = {
        "hrv": ["rmssd", "HRV_RMSSD", "hrv_rmssd"],
        "rhr": ["resting_hr", "RHR", "heart_rate__resting_heart_rate"],
        "sleep": ["minutesAsleep", "SLEEP_MINUTES_ASLEEP", "sleep__total_asleep_minutes"],
    }
    by_person: dict[str, list[tuple[date, dict[str, float]]]] = defaultdict(list)
    for (person, day), vals in features.items():
        by_person[person].append((day, vals))
    output = []
    for person, series in by_person.items():
        series.sort()
        for i, (day, values) in enumerate(series):
            prior = series[max(0, i-min_baseline_days):i]
            if len(prior) < min_baseline_days:
                continue
            component_values: dict[str, float] = {}
            for component, aliases in metric_candidates.items():
                current = next((values[a] for a in aliases if a in values), None)
                baseline = [next((v[a] for a in aliases if a in v), None) for _, v in prior]
                baseline = [v for v in baseline if v is not None]
                if current is None or len(baseline) < min_baseline_days:
                    continue
                center = statistics.median(baseline)
                mad = statistics.median(abs(v-center) for v in baseline)
                scale = 1.4826 * mad
                if scale <= 1e-12:
                    continue
                z = max(-3.0, min(3.0, (current-center)/scale))
                component_values[component] = -z if component == "rhr" else z
            if len(component_values) < 2:
                continue
            score = 50.0 + 50.0 * statistics.mean(component_values.values()) / 3.0
            output.append({"dataset_id": dataset_id, "participant_id": person, "date": day.isoformat(),
                           "recovery_like_relative_composite": round(max(0.0, min(100.0, score)), 3),
                           "components_used": json.dumps(component_values, ensure_ascii=False),
                           "baseline_days": len(prior), "derivation": "unvalidated clipped robust deviation composite; HRV/sleep oriented upward, RHR oriented downward",
                           "label_type": "synthetic_derived_engineering_proxy_not_observed_outcome",
                           "synthetic": True, "clinical_validation": False,
                           "allowed_use": "UI/rule-engine smoke tests only; exclude from all training/evaluation"})
    write_csv(out_path, output, list(output[0]) if output else ["dataset_id", "participant_id", "date"])
    return {"rows": len(output), "participants": len({r["participant_id"] for r in output}),
            "status": "SYNTHETIC_DERIVED_PROXY_NOT_TRAINING_LABEL", "output": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create source-aware expanded research bundle; synthetic proxy is isolated from labels.")
    parser.add_argument("--fatigue-root", type=Path, default=Path("dataset/fatigue_zenodo_8018238"))
    parser.add_argument("--harmonized-root", type=Path, default=Path("dataset/harmonized"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/master"))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--make-engineering-proxy", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fatigue_observations, fatigue_features, fatigue_stats = build_activity_observations(args.fatigue_root)
    fatigue_labels, fatigue_survey_stats = build_survey_labels(args.fatigue_root)
    life_observations, life_features, life_stats = build_lifesnaps_daily(args.harmonized_root)
    lifesnaps_labels, lifesnaps_survey_stats = build_lifesnaps_surveys(args.harmonized_root)
    all_labels = fatigue_labels + lifesnaps_labels

    obs_fields = ["dataset_id", "dataset_version", "participant_id", "date", "metric_code", "metric_description",
                  "value", "unit", "raw_value", "raw_column", "source_file", "source_row", "synthetic"]
    label_fields = ["dataset_id", "dataset_version", "participant_id", "survey_domain", "survey_date", "label_date",
                    "target_period_start", "target_period_end", "label_name", "question_text", "label_raw",
                    "label_numeric_source", "answer_text", "answer_encoding", "label_semantics",
                    "fatigue_related_field", "source_file", "source_row", "synthetic", "clinical_validation",
                    "feature_window_relation"]
    write_csv(args.output / "observations_long.csv", fatigue_observations + life_observations, obs_fields)
    write_csv(args.output / "labels_long.csv", all_labels, label_fields)
    joined_fatigue = build_fatigue_examples(root=args.fatigue_root, labels=fatigue_labels,
                                            activity=fatigue_features, lookback=args.lookback_days,
                                            min_sensor_features=5,
                                            out_path=args.output / "fatigue_daily_training_examples.csv")
    lifesnaps_daily_labels = [row for row in lifesnaps_labels if row["label_name"] in {
        "STAI_STRESS_SCORE", "PANAS_POSITIVE_AFFECT", "PANAS_NEGATIVE_AFFECT"}]
    joined_lifesnaps = wide_windows_to_labels(lifesnaps_daily_labels, {"lifesnaps_rais": life_features},
                                              args.lookback_days, args.output / "lifesnaps_survey_feature_windows.csv")

    proxies = {}
    if args.make_engineering_proxy:
        proxies["fatigue_remote_wearable_zenodo_8018238"] = build_recovery_proxy(
            fatigue_features, "fatigue_remote_wearable_zenodo_8018238",
            args.output / "synthetic_engineering_only" / "fatigue_cohort_recovery_like_proxy.csv")
        proxies["lifesnaps_rais"] = build_recovery_proxy(
            life_features, "lifesnaps_rais",
            args.output / "synthetic_engineering_only" / "lifesnaps_recovery_like_proxy.csv")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Expanded source-aware research aggregation; not clinical validation.",
        "source_datasets": {
            "fatigue_zenodo_8018238": {"source_record": "https://zenodo.org/records/8018238", "license": "CC BY 4.0",
                                        "citation": "Rao et al. 2023, DOI 10.3389/fdgth.2023.1099456",
                                        "wearables": fatigue_stats, "surveys": fatigue_survey_stats,
                                        "limits": ["questionnaire fatigue is self-report", "units not stated for many aggregates",
                                                   "disease/healthy cohort mix", "no Recovery outcome"]},
            "lifesnaps_rais": {"source_record": "https://zenodo.org/records/7229547", "observations": life_stats,
                               "surveys": lifesnaps_survey_stats,
                               "limits": ["wearable definitions and units need codebook audit", "STAI/PANAS/SEMA are distinct self-report types"]},
        },
        "outputs": {"observations_long_rows": len(fatigue_observations) + len(life_observations),
                    "labels_long_rows": len(all_labels), "fatigue_training_examples": joined_fatigue,
                    "lifesnaps_feature_windows": joined_lifesnaps, "synthetic_proxies": proxies},
        "identity_policy": "participant IDs are namespaced by dataset; never cross-join IDs across cohorts",
        "synthetic_policy": "only generated engineering proxy files under synthetic_engineering_only; never append to observed labels or training examples",
    }
    (args.output / "master_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
