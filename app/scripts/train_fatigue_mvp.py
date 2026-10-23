from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


SEVERITY = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}
EXCLUDED_SENSOR_COLUMNS = {"user_id", "date"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def build_examples(root: Path, output: Path, lookback_days: int = 7, min_sensor_features: int = 5) -> dict[str, Any]:
    if lookback_days < 1 or min_sensor_features < 1:
        raise ValueError("lookback_days and min_sensor_features must be positive")
    activity_path = root / "day_activity_features.csv"
    survey_path = root / "daily_surveys_one_click_responses.csv"
    activity_fields, activity_rows = read_csv(activity_path)
    _, survey_rows = read_csv(survey_path)

    activity_by_user: dict[str, dict[date, dict[str, str]]] = defaultdict(dict)
    for row in activity_rows:
        try:
            day = date.fromisoformat(row["date"][:10])
        except (ValueError, TypeError, KeyError):
            continue
        participant = row.get("user_id", "").strip()
        if participant:
            activity_by_user[participant][day] = row

    # The survey is submitted on the day after it asks about. Use wearable days strictly before the fatigue target day.
    eligible_surveys: dict[tuple[str, date], dict[str, str]] = {}
    duplicates = 0
    for row in survey_rows:
        if row.get("one_click_response", "").strip().casefold() != "yes":
            continue
        label = row.get("tired_yesterday", "").strip().casefold()
        if label not in SEVERITY:
            continue
        try:
            survey_day = date.fromisoformat(row["survey_start_datetime"][:10])
        except (ValueError, TypeError, KeyError):
            continue
        fatigue_day = survey_day - timedelta(days=1)
        key = (row.get("user_id", "").strip(), fatigue_day)
        if not key[0]:
            continue
        prior = eligible_surveys.get(key)
        if prior:
            duplicates += 1
            if row.get("survey_end_datetime", "") > prior.get("survey_end_datetime", ""):
                eligible_surveys[key] = row
        else:
            eligible_surveys[key] = row

    sensor_metrics = [field for field in activity_fields if field not in EXCLUDED_SENSOR_COLUMNS]
    output_rows: list[dict[str, Any]] = []
    missing_sensor_target_prior = 0
    for (participant, fatigue_day), survey in sorted(eligible_surveys.items()):
        # Use only the seven calendar days preceding the day being labeled, never that day or later.
        history_days = [fatigue_day - timedelta(days=lag) for lag in range(lookback_days, 0, -1)]
        observations = [activity_by_user.get(participant, {}).get(day) for day in history_days]
        if not any(observations):
            missing_sensor_target_prior += 1
            continue
        out: dict[str, Any] = {
            "dataset_id": "fatigue_remote_wearable_zenodo_8018238",
            "participant_id": participant,
            "fatigue_date": fatigue_day.isoformat(),
            "survey_date": survey.get("survey_start_datetime", "")[:10],
            "target_name": "tired_yesterday",
            "target_category": survey["tired_yesterday"].strip().casefold(),
            "target_ordinal_proxy": SEVERITY[survey["tired_yesterday"].strip().casefold()],
            "target_source": "daily_surveys_one_click_responses.csv; question: Did you feel tired yesterday?",
            "target_source_row": survey.get("day_occurrence", ""),
            "feature_window_start": history_days[0].isoformat(),
            "feature_window_end_exclusive": fatigue_day.isoformat(),
            "lookback_days": lookback_days,
            "temporal_rule": "features only from calendar days before fatigue_date; label asks about day before survey submission",
            "observed_feature_days": sum(row is not None for row in observations),
            "synthetic": False,
            "clinical_validation": False,
        }
        populated = 0
        for metric in sensor_metrics:
            vals: list[float] = []
            for row in observations:
                if row is None:
                    continue
                try:
                    value = float(row.get(metric, ""))
                except (ValueError, TypeError):
                    continue
                if math.isfinite(value):
                    vals.append(value)
            out[metric] = statistics.mean(vals) if vals else None
            if vals:
                populated += 1
        out["observed_sensor_metrics"] = populated
        out["eligible_for_modeling"] = populated >= min_sensor_features
        output_rows.append(out)

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0]) if output_rows else ["dataset_id", "participant_id", "fatigue_date", "target_category"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "output": str(output), "rows": len(output_rows),
        "eligible_rows": sum(bool(row["eligible_for_modeling"]) for row in output_rows),
        "participants": len({row["participant_id"] for row in output_rows}),
        "ordinal_proxy_counts": dict(Counter(row["target_category"] for row in output_rows)),
        "survey_duplicate_user_fatigue_days": duplicates,
        "labels_without_prior_sensor_rows": missing_sensor_target_prior,
        "lookback_days": lookback_days, "min_sensor_metrics": min_sensor_features,
        "label_semantics": "five-category self-reported tiredness; ordinal proxy mapping only, not clinical fatigue diagnosis or recovery label",
        "license": "CC BY 4.0; attribute Zenodo DOI 10.5281/zenodo.8018238",
        "clinical_validation": False,
    }


def train_and_evaluate(dataset: Path, output: Path, folds: int = 5) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import GroupKFold
    import joblib

    _, rows = read_csv(dataset)
    rows = [row for row in rows if row.get("eligible_for_modeling", "").casefold() == "true"]
    if not rows:
        raise ValueError("No eligible training examples")
    metadata = {"dataset_id", "participant_id", "fatigue_date", "survey_date", "target_name", "target_category",
                "target_ordinal_proxy", "target_source", "target_source_row", "feature_window_start",
                "feature_window_end_exclusive", "lookback_days", "temporal_rule", "observed_feature_days",
                "synthetic", "clinical_validation", "observed_sensor_metrics", "eligible_for_modeling"}
    feature_names = [name for name in rows[0] if name not in metadata]
    people = [row["participant_id"] for row in rows]
    y = [float(row["target_ordinal_proxy"]) for row in rows]
    x: list[list[float]] = []
    for row in rows:
        features = []
        for name in feature_names:
            try:
                value = float(row.get(name, ""))
            except (ValueError, TypeError):
                value = math.nan
            features.append(value if math.isfinite(value) else math.nan)
        x.append(features)
    unique_people = sorted(set(people))
    fold_count = min(folds, len(unique_people))
    if fold_count < 2:
        raise ValueError("Need at least two participant groups")

    predictions = [math.nan] * len(rows)
    baseline_predictions = [math.nan] * len(rows)
    fold_reports = []
    final_model = None
    final_active_features: list[int] = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=fold_count).split(x, y, people), start=1):
        train_y = [y[index] for index in train_idx]
        baseline = statistics.median(train_y)
        active_features = [column for column in range(len(feature_names))
                           if len({x[index][column] for index in train_idx if math.isfinite(x[index][column])}) > 1]
        if not active_features:
            predicted = [baseline] * len(test_idx)
            model = None
        else:
            model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=7,
                                                   l2_regularization=1.0, random_state=42)
            model.fit([[x[index][column] for column in active_features] for index in train_idx], train_y)
            predicted = model.predict([[x[index][column] for column in active_features] for index in test_idx])
        if fold == fold_count:
            final_active_features = active_features
            final_model = model
        for index, value in zip(test_idx, predicted):
            predictions[index] = float(value)
            baseline_predictions[index] = baseline
        fold_reports.append({"fold": fold, "train_participants": len({people[i] for i in train_idx}),
                             "test_participants": len({people[i] for i in test_idx}), "test_rows": len(test_idx)})

    model_mae = mean_absolute_error(y, predictions)
    baseline_mae = mean_absolute_error(y, baseline_predictions)
    model_rmse = math.sqrt(mean_squared_error(y, predictions))
    baseline_rmse = math.sqrt(mean_squared_error(y, baseline_predictions))
    within_one = sum(abs(y[i] - predictions[i]) <= 1 for i in range(len(y))) / len(y)
    model_path = output.with_name("fatigue_tired_yesterday_model.joblib")
    joblib.dump({"model": final_model, "active_feature_names": [feature_names[i] for i in final_active_features],
                 "all_feature_names": feature_names, "target_mapping": SEVERITY,
                 "target": "tired_yesterday ordinal proxy 0-4", "source_dataset": "Zenodo 8018238",
                 "lookback_days": int(rows[0]["lookback_days"])}, model_path)
    report = {
        "purpose": "Exploratory self-reported tiredness proxy benchmark; not a clinical model or recovery predictor.",
        "dataset": "Zenodo 8018238, CC BY 4.0",
        "rows": len(rows), "participants": len(unique_people), "features": len(feature_names),
        "feature_names": feature_names,
        "target": "tired_yesterday mapped to ordinal 0..4 from source response categories; retain original category in prepared data",
        "split": f"GroupKFold({fold_count}), participant grouped; every participant held out once",
        "model": "HistGradientBoostingRegressor", "baseline": "training-fold median ordinal score",
        "model_path": str(model_path), "deployable": False,
        "model_mae_ordinal": model_mae, "baseline_mae_ordinal": baseline_mae,
        "model_rmse_ordinal": model_rmse, "baseline_rmse_ordinal": baseline_rmse,
        "within_one_category_fraction": within_one,
        "beats_baseline_mae": model_mae < baseline_mae,
        "folds": fold_reports,
        "limitations": ["The target is self-reported tiredness for the prior day, not a validated clinical fatigue diagnosis.",
                        "Ordinal 0..4 mapping is a transparent proxy ordering of categorical responses, not a validated interval scale.",
                        "Dataset contains healthy participants and people with Sjogren's disease/SLE; population shift and disease context matter.",
                        "Participant holdout is not temporal or external validation.",
                        "No model-generated labels or synthetic physiology are used.",
                        "Do not use for recovery score, medical advice, or alerts."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and train an exploratory wearable-to-self-reported-fatigue MVP.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/fatigue_zenodo_8018238"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/harmonized/fatigue_mvp"))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--min-sensor-metrics", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    data_path = args.output_dir / "fatigue_wearable_examples.csv"
    prep = build_examples(args.dataset_root, data_path, args.lookback_days, args.min_sensor_metrics)
    report = train_and_evaluate(data_path, args.output_dir / "fatigue_model_evaluation.json", args.folds)
    result = {"preparation": prep, "evaluation": {k: v for k, v in report.items() if k != "feature_names" and k != "folds"}}
    (args.output_dir / "run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
