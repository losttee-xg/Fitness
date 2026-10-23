from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

TIME_FORMAT = "%d.%m.%y %H:%M"
SENSOR_FIELDS = ("HR", "HRV", "RESP", "Steps", "EnergyExpenditure", "ActivityCounts", "SkinTemperature", "BloodPerfusion")


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), TIME_FORMAT)
    except (ValueError, TypeError, AttributeError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(root: Path, out: Path) -> dict[str, Any]:
    survey_rows = read_csv(root / "fatiguePROs.csv")
    sensor_by_user_date: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    sensor_files = sorted(root.glob("subjectID_*.csv"))
    sensor_file_stats = {}
    for path in sensor_files:
        subject = path.stem.replace("subjectID_", "")
        rows = read_csv(path)
        valid = 0
        for row in rows:
            dt = parse_time(row.get("Timestamp", ""))
            if dt:
                sensor_by_user_date[(subject, dt.strftime("%Y-%m-%d"))].append(row)
                valid += 1
        sensor_file_stats[subject] = {"raw_rows": len(rows), "valid_timestamp_rows": valid}

    # Keep the explicitly scaled fatigue question as the target; other PROs remain separate outcomes.
    target_by_user_date: dict[tuple[str, str], dict[str, Any]] = {}
    other_question_counts: Counter[str] = Counter()
    for row in survey_rows:
        participant = str(row.get("SubjectID", "")).strip()
        dt = parse_time(row.get("DateTime", ""))
        if not participant or not dt:
            continue
        question = (row.get("PROquestion") or "").strip()
        answer = (row.get("PROanswer_value") or "").strip()
        choice = (row.get("PROanswer_choice") or "").strip()
        other_question_counts[question] += 1
        if question != "Describe fatigue on a scale of 1 to 10, where 1 means you don’t feel tired at all and 10 means the worst tiredness you can imagine":
            continue
        try:
            score = float(answer)
        except ValueError:
            continue
        target_by_user_date[(participant, dt.strftime("%Y-%m-%d"))] = {
            "target": score, "target_raw": answer, "timestamp": dt,
            "question": question, "timezone": (row.get("Timezone") or "")}

    all_features = sorted({field for rows in sensor_by_user_date.values() for row in rows for field in row if field in SENSOR_FIELDS})
    examples: list[dict[str, Any]] = []
    pair_counts = Counter()
    for (participant, target_day), target in sorted(target_by_user_date.items()):
        previous_day = (target["timestamp"].date() - timedelta(days=1)).strftime("%Y-%m-%d")
        prior = sensor_by_user_date.get((participant, previous_day), [])
        if not prior:
            continue
        parsed_values: dict[str, list[float]] = defaultdict(list)
        for record in prior:
            for field in all_features:
                raw = (record.get(field) or "").strip()
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if math.isfinite(value):
                    parsed_values[field].append(value)
        row: dict[str, Any] = {
            "dataset_id": "healthy_multisensor_fatigue_zenodo_4266157",
            "participant_id": participant,
            "target_date": target_day,
            "target_score": target["target"],
            "target_original_value": target["target_raw"],
            "target_question": target["question"],
            "survey_datetime": target["timestamp"].isoformat(),
            "timezone": target["timezone"],
            "feature_date": previous_day,
            "feature_window_relation": "complete_calendar_day_strictly_before_fatigue_survey_day",
            "synthetic": False,
            "clinical_validation": False,
        }
        total_samples = max(1, len(prior))
        for field in all_features:
            vals = parsed_values.get(field, [])
            row[f"{field}__mean"] = statistics.mean(vals) if vals else None
            row[f"{field}__std"] = statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)
            row[f"{field}__min"] = min(vals) if vals else None
            row[f"{field}__max"] = max(vals) if vals else None
            row[f"{field}__samples"] = len(vals)
            row[f"{field}__sample_coverage"] = len(vals) / total_samples
        row["sensor_rows_prior_day"] = len(prior)
        row["eligible_for_modeling"] = len(parsed_values.get("HR", [])) > 0
        examples.append(row)
        pair_counts["matched_survey_days"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(examples[0]) if examples else []
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(examples)
    return {"output": str(out), "survey_rows": len(survey_rows), "sensor_files": len(sensor_files),
            "sensor_participants": len(sensor_file_stats), "fatigue_question_days": len(target_by_user_date),
            "matched_days_with_previous_sensor_day": len(examples),
            "participants_matched": len({r["participant_id"] for r in examples}),
            "sensor_metrics": all_features, "other_survey_question_counts": dict(other_question_counts),
            "sensor_file_stats": sensor_file_stats,
            "rule": "target is direct self-reported 1-10 fatigue rating; predictors summarize only the complete prior calendar day; no generated labels",
            "license": "CC BY 4.0; Zenodo DOI 10.5281/zenodo.4266157"}


def evaluate(path: Path, folds: int = 5) -> dict[str, Any]:
    rows = [r for r in read_csv(path) if r.get("eligible_for_modeling", "").casefold() == "true"]
    if not rows:
        return {"status": "no_eligible_rows"}
    metadata = {"dataset_id", "participant_id", "target_date", "target_score", "target_original_value", "target_question",
                "survey_datetime", "timezone", "feature_date", "feature_window_relation", "synthetic",
                "clinical_validation", "sensor_rows_prior_day", "eligible_for_modeling"}
    feature_names = [k for k in rows[0] if k not in metadata]
    X = np.asarray([[float(r[k]) if r.get(k) not in (None, "") else np.nan for k in feature_names] for r in rows], dtype=float)
    y = np.asarray([float(r["target_score"]) for r in rows], dtype=float)
    groups = np.asarray([r["participant_id"] for r in rows])
    split = GroupKFold(n_splits=min(folds, len(set(groups))))
    preds = np.zeros(len(y)); baseline = np.zeros(len(y))
    fold_reports = []
    for fold, (tr, te) in enumerate(split.split(X, y, groups), start=1):
        baseline[te] = statistics.median(y[tr])
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.7,
                                                  random_state=42, n_jobs=1))
        model.fit(X[tr], y[tr]); preds[te] = model.predict(X[te])
        fold_reports.append({"fold": fold, "train_participants": len(set(groups[tr])),
                             "test_participants": len(set(groups[te])), "test_rows": len(te)})
    return {"status": "exploratory_external_cohort_group_cv", "target": "self-reported fatigue 1..10, original numeric answer",
            "rows": len(rows), "participants": len(set(groups)), "features": len(feature_names),
            "split": f"GroupKFold({len(fold_reports)}) by participant",
            "model": {"mae": float(mean_absolute_error(y, preds)), "rmse": float(math.sqrt(mean_squared_error(y, preds)))},
            "training_fold_median_baseline": {"mae": float(mean_absolute_error(y, baseline)), "rmse": float(math.sqrt(mean_squared_error(y, baseline)))},
            "beats_baseline_mae": float(mean_absolute_error(y, preds)) < float(mean_absolute_error(y, baseline)),
            "folds": fold_reports,
            "limitations": ["small healthy-adult pilot cohort", "survey scale is subjective; inspect original study instrument and data coverage",
                            "previous-day sensor features predict a later survey rating, not clinical fatigue or Recovery",
                            "do not tune repeatedly on this cohort and report as independent validation"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and externally evaluate public daily fatigue wearable cohort.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/fatigue_zenodo_4266157"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/harmonized/independent_fatigue_4266157"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    prep = prepare(args.dataset_root, args.output_dir / "fatigue_wearable_windows.csv")
    ev = evaluate(args.output_dir / "fatigue_wearable_windows.csv", args.folds)
    report = {"source": "Zenodo 4266157, CC BY 4.0", "citation": "De Luca et al. 2020, DOI 10.5281/zenodo.4266157",
              "preparation": prep, "evaluation": ev}
    (args.output_dir / "external_cohort_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"preparation": prep, "evaluation": ev}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
