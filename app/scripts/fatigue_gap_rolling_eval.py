from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

LABELS = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}
META = {"dataset_id", "participant_id", "target_name", "target_category", "target_date", "survey_date",
        "target_period_start", "target_period_end", "target_source", "target_source_file", "target_source_row",
        "feature_window_start", "feature_window_end_exclusive", "feature_window_relation", "lookback_days",
        "observed_days", "synthetic", "clinical_validation", "observed_sensor_metrics", "eligible_for_modeling"}


def read_data(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("target_name") == "tired_yesterday"
                and r.get("eligible_for_modeling", "").casefold() == "true"
                and r.get("synthetic", "false").casefold() != "true"]
    feature_names = [k for k in rows[0] if k not in META and k != "target_numeric_source"]
    X = np.asarray([[float(r[k]) if r.get(k) not in (None, "") else np.nan for k in feature_names] for r in rows], dtype=float)
    y = np.asarray([LABELS[r["target_category"].casefold()] for r in rows], dtype=int)
    groups = [r["participant_id"] for r in rows]
    dates = [date.fromisoformat(r["target_date"]) for r in rows]
    return rows, feature_names, X, y, groups, dates


def scores(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(math.sqrt(mean_squared_error(y, pred)))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Gap-aware rolling-origin known-user fatigue check.")
    parser.add_argument("--input", type=Path, default=Path("dataset/harmonized/master/fatigue_daily_training_examples.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/master/fatigue_gap_rolling_evaluation.json"))
    parser.add_argument("--gap-days", type=int, default=7)
    parser.add_argument("--minimum-train-rows", type=int, default=100)
    args = parser.parse_args()
    rows, feature_names, X_sensor, y, groups, dates = read_data(args.input)

    # Create only past-label features; never use a response from the same/future target day.
    by_person: dict[str, list[int]] = {}
    for i, person in enumerate(groups):
        by_person.setdefault(person, []).append(i)
    X_history = np.full((len(rows), 4), np.nan, dtype=float)
    for person, indices in by_person.items():
        ordered = sorted(indices, key=lambda i: dates[i])
        past: list[int] = []
        for ix in ordered:
            eligible = [j for j in past if dates[j] < dates[ix]]
            if eligible:
                recent = eligible[-3:]
                X_history[ix, 0] = y[eligible[-1]]
                X_history[ix, 1] = statistics.mean(y[j] for j in recent)
                X_history[ix, 2] = len(eligible)
                X_history[ix, 3] = (dates[ix] - dates[eligible[-1]]).days
            past.append(ix)
    X_combined = np.column_stack([X_sensor, X_history])

    sorted_dates = sorted(set(dates))
    cut_indices = [int(len(sorted_dates)*0.60), int(len(sorted_dates)*0.72), int(len(sorted_dates)*0.84)]
    cut_dates = [sorted_dates[min(i, len(sorted_dates)-1)] for i in cut_indices]
    ends = cut_dates[1:] + [sorted_dates[-1] + timedelta(days=1)]
    fold_results = []
    covered_test_indices: set[int] = set()
    for fold, (cut, end) in enumerate(zip(cut_dates, ends), start=1):
        test_start = cut
        train_end = cut - timedelta(days=args.gap_days)
        train = np.asarray([i for i, day in enumerate(dates) if day < train_end], dtype=int)
        test = np.asarray([i for i, day in enumerate(dates) if test_start <= day < end], dtype=int)
        test = np.asarray([i for i in test if i not in covered_test_indices], dtype=int)
        covered_test_indices.update(map(int, test))
        if len(train) < args.minimum_train_rows or not len(test):
            fold_results.append({"fold": fold, "status": "insufficient_rows", "train_rows": len(train), "test_rows": len(test)})
            continue

        train_persons = set(groups[i] for i in train)
        eligible_test = np.asarray([i for i in test if groups[i] in train_persons and np.isfinite(X_history[i, 0])], dtype=int)
        if not len(eligible_test):
            fold_results.append({"fold": fold, "status": "no_known_user_history", "train_rows": len(train), "test_rows": len(test)})
            continue
        base_train_median = int(np.rint(np.median(y[train])))
        p_last = np.asarray([int(np.rint(X_history[i, 0])) if np.isfinite(X_history[i, 0]) else base_train_median for i in eligible_test])
        p_mean3 = np.asarray([int(np.rint(X_history[i, 1])) if np.isfinite(X_history[i, 1]) else base_train_median for i in eligible_test])

        models = {}
        for name, X in (("wearable_only", X_sensor), ("wearable_plus_prior_self_report", X_combined)):
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  ExtraTreesRegressor(n_estimators=250, min_samples_leaf=10, max_features=0.7,
                                                      random_state=42, n_jobs=1))
            model.fit(X[train], y[train])
            pred = np.clip(np.rint(model.predict(X[eligible_test])), 0, 4).astype(int)
            models[name] = scores(y[eligible_test], pred)
        fold_results.append({"fold": fold, "status": "evaluated", "train_end_exclusive": train_end.isoformat(),
                             "test_start": test_start.isoformat(), "test_end_exclusive": end.isoformat(),
                             "gap_days": args.gap_days, "train_rows": len(train), "test_rows_all": len(test),
                             "test_rows_known_users_with_history": len(eligible_test),
                             "train_participants": len(train_persons),
                             "test_participants": len(set(groups[i] for i in eligible_test)),
                             "observed_class_counts_test": dict(Counter(map(int, y[eligible_test]))),
                             "last_self_report_baseline": scores(y[eligible_test], p_last),
                             "rolling3_self_report_baseline": scores(y[eligible_test], p_mean3),
                             "models": models})
    report = {"purpose": "diagnostic research evaluation; not deployment approval",
              "target": "five-category self-reported tiredness proxy, not clinical fatigue or Recovery",
              "rows": len(rows), "participants": len(set(groups)), "features": len(feature_names),
              "split": "three non-overlapping chronological test blocks; training target dates end at least gap_days before test block",
              "gap_days": args.gap_days, "feature_window_days": 7,
              "known_user_prediction_timing": "past survey labels used only if target date is strictly earlier than prediction target",
              "synthetic_included": False, "folds": fold_results,
              "overall_warning": ["User-level holdout and known-user future prediction are different product claims.",
                                  "This is one deterministic rolling-origin protocol and shares source cohort with development; not external prospective validation.",
                                  "No repeated tuning on these test blocks; use new prospective data for final confirmation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": fold_results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
