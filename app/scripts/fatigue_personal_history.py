from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

LABELS = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}
META = {"dataset_id", "participant_id", "target_name", "target_category", "target_date", "survey_date",
        "target_period_start", "target_period_end", "target_source", "target_source_file", "target_source_row",
        "feature_window_start", "feature_window_end_exclusive", "feature_window_relation", "lookback_days",
        "observed_days", "synthetic", "clinical_validation", "observed_sensor_metrics", "eligible_for_modeling"}


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "ordinal_mae": float(mean_absolute_error(y, pred)),
            "ordinal_rmse": float(math.sqrt(mean_squared_error(y, pred)))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Known-user fatigue prediction with strictly prior check-in labels.")
    parser.add_argument("--input", type=Path, default=Path("dataset/harmonized/master/fatigue_daily_training_examples.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/master/fatigue_personal_history_experiment.json"))
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("target_name") == "tired_yesterday"
                and r.get("eligible_for_modeling", "").casefold() == "true"
                and r.get("synthetic", "false").casefold() != "true"]
    feature_names = [k for k in rows[0] if k not in META and k != "target_numeric_source"]
    by_person: dict[str, list[int]] = defaultdict(list)
    y = np.asarray([LABELS[r["target_category"].casefold()] for r in rows], dtype=int)
    for i, row in enumerate(rows):
        by_person[row["participant_id"]].append(i)

    X: list[list[float]] = []
    prior_last = np.full(len(rows), np.nan)
    prior_mean3 = np.full(len(rows), np.nan)
    prior_n = np.zeros(len(rows), dtype=float)
    prior_gap = np.full(len(rows), np.nan)
    temporal_test: list[int] = []
    temporal_train: list[int] = []
    for person, indices in by_person.items():
        ordered = sorted(indices, key=lambda i: rows[i]["target_date"])
        label_history: list[tuple[date, int]] = []
        person_cut = max(3, min(len(ordered)-1, int(len(ordered)*0.8))) if len(ordered) >= 5 else len(ordered)
        for position, ix in enumerate(ordered):
            target_day = date.fromisoformat(rows[ix]["target_date"])
            earlier = [(d, value) for d, value in label_history if d < target_day]
            if earlier:
                prior_last[ix] = earlier[-1][1]
                prior_mean3[ix] = statistics.mean(value for _, value in earlier[-3:])
                prior_n[ix] = len(earlier)
                prior_gap[ix] = (target_day-earlier[-1][0]).days
            if position < person_cut:
                temporal_train.append(ix)
            elif len(ordered) >= 5:
                temporal_test.append(ix)
            label_history.append((target_day, int(y[ix])))
        # Inference on a test row must only use outcomes with target dates strictly earlier than it.
        for pos in range(person_cut, len(ordered)):
            ix = ordered[pos]
            target_day = date.fromisoformat(rows[ix]["target_date"])
            past_ix = [j for j in ordered[:pos] if date.fromisoformat(rows[j]["target_date"]) < target_day]
            if past_ix:
                prior_last[ix] = int(y[past_ix[-1]])
                prior_mean3[ix] = statistics.mean(int(y[j]) for j in past_ix[-3:])
                prior_n[ix] = len(past_ix)
                prior_gap[ix] = (target_day-date.fromisoformat(rows[past_ix[-1]]["target_date"])).days

    base_feature_names = feature_names
    for i, row in enumerate(rows):
        vals = []
        for name in base_feature_names:
            try:
                value = float(row.get(name, ""))
            except (TypeError, ValueError):
                value = math.nan
            vals.append(value if math.isfinite(value) else math.nan)
        vals.extend([prior_last[i], prior_mean3[i], prior_n[i], prior_gap[i]])
        X.append(vals)
    X_arr = np.asarray(X, dtype=float)

    train = np.asarray(temporal_train, dtype=int)
    test = np.asarray(temporal_test, dtype=int)
    y_train, y_test = y[train], y[test]
    predictions: dict[str, np.ndarray] = {}
    pop_median = int(np.rint(np.median(y_train)))
    predictions["population_median"] = np.full(len(test), pop_median, dtype=int)
    predictions["last_known_self_report"] = np.where(np.isfinite(prior_last[test]), np.rint(prior_last[test]), pop_median).astype(int)
    predictions["rolling3_self_report"] = np.where(np.isfinite(prior_mean3[test]), np.rint(prior_mean3[test]), pop_median).astype(int)

    ordinal_model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=7,
                                                    l2_regularization=2.0, random_state=42)
    ordinal_model.fit(X_arr[train], y_train)
    predictions["wearable_plus_prior_labels_ordinal"] = np.clip(np.rint(ordinal_model.predict(X_arr[test])), 0, 4).astype(int)
    class_model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, max_features=0.7,
                                                     class_weight="balanced", random_state=42, n_jobs=1))
    class_model.fit(X_arr[train], y_train)
    predictions["wearable_plus_prior_labels_classifier"] = class_model.predict(X_arr[test]).astype(int)

    report = {"purpose": "exploratory known-user prediction at a check-in; not for unseen users or clinical use",
              "target": "self-reported tired yesterday ordinal convenience mapping 0..4",
              "split": "per-participant chronological 80/20 holdout; participants may appear in both periods",
              "train_rows": len(train), "test_rows": len(test), "participants": len({rows[i]["participant_id"] for i in test}),
              "class_counts_test": dict(Counter(map(int, y_test))),
              "predictor_timing": "prior daily survey responses are included only when prior target_date is strictly earlier than current target_date; suitable only for a future check-in after prior response has arrived",
              "feature_groups": {"wearable_feature_count": len(feature_names), "added_prior_label_features": ["last_known_tiredness", "rolling3_tiredness_mean", "prior_label_count", "days_since_previous_label"]},
              "metrics": {name: score(y_test, pred) for name, pred in predictions.items()},
              "limitations": ["The latest self-report may dominate because tiredness is persistent; this is a check-in nowcast, not a pre-day forecast.",
                              "Single temporal holdout; repeated windows overlap and labels may be serially correlated.",
                              "Source label is subjective tiredness, not clinical fatigue or Recovery.",
                              "No synthetic data or model-generated labels are used.",
                              "Model must beat personal-history baselines on a preregistered primary metric and then pass prospective validation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"],
                      "train_rows": len(train), "test_rows": len(test), "test_participants": report["participants"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
