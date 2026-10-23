from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

META = {"dataset_id", "participant_id", "target_name", "target_category", "target_date", "survey_date",
        "target_period_start", "target_period_end", "target_source", "target_source_file", "target_source_row",
        "feature_window_start", "feature_window_end_exclusive", "feature_window_relation", "lookback_days",
        "observed_days", "synthetic", "clinical_validation", "observed_sensor_metrics", "eligible_for_modeling"}
LABELS = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r.get("target_name") == "tired_yesterday"
                and r.get("eligible_for_modeling", "").casefold() == "true"
                and r.get("synthetic", "false").casefold() != "true"]
    if not rows:
        raise ValueError("No real eligible tired_yesterday rows found")
    fields = [k for k in rows[0] if k not in META and k != "target_numeric_source"]
    X = np.asarray([[float(r[k]) if r.get(k) not in (None, "") else np.nan for k in fields] for r in rows], dtype=float)
    y = np.asarray([LABELS[r["target_category"].casefold()] for r in rows], dtype=int)
    groups = np.asarray([r["participant_id"] for r in rows], dtype=str)
    return X, y, groups, fields


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "ordinal_mae": float(mean_absolute_error(y, pred)),
            "ordinal_rmse": float(math.sqrt(mean_squared_error(y, pred)))}


def run_group_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> dict[str, Any]:
    n_splits = min(folds, len(set(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    names = ["majority_class", "median_ordinal", "hist_gradient_boosting_classifier",
             "hist_gradient_boosting_regressor_rounded", "extra_trees_balanced", "random_forest_balanced"]
    predictions = {name: np.zeros(len(y), dtype=int) for name in names}
    fold_reports = []
    for fold, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
        majority = Counter(y[train]).most_common(1)[0][0]
        median = int(np.clip(np.rint(np.median(y[train])), 0, 4))
        predictions["majority_class"][test] = majority
        predictions["median_ordinal"][test] = median

        classifiers = {
            "hist_gradient_boosting_classifier": HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=2.0, random_state=42),
            "extra_trees_balanced": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                ExtraTreesClassifier(n_estimators=300, min_samples_leaf=8, max_features=0.7,
                                     class_weight="balanced", random_state=42, n_jobs=1)),
            "random_forest_balanced": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features=0.7,
                                       class_weight="balanced_subsample", random_state=42, n_jobs=1)),
        }
        for name, model in classifiers.items():
            model.fit(X[train], y[train])
            predictions[name][test] = model.predict(X[test]).astype(int)
        reg = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=7,
                                             l2_regularization=1.0, random_state=42)
        reg.fit(X[train], y[train])
        predictions["hist_gradient_boosting_regressor_rounded"][test] = np.clip(np.rint(reg.predict(X[test])), 0, 4).astype(int)
        fold_reports.append({"fold": fold, "train_people": len(set(groups[train])),
                             "test_people": len(set(groups[test])), "train_rows": len(train),
                             "test_rows": len(test), "train_class_counts": dict(Counter(map(int, y[train]))),
                             "test_class_counts": dict(Counter(map(int, y[test])))})
    summary = {name: metrics(y, pred) for name, pred in predictions.items()}
    base_mae = summary["median_ordinal"]["ordinal_mae"]
    for report in summary.values():
        report["beats_median_baseline_ordinal_mae"] = report["ordinal_mae"] < base_mae
    return {"split": f"GroupKFold({n_splits}) by participant; each participant held out once",
            "folds": fold_reports, "metrics": summary}


def run_temporal_holdout(X: np.ndarray, y: np.ndarray, groups: np.ndarray, dates: list[str], fields: list[str],
                         test_fraction: float = 0.2) -> dict[str, Any]:
    # This test asks whether each participant's earlier history predicts their later answers.
    per_person: dict[str, list[int]] = {}
    for i, person in enumerate(groups):
        per_person.setdefault(person, []).append(i)
    train_ix: list[int] = []
    test_ix: list[int] = []
    for person, indices in per_person.items():
        ordered = sorted(indices, key=lambda i: dates[i])
        if len(ordered) < 5:
            continue
        cut = max(3, min(len(ordered)-1, int(len(ordered)*(1-test_fraction))))
        train_ix.extend(ordered[:cut])
        test_ix.extend(ordered[cut:])
    if not test_ix:
        return {"status": "insufficient_repeated_participant_history"}
    train_ix_a, test_ix_a = np.asarray(train_ix), np.asarray(test_ix)
    train_counts = Counter(map(int, y[train_ix_a]))
    majority = train_counts.most_common(1)[0][0]
    median = int(np.clip(np.rint(np.median(y[train_ix_a])), 0, 4))
    baseline_majority = np.full(len(test_ix_a), majority)
    baseline_median = np.full(len(test_ix_a), median)
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, max_features=0.7,
                                               class_weight="balanced", random_state=42, n_jobs=1))
    model.fit(X[train_ix_a], y[train_ix_a])
    pred = model.predict(X[test_ix_a]).astype(int)
    return {"status": "exploratory_within_person_temporal_holdout",
            "split": "last 20% of each participant's chronologically ordered survey responses; only participants with >=5 labels",
            "train_rows": len(train_ix_a), "test_rows": len(test_ix_a),
            "train_participants": len(set(groups[train_ix_a])), "test_participants": len(set(groups[test_ix_a])),
            "train_class_counts": dict(Counter(map(int, y[train_ix_a]))),
            "test_class_counts": dict(Counter(map(int, y[test_ix_a])),),
            "median_baseline": metrics(y[test_ix_a], baseline_median),
            "majority_baseline": metrics(y[test_ix_a], baseline_majority),
            "extra_trees": metrics(y[test_ix_a], pred),
            "limitations": ["Participant history may be used, unlike unseen-participant CV.",
                            "This is a single chronological holdout with overlapping survey windows.",
                            "Not external validation and not deployable without prospective evaluation."]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fatigue label models and baselines without synthetic data.")
    parser.add_argument("--input", type=Path, default=Path("dataset/harmonized/master/fatigue_daily_training_examples.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/master/fatigue_model_comparison.json"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    X, y, groups, fields = load(args.input)
    dates = []
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("target_name") == "tired_yesterday" and row.get("eligible_for_modeling", "").casefold() == "true" and row.get("synthetic", "false").casefold() != "true":
                dates.append(row["target_date"])
    report = {"purpose": "research-only model comparison; target is self-reported tiredness, not clinical fatigue or Recovery",
              "rows": len(y), "participants": len(set(groups)), "features": len(fields),
              "class_counts": dict(Counter(map(int, y))),
              "ordinal_mapping": LABELS,
              "evaluation": run_group_cv(X, y, groups, args.folds),
              "within_person_temporal": run_temporal_holdout(X, y, groups, dates, fields),
              "synthetic_rows_included": False,
              "recommendation": "Choose only by preregistered primary metric and untouched validation; report macro-F1, balanced accuracy and ordinal MAE together."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "class_counts": report["class_counts"],
                      "group_cv": report["evaluation"]["metrics"],
                      "within_person_temporal": report["within_person_temporal"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
