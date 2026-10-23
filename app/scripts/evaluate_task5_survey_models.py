from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_examples(path: Path, target: str) -> tuple[list[dict[str, float]], list[float], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"participant_id", "target_name", "target_value", "eligible_for_modeling"}
        if required - set(reader.fieldnames or []):
            raise ValueError(f"Missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = [row for row in reader if row["target_name"] == target and row["eligible_for_modeling"].casefold() == "true"]
    if not rows:
        return [], [], [], []
    blocked = {"target_value", "target_date", "target_source_row", "participant_id", "dataset_id", "target_name",
               "target_unit", "target_source", "feature_window_start", "feature_window_end_exclusive",
               "temporal_rule", "lookback_days", "synthetic", "clinical_validation", "eligible_for_modeling",
               "eligibility_rule", "feature_days_total"}
    names = [key for key in rows[0] if key not in blocked and any(key.endswith(suffix) for suffix in
             ("_mean", "_std", "_last", "_delta_first_last", "_n_days", "_coverage"))]
    examples: list[dict[str, float]] = []
    targets: list[float] = []
    people: list[str] = []
    dates: list[str] = []
    for row in rows:
        features: dict[str, float] = {}
        for name in names:
            raw = row.get(name, "")
            try:
                value = float(raw) if raw not in (None, "") else math.nan
            except ValueError:
                value = math.nan
            features[name] = value if math.isfinite(value) else math.nan
        try:
            y = float(row["target_value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y):
            continue
        examples.append(features)
        targets.append(y)
        people.append(row["participant_id"])
        dates.append(row["target_date"])
    return examples, targets, people, names


def evaluate_target(path: Path, target: str, folds: int = 5) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import GroupKFold

    x, y, groups, feature_names = load_examples(path, target)
    unique_people = sorted(set(groups))
    if len(unique_people) < max(6, folds):
        return {"target": target, "status": "insufficient_participants", "participants": len(unique_people), "rows": len(y)}
    fold_count = min(folds, len(unique_people))
    splitter = GroupKFold(n_splits=fold_count)
    predictions = [math.nan] * len(y)
    baselines = [math.nan] * len(y)
    fold_reports = []
    for fold, (train_ix, test_ix) in enumerate(splitter.split(x, y, groups), start=1):
        train_y = [y[index] for index in train_ix]
        baseline = statistics.median(train_y)
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=5,
                                               l2_regularization=1.0, random_state=42)
        model.fit([[x[index][name] for name in feature_names] for index in train_ix], train_y)
        pred = model.predict([[x[index][name] for name in feature_names] for index in test_ix])
        fold_people = sorted({groups[index] for index in test_ix})
        for index, value in zip(test_ix, pred):
            predictions[index] = float(value)
            baselines[index] = baseline
        fold_reports.append({"fold": fold, "train_participants": len({groups[index] for index in train_ix}),
                             "test_participants": len(fold_people), "test_rows": len(test_ix),
                             "test_participant_ids": fold_people})

    model_mae = mean_absolute_error(y, predictions)
    baseline_mae = mean_absolute_error(y, baselines)
    model_rmse = math.sqrt(mean_squared_error(y, predictions))
    baseline_rmse = math.sqrt(mean_squared_error(y, baselines))
    return {
        "target": target, "status": "exploratory_group_cross_validation",
        "rows": len(y), "participants": len(unique_people), "features": feature_names,
        "split": f"GroupKFold({fold_count}), participant-grouped; every participant held out once",
        "model": "HistGradientBoostingRegressor", "baseline": "training-fold median",
        "model_mae": model_mae, "model_rmse": model_rmse,
        "baseline_mae": baseline_mae, "baseline_rmse": baseline_rmse,
        "beats_baseline_mae": model_mae < baseline_mae,
        "folds": fold_reports,
        "limitations": ["STAI/PANAS are self-report questionnaire scores, not clinical diagnoses or direct fatigue/recovery ground truth.",
                        "Small cohort; metrics are exploratory and not external validation.",
                        "Features use the preceding seven calendar days only; missing values are not imputed.",
                        "No feature importance or causal interpretation is claimed."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Participant-held-out exploratory STAI/PANAS regression benchmark.")
    parser.add_argument("--features", type=Path, default=Path("dataset/harmonized/enriched/lifesnaps_survey_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/enriched/survey_model_evaluation.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--targets", nargs="+", default=["STAI_STRESS_SCORE", "PANAS_POSITIVE_AFFECT", "PANAS_NEGATIVE_AFFECT"])
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("folds must be at least 2")
    results = [evaluate_target(args.features, target, args.folds) for target in args.targets]
    report = {"purpose": "MVP research benchmark; no clinical or alert use", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
