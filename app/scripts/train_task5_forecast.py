from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def read_rows(path: Path, metric: str, dataset: str) -> list[dict[str, Any]]:
    import csv
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"dataset_id", "participant_id", "date", "metric_code", "value", "unit"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for row in reader:
            if row["dataset_id"] != dataset or row["metric_code"] != metric:
                continue
            try:
                day = date.fromisoformat(row["date"][:10])
                value = float(row["value"])
            except (ValueError, TypeError):
                continue
            if not math.isfinite(value) or row.get("synthetic", "false").casefold() == "true":
                continue
            rows.append({"participant": row["participant_id"], "date": day,
                         "value": value, "unit": row["unit"]})
    return rows


def make_examples(rows: list[dict[str, Any]], history_days: int, horizon: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[date, tuple[float, str]]] = defaultdict(dict)
    for row in rows:
        participant = row["participant"]
        day = row["date"]
        if day in grouped[participant]:
            raise ValueError(f"Duplicate {participant} {day}: daily panel must contain one row per metric/day")
        grouped[participant][day] = (row["value"], row["unit"])
    examples = []
    for participant, values in sorted(grouped.items()):
        units = {unit for _, unit in values.values()}
        if len(units) != 1:
            raise ValueError(f"Mixed units for participant {participant}: {units}")
        days = sorted(values)
        lookup = {day: value for day, (value, _) in values.items()}
        for target_day in days:
            target_date = target_day + timedelta(days=horizon)
            if target_date not in lookup:
                continue
            history_dates = [target_day - timedelta(days=lag) for lag in range(history_days - 1, -1, -1)]
            if any(day not in lookup for day in history_dates):
                continue
            history = [lookup[day] for day in history_dates]
            features = {f"lag_{i+1}": history[-(i+1)] for i in range(min(7, history_days))}
            features.update({
                "mean_7": sum(history[-min(7, history_days):]) / min(7, history_days),
                "mean_history": sum(history) / len(history),
                "min_history": min(history),
                "max_history": max(history),
                "weekday_sin": math.sin(2 * math.pi * target_day.weekday() / 7),
                "weekday_cos": math.cos(2 * math.pi * target_day.weekday() / 7),
            })
            examples.append({"participant": participant, "target_day": target_date,
                             "features": features, "target": lookup[target_date],
                             "last_value": history[-1]})
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a global, single-dataset time-series regressor with participant holdout.")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True, help="Never pool datasets with incompatible device/label semantics.")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--test-participant-fraction", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.history_days < 2 or args.horizon_days < 1 or not 0.1 <= args.test_participant_fraction <= 0.5:
        parser.error("history-days >=2, horizon-days >=1 and test fraction between 0.1 and 0.5 are required")
    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as error:
        raise SystemExit("Install optional training dependencies: python -m pip install -r requirements-task5.txt") from error

    rows = read_rows(args.panel, args.metric, args.dataset_id)
    examples = make_examples(rows, args.history_days, args.horizon_days)
    participants = sorted({row["participant"] for row in examples})
    if len(participants) < 6:
        raise SystemExit(f"Need at least 6 participants with usable windows; found {len(participants)}")
    features = sorted(examples[0]["features"])
    x = [[row["features"][name] for name in features] for row in examples]
    y = [row["target"] for row in examples]
    groups = [row["participant"] for row in examples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_participant_fraction, random_state=42)
    train_ix, test_ix = next(splitter.split(x, y, groups))
    train_people = {groups[index] for index in train_ix}
    test_people = {groups[index] for index in test_ix}
    if train_people & test_people:
        raise RuntimeError("Participant leakage detected")
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_leaf_nodes=7,
                                           l2_regularization=1.0, random_state=42)
    model.fit([x[i] for i in train_ix], [y[i] for i in train_ix])
    predictions = model.predict([x[i] for i in test_ix])
    target = [y[i] for i in test_ix]
    naive = [examples[i]["last_value"] for i in test_ix]
    mae = mean_absolute_error(target, predictions)
    naive_mae = mean_absolute_error(target, naive)
    rmse = math.sqrt(mean_squared_error(target, predictions))
    denominator = sum(abs(target[i] - naive[i]) for i in range(len(target))) / max(1, len(target))
    mase = mae / denominator if denominator > 0 else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.dataset_id}_{args.metric}_h{args.horizon_days}.joblib"
    joblib.dump({"model": model, "features": features, "dataset_id": args.dataset_id,
                 "metric_code": args.metric, "unit": rows[0]["unit"], "history_days": args.history_days,
                 "horizon_days": args.horizon_days, "split_seed": 42}, model_path)
    report = {
        "purpose": "Research baseline only; not clinical validation or production model approval.",
        "dataset_id": args.dataset_id, "metric_code": args.metric, "unit": rows[0]["unit"],
        "history_days": args.history_days, "horizon_days": args.horizon_days,
        "split": "GroupShuffleSplit by participant; no participant overlap",
        "participants_total": len(participants), "train_participants": len(train_people),
        "test_participants": len(test_people), "train_examples": len(train_ix), "test_examples": len(test_ix),
        "model": "HistGradientBoostingRegressor", "mae": mae, "rmse": rmse,
        "last_value_baseline_mae": naive_mae, "mase_vs_last_value": mase,
        "beats_last_value_baseline": mae < naive_mae,
        "model_path": str(model_path),
        "limitations": [
            "Use one dataset and one metric per run; never combine incompatible sources or units.",
            "Participant holdout tests new-person generalization, but does not replace a temporal external test.",
            "No prediction interval or calibrated uncertainty is included.",
            "Not suitable for fatigue, Recovery, clinical decisions or alerts without validated targets and labels.",
        ],
    }
    report_path = args.output_dir / f"{args.dataset_id}_{args.metric}_h{args.horizon_days}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
