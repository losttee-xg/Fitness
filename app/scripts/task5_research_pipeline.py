from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REQUIRED = {"dataset_id", "participant_id", "date", "metric_code", "value", "unit"}
LONG_REQUIRED = {"dataset_id", "participant_id", "metric_code", "value", "unit"}


def load_observations(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"parsed": 0, "skipped_missing_calendar_date": 0, "skipped_invalid": 0}
    for path in paths:
        if path.suffix.casefold() == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                source_rows = ((line_number, json.loads(line)) for line_number, line in enumerate(stream, start=1) if line.strip())
                for line, row in source_rows:
                    if LONG_REQUIRED - set(row):
                        raise ValueError(f"{path}:{line}: missing required long-format fields {sorted(LONG_REQUIRED - set(row))}")
                    timestamp = row.get("observed_at_utc")
                    if not timestamp:
                        counts["skipped_missing_calendar_date"] += 1
                        continue
                    try:
                        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            counts["skipped_missing_calendar_date"] += 1
                            continue
                        day = dt.astimezone(timezone.utc).date().isoformat()
                        value = float(row["value"])
                    except (TypeError, ValueError):
                        counts["skipped_invalid"] += 1
                        continue
                    if not math.isfinite(value) or not isinstance(row["value"], (int, float)):
                        counts["skipped_invalid"] += 1
                        continue
                    if row.get("metric_code") is None:
                        counts["skipped_invalid"] += 1
                        continue
                    rows.append({**row, "date": day, "value": value, "source_file": str(path), "source_row": line})
                    counts["parsed"] += 1
            continue

        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing required columns {sorted(missing)}")
            for line, row in enumerate(reader, start=2):
                try:
                    value = float(row["value"])
                    day = date.fromisoformat(row["date"][:10]).isoformat()
                except (TypeError, ValueError):
                    counts["skipped_invalid"] += 1
                    continue
                if not math.isfinite(value):
                    counts["skipped_invalid"] += 1
                    continue
                rows.append({**row, "date": day, "value": value, "source_file": str(path), "source_row": line})
                counts["parsed"] += 1
    return rows, counts


def build_panel(paths: list[Path], out: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    observations, counts = load_observations(paths)
    for row in observations:
        key = (row["dataset_id"], row["participant_id"], row["date"], row["metric_code"], row["unit"])
        grouped[key].append(row)
    panel: list[dict[str, Any]] = []
    for (dataset, participant, day, metric, unit), values in sorted(grouped.items()):
        panel.append({
            "dataset_id": dataset,
            "participant_id": participant,
            "date": day,
            "metric_code": metric,
            "value": statistics.median(item["value"] for item in values),
            "unit": unit,
            "observation_count": len(values),
            "aggregation": "daily_median",
            "source_rows": json.dumps([item["source_row"] for item in values], separators=(",", ":")),
            "synthetic": "false",
            "clinical_validation": "false",
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panel[0]) if panel else [
            "dataset_id", "participant_id", "date", "metric_code", "value", "unit",
            "observation_count", "aggregation", "source_rows", "synthetic", "clinical_validation"])
        writer.writeheader()
        writer.writerows(panel)
    return {"output": str(out), "panel_rows": len(panel),
            "participants": len({row["participant_id"] for row in panel}),
            "datasets": sorted({row["dataset_id"] for row in panel}),
            "source_ingest_counts": counts,
            "purpose": "Research engineering only; observations are not user records or clinical ground truth."}


def load_panel(path: Path, metric: str) -> dict[tuple[str, str], dict[date, float]]:
    panel: dict[tuple[str, str], dict[date, float]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset_id", "participant_id", "date", "metric_code", "value", "unit"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Panel is missing columns: {sorted(missing)}")
        for row in reader:
            if row["metric_code"] != metric:
                continue
            if row.get("synthetic", "false").casefold() == "true":
                continue
            try:
                day = date.fromisoformat(row["date"][:10])
                value = float(row["value"])
            except (ValueError, TypeError):
                continue
            if math.isfinite(value):
                key = (row["dataset_id"], row["participant_id"])
                if day in panel[key]:
                    raise ValueError(f"Duplicate daily target for {key} on {day}; deduplicate before training.")
                panel[key][day] = value
    return panel


def make_forecast_pairs(series: dict[date, float], history_days: int, horizon: int) -> list[tuple[date, list[float], float]]:
    by_day = dict(series)
    result = []
    for forecast_origin in sorted(by_day):
        target_day = forecast_origin + timedelta(days=horizon)
        history_days_list = [forecast_origin - timedelta(days=lag) for lag in range(history_days - 1, -1, -1)]
        if target_day not in by_day or any(day not in by_day for day in history_days_list):
            continue
        history = [by_day[day] for day in history_days_list]
        result.append((target_day, history, by_day[target_day]))
    return result


def metrics(y: list[float], pred: list[float], naive: list[float]) -> dict[str, float | None]:
    if not y:
        return {"n": 0, "mae": None, "rmse": None, "mase": None}
    mae = statistics.mean(abs(a - b) for a, b in zip(y, pred))
    rmse = math.sqrt(statistics.mean((a - b) ** 2 for a, b in zip(y, pred)))
    scale = statistics.mean(abs(a - b) for a, b in zip(y, naive))
    mase = mae / scale if scale > 0 else None
    return {"n": len(y), "mae": mae, "rmse": rmse, "mase_vs_last_value": mase}


def forecast_eval(path: Path, metric: str, min_history: int, horizon: int, test_fraction: float) -> dict[str, Any]:
    data = load_panel(path, metric)
    reports = []
    for key, series in sorted(data.items()):
        pairs = make_forecast_pairs(series, min_history, horizon)
        if len(pairs) < 3:
            continue
        cut = max(1, min(len(pairs) - 1, int(len(pairs) * (1 - test_fraction))))
        test = pairs[cut:]
        y = [row[2] for row in test]
        naive = [row[1][0] for row in test]
        rolling = [statistics.median(row[1]) for row in test]
        reports.append({
            "dataset_id": key[0], "participant_id": key[1], "train_pairs": cut,
            "test_pairs": len(test), "last_value": metrics(y, naive, naive),
            "rolling_median": metrics(y, rolling, naive),
            "warning": "Per-participant chronological backtest; not unseen-person external validation.",
        })
    return {
        "metric_code": metric, "horizon_days": horizon, "min_history_days": min_history,
        "split": "per-participant chronological holdout", "participants_evaluated": len(reports),
        "results": reports,
        "limitations": [
            "Only daily panels with consecutive dates are scored; missing days are not imputed.",
            "No model weights are fitted: these are naive benchmarks required before ML.",
            "Cross-dataset results are not pooled because devices/labels may not be comparable.",
            "This evaluation does not establish clinical or product performance.",
        ],
    }


def robust_anomaly(path: Path, metric: str, window: int, min_baseline: int, threshold: float) -> dict[str, Any]:
    data = load_panel(path, metric)
    results = []
    for (dataset, participant), series in sorted(data.items()):
        ordered = sorted(series.items())
        candidates = []
        for index, (day, value) in enumerate(ordered):
            history = [prior_value for _, prior_value in ordered[max(0, index-window):index]]
            if len(history) < min_baseline:
                continue
            center = statistics.median(history)
            mad = statistics.median(abs(item-center) for item in history)
            if mad <= 1e-12:
                candidates.append({"date": day.isoformat(), "status": "ABSTAIN_ZERO_MAD", "value": value,
                                   "baseline_median": center, "baseline_days": len(history)})
                continue
            robust_z = (value-center)/(1.4826*mad)
            candidates.append({"date": day.isoformat(), "status": "CANDIDATE" if abs(robust_z) >= threshold else "NO_CANDIDATE",
                               "value": value, "baseline_median": center, "baseline_mad": mad,
                               "robust_z": robust_z, "baseline_days": len(history),
                               "synthetic": False, "user_alert_sent": False})
        results.append({"dataset_id": dataset, "participant_id": participant, "candidates": candidates})
    return {"metric_code": metric, "window_days": window, "minimum_baseline_observations": min_baseline,
            "threshold": threshold, "status": "SHADOW_CANDIDATES_ONLY", "results": results,
            "limitations": [
                "Threshold is an exploratory statistical rule, not a validated health-event detector.",
                "Candidate frequency is not precision or false-positive rate without adjudicated labels.",
                "Autocorrelation, device changes, weekday effects and quality flags are not modeled by this simple baseline.",
                "No proactive messages are sent.",
            ]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 5 public-data normalization and offline baselines.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-panel", help="Aggregate normalized long observations into daily medians.")
    build.add_argument("--input", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    forecast = sub.add_parser("forecast", help="Run per-person chronological naive forecast benchmarks.")
    forecast.add_argument("--panel", type=Path, required=True)
    forecast.add_argument("--metric", required=True)
    forecast.add_argument("--horizon", type=int, default=1)
    forecast.add_argument("--min-history", type=int, default=7)
    forecast.add_argument("--test-fraction", type=float, default=0.3)
    forecast.add_argument("--output", type=Path, required=True)
    anomaly = sub.add_parser("anomaly", help="Generate offline robust-MAD candidates only.")
    anomaly.add_argument("--panel", type=Path, required=True)
    anomaly.add_argument("--metric", required=True)
    anomaly.add_argument("--window", type=int, default=28)
    anomaly.add_argument("--min-baseline", type=int, default=14)
    anomaly.add_argument("--threshold", type=float, default=3.0)
    anomaly.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build-panel":
        result = build_panel(args.input, args.output)
    elif args.command == "forecast":
        if not 0 < args.test_fraction < 1 or args.horizon < 1 or args.min_history < args.horizon:
            parser.error("Invalid forecast split, horizon, or minimum history")
        result = forecast_eval(args.panel, args.metric, args.min_history, args.horizon, args.test_fraction)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        result = robust_anomaly(args.panel, args.metric, args.window, args.min_baseline, args.threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.command == "build-panel" else json.dumps({"output": str(args.output), "metric": args.metric}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
