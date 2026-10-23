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


def read_panel(path: Path) -> dict[tuple[str, str], dict[date, dict[str, float]]]:
    result: dict[tuple[str, str], dict[date, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("dataset_id") != "lifesnaps_rais" or row.get("synthetic", "false").casefold() == "true":
                continue
            try:
                day, value = date.fromisoformat(row["date"][:10]), float(row["value"])
            except (ValueError, TypeError):
                continue
            if math.isfinite(value):
                result[(row["dataset_id"], row["participant_id"])][day][row["metric_code"]] = value
    return result


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    dx, dy = [a-mx for a in x], [b-my for b in y]
    denominator = math.sqrt(sum(a*a for a in dx) * sum(b*b for b in dy))
    return sum(a*b for a, b in zip(dx, dy)) / denominator if denominator > 0 else None


def associations(path: Path) -> dict[str, Any]:
    grouped = read_panel(path)
    pairs = [
        ("STEPS", "SLEEP_MINUTES_ASLEEP", 1, "next-day sleep minutes"),
        ("STEPS", "RHR", 1, "next-day resting heart rate"),
        ("STEPS", "HRV_RMSSD", 1, "next-day RMSSD"),
        ("STEPS", "RHR", 0, "same-day resting heart rate"),
    ]
    result = []
    for habit, outcome, lag, description in pairs:
        all_x: list[float] = []
        all_y: list[float] = []
        within_x: list[float] = []
        within_y: list[float] = []
        people = 0
        for (_, _), series in grouped.items():
            local_x: list[float] = []
            local_y: list[float] = []
            for day, values in series.items():
                target_day = day + timedelta(days=lag)
                if habit in values and target_day in series and outcome in series[target_day]:
                    local_x.append(values[habit])
                    local_y.append(series[target_day][outcome])
            if len(local_x) < 5:
                continue
            people += 1
            all_x.extend(local_x)
            all_y.extend(local_y)
            mx, my = statistics.mean(local_x), statistics.mean(local_y)
            within_x.extend(x-mx for x in local_x)
            within_y.extend(y-my for y in local_y)
        result.append({"habit_metric": habit, "outcome_metric": outcome, "lag_days": lag,
                       "description": description, "paired_observations": len(all_x), "participants_with_5plus_pairs": people,
                       "pooled_pearson_raw": pearson(all_x, all_y),
                       "pooled_pearson_after_participant_demeaning": pearson(within_x, within_y),
                       "interpretation": "exploratory association only; observational, confounded, non-causal; participant demeaning does not solve time-varying confounding"})
    return {"dataset_id": "lifesnaps_rais", "status": "descriptive_association_no_significance_claim",
            "results": result, "limitations": ["Pairs require both exact calendar dates; no imputation.",
            "Same-day behavior and next-day outcomes are associations, not causal effects.",
            "Do not generalize across participants or treat as personalized advice."]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute exploratory participant-aware habit-health associations.")
    parser.add_argument("--panel", type=Path, default=Path("dataset/harmonized/enriched/daily_panel.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/enriched/habit_associations.json"))
    args = parser.parse_args()
    report = associations(args.panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": report["results"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
