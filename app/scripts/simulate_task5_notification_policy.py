from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any



def policy_candidates(path: Path, max_per_7_days: int = 1, cooldown_days: int = 3) -> dict[str, Any]:
    if max_per_7_days < 1 or cooldown_days < 0:
        raise ValueError("max_per_7_days >= 1 and cooldown_days >= 0 required")
    grouped: dict[tuple[str, str, str], list[date]] = defaultdict(list)
    for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
        if row.get("status") != "CANDIDATE":
            continue
        grouped[(row.get("dataset_id", "lifesnaps_rais"), row["participant_id"], row.get("metric_code", ""))].append(date.fromisoformat(row["date"][:10]))
    # Current shadow anomaly files are participant-nested JSON, handled separately by caller conversion.
    return {"status": "policy_simulation_only", "groups": len(grouped), "policy": {"max_per_7_days": max_per_7_days, "cooldown_days": cooldown_days}, "limitations": ["No messages are generated or sent.", "Policy settings are placeholders; they require user research and prospective safety evaluation."]}


def select_candidates(records: list[dict[str, Any]], max_per_7_days: int = 1, cooldown_days: int = 3) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    recent: list[date] = []
    last_sent: date | None = None
    for record in sorted(records, key=lambda row: row["date"]):
        day = date.fromisoformat(record["date"][:10])
        recent = [prior for prior in recent if (day-prior).days < 7]
        if recent and (day-recent[-1]).days < cooldown_days:
            continue
        if len(recent) >= max_per_7_days:
            continue
        recent.append(day)
        last_sent = day
        selected.append({**record, "policy_status": "selected_in_simulation_only", "notification_sent": False})
    return selected


def simulate(json_paths: list[Path], output: Path, max_per_7_days: int, cooldown_days: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        metric = data.get("metric_code", path.stem)
        for group in data.get("results", []):
            key = (group.get("dataset_id", "lifesnaps_rais"), group.get("participant_id"))
            for item in group.get("candidates", []):
                if item.get("status") == "CANDIDATE":
                    grouped[key].append({"date": item["date"], "robust_z": item.get("robust_z"),
                                         "baseline_median": item.get("baseline_median"),
                                         "value": item.get("value"), "metric_code": metric})
    result = []
    for (dataset_id, participant_id), candidates in sorted(grouped.items()):
        chosen = select_candidates(candidates, max_per_7_days, cooldown_days)
        if chosen:
            result.append({"dataset_id": dataset_id, "participant_id": participant_id,
                           "candidates_total": len(candidates), "selected_total": len(chosen), "selected": chosen})
    report = {"status": "SIMULATION_ONLY_NO_NOTIFICATIONS_SENT",
              "policy": {"max_per_7_days": max_per_7_days, "cooldown_days": cooldown_days},
              "candidate_groups_with_selection": len(result), "candidate_total": sum(row["candidates_total"] for row in result),
              "selected_total": sum(row["selected_total"] for row in result), "groups": result,
              "limitations": ["This is a software-policy smoke test, not an evidence-based notification policy.",
                              "Candidates have no adjudicated event labels, so false-positive rate and user benefit are unknown.",
                              "No message content or notification is produced."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate a rate-limited policy over shadow anomaly candidates; sends nothing.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-7-days", type=int, default=1)
    parser.add_argument("--cooldown-days", type=int, default=3)
    args = parser.parse_args()
    report = simulate(args.input, args.output, args.max_per_7_days, args.cooldown_days)
    print(json.dumps({key: value for key, value in report.items() if key != "groups"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
