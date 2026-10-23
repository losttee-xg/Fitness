from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "app" / "scripts" / "analyze_task5_habits.py"
SPEC = importlib.util.spec_from_file_location("analyze_task5_habits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_habit_association_keeps_participants_separate_and_requires_pairs(tmp_path: Path) -> None:
    path = tmp_path / "panel.csv"
    rows = []
    for p in ("p1", "p2"):
        for day, steps, rhr in (("2024-01-01", 1000, 65), ("2024-01-02", 2000, 64),
                                ("2024-01-03", 3000, 63), ("2024-01-04", 4000, 62),
                                ("2024-01-05", 5000, 61), ("2024-01-06", 6000, 60)):
            rows.extend([
                {"dataset_id":"lifesnaps_rais","participant_id":p,"date":day,"metric_code":"STEPS","value":str(steps),"unit":"steps","synthetic":"False"},
                {"dataset_id":"lifesnaps_rais","participant_id":p,"date":day,"metric_code":"RHR","value":str(rhr),"unit":"bpm","synthetic":"False"},
            ])
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=MODULE.associations(path)
    rhr_results = [item for item in result["results"] if item["outcome_metric"] == "RHR"]
    assert all(item["paired_observations"] >= 5 for item in rhr_results)
    next_day=next(item for item in rhr_results if item["lag_days"]==1)
    assert next_day["participants_with_5plus_pairs"]==2
    assert "non-causal" in next_day["interpretation"]
