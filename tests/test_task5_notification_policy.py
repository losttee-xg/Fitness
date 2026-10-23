from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "app" / "scripts" / "simulate_task5_notification_policy.py"
SPEC = importlib.util.spec_from_file_location("simulate_task5_notification_policy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_policy_caps_frequency_and_never_marks_sent() -> None:
    rows = [{"date": f"2024-01-{day:02d}", "metric_code": "RHR"} for day in (1, 2, 4, 8, 10, 15)]
    selected = MODULE.select_candidates(rows, max_per_7_days=1, cooldown_days=3)
    dates = [row["date"] for row in selected]
    assert dates == ["2024-01-01", "2024-01-08", "2024-01-15"]
    assert all(row["notification_sent"] is False for row in selected)
