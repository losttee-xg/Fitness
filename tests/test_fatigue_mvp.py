from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "app" / "scripts" / "train_fatigue_mvp.py"
SPEC = importlib.util.spec_from_file_location("train_fatigue_mvp", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fatigue_target_uses_prior_sensor_days_only(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    write_csv(root / "day_activity_features.csv", [
        {"user_id":"u1","date":"2024-01-01","steps__sum":"100","heart_rate__mean":"60"},
        {"user_id":"u1","date":"2024-01-02","steps__sum":"200","heart_rate__mean":"62"},
        {"user_id":"u1","date":"2024-01-03","steps__sum":"99999","heart_rate__mean":"200"},
        {"user_id":"u1","date":"2024-01-04","steps__sum":"88888","heart_rate__mean":"210"},
    ])
    write_csv(root / "daily_surveys_one_click_responses.csv", [{
        "user_id":"u1","day_occurrence":"1","one_click_submitted_datetime":"2024-01-04 08:00:00",
        "one_click_response":"yes","survey_start_datetime":"2024-01-04 08:00:00",
        "survey_end_datetime":"2024-01-04 08:01:00","tired_yesterday":"quite_a_bit",
    }])
    result = MODULE.build_examples(root, tmp_path / "out.csv", lookback_days=2, min_sensor_features=1)
    with (tmp_path / "out.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert result["rows"] == 1
    assert rows[0]["fatigue_date"] == "2024-01-03"
    assert rows[0]["feature_window_end_exclusive"] == "2024-01-03"
    assert rows[0]["steps__sum"] == "150.0"
    assert rows[0]["target_category"] == "quite_a_bit"


def test_unknown_and_negative_responses_are_not_silently_labeled(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    write_csv(root / "day_activity_features.csv", [{"user_id":"u1","date":"2024-01-01","steps__sum":"100"}])
    write_csv(root / "daily_surveys_one_click_responses.csv", [{
        "user_id":"u1","day_occurrence":"1","one_click_submitted_datetime":"2024-01-02 08:00:00",
        "one_click_response":"no","survey_start_datetime":"2024-01-02 08:00:00",
        "survey_end_datetime":"2024-01-02 08:01:00","tired_yesterday":"unknown",
    }])
    result = MODULE.build_examples(root, tmp_path / "out.csv", lookback_days=1, min_sensor_features=1)
    assert result["rows"] == 0
