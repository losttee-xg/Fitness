from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from scripts.task5_research_pipeline import build_panel, make_forecast_pairs, robust_anomaly


class Task5PipelineTests(unittest.TestCase):
    def test_forecast_pairs_use_only_history_before_target(self):
        start = date(2026, 1, 1)
        series = {start + timedelta(days=i): float(i) for i in range(10)}
        pairs = make_forecast_pairs(series, history_days=3, horizon=1)
        target_day, history, target = pairs[0]
        self.assertEqual(history, [0.0, 1.0, 2.0])
        self.assertEqual(target, 3.0)
        self.assertEqual(target_day, start + timedelta(days=3))

    def test_forecast_pairs_skip_missing_calendar_days(self):
        start = date(2026, 1, 1)
        series = {start: 1.0, start + timedelta(days=1): 2.0,
                  start + timedelta(days=3): 4.0, start + timedelta(days=4): 5.0}
        self.assertEqual(make_forecast_pairs(series, history_days=2, horizon=1), [])

    def test_daily_panel_preserves_dataset_and_participant_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "in.csv"
            source.write_text(
                "dataset_id,participant_id,date,metric_code,value,unit\n"
                "ds-a,p1,2026-01-01,HR,60,bpm\n"
                "ds-a,p1,2026-01-01,HR,62,bpm\n"
                "ds-b,p1,2026-01-01,HR,80,bpm\n", encoding="utf-8")
            output = root / "panel.csv"
            result = build_panel([source], output)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(result["panel_rows"], 2)
            self.assertEqual({row["dataset_id"] for row in rows}, {"ds-a", "ds-b"})
            self.assertEqual([float(row["value"]) for row in rows], [61.0, 80.0])

    def test_jsonl_relative_timestamps_are_not_converted_to_calendar_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "research.jsonl"
            source.write_text(
                json.dumps({"dataset_id":"lab","participant_id":"p1","metric_code":"HR",
                            "value":70,"unit":"bpm","observed_at_utc":None,
                            "source_timestamp":12.0,"quality_flags":["relative_seconds_not_utc"]}) + "\n" +
                json.dumps({"dataset_id":"wear","participant_id":"p2","metric_code":"HR",
                            "value":70,"unit":"bpm","observed_at_utc":"2026-01-01T23:00:00-05:00",
                            "source_timestamp":1767326400,"quality_flags":[]}) + "\n",
                encoding="utf-8")
            output = root / "panel.csv"
            result = build_panel([source], output)
            self.assertEqual(result["source_ingest_counts"]["skipped_missing_calendar_date"], 1)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["date"], "2026-01-02")

    def test_anomaly_is_shadow_only_and_abstains_when_mad_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "panel.csv"
            lines = ["dataset_id,participant_id,date,metric_code,value,unit,synthetic"]
            start = date(2026, 1, 1)
            for i in range(16):
                lines.append(f"ds,p1,{start + timedelta(days=i)},RHR,{60 if i < 15 else 75},bpm,false")
            source.write_text("\n".join(lines), encoding="utf-8")
            output = robust_anomaly(source, "RHR", window=14, min_baseline=7, threshold=3.0)
            self.assertEqual(output["status"], "SHADOW_CANDIDATES_ONLY")
            self.assertFalse(any(item.get("user_alert_sent", False)
                                 for person in output["results"] for item in person["candidates"]))
            self.assertTrue(any(item["status"] == "ABSTAIN_ZERO_MAD"
                                for person in output["results"] for item in person["candidates"]))

    def test_synthetic_rows_are_excluded_from_anomaly_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "panel.csv"
            source.write_text(
                "dataset_id,participant_id,date,metric_code,value,unit,synthetic\n"
                "ds,p1,2026-01-01,RHR,60,bpm,false\n"
                "ds,p1,2026-01-02,RHR,61,bpm,false\n"
                "ds,p1,2026-01-03,RHR,200,bpm,true\n", encoding="utf-8")
            output = robust_anomaly(source, "RHR", window=14, min_baseline=1, threshold=2)
            result_days = [item["date"] for person in output["results"] for item in person["candidates"]]
            self.assertNotIn("2026-01-03", result_days)


if __name__ == "__main__":
    unittest.main()
