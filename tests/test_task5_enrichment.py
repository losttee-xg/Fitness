from __future__ import annotations

import csv
import importlib.util
from datetime import date
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "app" / "scripts" / "enrich_task5_features.py"
SPEC = importlib.util.spec_from_file_location("enrich_task5_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_survey_features_only_use_days_before_target(tmp_path: Path) -> None:
    obs = tmp_path / "observations.csv"
    labels = tmp_path / "labels.csv"
    out = tmp_path / "features.csv"
    write_csv(obs, [
        {"dataset_id":"lifesnaps_rais","participant_id":"p1","date":"2024-01-01","metric_code":"RHR","value":"60","unit":"bpm","synthetic":"False"},
        {"dataset_id":"lifesnaps_rais","participant_id":"p1","date":"2024-01-02","metric_code":"RHR","value":"64","unit":"bpm","synthetic":"False"},
        {"dataset_id":"lifesnaps_rais","participant_id":"p1","date":"2024-01-03","metric_code":"RHR","value":"100","unit":"bpm","synthetic":"False"},
        {"dataset_id":"lifesnaps_rais","participant_id":"p1","date":"2024-01-02","metric_code":"STEPS","value":"9000","unit":"steps","synthetic":"True"},
    ])
    write_csv(labels, [
        {"dataset_id":"lifesnaps_rais","participant_id":"p1","date":"2024-01-03","label_name":"STAI_STRESS_SCORE","label_value":"40","label_unit":"score_points","label_source":"stai.csv","label_status":"questionnaire_score_scored_not_clinical_diagnosis","source_row":"2","synthetic":"False"},
    ])
    report = MODULE.build_survey_features(obs, labels, out, lookback_days=2, min_observed_days=2)
    with out.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert report["rows"] == 1
    assert rows[0]["rhr_mean"] == "62.0"
    assert rows[0]["rhr_last"] == "64.0"
    assert rows[0]["steps_n_days"] == "0"
    assert rows[0]["rhr_n_days"] == "2"
    assert rows[0]["eligible_for_modeling"] == "True"


def test_unknown_or_non_questionnaire_targets_are_not_joined(tmp_path: Path) -> None:
    obs = tmp_path / "observations.csv"
    labels = tmp_path / "labels.csv"
    out = tmp_path / "features.csv"
    write_csv(obs, [{"dataset_id":"lifesnaps_rais","participant_id":"p1","date":"2024-01-01","metric_code":"RHR","value":"60","unit":"bpm","synthetic":"False"}])
    write_csv(labels, [{"dataset_id":"lifesnaps_rais","participant_id":"p1","date":"2024-01-02","label_name":"SEMA_TIRED","label_value":"1","label_unit":"binary_ema_flag","label_source":"SEMA","label_status":"self_report_flag_semantics_not_fatigue_scale","source_row":"2","synthetic":"False"}])
    report = MODULE.build_survey_features(obs, labels, out, lookback_days=7, min_observed_days=1)
    assert report["rows"] == 0


def test_catalog_is_explicitly_not_llm_generated(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    MODULE.write_catalog(path)
    assert '"created_by": "deterministic_project_metadata; not LLM-generated"' in path.read_text(encoding="utf-8")
