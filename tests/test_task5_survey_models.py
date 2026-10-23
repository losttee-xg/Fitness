from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "app" / "scripts" / "evaluate_task5_survey_models.py"
SPEC = importlib.util.spec_from_file_location("evaluate_task5_survey_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_loader_excludes_identifiers_and_target_leakage(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    rows = []
    for i in range(12):
        rows.append({"dataset_id":"lifesnaps_rais","participant_id":f"p{i%6}","target_date":f"2024-01-{i+1:02d}",
                     "target_name":"STAI_STRESS_SCORE","target_value":str(40+i),"target_unit":"score_points",
                     "target_source":"stai.csv","feature_window_start":"2023-12-25","feature_window_end_exclusive":"2024-01-01",
                     "target_source_row":str(i),"temporal_rule":"past only","lookback_days":"7","synthetic":"False",
                     "clinical_validation":"False","eligible_for_modeling":"True","eligibility_rule":"x","feature_days_total":"5",
                     "rhr_mean":"60","rhr_n_days":"4"})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    x,y,groups,names=MODULE.load_examples(path,"STAI_STRESS_SCORE")
    assert len(x)==12
    assert set(names)=={"rhr_mean","rhr_n_days"}
    assert "target_value" not in names and "participant_id" not in names
    assert len(set(groups))==6
