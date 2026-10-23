from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_clinocr import build_report, load_manifest, score_text


def test_clinocr_manifest_pairs_scan_and_ground_truth_and_separates_splits(tmp_path: Path):
    root=tmp_path/"ClinOCR-Bench"
    for subset in ("normal","handwriting","poor","rotated","tables","mixed"):
        scan_dir=root/"scans"/subset
        truth_dir=root/"ground_truth"/subset
        scan_dir.mkdir(parents=True)
        truth_dir.mkdir(parents=True)
        name=f"template_1_sample_1_{subset}"
        (scan_dir/f"{name}.jpg").write_bytes(b"img")
        (truth_dir/f"{name}.txt").write_text("Patient result: 64 bpm",encoding="utf-8")
    lookup=root/"oneshot_lookup.csv"
    lookup.write_text("subset,template,sample,role\nnormal,1,1,eval\n",encoding="utf-8")
    manifest=load_manifest(tmp_path)
    assert len(manifest)==6
    assert next(row for row in manifest if row["subset"]=="normal")["split"]=="test"
    assert all(row["ground_truth_path"].endswith(".txt") for row in manifest)
    output=tmp_path/"result.json"
    report=build_report(tmp_path,output)
    assert report["records_found"]==6
    assert report["mode"]=="dataset_inventory_only_no_model_calls"
    assert json.loads(output.read_text(encoding="utf-8"))["model_scores"] is None


def test_ocr_text_scoring_reports_exact_and_error_rates():
    exact=score_text("Heart rate 64 bpm", "Heart rate 64 bpm")
    wrong=score_text("Heart rate 64 bpm", "Heart rate 65 bpm")
    assert exact["exact_match"] is True
    assert exact["char_error_rate"]==0
    assert exact["word_error_rate"]==0
    assert wrong["exact_match"] is False
    assert wrong["char_error_rate"]>0
    assert wrong["word_error_rate"]>0


def test_wer_uses_word_edits_not_character_edits():
    result=score_text("one two three", "one too three")
    assert result["word_error_rate"] == 1 / 3
    assert result["char_error_rate"] > 0
