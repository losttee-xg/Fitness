from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


SUBSETS = {"normal", "handwriting", "poor", "rotated", "tables", "mixed"}


def load_manifest(root: Path) -> list[dict[str, Any]]:
    benchmark = root / "ClinOCR-Bench"
    lookup_path = benchmark / "oneshot_lookup.csv"
    roles: dict[str, str] = {}
    if lookup_path.is_file():
        with lookup_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                subset = row.get("subset") or ""
                template = row.get("template") or ""
                sample = row.get("sample") or ""
                role = row.get("role") or "unknown"
                if subset and template and sample:
                    doc_id = f"template_{template}_sample_{sample}_{subset}"
                    roles[doc_id] = role
    items = []
    for subset in sorted(SUBSETS):
        scan_dir = benchmark / "scans" / subset
        gt_dir = benchmark / "ground_truth" / subset
        for scan in sorted(scan_dir.glob("*.jpg")):
            truth = gt_dir / f"{scan.stem}.txt"
            if not truth.is_file():
                continue
            role = roles.get(scan.stem, "unknown")
            items.append({
                "doc_id": scan.stem,
                "subset": subset,
                "split": "train" if role in {"exemplar", "train"} else "test" if role in {"eval", "test"} else role,
                "image_path": str(scan),
                "ground_truth_path": str(truth),
            })
    return items


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _split_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        subset = record["subset"]
        split = record["split"]
        summary.setdefault(subset, {}).setdefault(split, 0)
        summary[subset][split] += 1
    return summary


def _sequence_edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def score_text(reference: str, hypothesis: str) -> dict[str, Any]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    char_err = edit_distance(ref, hyp)
    ref_words = ref.split()
    hyp_words = hyp.split()
    word_err = _sequence_edit_distance(ref_words, hyp_words)
    return {
        "char_error_rate": char_err / max(len(ref), 1),
        "word_error_rate": word_err / max(len(ref_words), 1),
        "exact_match": ref == hyp,
        "reference_chars": len(ref),
        "reference_words": len(ref_words),
    }


def score_run(records: list[dict[str, Any]], hypotheses_path: Path) -> dict[str, Any]:
    hypotheses = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    by_id = {item["doc_id"]: item for item in hypotheses}
    selected = [record for record in records if record["split"] == "test"]
    per_subset: dict[str, list[dict[str, Any]]] = {}
    missing = []
    for record in selected:
        doc_id = record["doc_id"]
        hypothesis = by_id.get(doc_id)
        if not hypothesis or not isinstance(hypothesis.get("transcription"), str):
            missing.append(doc_id)
            continue
        reference = Path(record["ground_truth_path"]).read_text(encoding="utf-8")
        score = score_text(reference, hypothesis["transcription"])
        per_subset.setdefault(record["subset"], []).append(score)
    all_scores = [score for scores in per_subset.values() for score in scores]
    def mean(key: str, scores: list[dict[str, Any]]) -> float | None:
        return sum(item[key] for item in scores) / len(scores) if scores else None
    return {
        "evaluated": len(all_scores),
        "expected_test_items": len(selected),
        "expected_test_items_for_run": len(by_id),
        "coverage": len(all_scores) / len(selected) if selected else 0,
        "missing_hypothesis_ids": missing,
        "macro_char_error_rate": mean("char_error_rate", all_scores),
        "macro_word_error_rate": mean("word_error_rate", all_scores),
        "exact_match_rate": mean("exact_match", all_scores),
        "by_subset": {subset:{
            "count":len(scores),"char_error_rate":mean("char_error_rate",scores),
            "word_error_rate":mean("word_error_rate",scores),"exact_match_rate":mean("exact_match",scores)
        } for subset,scores in sorted(per_subset.items())},
    }


def build_report(root: Path, output: Path) -> dict[str, Any]:
    records = load_manifest(root)
    report = {
        "benchmark": "ClinOCR-Bench",
        "synthetic": True,
        "license": "MIT",
        "mode": "dataset_inventory_only_no_model_calls",
        "records_found": len(records),
        "counts_by_subset_split": _split_summary(records),
        "license_file": str(root / "ClinOCR-Bench" / "LICENSE"),
        "release_zip_sha256": "ce1d231138050abf7f458ba5e6bd75c6ee2f3832b72f22296843da4c5ba45457",
        "records": records,
        "model_scores": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path)
    parser.add_argument("--live-gpt4o", action="store_true")
    parser.add_argument("--allow-billable-tests", action="store_true")
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()
    if args.live_gpt4o and not args.allow_billable_tests:
        parser.error("Pass --allow-billable-tests to explicitly opt into billable synthetic-image requests")
    if args.live_gpt4o and args.hypotheses:
        parser.error("Use either --live-gpt4o or --hypotheses, not both")
    report = build_report(args.root, args.output)
    if args.live_gpt4o:
        import base64
        import httpx

        api_key=os.environ.get("RALLY_OCR_API_KEY")
        base_url=os.environ.get("RALLY_OCR_BASE_URL","https://api.pinkyne.com/v1").rstrip("/")
        if not api_key:
            parser.error("--live-gpt4o requires RALLY_OCR_API_KEY; no key is read from XDrama or source files")
        if args.max_images < 1 or args.max_images > 328:
            parser.error("--max-images must be from 1 to 328")
        if not args.allow_billable_tests:
            parser.error("Pass --allow-billable-tests to explicitly opt into billable synthetic-image requests")
        hypotheses=[]
        for record in report["records"]:
            if record["split"] != "test":
                continue
            if len(hypotheses) >= args.max_images:
                break
            image_bytes=Path(record["image_path"]).read_bytes()
            encoded=base64.b64encode(image_bytes).decode("ascii")
            prompt="Transcribe all legible text verbatim, preserving line breaks and spelling. Return transcription in the required JSON object. Do not infer or normalize any content."
            response=httpx.post(f"{base_url}/chat/completions",headers={"Authorization":f"Bearer {api_key}"},timeout=90,json={
                "model":"gpt-4o","temperature":0,"max_tokens":6000,
                "messages":[{"role":"user","content":[{"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{encoded}","detail":"high"}}]}],
                "response_format":{"type":"json_schema","json_schema":{"name":"ocr_transcription","strict":True,"schema":{
                    "type":"object","properties":{"transcription":{"type":"string"}},
                    "required":["transcription"],"additionalProperties":False}}}})
            response.raise_for_status()
            data=response.json()
            content=data["choices"][0]["message"]["content"]
            payload=json.loads(content) if isinstance(content,str) else {"transcription":""}
            hypotheses.append({"doc_id":record["doc_id"],"transcription":payload.get("transcription",""),"usage":data.get("usage")})
        hypothesis_path=args.output.with_name(args.output.stem+"-hypotheses.json")
        hypothesis_path.write_text(json.dumps(hypotheses,ensure_ascii=False,indent=2),encoding="utf-8")
        report["mode"]="live_gpt4o_synthetic_test_only"
        report["hypotheses_path"]=str(hypothesis_path)
        report["model_scores"]=score_run(report["records"],hypothesis_path)
        report["model_scores"]["coverage_for_run"] = report["model_scores"]["evaluated"] / len(hypotheses) if hypotheses else 0
        report["total_usage"]={"requests":len(hypotheses),"total_tokens":sum(item.get("usage",{}).get("total_tokens",0) for item in hypotheses)}
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    elif args.hypotheses:
        report["mode"] = "scored_transcription_hypotheses"
        report["model_scores"] = score_run(report["records"], args.hypotheses)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
