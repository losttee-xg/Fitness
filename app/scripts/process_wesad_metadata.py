from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def inspect_e4_zip(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "signals": [], "error": None}
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                sample = archive.read(name)[:8192].decode("utf-8", errors="replace").splitlines()
                rows = list(csv.reader(sample))
                result["signals"].append({"member": name, "preview_rows": len(rows),
                                          "first_rows": rows[:4]})
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        result["error"] = type(error).__name__
    return result


def inspect_wesad(root: Path, output: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    participants: list[dict[str, Any]] = []
    for folder in sorted(root.glob("S*")):
        if not folder.is_dir():
            continue
        participant = folder.name
        readme = folder / f"{participant}_readme.txt"
        participant_info: dict[str, Any] = {"participant_id": participant, "notes": []}
        if readme.exists():
            for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" in line and not line.startswith("###"):
                    key, value = line.split(":", 1)
                    participant_info[key.strip()] = value.strip()
                elif line.strip() and not line.startswith("###"):
                    participant_info["notes"].append(line.strip())
        participants.append(participant_info)

        questionnaire = folder / f"{participant}_quest.csv"
        if questionnaire.exists():
            with questionnaire.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
                for row_number, row in enumerate(csv.reader(stream, delimiter=";"), start=1):
                    for column, raw in enumerate(row, start=1):
                        value = number(raw)
                        if value is None:
                            continue
                        records.append({"dataset_id": "wesad", "dataset_version": "1.0.0",
                                        "participant_id": participant, "source_file": questionnaire.name,
                                        "source_row": row_number, "source_column": column,
                                        "source_metric": "questionnaire_numeric_item", "metric_code": None,
                                        "metric_variant": None, "raw_value": raw, "raw_unit": None,
                                        "value": value, "unit": None, "source_timestamp": None,
                                        "observed_at_utc": None,
                                        "quality_flags": ["questionnaire_item_mapping_required"]})
        e4_zip = folder / f"{participant}_E4_Data.zip"
        if e4_zip.exists():
            # List/read embedded text data safely; never unpickle participant objects.
            nested = inspect_e4_zip(e4_zip)
            for signal in nested["signals"]:
                member = signal["member"]
                rows = signal["first_rows"]
                # Empatica files use start epoch, sample rate and signal values.
                if len(rows) < 3:
                    continue
                try:
                    epoch = number(rows[0][0])
                    rate = number(rows[1][0])
                except (IndexError, TypeError):
                    epoch, rate = None, None
                if epoch is None or rate is None or rate <= 0:
                    continue
                signal_name = Path(member).stem.casefold()
                unit_by_signal = {"bvp": "a.u.", "eda": "uS", "temp": "C", "hr": "bpm", "ibi": "s", "acc": "g"}
                unit = unit_by_signal.get(signal_name)
                # Keep cadence metadata only here; raw signal rows remain in source files to avoid enormous expansion.
                records.append({"dataset_id": "wesad", "dataset_version": "1.0.0",
                                "participant_id": participant, "source_file": f"{e4_zip.name}:{member}",
                                "source_row": None, "source_column": None, "source_metric": signal_name,
                                "metric_code": "RS-01" if signal_name == "hr" else None,
                                "metric_variant": None, "raw_value": None, "raw_unit": unit,
                                "value": None, "unit": unit, "source_timestamp": epoch,
                                "observed_at_utc": None,
                                "quality_flags": ["signal_sampling_metadata_only", f"sampling_rate_hz:{rate}", "raw_samples_not_expanded"]})

    output.mkdir(parents=True, exist_ok=True)
    records_file = output / "wesad_normalized_metadata.jsonl"
    with records_file.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "dataset_id": "wesad", "dataset_version": "1.0.0",
        "purpose": "engineering ingestion test only; not clinical validation or user data",
        "license": "scientific non-commercial use only; attribution required",
        "required_citation": "Schmidt et al., Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection, ICMI 2018, DOI: 10.1145/3242969.3242985",
        "participant_count": len(participants), "participants": participants,
        "normalized_metadata_records": len(records), "normalized_file": records_file.name,
        "pkl_policy": "S10.pkl and similar participant pickle files were not opened or deserialized.",
        "raw_signal_policy": "Embedded E4 ZIP contents inspected only for metadata preview; high-rate samples were not flattened into the merged file.",
        "notes": ["Do not mix participant IDs across datasets.", "No personal user observations were created.",
                  "CSV questionnaire items remain unmapped pending review of the official data dictionary."]
    }
    report_file = output / "wesad_normalization_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"participants": len(participants), "records": len(records),
                      "normalized_file": str(records_file), "report": str(report_file)}, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect_wesad(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
