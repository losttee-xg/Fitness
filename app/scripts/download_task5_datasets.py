from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.safe_extract_research_zip import safe_extract

DATASETS: dict[str, dict[str, Any]] = {
    "sleepaccel": {
        "title": "Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography",
        "version": "1.0.0",
        "url": "https://physionet.org/content/sleep-accel/get-zip/1.0.0/",
        "allowed_hosts": ["physionet.org", "www.physionet.org"],
        "license": "Open Data Commons Attribution License 1.0",
        "license_url": "https://physionet.org/content/sleep-accel/view-license/1.0.0/",
        "citation": "Walch et al., PhysioNet, DOI: 10.13026/hmhs-py35",
        "purpose": "Research/engineering evaluation; not clinical validation.",
        "limitations": ["31 participants", "one PSG lab night per included participant", "not a HealthKit export"],
    },
    "wearable_device": {
        "title": "Wearable Device Dataset from Induced Stress and Structured Exercise Sessions",
        "version": "1.0.1",
        "url": "https://physionet.org/content/wearable-device-dataset/get-zip/1.0.1/",
        "allowed_hosts": ["physionet.org", "www.physionet.org"],
        "license": "Open Data Commons Attribution License 1.0",
        "license_url": "https://physionet.org/content/wearable-device-dataset/view-license/1.0.1/",
        "citation": "Hongn et al., PhysioNet, DOI: 10.13026/he0v-tf17",
        "purpose": "Research/engineering evaluation; not clinical validation.",
        "limitations": ["Empatica E4, not Apple Watch", "structured sessions", "inspect data_constraints.txt"],
    },
    "pamap2": {
        "title": "PAMAP2 Physical Activity Monitoring",
        "version": "UCI-231",
        "url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "allowed_hosts": ["archive.ics.uci.edu"],
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": "Reiss and Stricker, Creating and Benchmarking a New Dataset for Physical Activity Monitoring, 2012",
        "purpose": "Activity/HR engineering evaluation; not clinical validation.",
        "limitations": ["9 participants", "research IMU data, not smartwatch export"],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset_id: str, root: Path, timeout: int = 60) -> dict[str, Any]:
    spec = DATASETS[dataset_id]
    raw_dir = root / "archives" / dataset_id
    extracted_dir = root / "extracted" / dataset_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty extraction directory: {extracted_dir}")
    archive = raw_dir / f"{dataset_id}.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")

    temp_path = raw_dir / f"{dataset_id}.download.part"
    final_host = ""
    try:
        for attempt in range(5):
            offset = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {"User-Agent": "FitnessResearchPipeline/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(spec["url"], headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    final_host = (urllib.parse.urlparse(response.geturl()).hostname or "").lower()
                    if final_host not in spec["allowed_hosts"]:
                        raise ValueError(f"Download redirected to unapproved host: {final_host}")
                    status = getattr(response, "status", 200)
                    if offset and status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError(f"Unexpected range response: {content_range}")
                        mode = "ab"
                    else:
                        mode = "wb"
                        offset = 0
                    total = int(response.headers.get("Content-Length", "0") or 0) + offset
                    with temp_path.open(mode) as tmp:
                        downloaded = offset
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            tmp.write(block)
                            downloaded += len(block)
                            if downloaded % (64 * 1024 * 1024) < len(block):
                                print(f"{dataset_id}: downloaded {downloaded:,} bytes" + (f" / {total:,}" if total else ""), flush=True)
                    current_size = temp_path.stat().st_size
                    if total and current_size < total:
                        raise TimeoutError(f"Download incomplete: {current_size} of {total} bytes")
                    break
            except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
                if attempt == 4:
                    raise
                print(f"{dataset_id}: attempt {attempt + 1} interrupted ({error}); retrying from saved byte offset", flush=True)
                time.sleep(min(2 ** attempt, 15))
        if not zipfile.is_zipfile(temp_path):
            raise ValueError(f"Downloaded content is not a ZIP archive; partial file retained at {temp_path}")
        with zipfile.ZipFile(temp_path) as archive_file:
            bad_member = archive_file.testzip()
            if bad_member:
                raise ValueError(f"ZIP CRC check failed: {bad_member}; archive retained at {temp_path}")
        archive_sha256 = sha256_file(temp_path)
        temp_path.replace(archive)
        temp_path = None
        extracted_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = safe_extract(archive, extracted_dir)
        return {
            "dataset_id": dataset_id,
            "title": spec["title"],
            "version": spec["version"],
            "source_url": spec["url"],
            "final_host": final_host,
            "license": spec["license"],
            "license_url": spec["license_url"],
            "citation": spec["citation"],
            "purpose": spec["purpose"],
            "limitations": spec["limitations"],
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_path": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "extracted_path": str(extracted_dir),
            "extracted_files": len(extracted_files),
            "files": extracted_files,
        }
    except (urllib.error.URLError, TimeoutError):
        raise
    except Exception:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Download allowlisted open research datasets and safely extract them.")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS), required=True,
                        help="Repeat to download multiple datasets. Gated/restricted datasets are intentionally excluded.")
    parser.add_argument("--root", type=Path, default=Path("data/research/task5"))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset_id in dict.fromkeys(args.dataset):
        results.append(download(dataset_id, args.root, args.timeout))
    manifest = {"purpose": "public research data for engineering/model feasibility only; not clinical validation or user data",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": results}
    manifest_path = args.root / "download_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior.setdefault("datasets", []).extend(results)
        prior["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest = prior
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "datasets": [
        {"dataset_id": row["dataset_id"], "archive_bytes": row["archive_bytes"],
         "extracted_files": row["extracted_files"], "sha256": row["archive_sha256"]} for row in results
    ]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
