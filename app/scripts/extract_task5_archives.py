from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


MAX_MEMBER_BYTES = 10 * 1024 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_path(root: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    destination = (root / Path(*member.parts)).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"Archive member escapes extraction root: {member_name}")
    return destination


def extract_members(archive: zipfile.ZipFile, target: Path, include: callable) -> list[dict[str, object]]:
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for info in archive.infolist():
        if info.is_dir() or not include(info.filename):
            continue
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"Refusing unusually large member: {info.filename} ({info.file_size} bytes)")
        destination = safe_member_path(target, info.filename)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        copied = 0
        with archive.open(info) as source, destination.open("xb") as output:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                copied += len(block)
        if copied != info.file_size:
            raise IOError(f"Member size mismatch for {info.filename}: {copied} != {info.file_size}")
        entries.append({"archive_path": info.filename, "bytes": copied,
                        "sha256": digest.hexdigest(), "path": str(destination)})
    return entries


def extract_outer_nested(outer_path: Path, nested_name: str, target: Path,
                         include: callable) -> list[dict[str, object]]:
    with zipfile.ZipFile(outer_path) as outer:
        bad = outer.testzip()
        if bad:
            raise ValueError(f"Outer archive CRC failed: {bad}")
        matches = [info for info in outer.infolist() if info.filename == nested_name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {nested_name} in outer archive; got {len(matches)}")
        import tempfile
        temp_dir = target.parent / f".{target.name}_nested_work"
        temp_dir.mkdir(parents=True, exist_ok=True)
        nested_path = temp_dir / nested_name
        if nested_path.exists():
            raise FileExistsError(f"Refusing to overwrite nested archive work file: {nested_path}")
        info = matches[0]
        digest = hashlib.sha256()
        copied = 0
        with outer.open(info) as source, nested_path.open("xb") as output:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                copied += len(block)
        if copied != info.file_size:
            raise IOError(f"Nested archive size mismatch: {copied} != {info.file_size}")
        if not zipfile.is_zipfile(nested_path):
            raise ValueError(f"Nested member {nested_name} is not a valid ZIP")
        with zipfile.ZipFile(nested_path) as nested:
            bad = nested.testzip()
            if bad:
                raise ValueError(f"Nested archive CRC failed: {bad}")
            entries = extract_members(nested, target, include)
        nested_path.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        return [{"nested_archive": nested_name, "nested_archive_sha256": digest.hexdigest(),
                 "nested_archive_bytes": copied, "extracted": entries}]


def pamap_include(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".dat") or lowered.endswith(".pdf") or lowered.endswith(".txt") or lowered.endswith(".md")


def ppg_metadata_include(name: str) -> bool:
    lowered = name.lower()
    base = PurePosixPath(lowered).name
    return base.endswith("_activity.csv") or base.endswith("_quest.csv") or base in {
        "ppg_fieldstudy_readme.pdf", "readme.pdf"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely extract selected members from downloaded research ZIP archives.")
    parser.add_argument("--pamap-outer", type=Path)
    parser.add_argument("--ppg-outer", type=Path)
    parser.add_argument("--target-root", type=Path, default=Path("data/research/task5/extracted"))
    args = parser.parse_args()
    results = []
    if args.pamap_outer:
        target = args.target_root / "pamap2"
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {target}")
        results.extend(extract_outer_nested(args.pamap_outer, "PAMAP2_Dataset.zip", target, pamap_include))
    if args.ppg_outer:
        target = args.target_root / "ppg_dalia_metadata_only"
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {target}")
        results.extend(extract_outer_nested(args.ppg_outer, "data.zip", target, ppg_metadata_include))
    if not results:
        parser.error("Provide at least one outer ZIP archive")
    manifest = {"purpose": "safe research data inspection; not clinical validation",
                "ppg_policy": "Pickle and HDF5 sensor payloads were not extracted or loaded; only metadata CSV/PDF were selected.",
                "datasets": results}
    args.target_root.mkdir(parents=True, exist_ok=True)
    output = args.target_root / "extraction_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "datasets": [
        {"nested_archive": row["nested_archive"], "files_extracted": len(row["extracted"])} for row in results
    ]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
