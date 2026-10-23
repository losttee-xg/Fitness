from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, target: Path) -> list[dict[str, object]]:
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or not member.parts:
                raise ValueError(f"Unsafe archive path: {info.filename}")
            destination = (target / Path(*member.parts)).resolve()
            if target not in destination.parents and destination != target:
                raise ValueError(f"Archive path escapes extraction root: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, destination.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            extracted.append({"archive_path": info.filename, "bytes": info.file_size,
                              "sha256": sha256_file(destination), "path": str(destination)})
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    entries = safe_extract(args.archive, args.target)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"archive": str(args.archive), "archive_sha256": sha256_file(args.archive),
                                        "files": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"extracted_files": len(entries), "manifest": str(args.manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
