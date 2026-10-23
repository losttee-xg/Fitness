from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from rally_health.importer import import_workbooks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Rally Health Excel workbooks into normalized JSON catalogs.")
    parser.add_argument("--source", type=Path, default=Path.cwd(), help="Directory containing the five source workbooks")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "normalized", help="Output directory")
    args = parser.parse_args()
    report = import_workbooks(args.source, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
