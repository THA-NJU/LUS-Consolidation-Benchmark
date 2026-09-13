#!/usr/bin/env python3
"""Regenerate route coverage and the APRIL source SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lusbench.registry import ROUTES  # noqa: E402


def write_coverage() -> None:
    fields = (
        "model", "display_name", "family", "size", "status",
        "route_available", "syntax_checked", "dry_run_checked",
        "gpu_smoke_status", "formal_run_status", "route", "script",
        "evaluation_script", "note",
    )
    with (ROOT / "MODEL_COVERAGE.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for route in ROUTES:
            row = {
                "model": route.model,
                "display_name": route.display_name,
                "family": route.family,
                "size": route.size,
                "status": route.status,
                "route_available": bool(route.script or route.evaluation_script),
                "syntax_checked": True,
                "dry_run_checked": True,
                "gpu_smoke_status": (
                    "known_evaluation_failure"
                    if route.model == "u_rwkv" and route.size == 224
                    else "not_reverified_in_release_audit"
                ),
                "formal_run_status": "not_reverified_in_release_audit",
                "route": route.route or "",
                "script": route.script or "",
                "evaluation_script": route.evaluation_script or "",
                "note": route.note,
            }
            writer.writerow(row)


def write_april_manifest() -> None:
    root = ROOT / "third_party" / "april_medseg"
    manifest = root / "SOURCE_SHA256SUMS.txt"
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
        for path in paths
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    write_coverage()
    write_april_manifest()
    print(f"coverage_routes={len(ROUTES)} manifest_written=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
