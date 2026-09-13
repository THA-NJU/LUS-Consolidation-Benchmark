#!/usr/bin/env python3
"""Capture Git or source-snapshot provenance without requiring Git metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", "outputs", "runs", "wandb",
    "checkpoints", "pretrained", "Evaluation", "evaluation_results",
}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".pyc", ".log"}


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        yield path, relative


def snapshot_record(root: Path) -> dict[str, object]:
    files = []
    tree = hashlib.sha256()
    for path, relative in source_files(root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name = relative.as_posix()
        files.append({"path": name, "sha256": digest, "bytes": path.stat().st_size})
        tree.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {
        "source_form": "source_snapshot",
        "provenance_status": "snapshot_only",
        "file_count": len(files),
        "tree_sha256": tree.hexdigest(),
        "files": files,
    }


def git_record(root: Path) -> dict[str, object] | None:
    try:
        inside = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if inside != "true":
            return None
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "source_form": "git_checkout",
        "provenance_status": "verified_git_commit",
        "commit": commit,
        "dirty": dirty,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declared-upstream-revision")
    args = parser.parse_args()
    root = args.source.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Source directory does not exist: {root}")
    record = git_record(root) or snapshot_record(root)
    record["source_name"] = root.name
    if args.declared_upstream_revision:
        record["declared_upstream_revision"] = args.declared_upstream_revision
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
