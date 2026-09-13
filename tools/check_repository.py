#!/usr/bin/env python3
"""Static release audit; no GPU or model framework import is required."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from lusbench.registry import ROUTES, model_count  # noqa: E402


FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".dcm", ".pyc", ".log"}
FORBIDDEN_PARTS = {"__pycache__", "output", "outputs", "runs", "wandb", "checkpoints", "Evaluation"}
PRIVATE_PATH = re.compile(r"/(?:home/[^/]+|root/autodl-tmp|workspace/scratch)/")
EXPECTED_COUNTS = {
    "512": {"train": 16131, "val": 2111, "test": 1539},
    "224": {"train": 26017, "val": 2796, "test": 3557},
}
EXPECTED_PROTOCOL = {
    "max_epochs": 600,
    "early_stopping_patience": 15,
    "validation_interval": 1,
    "warmup_epochs": 10,
    "samples_per_epoch": None,
    "seed": 42,
    "balanced_sampler": False,
}


def check_files(errors: list[str]) -> None:
    hashes: dict[str, list[Path]] = defaultdict(list)
    paths = list(ROOT.rglob('*'))
    try:
        top = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'],
                             text=True, capture_output=True, check=True).stdout.strip()
        if Path(top).resolve() == ROOT.resolve():
            listed = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                                    text=True, capture_output=True, check=True).stdout
            paths = [ROOT / name for name in listed.split('\0') if name]
    except (OSError, subprocess.CalledProcessError):
        pass
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in {'.git', '.external', '.venvs'} for part in relative.parts):
            continue
        if relative.as_posix() == 'configs/runtime.local.json':
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden release file: {relative}")
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden release directory: {relative}")
        if path.suffix in {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            if PRIVATE_PATH.search(text):
                errors.append(f"development-machine absolute path: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
            except SyntaxError as exc:
                errors.append(f"syntax error: {relative}: {exc}")
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(relative)

    for paths in hashes.values():
        if len(paths) < 2:
            continue
        # License copies are intentional because each third-party subtree must
        # remain redistributable on its own.
        if all(path.name == "LICENSE" for path in paths):
            continue
        errors.append("duplicate file bytes: " + ", ".join(map(str, paths)))


def check_registry(errors: list[str]) -> None:
    if model_count() != 36:
        errors.append(f"model registry contains {model_count()} unique models, expected 36")
    keys = [(route.model, route.size) for route in ROUTES]
    if len(keys) != len(set(keys)):
        errors.append("duplicate model/size keys in registry")
    for route in ROUTES:
        if route.script and not (ROOT / route.script).is_file():
            errors.append(f"missing training backend: {route.model}/{route.size}: {route.script}")
        if route.evaluation_script and not (ROOT / route.evaluation_script).is_file():
            errors.append(f"missing evaluation backend: {route.model}/{route.size}: {route.evaluation_script}")
        if route.status == "evaluation_only" and (route.route or route.script):
            errors.append(f"evaluation-only route exposes training: {route.model}/{route.size}")


def check_protocol(errors: list[str]) -> None:
    protocol = json.loads((ROOT / "configs/benchmark_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("released_split_counts") != EXPECTED_COUNTS:
        errors.append("benchmark_protocol.json split counts changed")
    for key, expected in EXPECTED_PROTOCOL.items():
        actual = protocol.get("training", {}).get(key)
        if actual != expected:
            errors.append(f"protocol {key}={actual!r}, expected {expected!r}")
    manifest = json.loads((ROOT / "configs/dataset_manifest.json").read_text(encoding="utf-8"))
    actual_counts = {size: value["counts"] for size, value in manifest["sizes"].items()}
    if actual_counts != EXPECTED_COUNTS:
        errors.append("dataset_manifest.json split counts changed")


def check_april_subset(errors: list[str]) -> None:
    root = ROOT / "third_party/april_medseg"
    required = (
        "LICENSE", "PROVENANCE.md", "medseg/model_builder.py",
        "medseg/kernels/wkv/wkv_op.cpp", "medseg/kernels/wkv/wkv_cuda.cu",
        "medseg/models/networks/rwkv/u_rwkv.py",
        "medseg/models/networks/rwkv/rwkv_unet.py",
    )
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"APRIL subset missing {relative}")

    init_path = root / "medseg/models/networks/__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    registry = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_SPECIAL_ARCHS":
            registry = ast.literal_eval(node.value)
            break
    if not isinstance(registry, dict) or len(registry) != 17:
        errors.append("APRIL lazy registry must contain exactly 17 architectures")
        return
    for model, entry in registry.items():
        module_name, _ = entry
        path = root.joinpath(*module_name.split(".")).with_suffix(".py")
        if not path.is_file():
            errors.append(f"APRIL architecture module missing for {model}: {module_name}")


def main() -> int:
    errors: list[str] = []
    check_files(errors)
    check_registry(errors)
    check_protocol(errors)
    check_april_subset(errors)
    if errors:
        print("REPOSITORY CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    training_routes = sum(route.script is not None for route in ROUTES)
    evaluation_routes = sum(route.evaluation_script is not None for route in ROUTES)
    print(
        "REPOSITORY CHECK PASSED: "
        f"models={model_count()} routes={len(ROUTES)} "
        f"training_routes={training_routes} evaluation_routes={evaluation_routes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
