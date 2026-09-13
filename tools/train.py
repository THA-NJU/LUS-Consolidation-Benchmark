#!/usr/bin/env python3
"""Unified dispatcher for all released LUSBench training routes."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lusbench.registry import ROUTES, find_route, model_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--size", type=int, choices=(224, 512))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--run-mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--april-root", type=Path, default=ROOT / "third_party" / "april_medseg")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser.parse_args()


def list_routes() -> None:
    print(f"{model_count()} registered models")
    print(f"{'model':<28} {'size':<5} {'family':<16} {'status':<18} note")
    for route in ROUTES:
        print(f"{route.model:<28} {route.size:<5} {route.family:<16} {route.status:<18} {route.note}")


def smoke_args(route: str) -> list[str]:
    if route == "sam512":
        return ["--epochs", "1", "--patience", "1", "--max-train-samples", "8",
                "--max-val-samples", "4", "--max-test-samples", "4"]
    if route == "s2denet":
        return ["--epochs", "1", "--patience", "1", "--warmup-epochs", "0", "--workers", "0"]
    if route in {"yolo_family", "usfm224"}:
        return ["--epochs", "1", "--patience", "1", "--warmup-epochs", "0"]
    return []


def build_command(args: argparse.Namespace) -> list[str]:
    if not args.model or not args.size or not args.data_root:
        raise SystemExit("--model, --size, and --data-root are required unless --list is used")
    route = find_route(args.model, args.size)
    if not route.route or not route.script:
        raise SystemExit(f"{route.display_name} Size_{route.size} has no released training route: {route.note}")

    script = ROOT / route.script
    output = args.output_dir / str(args.size) / args.model
    python = sys.executable
    if route.route in {"april_general", "april_additional224", "april_additional512", "mamba512"}:
        kind = "april_general224" if route.route == "april_additional224" else route.route
        command = [python, str(ROOT / "tools" / "_run_configured.py"),
                   "--kind", kind, "--script", str(script), "--model", args.model,
                   "--size", str(args.size), "--data-root", str(args.data_root),
                   "--output-dir", str(output), "--run-mode", args.run_mode,
                   "--april-root", str(args.april_root)]
    elif route.route == "monai512":
        command = [python, str(script), "--april-root", str(args.april_root),
                   "--data-root", str(args.data_root), "--output-dir", str(output),
                   "--models", args.model, "--run-mode", args.run_mode]
    elif route.route == "monai224":
        family = "monai" if args.model.startswith("monai_") else "mamba"
        command = [python, str(script), "--april-root", str(args.april_root),
                   "--data-root", str(args.data_root), "--output-dir", str(output),
                   "--family", family, "--models", args.model, "--run-mode", args.run_mode]
    elif route.route == "rwkv512":
        command = [python, str(script), "--april-root", str(args.april_root),
                   "--data-root", str(args.data_root), "--output-dir", str(output),
                   "--models", args.model, "--run-mode", args.run_mode]
    elif route.route == "rwkv224":
        command = [python, str(script), "--april-root", str(args.april_root),
                   "--base512-script", str(ROOT / "experiments/rwkv/train_512.py"),
                   "--data-root", str(args.data_root), "--output-dir", str(output),
                   "--model", args.model, "--run-mode", args.run_mode]
    elif route.route == "sam512":
        command = [python, str(script), "--data-root", str(args.data_root), "--output-dir", str(output)]
    elif route.route == "sam224":
        command = [python, str(script), "--model", args.model, "--data-root", str(args.data_root),
                   "--output-dir", str(output), "--run-mode", args.run_mode]
    elif route.route == "sam3":
        command = [python, str(script), "--data-root", str(args.data_root),
                   "--output-dir", str(output), "--run-mode", args.run_mode]
    elif route.route in {"segformer_native", "yolo26s_native"}:
        kind = "segformer" if route.route == "segformer_native" else "yolo26s"
        command = [python, str(ROOT / "tools" / "_run_native_training.py"),
                   "--kind", kind, "--script", str(script), "--size", str(args.size),
                   "--data-root", str(args.data_root), "--output-dir", str(output),
                   "--run-mode", args.run_mode]
    elif route.route == "yolo_family":
        command = [python, str(script), "--model", args.model, "--size", str(args.size),
                   "--data-root", str(args.data_root), "--run-root", str(output / "runs"),
                   "--evaluation-root", str(output / "evaluation")]
    elif route.route == "s2denet":
        command = [python, str(script), "--size", str(args.size), "--data-root", str(args.data_root),
                   "--output-root", str(args.output_dir)]
    elif route.route == "usfm224":
        command = [python, str(script), "--data-root", str(args.data_root), "--output-root", str(output)]
    else:
        raise AssertionError(route.route)

    if args.run_mode == "smoke":
        command.extend(smoke_args(route.route))
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    return command


def main() -> int:
    args = parse_args()
    if args.list:
        list_routes()
        return 0
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    os.chdir(ROOT)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
