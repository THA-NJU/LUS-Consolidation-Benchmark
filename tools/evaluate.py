#!/usr/bin/env python3
"""Unified selector for released checkpoint-evaluation backends.

Evaluation backends have family-specific checkpoint layouts.  This command
selects the canonical backend and forwards arguments after ``--`` unchanged.
Use ``--show-command`` to verify the resolved backend before a GPU run.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lusbench.registry import ROUTES, find_route  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--size", type=int, choices=(224, 512))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show-command", action="store_true")
    parser.add_argument("--sample", "--smoke", dest="smoke", action="store_true", help="Use the shared five-patient adapter; available for all 36 models on either track")
    parser.add_argument("--pth", type=Path)
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("forwarded", nargs=argparse.REMAINDER,
                        help="Backend-specific arguments after --")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print(f"{'model':<28} {'size':<5} evaluator")
        for route in ROUTES:
            print(f"{route.model:<28} {route.size:<5} {route.evaluation_script or '-'}")
        return 0
    if not args.model or not args.size:
        raise SystemExit("--model and --size are required unless --list is used")
    route = find_route(args.model, args.size)
    if args.smoke:
        if not all((args.pth, args.subset, args.out)):
            raise SystemExit('--sample requires --pth, --subset and --out')
        aliases = {'fpn_resnet34': 'Resnet34_fpn', 'deeplabv3plus_resnet34': 'Resnet34_DLV3', 'usfm_transfer': 'USFM'}
        command = [sys.executable, str(ROOT / 'tools/smoke.py'), '--models', aliases.get(args.model, args.model),
                   '--size', str(args.size), '--pth', str(args.pth), '--subset', str(args.subset), '--out', str(args.out)]
        if args.runtime_config:
            command += ['--runtime-config', str(args.runtime_config)]
        print(shlex.join(command), flush=True)
        return 0 if args.show_command else subprocess.run(command, cwd=ROOT).returncode
    if not route.evaluation_script:
        raise SystemExit(f"No standalone evaluator for {route.model}/{route.size}: {route.note}")
    forwarded = list(args.forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = [sys.executable, str(ROOT / route.evaluation_script), *forwarded]
    print(shlex.join(command), flush=True)
    if args.show_command:
        return 0
    os.chdir(ROOT)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
