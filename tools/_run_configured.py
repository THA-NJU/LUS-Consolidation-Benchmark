#!/usr/bin/env python3
"""Compatibility adapter for reviewed scripts that still use module globals."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kind",
        choices=("april_general", "april_general224", "april_additional512", "mamba512"),
        required=True,
    )
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--size", type=int, choices=(224, 512), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    p.add_argument("--april-root", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.april_root.resolve()))
    spec = importlib.util.spec_from_file_location("_lusbench_configured_entry", args.script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {args.script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(args.script.resolve().parent))
    spec.loader.exec_module(module)
    module.MODELS_TO_RUN = [args.model]
    module.RUN_MODE = args.run_mode
    module.OUTPUT_ROOT = args.output_dir.resolve()
    module.PROJECT_ROOT = args.april_root.resolve()
    if args.run_mode == "smoke":
        module.NUM_WORKERS = 0
        if args.model == 'pvtb2_emcad':
            # One-batch diagnostics need a real update, not AMP scale calibration.
            module.USE_AMP = False
            print('Smoke PVT-EMCAD uses FP32; formal training retains its AMP setting.', flush=True)
        for spec in getattr(module, 'MODEL_SPECS', {}).values():
            if isinstance(spec, dict):
                spec['batch_size'] = min(int(spec.get('batch_size', 2)), 2)
                spec['grad_accum'] = 1
                spec['warmup_epochs'] = 0
        if hasattr(module, 'get_model_spec'):
            original_spec = module.get_model_spec
            def tiny_spec(name):
                spec = dict(original_spec(name))
                spec.update(batch_size=min(int(spec.get('batch_size', 2)), 2), grad_accum=1, warmup_epochs=0)
                return spec
            module.get_model_spec = tiny_spec
    if args.kind in {"april_general", "april_general224"}:
        # These trainers rebuild OUTPUT_ROOT from OUTPUT_BASE for each task.
        module.OUTPUT_BASE = args.output_dir.resolve()
        module.RUN_SIZES = [args.size]
        module.DATA_ROOTS = {args.size: args.data_root.resolve()}
    elif args.kind == "april_additional512":
        if args.size != 512:
            raise ValueError("april_additional512 accepts only Size_512")
        module.DATA_ROOT = args.data_root.resolve()
    else:
        if args.size != 512:
            raise ValueError("mamba512 adapter accepts only Size_512")
        module.DATA_ROOT = args.data_root.resolve()
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
