#!/usr/bin/env python3
"""Configure native SegFormer/YOLO26s scripts without editing release files."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("segformer", "yolo26s"), required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(224, 512), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--model-weights", type=Path, default=None)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("_lusbench_native_training", args.script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {args.script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(args.script.resolve().parent))
    spec.loader.exec_module(module)

    module.DATA_ROOT = args.data_root.expanduser().resolve()
    module.OUTPUT_ROOT = args.output_dir.expanduser().resolve()
    if args.run_mode == "smoke":
        module.EPOCHS = 1
        module.PATIENCE = 1
        module.WARMUP_EPOCHS = 0
        module.NUM_WORKERS = 0
        module.MAX_SAMPLES_PER_SPLIT = 8
        module.BATCH_SIZE = 2
        module.VAL_BATCH_SIZE = 2
        module.GRAD_ACCUM_STEPS = 1
        module.SMOKE_RANDOM_INIT = True
        module.SMOKE_CHECK = True
    else:
        module.EPOCHS = 600
        module.PATIENCE = 15
        module.WARMUP_EPOCHS = 10
        module.MAX_SAMPLES_PER_SPLIT = None

    if args.kind == "segformer":
        if args.base_model is not None:
            module.MODEL_NAME_OR_PATH = str(args.base_model.expanduser().resolve())
    else:
        module.CACHE_ROOT = (module.OUTPUT_ROOT / "_semantic_cache").resolve()
        if args.model_weights is not None:
            module.MODEL_WEIGHTS = args.model_weights.expanduser().resolve()
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
