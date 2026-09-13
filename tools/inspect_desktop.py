#!/usr/bin/env python3
"""Inspect local checkpoints and prepare five 512 test cases; never run inference."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path


def select_cases(root, count=5, seed=42):
    images = root / 'test' / 'images'
    masks = root / 'test' / 'masks'
    files = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
    if len(files) != 1539:
        raise ValueError(f'Expected current 512 test inventory: 1539 images; found {len(files)} at {images}')
    groups = {}
    for p in files:
        m = re.match(r'^(p\d+)_', p.stem, re.I)
        if not m:
            raise ValueError(f'Cannot determine patient ID from {p.name}')
        groups.setdefault(m.group(1).lower(), []).append(p)
    if len(groups) != 11:
        raise ValueError(f'Expected 11 test patients, found {len(groups)}')
    rng = random.Random(seed)
    patients = sorted(rng.sample(sorted(groups), count))
    selected = []
    for patient in patients:
        image = rng.choice(groups[patient])
        matches = [p for p in masks.glob(image.stem + '.*') if p.is_file() and p.suffix.lower() in {'.png', '.tif', '.tiff', '.bmp'}]
        if len(matches) != 1:
            raise ValueError(f'Expected one reference mask for {image.name}, found {len(matches)}')
        selected.append({'patient': patient, 'image_id': image.stem, 'image': str(image.resolve()), 'mask': str(matches[0].resolve())})
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pth-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('./desktop_inspection'))
    p.add_argument('--prepare-dir', type=Path, help='Optional output root, e.g. ./outputs/validation; creates subset512/test only')
    args = p.parse_args()
    if not args.pth_root.is_dir():
        p.error(f'Pth directory does not exist: {args.pth_root}')
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'python': sys.executable, 'python_version': sys.version, 'pth_root': str(args.pth_root.resolve()), 'data_root': str(args.data_root.resolve()), 'files': [], 'errors': [], 'packages': {}, 'inference_executed': False}
    for name in ['torch', 'torchvision', 'monai', 'transformers', 'timm', 'ultralytics', 'segmentation-models-pytorch', 'mamba-ssm', 'numpy', 'Pillow']:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    with zipfile.ZipFile(args.out / 'desktop_sources.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for directory, dirs, names in os.walk(args.pth_root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules'})
            for name in sorted(names):
                f = Path(directory) / name
                try:
                    stat = f.stat()
                    rel = f.relative_to(args.pth_root).as_posix()
                    report['files'].append({'path': rel, 'bytes': stat.st_size, 'symlink': f.is_symlink()})
                    if f.suffix.lower() in {'.py', '.sh'} and stat.st_size <= 2_000_000 and not f.is_symlink():
                        z.write(f, 'Pth/' + rel)
                except OSError as exc:
                    report['errors'].append(str(exc))
                if len(report['files']) % 100 == 0:
                    print(f"Inspected {len(report['files'])} files (weights not read)", flush=True)
        try:
            cases = select_cases(args.data_root)
            for c in cases:
                c['image_sha256'] = hashlib.sha256(Path(c['image']).read_bytes()).hexdigest()
                c['mask_sha256'] = hashlib.sha256(Path(c['mask']).read_bytes()).hexdigest()
            report['selected_cases'] = cases
            report['selection'] = {'size': 512, 'split': 'test', 'count': 5, 'distinct_patients': 5, 'seed': 42}
            if args.prepare_dir:
                subset = args.prepare_dir / 'subset512'
                if subset.exists():
                    raise ValueError(f'{subset} already exists; refusing to overwrite a prior selection')
                for c in cases:
                    for key, folder in [('image', 'images'), ('mask', 'masks')]:
                        dest = subset / 'test' / folder / Path(c[key]).name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(c[key], dest)
                (subset / 'case_manifest.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
                report['prepared_subset'] = str(subset.resolve())
        except (OSError, ValueError) as exc:
            report['errors'].append('Subset preparation: ' + str(exc))
        z.writestr('report.json', json.dumps(report, indent=2))
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Done: {len(report['files'])} files inspected; {len(report['errors'])} errors. Return {args.out / 'desktop_sources.zip'}", flush=True)
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
