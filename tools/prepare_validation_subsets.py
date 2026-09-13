#!/usr/bin/env python3
"""Lock independent per-track test cases and separate tiny training/validation sets.

No source images are modified. This script copies selected images/references into
a new validation directory. Test images are never used as training inputs.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, split):
    images = root / split / 'images'
    masks = root / split / 'masks'
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f'Expected {images} and {masks}')
    result = {}
    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in EXTENSIONS:
            continue
        patient = image.stem.split('_', 1)[0]
        if not patient.startswith('p') or not patient[1:].isdigit():
            raise ValueError(f'Cannot read patient ID from {image.name}')
        if (masks / image.name).is_file():
            mask = masks / image.name
        else:
            matches = [p for p in masks.glob(image.stem + '.*') if p.suffix.lower() in EXTENSIONS]
            if len(matches) != 1:
                raise ValueError(f'Expected one matching reference for {image.name}')
            mask = matches[0]
        result.setdefault(patient, []).append((image, mask))
    return result


def choose(patients, count, seed):
    if len(patients) < count:
        raise ValueError(f'Require {count} distinct patients; found {len(patients)}')
    generator = random.Random(seed)
    ids = sorted(generator.sample(sorted(patients), count))
    return [(patient, *generator.choice(patients[patient])) for patient in ids]


def preserve_existing_test(manifest, patients):
    records = json.loads(manifest.read_text())
    if len(records) != 5 or len({row['patient'] for row in records}) != 5:
        raise ValueError('Existing 512 test manifest must contain five distinct patients')
    output = []
    for row in records:
        matches = [(im, gt) for im, gt in patients.get(row['patient'], []) if im.stem == row['image_id']]
        if len(matches) != 1:
            raise ValueError(f'Existing 512 case absent from this test split: {row["image_id"]}')
        image, mask = matches[0]
        if sha256(image) != row['image_sha256'] or sha256(mask) != row['mask_sha256']:
            raise ValueError(f'Existing 512 case content changed: {image}')
        output.append((row['patient'], image, mask))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data512', type=Path, required=True)
    p.add_argument('--data224', type=Path, required=True)
    p.add_argument('--existing512-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; preserve it or choose a new directory')
    plans = {}
    for size, root in [(512, args.data512), (224, args.data224)]:
        root = root.expanduser().resolve()
        print(f'Checking {size} patient partitions: {root}', flush=True)
        inventories = {split: inventory(root, split) for split in ['train', 'val', 'test']}
        for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            overlap = set(inventories[first]) & set(inventories[second])
            if overlap:
                raise ValueError(f'{size}: patient leakage between {first}/{second}: {sorted(overlap)}')
        plans[size] = {
            'train': choose(inventories['train'], 2, 10000 + size),
            'val': choose(inventories['val'], 2, 20000 + size),
            'test': (preserve_existing_test(args.existing512_manifest, inventories['test'])
                     if size == 512 else choose(inventories['test'], 5, 224)),
        }
    # All selections and partition checks pass before any copies are written.
    report = {'tracks': {}, 'test_selection': '512 preserves previous manifest; 224 independently samples patients then images with seed 224',
              'training_selection': 'two training patients and two validation patients per track; test never used for training'}
    for size, plan in plans.items():
        track = args.output / str(size)
        records = {}
        for split, selected in plan.items():
            records[split] = []
            for folder in ['images', 'masks']:
                (track / split / folder).mkdir(parents=True, exist_ok=True)
            for patient, image, mask in selected:
                shutil.copyfile(image, track / split / 'images' / image.name)
                shutil.copyfile(mask, track / split / 'masks' / mask.name)
                records[split].append({'patient': patient, 'image_id': image.stem,
                                       'image': str(image), 'mask': str(mask),
                                       'image_sha256': sha256(image), 'mask_sha256': sha256(mask)})
        (track / 'case_manifest.json').write_text(json.dumps(records['test'], indent=2))
        (track / 'split_manifests.json').write_text(json.dumps(records, indent=2))
        report['tracks'][str(size)] = records
        print(f'{size}: train 2 patients, val 2 patients, test 5 patients selected', flush=True)
    (args.output / 'selection_report.json').write_text(json.dumps(report, indent=2))
    print(f'Prepared: {args.output}', flush=True)


if __name__ == '__main__':
    main()
