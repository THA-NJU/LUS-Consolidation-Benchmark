#!/usr/bin/env python3
"""Copy five test images from distinct patients for checkpoint inference."""
import argparse
import json
import shutil
from pathlib import Path

from prepare_validation_subsets import inventory, choose, sha256


def prepare(data_root, size, output):
    data_root = Path(data_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if size not in (224, 512):
        raise ValueError('size must be 224 or 512')
    if output.exists():
        raise FileExistsError('Output already exists; choose a new directory')
    selected = choose(inventory(data_root, 'test'), 5, size)
    records = []
    for patient, image, mask in selected:
        records.append({'patient': patient, 'image_id': image.stem,
                        'image': image.name, 'mask': mask.name,
                        'image_sha256': sha256(image), 'mask_sha256': sha256(mask)})
    for folder in ('images', 'masks'):
        (output / 'test' / folder).mkdir(parents=True)
    for (_, image, mask), row in zip(selected, records):
        for source, folder, key in ((image, 'images', 'image'), (mask, 'masks', 'mask')):
            dest = output / 'test' / folder / source.name
            shutil.copyfile(source, dest)
            if sha256(dest) != row[key + '_sha256']:
                raise RuntimeError(f'Input changed during copying: {source}')
    (output / 'case_manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'{size}: copied five test images from five patients to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--size', type=int, choices=(224, 512), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_root, args.size, args.output)


if __name__ == '__main__':
    main()
