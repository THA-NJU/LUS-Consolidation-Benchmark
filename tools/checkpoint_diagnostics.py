"""Compute native metrics from the actual saved smoke masks."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_signature(args):
    digest = hashlib.sha256()
    files = []
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in {'.external', '.venvs', '.git', '__pycache__'})
        files.extend(Path(directory) / n for n in names if n.endswith('.py'))
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        digest.update(relative.as_posix().encode())
        digest.update(path.read_bytes())
    if getattr(args, 'runtime_config', None):
        digest.update(Path(args.runtime_config).read_bytes())
    return digest.hexdigest()


def finish(args, cases, dest):
    import numpy as np
    from PIL import Image
    path = ROOT / 'evaluation/native/segmentation_eval_common.py'
    spec = importlib.util.spec_from_file_location('smoke_metric_backend', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    rows = []
    for case in cases:
        with Image.open(dest / 'masks' / (case['image_id'] + '.png')) as im:
            pred = np.asarray(im) > 0
        with Image.open(args.subset / 'test/masks' / Path(case['mask']).name) as im:
            gt = np.asarray(im) > 0
        if pred.shape != (args.size, args.size) or gt.shape != pred.shape:
            raise ValueError('Saved mask/reference shape does not match selected track')
        rows.append(native.compute_case(case['image_id'], pred, gt, float('nan')))
    native.write_cases(dest / 'metrics.csv', rows)
    summary = native.summarize(rows, 'test_smoke')
    summary['scope'] = 'Five selected cases only; no benchmark performance or timing estimate'
    result = json.loads((dest / 'result.json').read_text())
    result.update(metrics=summary, metric_protocol='hd_empty_track_side_v1',
                  code_signature=code_signature(args),
                  checkpoint_sha256=file_hash(result['checkpoint']),
                  size=args.size, selected_cases=cases)
    # Record import locations so source fallback is observable in feedback.
    result['module_origins'] = {name: getattr(sys.modules[name], '__file__', None)
                                for name in ('medseg', 'usdsgen', 'mmseg', 'sam2', 'sam3', 'ultralytics')
                                if name in sys.modules}
    (dest / 'result.json').write_text(json.dumps(result, indent=2, default=str))
