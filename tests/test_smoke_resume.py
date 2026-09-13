"""Regression checks for safe resume and five-patient isolation; no GPU needed."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from smoke_512 import MODELS, validate_subset
from smoke_runtime import archive_partial, environment_for, validate_output


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dest = self.root / '512/example'
        (self.dest / 'masks').mkdir(parents=True)
        self.cases, rows = [], []
        for i in range(5):
            identifier = f'p{i:03d}_001'
            path = self.dest / 'masks' / (identifier + '.png')
            Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).save(path)
            self.cases.append({'image_id': identifier, 'patient': f'p{i:03d}'})
            rows.append({'image_id': identifier, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        checkpoint = self.root / 'checkpoint.fixture'
        checkpoint.write_bytes(b'trained fixture')
        (self.dest / 'metrics.csv').write_text('filename,hd,hd95\n')
        self.result = {'model': 'example', 'status': 'complete', 'cases': rows,
                       'size': 512, 'metric_protocol': 'hd_empty_track_side_v1',
                       'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        self.write_result()

    def tearDown(self):
        self.temporary.cleanup()

    def write_result(self):
        (self.dest / 'result.json').write_text(json.dumps(self.result))

    def valid(self):
        return validate_output(self.dest, self.cases, 'example')[0]

    def test_complete_current_results_are_reusable(self):
        self.assertTrue(self.valid())

    def test_modified_mask_is_rejected(self):
        path = self.dest / 'masks/p000_001.png'
        Image.fromarray(np.full((512, 512), 255, dtype=np.uint8)).save(path)
        self.assertFalse(self.valid())

    def test_wrong_patient_is_rejected(self):
        self.cases[0]['image_id'] = 'p999_001'
        self.assertFalse(self.valid())

    def test_partial_results_are_archived_without_loss(self):
        (self.dest / 'result.json').unlink()
        self.assertFalse(self.valid())
        before = {p.name: p.read_bytes() for p in (self.dest / 'masks').iterdir()}
        backup = Path(archive_partial(self.dest))
        self.assertFalse(self.dest.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in (backup / 'masks').iterdir()})

    def test_nonbinary_mask_rejected_even_with_updated_hash(self):
        path = self.dest / 'masks/p000_001.png'
        Image.fromarray(np.full((512, 512), 100, dtype=np.uint8)).save(path)
        self.result['cases'][0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_result()
        self.assertFalse(self.valid())

    def test_changed_input_hash_is_rejected(self):
        self.result['selected_cases'] = [dict(c) for c in self.cases]
        self.result['selected_cases'][0]['image_sha256'] = 'old_input_hash'
        self.write_result()
        self.assertFalse(self.valid())

    def test_subset_requires_different_patients(self):
        subset = self.root / 'subset512'
        subset.mkdir()
        cases = [dict(c) for c in self.cases]
        cases[1]['patient'] = cases[0]['patient']
        (subset / 'case_manifest.json').write_text(json.dumps(cases))
        with self.assertRaisesRegex(ValueError, 'five different patients'):
            validate_subset(subset)

    def test_environment_selection_preserves_venv_path(self):
        config = self.root / 'runtime.json'
        config.write_text(json.dumps({'python': {'usfm': 'env/bin/python'}, 'sources': {'usfm': 'source/usfm', 'usfm_mmseg': 'source/mmseg'}}))
        args = SimpleNamespace(runtime_config=config)
        python, env = environment_for(args, 'USFM', {})
        self.assertEqual(Path(python), self.root / 'env/bin/python')
        self.assertIn(str(self.root / 'source/mmseg'), env['PYTHONPATH'])

    def test_track_mismatch_cannot_reuse_results(self):
        self.assertFalse(validate_output(self.dest, self.cases, 'example', 224)[0])

    def test_changed_checkpoint_cannot_reuse_results(self):
        Path(self.result['checkpoint']).write_bytes(b'different checkpoint')
        self.assertFalse(self.valid())

    def test_legacy_metric_protocol_requires_new_run(self):
        self.result.pop('metric_protocol')
        self.write_result()
        self.assertFalse(self.valid())

    def test_all_36_unique_routes_registered(self):
        self.assertEqual(len(MODELS), 36)
        self.assertEqual(len(set(MODELS)), 36)


if __name__ == '__main__':
    unittest.main()
