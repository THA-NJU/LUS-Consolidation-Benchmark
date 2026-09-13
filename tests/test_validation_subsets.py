"""Patient-independent selection and input integrity checks, without GPU imports."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import prepare_validation_subsets as subset


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.roots = {size: self.root / str(size) for size in (512, 224)}
        for size, root in self.roots.items():
            for split, ids in [('train', range(1, 4)), ('val', range(10, 13)), ('test', range(20, 31))]:
                for folder in ('images', 'masks'):
                    directory = root / split / folder
                    directory.mkdir(parents=True)
                    for patient in ids:
                        for frame in range(2):
                            # Selection copies opaque bytes; image decoding belongs to the evaluator.
                            (directory / f'p{patient:03d}_{frame:03d}.png').write_bytes(
                                f'{size}-{split}-{folder}-{patient}-{frame}'.encode())
        self.manifest = self.root / 'previous512.json'
        self.records = []
        for patient, image, mask in subset.choose(subset.inventory(self.roots[512], 'test'), 5, 512):
            self.records.append({'patient': patient, 'image_id': image.stem,
                                 'image_sha256': subset.sha256(image), 'mask_sha256': subset.sha256(mask)})
        self.manifest.write_text(json.dumps(self.records))

    def tearDown(self):
        self.temp.cleanup()

    def run_selection(self, output):
        args = ['prepare_validation_subsets.py', '--data512', str(self.roots[512]),
                '--data224', str(self.roots[224]), '--existing512-manifest', str(self.manifest),
                '--output', str(output)]
        with patch.object(sys, 'argv', args):
            subset.main()

    def test_independent_224_selection_and_preserved_512_hashes(self):
        first = self.root / 'selected_a'
        self.run_selection(first)
        cases512 = json.loads((first / '512/case_manifest.json').read_text())
        self.assertEqual([c['image_id'] for c in cases512], [c['image_id'] for c in self.records])
        for row in cases512:
            copied = first / '512/test/images' / (row['image_id'] + '.png')
            self.assertEqual(subset.sha256(copied), row['image_sha256'])
        before224 = (first / '224/case_manifest.json').read_bytes()
        self.records.reverse()
        self.manifest.write_text(json.dumps(self.records))
        second = self.root / 'selected_b'
        self.run_selection(second)
        self.assertEqual(before224, (second / '224/case_manifest.json').read_bytes())
        selected = json.loads((first / '224/split_manifests.json').read_text())
        ids = {split: {row['patient'] for row in rows} for split, rows in selected.items()}
        self.assertEqual(len(ids['test']), 5)
        self.assertFalse(ids['test'] & (ids['train'] | ids['val']))

    def test_leakage_rejected_before_copy(self):
        root = self.roots[224]
        for folder in ('images', 'masks'):
            (root / 'train' / folder / 'p020_999.png').write_bytes(b'leaked patient')
        output = self.root / 'leaked'
        with self.assertRaisesRegex(ValueError, 'patient leakage'):
            self.run_selection(output)
        self.assertFalse(output.exists())

    def test_changed_existing_input_rejected(self):
        row = self.records[0]
        (self.roots[512] / 'test/images' / (row['image_id'] + '.png')).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            self.run_selection(self.root / 'changed')

    def test_existing_output_never_overwritten(self):
        output = self.root / 'existing'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('existing results')
        with self.assertRaises(SystemExit):
            self.run_selection(output)
        self.assertEqual(marker.read_text(), 'existing results')


if __name__ == '__main__':
    unittest.main()
