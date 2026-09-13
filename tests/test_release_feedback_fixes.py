import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from checkpoint_tracking import CheckpointTracker


class CheckpointTrackingTests(unittest.TestCase):
    def setUp(self):
        self.report = {'saved_checkpoints': [], 'reloaded_checkpoints': []}
        self.tracker = CheckpointTracker(self.report)

    def test_atomic_rename_and_buffer_copy_are_recognized(self):
        with tempfile.TemporaryDirectory() as folder:
            before, after = Path(folder) / 'best.tmp', Path(folder) / 'best.pth'
            before.write_bytes(b'checkpoint after update')
            self.tracker.saved(before)
            os.replace(before, after)
            self.tracker.loaded(after, self.tracker.digest(after))
            buffer = io.BytesIO(b'ultralytics buffered checkpoint')
            self.tracker.saved(buffer)
            after.write_bytes(buffer.getvalue())
            self.tracker.loaded(after, self.tracker.digest(after))
            self.assertEqual(len(self.report['reloaded_checkpoints']), 2)

    def test_reused_filename_with_different_content_is_not_a_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'best.pth'
            target.write_bytes(b'updated')
            self.tracker.saved(target)
            target.write_bytes(b'unrelated old initializer')
            self.tracker.loaded(target, self.tracker.digest(target))
            self.assertEqual(self.report['reloaded_checkpoints'], [])

    def test_file_buffer_is_flushed_without_changing_cursor(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'best.pth').open('wb') as stream:
                stream.write(b'updated')
                position = stream.tell()
                self.tracker.saved(stream)
                self.assertEqual(position, stream.tell())
            self.assertEqual(len(self.tracker.saved_digests), 1)


class NativeImportTests(unittest.TestCase):
    def test_numeric_metrics_import_without_matplotlib(self):
        import numpy as np
        for relative in ('evaluation/native/segmentation_eval_common.py',
                         'experiments/yolo26s/segmentation_eval_common.py'):
            with self.subTest(relative=relative), patch.dict(sys.modules,
                    {'matplotlib': None, 'matplotlib.pyplot': None, 'cv2': types.ModuleType('cv2')}):
                spec = importlib.util.spec_from_file_location('metric_fixture', ROOT / relative)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                gt = np.zeros((224, 224), dtype=bool)
                gt[10, 10] = True
                self.assertEqual(module._hd_hd95(np.zeros_like(gt), gt), (224.0, 224.0))

    def test_native_loader_resolves_sibling_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'local_fixture.py').write_text('VALUE = 42\n')
            script = root / 'entry.py'
            script.write_text('from local_fixture import VALUE\ndef main():\n    assert VALUE == 42\n    assert SMOKE_CHECK\n')
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/_run_native_training.py'),
                '--kind', 'yolo26s', '--script', str(script), '--size', '512',
                '--data-root', str(root), '--output-dir', str(root), '--run-mode', 'smoke'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required for the actual CPU optimizer probe')
    def test_actual_cpu_update_atomic_save_and_buffer_reload(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = root / 'entry.py'
            (root / 'local_fixture.py').write_text('VALUE = 42\n')
            script.write_text('''import io, os, torch
from pathlib import Path
from local_fixture import VALUE
root = Path(__file__).parent
net = torch.nn.Linear(2, 1)
opt = torch.optim.AdamW(net.parameters(), lr=0.01)
net(torch.ones(2, 2)).sum().backward()
opt.step()
torch.save(net.state_dict(), root / 'best.tmp')
os.replace(root / 'best.tmp', root / 'best.pth')
net.load_state_dict(torch.load(root / 'best.pth', weights_only=True))
buffer = io.BytesIO()
torch.save(net.state_dict(), buffer)
(root / 'buffer.pth').write_bytes(buffer.getvalue())
net.load_state_dict(torch.load(root / 'buffer.pth', weights_only=True))
''')
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/training_probe.py'),
                '--report', str(root / 'report.json'), '--', str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['optimizer_steps'], 1)
            self.assertEqual(report['changed_steps'], 1)
            self.assertEqual(len(report['reloaded_checkpoints']), 2)


if __name__ == '__main__':
    unittest.main()
