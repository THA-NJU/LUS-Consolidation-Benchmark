import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_release_checks import training_plan, extra_training_args
from smoke_512 import MODELS
from train import build_command
from runtime_config import source


class ReleaseRoutesTests(unittest.TestCase):
    def test_both_tracks_have_36_registered_models(self):
        from lusbench.registry import ROUTES
        for size in (512, 224):
            self.assertEqual(len([r for r in ROUTES if r.size == size]), 36)
        self.assertEqual(len(MODELS), 36)

    def test_minimal_plan_covers_every_available_native_backend(self):
        from lusbench.registry import ROUTES
        key = lambda r: (r.route, r.script, r.family, r.size)
        self.assertEqual({key(r) for r in training_plan()}, {key(r) for r in ROUTES if r.script})
        for size in (512, 224):
            self.assertIn(('yolo26m-sem', size), {(r.model, r.size) for r in training_plan()})

    def test_all_training_commands_use_separate_subset_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sam3 = root / 'sam3.fixture'
            sam3.touch()
            args = SimpleNamespace(pth=root / 'weights', runtime_config=None, sam3_initial_checkpoint=sam3)
            for route in training_plan():
                with self.subTest(model=route.model, size=route.size):
                    folder = root / 'results' / str(route.size) / route.model
                    extras, _ = extra_training_args(args, route, folder)
                    subset = root / 'subsets' / str(route.size)
                    command = build_command(SimpleNamespace(model=route.model, size=route.size,
                        data_root=subset, output_dir=folder, run_mode='smoke',
                        april_root=source(args, 'april'), extra=extras))
                    self.assertEqual(command[command.index('--data-root') + 1], str(subset))
                    self.assertNotIn('--preflight-only', command)
                    self.assertNotIn('--eval-only', command)
                    self.assertTrue(Path(command[1]).is_file())

    def test_missing_sam3_initialization_is_explicitly_blocked(self):
        route = next(r for r in training_plan() if r.model == 'sam3')
        args = SimpleNamespace(pth=ROOT / 'weights', runtime_config=None, sam3_initial_checkpoint=None)
        with self.assertRaisesRegex(FileNotFoundError, 'standard sam3.pt'):
            extra_training_args(args, route, ROOT / 'outputs')

    def test_default_sources_belong_to_current_checkout(self):
        args = SimpleNamespace(runtime_config=None)
        for name in ('sam2', 'sam3', 'medsam', 'samus', 'usfm', 'usfm_mmseg', 's2denet', 'april'):
            self.assertTrue(source(args, name).is_relative_to(ROOT))


if __name__ == '__main__':
    unittest.main()
