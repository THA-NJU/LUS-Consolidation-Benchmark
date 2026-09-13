"""Keep native runtime reports outside release sources without hiding tracked files."""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import check_repository


class OutputRoutingTests(unittest.TestCase):
    def test_april_dispatcher_sets_base_for_each_track_and_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / 'fixture.py'
            script.write_text('''from pathlib import Path
OUTPUT_BASE = Path(__file__).parent / 'wrong_source_output'
def main():
    assert OUTPUT_BASE == OUTPUT_ROOT
    assert len(RUN_SIZES) == 1
    result = OUTPUT_BASE / f'size{RUN_SIZES[0]}' / 'result.txt'
    result.parent.mkdir(parents=True)
    result.write_text(RUN_MODE)
''')
            for size in (224, 512):
                for mode in ('smoke', 'formal'):
                    for kind in ('april_general', 'april_general224'):
                        destination = root / f'{size}_{mode}_{kind}'
                        command = [sys.executable, '-B', str(ROOT / 'tools/_run_configured.py'),
                            '--kind', kind, '--script', str(script), '--model', 'nnunet_2d',
                            '--size', str(size), '--data-root', str(root / 'data'),
                            '--output-dir', str(destination), '--run-mode', mode, '--april-root', str(root)]
                        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual((destination / f'size{size}' / 'result.txt').read_text(), mode)
            self.assertFalse((root / 'wrong_source_output').exists())

    @unittest.skipUnless(shutil.which('git'), 'Git required for release file selection')
    def test_untracked_outputs_ignored_but_tracked_outputs_and_source_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
            git('init')
            (root / '.gitignore').write_text((ROOT / '.gitignore').read_text())
            report = root / 'experiments/april/output/result.json'
            report.parent.mkdir(parents=True)
            private = '/' + 'home' + '/fixture/data'
            report.write_text('{"data": "' + private + '"}')
            with patch.object(check_repository, 'ROOT', root):
                errors = []
                check_repository.check_files(errors)
                self.assertEqual(errors, [])
                git('add', '-f', 'experiments/april/output/result.json')
                check_repository.check_files(errors)
                self.assertTrue(any('forbidden release directory' in e for e in errors), errors)
                self.assertTrue(any('development-machine absolute path' in e for e in errors), errors)
                (root / 'source.py').write_text('DATA = ' + repr(private))
                errors = []
                check_repository.check_files(errors)
                self.assertTrue(any('source.py' in e and 'absolute path' in e for e in errors), errors)


if __name__ == '__main__':
    unittest.main()
