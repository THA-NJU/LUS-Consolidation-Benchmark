import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from setup_sources import checkout_changes


@unittest.skipUnless(shutil.which('git'), 'Git required')
class SourceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init')
        self.cache = self.root / 'models with spaces/__pycache__/module.cpython-310.pyc'
        self.cache.parent.mkdir(parents=True)
        self.cache.write_bytes(b'bytecode fixture')

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True)

    def test_untracked_bytecode_is_allowed_and_retained(self):
        self.assertEqual(checkout_changes(self.root), [])
        self.assertTrue(self.cache.exists())

    def test_untracked_source_and_other_cache_contents_remain_visible(self):
        source = self.cache.parent / 'unexpected.py'
        source.write_text('VALUE = 1')
        (self.root / 'module.pyc').write_bytes(b'outside cache')
        changes = checkout_changes(self.root)
        self.assertEqual(len(changes), 2)
        self.assertTrue(any('unexpected.py' in item for item in changes))
        self.assertTrue(any(item == '?? module.pyc' for item in changes))

    def test_tracked_bytecode_and_source_edits_are_not_ignored(self):
        source = self.root / 'source.py'
        source.write_text('VALUE = 1')
        self.git('add', '.')
        source.write_text('VALUE = 2')
        changes = checkout_changes(self.root)
        self.assertEqual(len(changes), 2)
        self.assertTrue(any('module.cpython-310.pyc' in item for item in changes))
        self.assertTrue(any(item.startswith('AM ') and 'source.py' in item for item in changes))


if __name__ == '__main__':
    unittest.main()
