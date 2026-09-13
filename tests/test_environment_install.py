import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ManualEnvironmentTests(unittest.TestCase):
    def test_legacy_launcher_stops_with_manual_guidance(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/setup_environments.py'),
            '--interpreters', 'nonexistent-file.json'], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 2)
        self.assertIn('No packages were installed or changed', result.stdout)
        self.assertIn('docs/ENVIRONMENTS.md', result.stdout.replace(chr(92), '/'))
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
