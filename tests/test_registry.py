from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lusbench.registry import ROUTES, find_route, model_count


class RegistryTests(unittest.TestCase):
    def test_unique_model_size_routes(self):
        keys = [(route.model, route.size) for route in ROUTES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_frozen_model_count(self):
        self.assertEqual(model_count(), 36)

    def test_frozen_counts_are_represented_by_both_tracks(self):
        self.assertEqual(find_route("sam3", 512).size, 512)
        self.assertEqual(find_route("sam3", 224).size, 224)

    def test_legacy_models_are_evaluation_only(self):
        for model in ("fpn_resnet34", "deeplabv3plus_resnet34"):
            for size in (512, 224):
                route = find_route(model, size)
                self.assertEqual(route.status, "evaluation_only")
                self.assertIsNone(route.script)


if __name__ == "__main__":
    unittest.main()
