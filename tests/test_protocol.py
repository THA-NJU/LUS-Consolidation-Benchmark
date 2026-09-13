from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    FIXED_THRESHOLD_BACKENDS = (
        "experiments/april/train_general.py",
        "experiments/april/train_additional_512.py",
        "experiments/april/train_additional_224.py",
        "experiments/mamba/train_512.py",
    )

    def test_release_counts(self):
        protocol = json.loads((ROOT / "configs/benchmark_protocol.json").read_text())
        self.assertEqual(
            protocol["released_split_counts"],
            {
                "512": {"train": 16131, "val": 2111, "test": 1539},
                "224": {"train": 26017, "val": 2796, "test": 3557},
            },
        )

    def test_formal_schedule(self):
        protocol = json.loads((ROOT / "configs/benchmark_protocol.json").read_text())
        training = protocol["training"]
        self.assertEqual(training["max_epochs"], 600)
        self.assertEqual(training["early_stopping_patience"], 15)
        self.assertEqual(training["warmup_epochs"], 10)
        self.assertIsNone(training["samples_per_epoch"])
        self.assertFalse(training["balanced_sampler"])

    def test_fixed_evaluation_threshold(self):
        protocol = json.loads((ROOT / "configs/benchmark_protocol.json").read_text())
        self.assertEqual(protocol["evaluation"]["threshold_default"], 0.5)
        self.assertTrue(protocol["evaluation"]["threshold_policy"].startswith("fixed"))

        prohibited = (
            "THRESHOLD_SELECTION_METRIC",
            "THRESHOLDS =",
            "def search_thresholds(",
            "def choose_best_threshold(",
            '"selected_threshold"',
            '"val_at_selected_threshold"',
        )
        for relative_path in self.FIXED_THRESHOLD_BACKENDS:
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            for marker in prohibited:
                self.assertNotIn(marker, source, f"{marker!r} remains in {relative_path}")
            self.assertIn('"threshold_policy": "fixed"', source)
            self.assertIn("BALANCE_POSITIVE_IMAGES = False", source)

        segformer_eval = (
            ROOT / "evaluation/native/evaluate_segformer_b2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("FIXED_THRESHOLD = 0.5", segformer_eval)
        self.assertNotIn("checkpoint.get(\"best_threshold\"", segformer_eval)


if __name__ == "__main__":
    unittest.main()
