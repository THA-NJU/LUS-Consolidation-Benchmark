"""Run real HD functions from every native backend without importing GPU libraries."""
import ast
import math
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
ROUTES = [
    ('evaluation/common/unified_native_segmentation_eval.py', 'surface_distances', True),
    ('evaluation/families/evaluate_autodl_easy_models.py', 'surface_distances', True),
    ('evaluation/native/segmentation_eval_common.py', '_hd_hd95', False),
    ('experiments/yolo26s/segmentation_eval_common.py', '_hd_hd95', False),
    ('experiments/yolo/train_family.py', 'hd_hd95', False),
    ('experiments/s2denet/train.py', 'hd_metrics', False),
    ('experiments/usfm/evaluate_zeroshot.py', 'hd_metrics', False),
    ('evaluation/legacy_resnet.py', 'hd_metrics', False),
]


def backend(relative, function, gt_first):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {function, '_surface', 'surface', 'mask_surface'}
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np, math=math, ndimage=ndimage,
                     binary_erosion=ndimage.binary_erosion,
                     distance_transform_edt=ndimage.distance_transform_edt,
                     CC_STRUCTURE_8=np.ones((3, 3), dtype=bool))
    ast.fix_missing_locations(tree := ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), namespace)
    fn = namespace[function]
    return lambda gt, pred: fn(gt, pred) if gt_first else fn(pred, gt)


class DistanceTests(unittest.TestCase):
    def test_native_aggregate_keeps_penalized_cases(self):
        for relative, function in [('experiments/yolo/train_family.py', 'summarize'),
                                   ('experiments/s2denet/train.py', 'summarize_cases')]:
            parsed = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
            nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
            nodes += [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == function]
            module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
            namespace = {'np': np}
            exec(compile(module, relative, 'exec'), namespace)
            for size in (224, 512):
                with self.subTest(backend=relative, size=size):
                    overlap = dict(metric_included=1, pred_pixels=1, dice=1, iou=1,
                                   recall=1, precision=1, hd=0, hd95=0, cc_delta=0,
                                   abs_cc_delta=0, inference_time_ms=1)
                    empty = dict(overlap, pred_pixels=0, dice=0, iou=0, recall=0,
                                 precision=0, hd=size, hd95=size, cc_delta=1, abs_cc_delta=1)
                    excluded = dict(overlap, metric_included=0, hd=float('nan'), hd95=float('nan'))
                    result = namespace[function]([overlap, empty, excluded], 'test')
                    self.assertEqual(result['hd_valid_count'], 2)
                    self.assertEqual(result['hd95_valid_count'], 2)
                    self.assertEqual(result['empty_prediction_count'], 1)
                    self.assertEqual(result['gt_empty_excluded_count'], 1)
                    self.assertEqual(result['hd_mean'], size / 2)
                    self.assertEqual(result['hd_std'], size / 2)

    def test_all_backends_use_track_side_for_empty_predictions(self):
        for path, name, order in ROUTES:
            fn = backend(path, name, order)
            for size in (224, 512):
                with self.subTest(backend=path, size=size):
                    gt = np.zeros((size, size), dtype=bool)
                    gt[size // 2, size // 2] = True
                    pred = np.zeros_like(gt)
                    self.assertEqual(fn(gt, pred), (float(size), float(size)))

    def test_nonempty_surface_distances_are_unchanged(self):
        for path, name, order in ROUTES:
            fn = backend(path, name, order)
            for size in (224, 512):
                with self.subTest(backend=path, size=size):
                    gt = np.zeros((size, size), dtype=bool)
                    gt[0, 0] = True
                    self.assertEqual(fn(gt, gt), (0.0, 0.0))
                    pred = np.zeros_like(gt)
                    pred[0, 1] = True
                    self.assertEqual(fn(gt, pred), (1.0, 1.0))
                    pred[0, 1] = False
                    pred[-1, -1] = True
                    expected = math.sqrt(2) * (size - 1)
                    for actual in fn(gt, pred):
                        self.assertAlmostEqual(actual, expected)
                        self.assertGreater(actual, size)  # Real distances are never clipped.

    def test_empty_penalty_participates_in_population_statistics(self):
        for path, name, order in ROUTES:
            fn = backend(path, name, order)
            for size in (224, 512):
                with self.subTest(backend=path, size=size):
                    gt = np.zeros((size, size), dtype=bool)
                    gt[0, 0] = True
                    values = np.asarray([fn(gt, gt), fn(gt, np.zeros_like(gt))])
                    self.assertEqual(int(np.isfinite(values).sum()), 4)
                    np.testing.assert_array_equal(values.mean(axis=0), [size / 2, size / 2])
                    np.testing.assert_array_equal(values.std(axis=0, ddof=0), [size / 2, size / 2])


if __name__ == '__main__':
    unittest.main()
