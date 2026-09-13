"""Test config-only loading control flow without installing GPU dependencies.

The constructor and normalization function are extracted from the actual source;
the external model/config classes are test doubles. This does not test GPU inference.
"""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class ModuleDouble:
    fail_load = False

    def __init__(self):
        pass

    def load_state_dict(self, state, strict):
        self.loaded_state, self.strict = state, strict
        if self.fail_load:
            raise RuntimeError('incompatible checkpoint')


class ConfigDouble:
    @staticmethod
    def from_dict(data):
        return data


class NetworkDouble:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise AssertionError('Must not load initialization weights for a complete trained state')


def load_constructor():
    source = ROOT / 'evaluation/native/evaluate_segformer_b2.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
              and n.name in {'SegFormerBinaryWrapper', 'normalize_state_dict'}]
    namespace = dict(nn=SimpleNamespace(Module=ModuleDouble), Path=Path, json=json,
                     SegformerConfig=ConfigDouble, SegformerForSemanticSegmentation=NetworkDouble)
    ast.fix_missing_locations(tree := ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(source), 'exec'), namespace)
    return namespace['SegFormerBinaryWrapper']


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.base = Path(self.temp.name)
        self.config = dict(model_type='segformer', depths=[3, 4, 6, 3],
                           hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768,
                           num_labels=150, id2label={str(i): str(i) for i in range(150)},
                           label2id={str(i): i for i in range(150)})
        self.write_config()
        self.wrapper = load_constructor()

    def tearDown(self):
        self.temp.cleanup()
        ModuleDouble.fail_load = False

    def write_config(self):
        (self.base / 'config.json').write_text(json.dumps(self.config))

    def test_config_only_preserves_b2_architecture_and_replaces_label_maps(self):
        token = object()
        model = self.wrapper(str(self.base), True, trained_state={'module.net.segformer.weight': token})
        self.assertEqual(model.net.config['depths'], [3, 4, 6, 3])
        self.assertEqual(model.net.config['decoder_hidden_size'], 768)
        self.assertEqual(model.net.config['num_labels'], 2)
        self.assertEqual(model.net.config['id2label'], {0: 'background', 1: 'consolidation'})
        self.assertEqual(model.net.config['label2id'], {'background': 0, 'consolidation': 1})
        self.assertEqual(model.loaded_state, {'net.segformer.weight': token})
        self.assertTrue(model.strict)
        self.assertEqual([p.name for p in self.base.iterdir()], ['config.json'])

    def test_incompatible_trained_state_stops_constructor(self):
        ModuleDouble.fail_load = True
        with self.assertRaisesRegex(RuntimeError, 'incompatible checkpoint'):
            self.wrapper(str(self.base), True, trained_state={})

    def test_other_model_config_is_rejected(self):
        self.config['model_type'] = 'unrelated_model'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'Not a SegFormer'):
            self.wrapper(str(self.base), True, trained_state={})

    def test_missing_local_json_does_not_attempt_network(self):
        (self.base / 'config.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.wrapper(str(self.base), True, trained_state={})


if __name__ == '__main__':
    unittest.main()
