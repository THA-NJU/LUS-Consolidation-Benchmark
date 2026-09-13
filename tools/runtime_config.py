"""Explicit, relocatable source and interpreter configuration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    **{m: 'ssm' for m in ('mamba_unet', 'vm_unet_v2', 'nnmamba_2d', 'swin_umamba', 'u_rwkv', 'rwkv_unet')},
    'USFM': 'usfm', 'sam2': 'sam', 'medsam': 'sam', 'samus': 'samus', 'sam3': 'sam3',
}


def configuration(args):
    path = getattr(args, 'runtime_config', None)
    if path is None:
        return {}, ROOT
    path = Path(path).expanduser().absolute()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Runtime configuration must be an object')
    return data, path.parent


def configured_path(value, base):
    path = Path(value).expanduser()
    # Keep interpreter symlinks intact so virtual environments remain active.
    return path.absolute() if path.is_absolute() else (base / path).absolute()


def source(args, name):
    data, base = configuration(args)
    value = data.get('sources', {}).get(name)
    if value:
        return configured_path(value, base)
    return ROOT / ('third_party/april_medseg' if name == 'april' else '.external/' + name)


def interpreter(args, model, default):
    data, base = configuration(args)
    mapping = data.get('python', {})
    value = mapping.get(model, mapping.get(GROUPS.get(model, 'base'), mapping.get('base')))
    return str(configured_path(value, base)) if value else default
