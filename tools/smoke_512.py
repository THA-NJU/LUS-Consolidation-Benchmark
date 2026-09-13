#!/usr/bin/env python3
"""Five fixed patients per selected track: checkpoint inference and binary PNGs."""
import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'monai': ('CNN', ['monai_unet', 'monai_vnet', 'monai_attention_unet', 'monai_unetplusplus']),
    'general': ('', ['nnunet_2d', 'aau_net', 'swinunet', 'nnformer_2d', 'sepnet', 'nulite', 'ukan', 'xlstm_unet_bot']),
    'additional': ('', ['mednext', 'pvtb2_emcad', 'rolling_unet']),
    'mamba': ('Mamba', ['mamba_unet', 'vm_unet_v2', 'nnmamba_2d', 'swin_umamba']),
    'rwkv': ('RWKV', ['u_rwkv', 'rwkv_unet']),
    'segformer': ('Transformer', ['segformer_b2']),
}
from smoke_adapters import EXTRA_MODELS
from runtime_config import source
MODELS = [m for _, ms in GROUPS.values() for m in ms] + EXTRA_MODELS


def load(relative, name):
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_subset(subset):
    cases = json.loads((subset / 'case_manifest.json').read_text())
    if len(cases) != 5 or len({c['patient'] for c in cases}) != 5:
        raise ValueError('Require exactly five different patients in the existing manifest')
    for key, folder in [('image', 'images'), ('mask', 'masks')]:
        expected = {Path(c[key]).name for c in cases}
        actual = {p.name for p in (subset / 'test' / folder).iterdir() if p.is_file()}
        if actual != expected:
            raise ValueError(f'Unexpected files in subset/test/{folder}')
        for c in cases:
            path = subset / 'test' / folder / Path(c[key]).name
            if hashlib.sha256(path.read_bytes()).hexdigest() != c[key + '_sha256']:
                raise ValueError(f'Changed selected file: {path}')
    return cases


def worker(args):
    if args.worker in EXTRA_MODELS:
        from smoke_adapters import run
        return run(args, validate_subset(args.subset))
    import numpy as np
    import torch
    from PIL import Image
    cases = validate_subset(args.subset)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this interpreter; refusing a silent CPU run')
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    common = load('evaluation/common/unified_native_segmentation_eval.py', 'unified_native_segmentation_eval')
    group, family = next((g, f) for g, (f, ms) in GROUPS.items() if args.worker in ms)
    model_name = args.worker
    april = args.april_root or source(args, 'april')
    april = april.resolve()
    sys.path.insert(0, str(april))
    dataset = common.NativeSegmentationDataset(args.subset / 'test', args.size, in_channels=3)
    amp_name = 'bf16'
    custom_prob = None
    if group == 'monai':
        mod = load('evaluation/families/evaluate_monai_family.py', 'smoke_monai')
        model, config = mod.build_monai_model(model_name, args.size)
    elif group == 'general':
        mod = load('evaluation/families/evaluate_autodl_easy_models.py', 'smoke_general_eval')
        train = load('experiments/april/train_general.py', 'smoke_general_train')
        train.PROJECT_ROOT = april
        train.PRETRAINED = False
        train.set_task_context(args.size, args.subset, args.out)
        model, config = train.build_april_model(model_name)
        family = mod.MODEL_FAMILY[model_name]
        amp_name = train.get_model_spec(model_name).get('amp_dtype', train.DEFAULT_AMP_DTYPE)
        dataset = mod.make_eval_dataset(train, args.subset, 'test', args.size)
        amp = train.resolve_amp_dtype(amp_name, device)
        custom_prob = lambda x: mod.model_probabilities(train, model, x, amp)
    elif group == 'additional':
        mod = load('evaluation/families/evaluate_mednext_pvtb2_rolling' + ('_224' if args.size == 224 else '') + '.py', 'smoke_additional')
        model, config = mod.build_model(model_name, april)
        family = {'mednext': 'CNN', 'pvtb2_emcad': 'Transformer', 'rolling_unet': 'other'}[model_name]
        amp_name = mod.MODEL_SPECS[model_name]['amp_dtype']
    elif group == 'mamba':
        mod = load('evaluation/families/evaluate_mamba_family.py', 'smoke_mamba')
        model, config = mod.build_mamba_model(model_name, args.size, april)
    elif group == 'rwkv':
        mod = load('evaluation/families/evaluate_rwkv_family.py', 'smoke_rwkv')
        mod.configure_isolated_extension_cache(model_name, args.size, args.out)
        model, config = mod.build_rwkv_model(model_name, args.size, april)
    else:
        mod = load('evaluation/native/evaluate_segformer_b2.py', 'smoke_segformer')
        checkpoint = args.pth / 'Transformer' / str(args.size) / 'Segformer/best_model.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = json.loads((checkpoint.parent / 'run_config.json').read_text()) if (checkpoint.parent / 'run_config.json').is_file() else {}
        config.update(payload.get('config', {}))
        base = args.segformer_base or str(ROOT / 'configs/segformer-b2')
        if not base:
            raise ValueError('SegFormer base config missing; supply --segformer-base with the original local model directory')
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            candidates = [ROOT / base_path]
            found = list(dict.fromkeys(c.resolve() for c in candidates if (c / 'config.json').is_file()))
            if len(found) != 1:
                raise ValueError(f'SegFormer base directory unresolved or ambiguous: {found}. Pass --segformer-base with the original absolute directory.')
            base_path = found[0]
        if not (base_path / 'config.json').is_file():
            raise FileNotFoundError(f'SegFormer config missing: {base_path / "config.json"}')
        model = mod.SegFormerBinaryWrapper(
            str(base_path), True, trained_state=payload.get('model_state', payload))
        print('SegFormer: local architecture config + strictly loaded trained checkpoint; no base weights loaded', flush=True)
        mean = config.get('image_mean', [0.10063751267950466] * 3)
        std = config.get('image_std', [0.14586819260714984] * 3)
        pairs = mod.list_pairs(args.subset, 'test')
        dataset = mod.EvalDataset(pairs, mean, std, [1, 255], args.size)
        amp_name = 'fp16'
        def custom_prob(x):
            with mod.autocast_context(device, amp_name):
                logits = model(x)
                logits = torch.nn.functional.interpolate(logits.float(), size=(args.size, args.size), mode='bilinear', align_corners=False)
                return torch.softmax(logits, dim=1)[:, 1]
    if group != 'segformer':
        checkpoint = args.pth / family / str(args.size) / model_name / 'best_model.pth'
        if group == 'general':
            mod.load_model_checkpoint(model, checkpoint, torch.device('cpu'))
        else:
            common.load_model_checkpoint(model, checkpoint, torch.device('cpu'))
    model.to(device).eval()
    amp = common.resolve_amp_dtype(amp_name, device)
    dest = args.out / str(args.size) / model_name
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'masks').mkdir()
    rows = []
    expected = {c['image_id'] for c in cases}
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            if isinstance(sample, dict):
                x, name = sample['image'], sample['case_name']
            else:
                x, _, name = sample
            if Path(name).stem not in expected:
                raise ValueError(f'Unexpected dataset item {name}')
            prob = custom_prob(x[None].to(device)) if custom_prob else common.foreground_probabilities(model, x[None].to(device), device, amp)
            if tuple(prob.shape) != (1, args.size, args.size) or not torch.isfinite(prob).all():
                raise ValueError(f'Prediction must be finite and have shape 1x{args.size}x{args.size}')
            pred = (prob[0].cpu().numpy() >= .5).astype(np.uint8)
            target = dest / 'masks' / (Path(name).stem + '.png')
            Image.fromarray(pred * 255).save(target)
            # GT is used only for this diagnostic Dice, never for prediction.
            case = next(c for c in cases if c['image_id'] == Path(name).stem)
            gt_file = args.subset / 'test/masks' / Path(case['mask']).name
            gt_raw = np.asarray(Image.open(gt_file))
            if gt_raw.ndim != 2 or not set(np.unique(gt_raw)).issubset({0, 1, 255}):
                raise ValueError('Reference is not an explicitly binary mask')
            gt = gt_raw > 0
            denominator = int(gt.sum()) + int(pred.sum())
            rows.append({'image_id': Path(name).stem, 'pred_pixels': int(pred.sum()), 'dice': 2 * int((gt & (pred > 0)).sum()) / denominator if denominator else 1.0, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
            print(f'[{model_name}] {index+1}/5 saved {target.name}', flush=True)
    if len(rows) != 5 or {r['image_id'] for r in rows} != expected:
        raise ValueError('Did not produce exactly the selected five cases')
    (dest / 'result.json').write_text(json.dumps({'model': model_name, 'size': args.size, 'checkpoint': str(checkpoint), 'amp': amp_name, 'python': sys.executable, 'config': config, 'cases': rows, 'status': 'complete', 'selected_cases': cases, 'april_source': str(april), 'scope': 'Five-case diagnostic only; not a new benchmark estimate'}, indent=2, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pth', type=Path, required=True)
    p.add_argument('--size', type=int, choices=(512, 224), default=512)
    p.add_argument('--runtime-config', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--worker', choices=MODELS, help=argparse.SUPPRESS)
    p.add_argument('--segformer-base')
    p.add_argument('--april-root', type=Path)
    p.add_argument('--python-map', type=Path, help='Optional JSON mapping model names to Python executables')
    p.add_argument('--timeout', type=int, default=900, help='Seconds per model including extension build')
    args = p.parse_args()
    args.pth = args.pth.expanduser().absolute()
    args.out = args.out.expanduser().absolute()
    args.subset = args.subset.expanduser().absolute()
    if args.worker:
        worker(args)
        from checkpoint_diagnostics import finish
        finish(args, validate_subset(args.subset), args.out / str(args.size) / args.worker)
        return 0
    cases = validate_subset(args.subset)
    from smoke_runtime import run
    return run(args, cases, MODELS)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
