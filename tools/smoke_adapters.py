"""Track-aware inference adapters using the supplied native evaluation code.

No training entry point is called. Ground truth is read only for diagnostics;
SAM prompts are fixed full-image boxes and SAM3 uses its fixed training text.
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from runtime_config import source

ROOT = Path(__file__).resolve().parents[1]
EXTRA_MODELS = ['Resnet34_fpn', 'Resnet34_DLV3', 'sam2', 'medsam', 'samus',
                'sam3', 'USFM', 's2denet', 'yolo11s-seg', 'yolo11m-seg',
                'yolo11l-seg', 'yolo26s-sem', 'yolo26m-sem', 'yolo26l-sem']


def load(relative, name):
    path = ROOT / relative
    for directory in (ROOT, ROOT / 'evaluation/common', path.parent):
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_root(candidates, sentinel):
    for candidate in candidates:
        if candidate and (Path(candidate) / sentinel).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError(f'Missing {sentinel}; searched: {candidates}')


def prepare(args, dest):
    """Return a native dataset, image-only predictor, and reproducibility record."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    name, size = args.worker, args.size
    device = torch.device('cuda:0')
    amp_name = 'bf16'
    record = {'threshold': 0.5}

    if name.startswith('Resnet34_'):
        native = load('evaluation/legacy_resnet.py', 'smoke_resnet_native')
        architecture = 'FPN' if name.endswith('fpn') else 'DeepLabV3Plus'
        directory = args.pth / 'CNN' / str(size) / name
        checkpoint = require_file(directory / (architecture + '_resnet34_best.pth' if size == 512 else 'best.pth'))
        settings = read_json(directory / ('evaluation_settings.json' if size == 512 else 'training_settings.json'))
        raw, _, _ = native.extract_state_dict(native.torch_load_cpu(checkpoint))
        state = native.strip_common_wrappers(raw)
        in_channels, out_channels, _, _ = native.infer_model_dimensions(state)
        model, constructor = native.build_model(architecture, in_channels, out_channels)
        native.load_state_strict(model, state, checkpoint)
        mode = settings.get('decode_mode', 'softmax_argmax')
        channel = int(settings.get('consolidation_channel', 2))
        mode, channel = native.resolve_decode(out_channels, mode, channel)
        normalization = settings.get('normalization', 'imagenet')
        dataset = native.BenchmarkDataset(args.subset, 'test', in_channels, normalization,
                                           settings.get('dataset_mean') or 0.100638,
                                           settings.get('dataset_std') or 0.145868)
        def decode(x):
            logits = native.extract_logits(model(x)).float()
            if logits.shape[-2:] != (size, size):
                logits = F.interpolate(logits, (size, size), mode='bilinear', align_corners=False)
            return native.decode_prediction(logits, mode, channel, .5)[1]
        record.update(normalization=normalization, decode_mode=mode, foreground_channel=channel,
                      constructor=constructor, settings_source=str(directory / 'evaluation_settings.json'))

    elif name == 'USFM':
        native = load('experiments/usfm/evaluate_zeroshot.py', 'smoke_usfm_native')
        checkpoint = require_file(args.pth / 'Transformer' / str(size) / 'USFM' / ('best_usfm_decoder.pth' if size == 512 else 'best.pth'))
        raw, _ = native.extract_state_dict(native.torch_load_cpu(checkpoint))
        classes, _ = native.infer_output_classes(native.strip_uniform_prefixes(raw))
        expected_classes = 3 if size == 512 else 2
        if classes != expected_classes:
            raise ValueError(f'{size} USFM expected {expected_classes} classes; checkpoint has {classes}')
        foreground_channel = 2 if size == 512 else 1
        model = native.USFMModel(classes)
        native.load_checkpoint_strict(model, raw, checkpoint)
        dataset = native.BenchmarkDataset(args.subset, 'test', False)
        def decode(x):
            logits = model(x).float()
            logits = F.interpolate(logits, (size, size), mode='bilinear', align_corners=False)
            return native.decode_logits(logits, 'argmax', foreground_channel, .5)[1]
        record.update(input_size=224, normalization='ImageNet', foreground_channel=foreground_channel, decode_mode='argmax')

    elif name == 's2denet':
        native = load('experiments/s2denet/train.py', 'smoke_s2denet_native')
        directory = args.pth / 'other' / str(size) / 's2denet'
        checkpoint = require_file(directory / 'best.pt')
        settings = read_json(directory / 'evaluation_settings.json')
        native_source = source_root([source(args, 's2denet')], 's2denet/model.py')
        model = native.import_official_model(native_source)(in_channels=8, num_class=1, return_edge=True)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(payload.get('model_state_dict', payload), strict=True)
        mean, std = settings.get('image_mean', .100638), settings.get('image_std', .145868)
        dataset = native.ConsolidationDataset(native.list_pairs(args.subset, 'test', size),
                                              size, False, 0.0, mean, std)
        def decode(x):
            probability, _ = native.model_forward(model, x)
            native.validate_probability_output('smoke mask', probability)
            return probability[:, 0] >= .5
        record.update(source=str(native_source), image_mean=mean, image_std=std)

    elif name in {'sam2', 'medsam', 'samus'}:
        native = load(f'experiments/foundation/train_{name}_512.py', f'smoke_{name}_native')
        directory = args.pth / 'SAM' / str(size) / name
        checkpoint = require_file(directory / 'best_model.pth')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        saved = dict(read_json(directory / 'result.json').get('args', {}))
        saved.update(payload.get('args', {}))
        input_size = int(saved.get('model_input_size', 256 if name == 'samus' else 1024))
        if name == 'sam2':
            native_source = source_root([source(args, 'sam2')], 'sam2/build_sam.py')
            sys.path.insert(0, str(native_source))
            from sam2.build_sam import build_sam2
            cfg = saved.get('model_cfg', 'configs/sam2.1/sam2.1_hiera_b+.yaml')
            base = build_sam2(cfg, None, device='cpu', mode='eval', apply_postprocessing=False)
            model = native.SAM2FixedFullBox(base)
            model.sam.load_state_dict(payload['model'], strict=True)
            record['model_cfg'] = cfg
        elif name == 'medsam':
            native_source = source_root([source(args, 'medsam')], 'segment_anything/build_sam.py')
            registry, _ = native.import_medsam_registry(native_source)
            # This MedSAM builder unconditionally converts checkpoint to Path.
            # Supply the complete trained state, rather than a missing initialization file.
            with tempfile.TemporaryDirectory(prefix='medsam_init_', dir=dest) as temporary:
                state_file = Path(temporary) / 'trained_state.pth'
                torch.save(payload['model'], state_file)
                base = registry['vit_b'](checkpoint=str(state_file))
            model = native.MedSAMFixedFullBox(base, input_size)
            model.sam.load_state_dict(payload['model'], strict=True)
        else:
            native_source = source_root([source(args, 'samus')], 'models/segment_anything_samus/build_sam_us.py')
            registry, _ = native.import_samus_registry(native_source)
            base = registry['vit_b'](args=SimpleNamespace(encoder_input_size=input_size), checkpoint=None)
            model = native.SAMUSFixedFullBox(base, input_size)
            model.samus.load_state_dict(payload['model'], strict=True)
        del payload
        dataset = native.ConsolidationDataset(args.subset / 'test', (1, 255), 0, input_size, False)
        amp_name = saved.get('amp_dtype', 'auto')
        amp_name = native.choose_amp_config(amp_name).name
        def decode(x):
            logits, _ = model(x)
            logits = F.interpolate(logits.float(), (size, size), mode='bilinear', align_corners=False)
            return torch.sigmoid(logits[:, 0]) >= .5
        record.update(source=str(native_source), input_size=input_size, prompt='fixed_full_image_box_no_gt')

    elif name == 'sam3':
        native = load('evaluation/foundation/evaluate_sam3_official.py', 'smoke_sam3_native')
        native_source = source_root([source(args, 'sam3')], 'sam3/model_builder.py')
        checkpoint = require_file(args.pth / 'sam3' / str(size) / 'best_dice.pt')
        sys.path.insert(0, str(native_source))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        bpe = require_file(native_source / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz')
        model = build_sam3_image_model(bpe_path=str(bpe), device='cpu', eval_mode=True,
                                      checkpoint_path=None, load_from_HF=False, enable_segmentation=True)
        native.strict_load_finetuned_state(model, checkpoint, 'auto')
        model.to(device).eval()
        processor = Sam3Processor(model, resolution=1008, device=str(device), confidence_threshold=.5)
        def predict_image(path):
            with Image.open(path) as handle:
                image = handle.convert('RGB')
            with native.amp_context('bf16'):
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt='foreground')
            return native.union_prediction(output, size, .5)
        record.update(checkpoint=str(checkpoint), source=str(native_source), prompt='foreground', amp='bf16')
        return None, predict_image, record

    elif name.startswith('yolo'):
        from ultralytics import YOLO
        native = load('experiments/yolo/train_family.py', 'smoke_yolo_native')
        folder = 'yolo26s_sem' if name == 'yolo26s-sem' else name
        directory = args.pth / 'Yolo' / str(size) / folder
        checkpoint = require_file(directory / 'best_dice.pt')
        settings = read_json(directory / 'run_config.json')
        instance = name.startswith('yolo11')
        model = YOLO(str(checkpoint), task='segment' if instance else 'semantic')
        conf, iou = float(settings.get('instance_conf', .25)), float(settings.get('instance_iou', .70))
        def predict_image(path):
            kwargs = dict(imgsz=size, batch=1, device='0', verbose=False, stream=False, save=False)
            if instance:
                kwargs.update(conf=conf, iou=iou, retina_masks=True)
            results = model.predict(source=str(path), **kwargs)
            if len(results) != 1:
                raise ValueError('YOLO must return one result for one image')
            return (native.instance_prediction(results[0], size, conf) if instance
                    else native.semantic_prediction(results[0], size))
        record.update(checkpoint=str(checkpoint), task='segment' if instance else 'semantic',
                      instance_conf=conf if instance else None, instance_iou=iou if instance else None)
        return None, predict_image, record
    else:
        raise ValueError(name)

    model.to(device).eval()
    record.update(checkpoint=str(checkpoint), amp=amp_name)
    def predict_tensor(x):
        use_amp = amp_name in {'bf16', 'fp16'}
        with torch.autocast('cuda', dtype=torch.bfloat16 if amp_name == 'bf16' else torch.float16,
                            enabled=use_amp):
            pred = decode(x[None].to(device))
        if tuple(pred.shape) != (1, size, size):
            raise ValueError(f'Unexpected native prediction shape: {tuple(pred.shape)}')
        return pred[0].detach().cpu().numpy()
    return dataset, predict_tensor, record


def run(args, cases):
    import numpy as np
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable: {sys.executable}')
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    size = args.size
    dest = args.out / str(size) / args.worker
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'masks').mkdir()
    print(f'Loading {args.worker} checkpoint with {sys.executable}', flush=True)
    dataset, predict, record = prepare(args, dest)
    expected = {c['image_id']: c for c in cases}
    if dataset is not None and len(dataset) != 5:
        raise ValueError(f'Dataset length {len(dataset)} differs from locked five-case subset')
    rows = []
    with torch.inference_mode():
        for i in range(5):
            if dataset is None:
                case = cases[i]
                name = Path(case['image']).name
                pred = predict(args.subset / 'test/images' / name)
            else:
                sample = dataset[i]
                if isinstance(sample, dict):
                    x, name = sample['image'], sample.get('name', sample.get('case_name'))
                else:
                    x, _, name = sample
                pred = predict(x)
            identifier = Path(name).stem
            if identifier not in expected or any(r['image_id'] == identifier for r in rows):
                raise ValueError(f'Unexpected or duplicate case: {name}')
            pred = np.asarray(pred)
            if pred.shape != (size, size) or not np.isfinite(pred).all() or not set(np.unique(pred)).issubset({0, 1}):
                raise ValueError(f'Prediction must be a finite binary {size}x{size} mask')
            pred = pred.astype(bool)
            target = dest / 'masks' / (identifier + '.png')
            Image.fromarray(pred.astype(np.uint8) * 255).save(target)
            # This reference is never passed into model/predictor or prompt builder.
            gt_path = args.subset / 'test/masks' / Path(expected[identifier]['mask']).name
            gt = np.asarray(Image.open(gt_path))
            if gt.shape != (size, size) or not set(np.unique(gt)).issubset({0, 1, 255}):
                raise ValueError(f'Invalid binary reference: {gt_path}')
            gt = gt > 0
            denominator = int(gt.sum()) + int(pred.sum())
            rows.append({'image_id': identifier, 'pred_pixels': int(pred.sum()),
                         'dice': 2 * int((gt & pred).sum()) / denominator if denominator else 1.0,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
            print(f'[{args.worker}] {i+1}/5 saved {target.name}', flush=True)
    result = dict(record, model=args.worker, size=size, python=sys.executable, cases=rows,
                  selected_cases=cases, status='complete', scope='Five-case diagnostic only; not a benchmark estimate')
    (dest / 'result.json').write_text(json.dumps(result, indent=2, default=str))
