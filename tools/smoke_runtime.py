"""Isolated per-environment execution, verified resume, and feedback collection."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


def environment_for(args, model, overrides):
    from runtime_config import interpreter, source, configuration, configured_path
    python = os.path.abspath(os.path.expanduser(overrides.get(model, interpreter(args, model, sys.executable))))
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'VIRTUAL_ENV', 'PYTHONPATH'):
        env.pop(key, None)
    env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1')
    paths = [Path(__file__).resolve().parents[1], source(args, 'april')]
    if model == 'USFM':
        paths += [source(args, 'usfm'), source(args, 'usfm_mmseg')]
    if model in {'sam2', 'medsam', 'samus', 'sam3'}:
        paths += [source(args, model)]
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    data, base = configuration(args)
    if data.get('cuda_home'):
        toolkit = configured_path(data['cuda_home'], base)
        env['CUDA_HOME'] = str(toolkit)
        env['PATH'] = str(toolkit / 'bin') + os.pathsep + env['PATH']
    env.setdefault('MAX_JOBS', '4')
    return python, env


def validate_output(dest, cases, model, size=512, expected_signature=None):
    """Validate previous masks, case identity, and file hashes before skipping."""
    import numpy as np
    from PIL import Image
    try:
        result = json.loads((dest / 'result.json').read_text())
        if result.get('metric_protocol') != 'hd_empty_track_side_v1':
            return False, 'Output predates track-side HD diagnostics'
        if expected_signature is not None and result.get('code_signature') != expected_signature:
            return False, 'Code or runtime configuration changed'
        if result.get('status') != 'complete' or result.get('model') != model or result.get('size', 512) != size:
            return False, 'Incomplete or mismatched result.json'
        expected = {c['image_id'] for c in cases}
        rows = result.get('cases', [])
        if len(rows) != 5 or {r['image_id'] for r in rows} != expected:
            return False, 'Saved case set differs from locked subset'
        if 'selected_cases' in result and result['selected_cases'] != cases:
            return False, 'Saved input hashes or manifest changed'
        if not (dest / 'metrics.csv').is_file():
            return False, 'Missing native metric diagnostics'
        from checkpoint_diagnostics import file_hash
        if result.get('checkpoint_sha256') != file_hash(result['checkpoint']):
            return False, 'Trained checkpoint changed'
        actual = {p.name for p in (dest / 'masks').iterdir() if p.is_file()}
        if actual != {name + '.png' for name in expected}:
            return False, 'Expected exactly five mask PNGs'
        for row in rows:
            path = dest / 'masks' / (row['image_id'] + '.png')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                return False, f'Changed mask: {path.name}'
            with Image.open(path) as handle:
                mask = np.asarray(handle)
            if mask.shape != (size, size) or not set(np.unique(mask)).issubset({0, 255}):
                return False, f'Invalid binary mask: {path.name}'
        return True, 'Five matching binary masks and hashes verified'
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, str(error)


def archive_partial(dest):
    # Rename only the model's exact output directory, within the same parent.
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target = dest.with_name(dest.name + '_incomplete_' + stamp)
    dest.rename(target)
    return str(target)


def stop_process(process):
    if process.poll() is None:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(args, cases, all_models):
    from checkpoint_diagnostics import code_signature
    signature = code_signature(args)
    args.out.mkdir(parents=True, exist_ok=True)
    logs = args.out / 'logs'
    logs.mkdir(exist_ok=True)
    overrides = json.loads(args.python_map.read_text()) if args.python_map else {}
    if not isinstance(overrides, dict) or any(k not in all_models or not isinstance(v, str) for k, v in overrides.items()):
        raise ValueError('--python-map must map supported model names to absolute Python executable paths')
    report = {'selected_cases': cases, 'python': sys.executable, 'results': [],
              'supported_models': all_models, 'pending_adapters': [],
              'size': args.size, 'scope': 'five fixed patients in selected track; no training; diagnostic masks only'}
    for i, model in enumerate(args.models, 1):
        dest = args.out / str(args.size) / model
        result = {'model': model}
        valid, reason = validate_output(dest, cases, model, args.size, signature)
        if valid:
            result.update(status='reused', detail=reason)
            print(f'[{i}/{len(args.models)}] {model}: reused (5/5 masks verified)', flush=True)
        else:
            if dest.exists():
                result['archived_incomplete'] = archive_partial(dest)
                print(f'[{model}] incomplete output archived: {reason}', flush=True)
            python, env = environment_for(args, model, overrides)
            result['python'] = python
            log = logs / (model + '.log')
            if log.exists():
                log.rename(log.with_name(log.stem + '_previous_' + str(time.time_ns()) + '.log'))
            print(f'[{i}/{len(args.models)}] {model}: starting (up to {args.timeout}s)\n  Python: {python}', flush=True)
            command = [python, '-u', str(Path(__file__).with_name('smoke_512.py')), '--worker', model,
                       '--pth', str(args.pth), '--out', str(args.out), '--subset', str(args.subset), '--size', str(args.size)]
            if args.runtime_config:
                command += ['--runtime-config', str(args.runtime_config)]
            if args.segformer_base:
                command += ['--segformer-base', args.segformer_base]
            if args.april_root:
                command += ['--april-root', str(args.april_root)]
            process = None
            try:
                if not Path(python).is_file():
                    raise FileNotFoundError(f'Python environment missing: {python}; supply --python-map')
                with log.open('w') as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=env, cwd=str(Path(__file__).resolve().parents[1]),
                                               start_new_session=True)
                    started = time.monotonic()
                    offset = 0
                    while process.poll() is None:
                        remaining = args.timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            stop_process(process)
                            result['detail'] = 'Timeout; worker process group stopped'
                            break
                        try:
                            process.wait(timeout=min(20, remaining))
                        except subprocess.TimeoutExpired:
                            print(f'[{model}] running {int(time.monotonic()-started)}s; log: {log}', flush=True)
                        with log.open('r', errors='replace') as reader:
                            reader.seek(offset)
                            lines = reader.read().splitlines()
                            offset = reader.tell()
                        for line in lines[-6:]:
                            if line.strip():
                                print('  ' + line[:500], flush=True)
                result['returncode'] = process.returncode
                valid, reason = validate_output(dest, cases, model, args.size, signature)
                result['status'] = 'complete' if process.returncode == 0 and valid else 'failed'
                if not valid:
                    result['output_check'] = reason
                if result['status'] == 'failed':
                    print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]), flush=True)
            except KeyboardInterrupt:
                if process:
                    stop_process(process)
                raise
            except Exception as error:
                if process:
                    stop_process(process)
                result.update(status='failed', detail=f'{type(error).__name__}: {error}')
                with log.open('a') as stream:
                    stream.write('\n' + result['detail'] + '\n')
                print(result['detail'], flush=True)
            print(f'[{model}] {result["status"]}', flush=True)
        report['results'].append(result)
        (args.out / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    # Include all prior complete results, even after a one-model retry.
    report['coverage'] = {m: validate_output(args.out / str(args.size) / m, cases, m, args.size, signature)[0] for m in all_models}
    report['complete_model_count'] = sum(report['coverage'].values())
    (args.out / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    archive = args.out / 'smoke_feedback.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('smoke_report.json', json.dumps(report, indent=2))
        for model in all_models:
            for path, name in [(logs / (model + '.log'), 'logs/' + model + '.log'),
                               (args.out / str(args.size) / model / 'result.json', 'results/' + model + '.json'),
                               (args.out / str(args.size) / model / 'metrics.csv', 'metrics/' + model + '.csv')]:
                if path.is_file():
                    z.write(path, name)
        for path in args.pth.rglob('*.json'):
            if str(args.size) in path.parts and path.name in {'run_settings.json', 'run_config.json', 'evaluation_settings.json', 'training_settings.json', 'result.json'} and path.stat().st_size < 1_000_000:
                z.write(path, 'Pth/' + path.relative_to(args.pth).as_posix())
    print(f'Verified output coverage: {report["complete_model_count"]}/{len(all_models)} models. Return {archive}', flush=True)
    return int(any(r['status'] not in {'complete', 'reused'} for r in report['results']))
