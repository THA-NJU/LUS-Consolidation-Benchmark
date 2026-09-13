#!/usr/bin/env python3
"""Dual-track checkpoint evaluation and minimal native-training diagnostics."""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lusbench.registry import ROUTES
from runtime_config import source
from smoke_runtime import environment_for, stop_process
from train import build_command

ALIASES = {'fpn_resnet34': 'Resnet34_fpn', 'deeplabv3plus_resnet34': 'Resnet34_DLV3', 'usfm_transfer': 'USFM'}


def training_plan(all_models=False):
    selected, seen = [], set()
    for route in ROUTES:
        if not route.script:
            continue
        # Exercise each distinct native backend/family/track once by default.
        key = (route.route, route.script, route.family, route.size)
        if all_models or key not in seen:
            selected.append(route)
            seen.add(key)
    # YOLO instance and semantic trainers have distinct loss and data paths.
    for route in ROUTES:
        if route.model == 'yolo26m-sem' and route not in selected:
            selected.append(route)
    return selected


def extra_training_args(args, route, folder):
    name, size = route.model, route.size
    pth = args.pth
    extras, probe = [], []
    if route.route == 'segformer_native':
        extras = ['--base-model', str(ROOT / 'configs/segformer-b2')]
    elif route.route == 'yolo26s_native':
        extras = ['--model-weights', str(pth / 'Yolo' / str(size) / 'yolo26s_sem/best_dice.pt')]
    elif route.route == 'yolo_family':
        extras = ['--weights', str(pth / 'Yolo' / str(size) / name / 'best_dice.pt'),
                  '--batch-size', '2', '--effective-batch-size', '2', '--eval-batch-size', '2',
                  '--workers', '0', '--no-amp', '--labels', '1', '255', '--cache-root', str(folder / 'cache')]
    elif route.route == 's2denet':
        extras = ['--repo-root', str(source(args, 's2denet')), '--micro-batch', '2']
    elif route.route == 'usfm224':
        extras = ['--checkpoint', str(pth / 'Transformer/512/USFM/best_usfm_decoder.pth'),
                  '--batch-size', '2', '--grad-accum', '1', '--workers', '0', '--skip-count-check']
    elif route.route in {'sam512', 'sam224'}:
        checkpoint = pth / 'SAM' / str(size) / name / 'best_model.pth'
        if name in {'medsam', 'samus'}:
            target = folder / 'initial_state.pth'
            probe = ['--initial-state', str(checkpoint), '--initial-target', str(target)]
            checkpoint = target
            extras += [f'--{name}-source-dir', str(source(args, name))]
        extras += ['--checkpoint', str(checkpoint), '--batch-size', '2', '--grad-accum', '1',
                   '--num-workers', '0', '--warmup-epochs', '0', '--epochs', '1', '--patience', '1']
        if size == 224:
            extras += ['--skip-rich-eval']
        else:
            extras += ['--foreground-values', '1,255', '--min-mask-pixels', '1']
    elif route.route == 'sam3':
        if not args.sam3_initial_checkpoint or not args.sam3_initial_checkpoint.is_file():
            raise FileNotFoundError('SAM3 native interactive training requires --sam3-initial-checkpoint pointing to standard sam3.pt')
        extras = ['--checkpoint', str(args.sam3_initial_checkpoint),
                  '--sam3-source-dir', str(source(args, 'sam3')), '--epochs', '1',
                  '--warmup-epochs', '0', '--patience', '1', '--num-workers', '0',
                  '--grad-accum', '1', '--min-mask-pixels', '1', '--foreground-values', '1,255']
    return extras, probe


def execute(command, env, log, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Log: ' + str(log), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        try:
            while process.poll() is None:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    stop_process(process)
                    return 124
                try:
                    process.wait(timeout=min(20, remaining))
                except subprocess.TimeoutExpired:
                    print(f'  running {int(time.monotonic()-start)}s', flush=True)
        except BaseException:
            stop_process(process)
            raise
    if process.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-16:]), flush=True)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pth', type=Path, required=True)
    parser.add_argument('--subsets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--phase', choices=('evaluate', 'train', 'all'), default='all')
    parser.add_argument('--all-training-models', action='store_true')
    parser.add_argument('--evaluation-models', nargs='+', help='Evaluate only these smoke model IDs')
    parser.add_argument('--training-jobs', nargs='+', help='Run only SIZE_MODEL tags from the training plan')
    parser.add_argument('--sam3-initial-checkpoint', type=Path)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--training-timeout', type=int, default=1800)
    args = parser.parse_args()
    for key in ('pth', 'subsets', 'out', 'runtime_config', 'sam3_initial_checkpoint'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().absolute())
    if args.training_jobs:
        available = {f'{r.size}_{r.model}' for r in training_plan(args.all_training_models)}
        unknown = set(args.training_jobs) - available
        if unknown:
            parser.error('Unknown training jobs: ' + ', '.join(sorted(unknown)))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'evaluation': [], 'training': [], 'public_release_ready': False,
              'scope': '36 models per track for inference; representative native training backends by default',
              'training_coverage': 'all available routes' if args.all_training_models else 'one model per backend/family/track plus both YOLO task kinds'}
    report['requested_evaluation_models'] = args.evaluation_models or 'all'
    report['requested_training_jobs'] = args.training_jobs or 'all in selected plan'
    report['scope'] = 'Only the selected checks in this invocation; earlier results are not revalidated.'
    if args.phase in {'evaluate', 'all'}:
        for size in (512, 224):
            print(f'=== {size}: selected-model evaluation ===', flush=True)
            command = [sys.executable, '-u', str(ROOT / 'tools/smoke.py'), '--size', str(size),
                       '--pth', str(args.pth), '--subset', str(args.subsets / str(size)),
                       '--out', str(args.out / f'evaluate_{size}'), '--timeout', str(args.timeout)]
            if args.evaluation_models:
                command += ['--models', *args.evaluation_models]
            if args.runtime_config:
                command += ['--runtime-config', str(args.runtime_config)]
            # Smoke runner already provides per-model progress, isolation and timeout.
            code = subprocess.run(command, cwd=ROOT).returncode
            report['evaluation'].append({'size': size, 'returncode': code})
    if args.phase in {'train', 'all'}:
        plan = training_plan(args.all_training_models)
        if args.training_jobs:
            plan = [r for r in plan if f'{r.size}_{r.model}' in args.training_jobs]
        for index, route in enumerate(plan, 1):
            tag = f'{route.size}_{route.model}'
            folder = args.out / 'train' / tag
            if folder.exists():
                folder = folder.with_name(folder.name + '_retry_' + str(time.time_ns()))
            folder.mkdir(parents=True)
            result = {'model': route.model, 'size': route.size, 'output': str(folder)}
            print(f'=== training {index}/{len(plan)}: {tag} ===', flush=True)
            try:
                extras, probe = extra_training_args(args, route, folder)
                params = SimpleNamespace(model=route.model, size=route.size,
                                         data_root=args.subsets / str(route.size),
                                         output_dir=folder / 'native', run_mode='smoke',
                                         april_root=source(args, 'april'), extra=extras)
                native = build_command(params)
                python, env = environment_for(args, ALIASES.get(route.model, route.model), {})
                command = [python, '-u', str(ROOT / 'tools/training_probe.py'),
                           '--report', str(folder / 'training_check.json'), *probe, '--', *native[1:]]
                code = execute(command, env, folder / 'training.log', args.training_timeout)
                result.update(returncode=code, status='complete' if code == 0 else 'failed')
            except Exception as error:
                result.update(status='blocked', detail=f'{type(error).__name__}: {error}')
                print(result['detail'], flush=True)
            report['training'].append(result)
            (args.out / 'release_report.json').write_text(json.dumps(report, indent=2))
    report['selected_checks_passed'] = (all(r['returncode'] == 0 for r in report['evaluation'])
                                       and all(r['status'] == 'complete' for r in report['training']))
    report['release_note'] = 'Runtime results require review, including import locations and independent environment installation; not automatic publication approval.'
    (args.out / 'release_report.json').write_text(json.dumps(report, indent=2))
    archive = args.out / 'release_feedback.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.out.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.csv', '.log'} and path.stat().st_size < 5_000_000:
                z.write(path, path.relative_to(args.out).as_posix())
    print(f'Return {archive}', flush=True)
    return 0 if report['selected_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
