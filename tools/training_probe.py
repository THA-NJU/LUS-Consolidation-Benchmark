#!/usr/bin/env python3
"""Instrument an actual native training entry point for a minimal runtime check.

This does not replace its model, loss, optimizer, or checkpoint implementation.
Use a separate tiny train/val/test dataset and output directory.
"""
import argparse
from checkpoint_tracking import CheckpointTracker
import json
import os
import runpy
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--initial-target', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Expected a native Python script and its arguments after --')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    import torch
    os.environ['LUSBENCH_TRAINING_PROBE'] = '1'
    if args.initial_state:
        payload = torch.load(args.initial_state, map_location='cpu', weights_only=False)
        state = payload.get('model', payload)
        args.initial_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.initial_target)
        del payload, state
    report = {'status': 'running', 'optimizer_steps': 0, 'changed_steps': 0,
              'backward_calls': 0, 'nonfinite_backward_inputs': 0,
              'nonfinite_gradient_steps': 0, 'saved_checkpoints': [], 'reloaded_checkpoints': [],
              'native_script': command[0], 'python': sys.executable,
              'scope': 'Native training execution on tiny split subsets; not convergence or full-run reproduction'}
    original_save, original_load = torch.save, torch.load

    tracker = CheckpointTracker(report)

    def save(payload, target, *a, **kw):
        result = original_save(payload, target, *a, **kw)
        if report['changed_steps']:
            tracker.saved(target)
        return result

    def load(target, *a, **kw):
        digest = tracker.digest(target)
        result = original_load(target, *a, **kw)
        tracker.loaded(target, digest)
        return result

    torch.save, torch.load = save, load
    original_backward = torch.autograd.backward
    def backward(tensors, *a, **kw):
        values = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
        report['backward_calls'] += 1
        report['nonfinite_backward_inputs'] += int(any(not bool(torch.isfinite(t.detach()).all()) for t in values))
        return original_backward(tensors, *a, **kw)
    torch.autograd.backward = backward

    active_optimizers = set()

    def wrap(original):
        def step(optimizer, *a, **kw):
            # AdamW can call Adam.step internally in newer Torch versions.
            if id(optimizer) in active_optimizers:
                return original(optimizer, *a, **kw)
            tracked = None
            finite = True
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    grad = param.grad.detach()
                    values = grad.coalesce().values() if grad.is_sparse else grad
                    finite = finite and bool(torch.isfinite(values).all())
                    if tracked is None and not grad.is_sparse and bool(values.abs().max() > 0):
                        tracked = (param, param.detach().clone())
            active_optimizers.add(id(optimizer))
            try:
                result = original(optimizer, *a, **kw)
            finally:
                active_optimizers.discard(id(optimizer))
            report['optimizer_steps'] += 1
            report['nonfinite_gradient_steps'] += int(not finite)
            if tracked is not None and not torch.equal(tracked[0].detach(), tracked[1]):
                report['changed_steps'] += 1
            print(f'[training probe] step={report["optimizer_steps"]} changed={report["changed_steps"]} finite={finite}', flush=True)
            return result
        return step

    for name in ('Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 'Adamax'):
        cls = getattr(torch.optim, name)
        cls.step = wrap(cls.step)
    exit_code = 0
    try:
        sys.path.insert(0, str(Path(command[0]).resolve().parent))
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        if error.code is not None and not isinstance(error.code, int):
            report['error'] = str(error.code)
            print(error.code, file=sys.stderr, flush=True)
    except Exception:
        exit_code = 1
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        passed = (exit_code == 0 and report['changed_steps'] > 0
                  and report['backward_calls'] > 0 and report['nonfinite_backward_inputs'] == 0
                  and report['nonfinite_gradient_steps'] == 0
                  and bool(report['saved_checkpoints']) and bool(report['reloaded_checkpoints']))
        report.update(status='complete' if passed else 'failed', native_returncode=exit_code)
        args.report.write_text(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
