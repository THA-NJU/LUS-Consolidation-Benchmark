#!/usr/bin/env python3
"""Fetch reviewed external source commits into this checkout, without weights."""
import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, capture=False):
    return subprocess.run(['git', '-C', str(path), *args], check=True,
                          text=True, capture_output=capture)


def checkout_changes(target):
    """Ignore only untracked Python bytecode, never source or tracked changes."""
    status = git(target, 'status', '--porcelain=v1', '-z', '--untracked-files=all', capture=True).stdout
    records = iter(status.split('\0'))
    changes = []
    for record in records:
        if not record:
            continue
        state, name = record[:2], record[3:]
        if 'R' in state or 'C' in state:
            next(records, None)  # NUL format includes a second path for renames/copies.
        path = PurePosixPath(name)
        if state == '??' and '__pycache__' in path.parts and path.suffix == '.pyc':
            continue
        changes.append(record)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / '.external')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / 'configs/external_sources.json').read_text())
    for index, (name, spec) in enumerate(sources.items(), 1):
        target = args.root / name
        print(f'[{index}/{len(sources)}] {name}: {spec["commit"]}', flush=True)
        if target.exists():
            if not (target / '.git').is_dir():
                raise RuntimeError(f'Refusing existing non-Git source directory: {target}')
            head = subprocess.run(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True, capture_output=True)
            if head.returncode:
                remote = git(target, 'remote', 'get-url', 'origin', capture=True).stdout.strip()
                dirty = git(target, 'status', '--porcelain', capture=True).stdout.strip()
                if remote != spec['url'] or dirty:
                    raise RuntimeError(f'Refusing unknown partial source checkout: {target}')
                git(target, 'fetch', '--depth', '1', 'origin', spec['commit'])
                git(target, 'checkout', '--detach', 'FETCH_HEAD')
            current = git(target, 'rev-parse', 'HEAD', capture=True).stdout.strip()
            dirty = checkout_changes(target)
            if current != spec['commit'] or dirty:
                raise RuntimeError(f'{target} differs from pinned checkout: HEAD={current}; '
                                   f'expected={spec["commit"]}; changes={dirty[:10]}. '
                                   'Preserve these changes and inspect them before proceeding.')
        else:
            target.mkdir()
            git(target, 'init')
            git(target, 'remote', 'add', 'origin', spec['url'])
            git(target, 'fetch', '--depth', '1', 'origin', spec['commit'])
            git(target, 'checkout', '--detach', 'FETCH_HEAD')
        for required in spec.get('required', []):
            if not (target / required).is_file():
                raise FileNotFoundError(target / required)
    print('External sources prepared. Model weights remain a separate input.', flush=True)


if __name__ == '__main__':
    main()
