#!/usr/bin/env python3
"""Automatic environment reconstruction is retired; use manual family setup."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    print('No packages were installed or changed.')
    print('Create separate model-family environments following: ' + str(ROOT / 'docs/ENVIRONMENTS.md'))
    print('Map existing Python executables in configs/runtime.local.json.')
    print('observed_runtime_versions.json is an inventory, not an installable lock file.')
    print('Run tools/run_release_checks.py with --runtime-config after manual setup.')
    # Stop legacy automatic launchers before they use a partially built environment.
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
