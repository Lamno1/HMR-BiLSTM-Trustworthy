"""Run the public CPU reviewer workflow from a saved results snapshot."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=['all', 'tables', 'ranking', 'figures'], default='all')
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'paper_results/v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reviewer')
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    output = args.output.resolve()
    if not (snapshot / 'manifest.json').is_file():
        parser.error(f'Snapshot manifest is missing from {snapshot}')
    if output.is_relative_to(snapshot) or output.is_relative_to(ROOT / 'paper_results'):
        parser.error('Choose an output directory outside the versioned snapshot')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'logs').mkdir(exist_ok=True)

    steps = {
        'tables': ['-m', 'validation.check_bel_tables', '--public-snapshot', str(snapshot)],
        'ranking': ['-m', 'validation.audit_selection_rankings',
                    '--results-dir', str(snapshot / 'results'), '--out-dir', str(output / 'ranking')],
        'figures': ['-m', 'validation.plot_submission_figures',
                    '--tables-dir', str(snapshot / 'results/tables'), '--out-dir', str(output / 'figures')],
    }
    selected = list(steps) if args.step == 'all' else [args.step]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    git_status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
    versions = {}
    for package in ['numpy', 'scipy', 'scikit-learn', 'matplotlib']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    report = {
        'status': 'running', 'code_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'working_tree_dirty': bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
        'python': sys.version, 'platform': platform.platform(), 'packages': versions,
        'snapshot': str(snapshot), 'output': str(output), 'steps': [],
    }
    report_path = output / 'reviewer_report.json'
    write_report(report_path, report)
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')

    for index, step in enumerate(selected, start=1):
        print(f'[{index}/{len(selected)}] {step}', flush=True)
        command = [sys.executable, *steps[step]]
        log_path = output / 'logs' / f'{step}.txt'
        entry = {'name': step, 'command': command, 'log': str(log_path)}
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True,
                                     encoding='utf-8', env=environment)
            entry['exit_code'] = process.returncode
            log_path.write_text(process.stdout + process.stderr, encoding='utf-8')
            if process.returncode:
                raise RuntimeError(f'Command exited with code {process.returncode}')
            if step == 'tables':
                print(process.stdout.strip().splitlines()[-1])
            elif step == 'ranking':
                actual = json.loads((output / 'ranking/ranking_diagnostics.json').read_text())
                expected = json.loads((snapshot / 'results/tables/ranking_diagnostics.json').read_text())
                entry['matches_snapshot'] = actual == expected
                if not entry['matches_snapshot']:
                    raise RuntimeError('Ranking diagnostics differ from the published snapshot')
                print('Ranking diagnostics match the snapshot.')
                print(f"Mean selection regret: {actual['mean_validation_regret']:.6f}")
            else:
                expected_files = [f'{name}.{extension}'
                                  for name in ['confusion_matrix_seeds', 'transfer_auc', 'validation_agreement']
                                  for extension in ['pdf', 'svg', 'png']]
                missing = [name for name in expected_files if not (output / 'figures' / name).is_file()]
                if missing:
                    entry['missing_files'] = missing
                    raise RuntimeError(f'Missing figure files: {missing}')
                print('Created three figures in PDF, SVG and PNG formats.')
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as error:
            entry['status'] = 'failed'
            entry['error'] = str(error)
            report['steps'].append(entry)
            report['status'] = 'failed'
            write_report(report_path, report)
            print(f'{step} failed: {error}', file=sys.stderr)
            print(f'Details: {log_path}', file=sys.stderr)
            raise SystemExit(entry.get('exit_code') or 1) from error
        entry['status'] = 'passed'
        report['steps'].append(entry)
        write_report(report_path, report)

    report['status'] = 'passed'
    write_report(report_path, report)
    print(f'Reviewer workflow passed. Report: {report_path}')


if __name__ == '__main__':
    main()
