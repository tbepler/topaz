"""Pre-release checks for topaz, run locally (not in CI).

Usage:
    python test/release/checks.py mutants [--mutant M1 ...] [-m "not slow"]

`mutants` is the acceptance gate for the pytest suite: each patch in test/mutants/ re-introduces
a real regression into a copy of the working-tree topaz/ package, the suite runs against that
copy, and every catcher listed in mutants.json must fail with an assertion failure. A control run
on the unpatched copy must pass first, so a suite that is already red can't "catch" anything.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MUTANTS_DIR = REPO_ROOT / 'test' / 'mutants'


def load_mutants(ids=None):
    with open(MUTANTS_DIR / 'mutants.json') as f:
        mutants = json.load(f)['mutants']
    if ids:
        unknown = set(ids) - {m['id'] for m in mutants}
        if unknown:
            raise ValueError(f'unknown mutant ids: {sorted(unknown)}')
        mutants = [m for m in mutants if m['id'] in ids]
    return mutants


def copy_topaz(dest):
    """Copy the working-tree topaz/ package into dest (pretrained weights are symlinked, not copied)."""
    src = REPO_ROOT / 'topaz'
    shutil.copytree(src, dest / 'topaz', ignore=shutil.ignore_patterns('__pycache__', 'pretrained'))
    os.symlink(src / 'pretrained', dest / 'topaz' / 'pretrained')


def apply_patch(patch, root):
    for check in (['--check'], []):
        proc = subprocess.run(['git', 'apply'] + check + [str(patch)], cwd=root, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'{patch.name} does not apply to the current topaz/ (regenerate it):\n{proc.stderr}')


def run_pytest(root, marker_expr, junit_path):
    """Run the suite in this repo against the topaz package under `root`."""
    env = dict(os.environ, PYTHONPATH=str(root), TOPAZ_EXPECT_ROOT=str(root / 'topaz'), PYTHONNOUSERSITE='1')
    cmd = ['pytest', '-q', '-p', 'no:cacheprovider', '-m', marker_expr, f'--junitxml={junit_path}']
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return proc


def parse_junit(path):
    """{'<classname>::<name>': 'passed' | 'failed' | 'error' | 'skipped'}"""
    results = {}
    for case in ET.parse(path).getroot().iter('testcase'):
        key = f"{case.get('classname')}::{case.get('name')}"
        if case.find('failure') is not None:
            results[key] = 'failed'
        elif case.find('error') is not None:
            results[key] = 'error'
        elif case.find('skipped') is not None:
            results[key] = 'skipped'
        else:
            results[key] = 'passed'
    return results


def judge(mutant, proc, results):
    """Caught only if pytest exited non-zero and every catcher matched at least one test,
    all of whose matches failed by assertion (not error/skip)."""
    rows = []
    for catcher in mutant['catchers']:
        matches = {k: v for k, v in results.items() if catcher in k}
        if not matches:
            status = 'missing'
        elif all(v == 'failed' for v in matches.values()):
            status = 'caught'
        elif any(v == 'error' for v in matches.values()):
            status = 'error'
        else:
            status = 'survived'
        rows.append({'mutant': mutant['id'], 'catcher': catcher, 'status': status, 'n_tests': len(matches)})
    if not rows:
        rows.append({'mutant': mutant['id'], 'catcher': '(none listed)', 'status': 'survived', 'n_tests': 0})
    caught = proc.returncode != 0 and all(r['status'] == 'caught' for r in rows)
    return caught, rows


def mutants(ids=None, marker_expr='not slow', verbose=True):
    """Run the mutant gate. Returns (summary rows, catcher rows) as lists of dicts."""
    selected = load_mutants(ids)
    summary, detail = [], []
    with tempfile.TemporaryDirectory(prefix='topaz-mutants-') as tmp:
        tmp = Path(tmp)

        control = tmp / 'control'
        control.mkdir()
        copy_topaz(control)
        proc = run_pytest(control, marker_expr, tmp / 'control.xml')
        if proc.returncode != 0:
            raise RuntimeError('control run (unpatched topaz) failed; fix the suite before running mutants:\n'
                               + proc.stdout[-3000:] + proc.stderr[-3000:])
        n_control = len(parse_junit(tmp / 'control.xml'))
        if verbose:
            print(f'control: {n_control} tests passed', flush=True)

        for mutant in selected:
            root = tmp / mutant['id']
            root.mkdir()
            copy_topaz(root)
            apply_patch(MUTANTS_DIR / mutant['patch'], root)
            junit = tmp / f"{mutant['id']}.xml"
            proc = run_pytest(root, marker_expr, junit)
            if not junit.exists():
                raise RuntimeError(f"pytest produced no report for {mutant['id']}:\n{proc.stdout[-3000:]}{proc.stderr[-3000:]}")
            results = parse_junit(junit)
            caught, rows = judge(mutant, proc, results)
            n_failed = sum(v == 'failed' for v in results.values())
            summary.append({'mutant': mutant['id'], 'caught': caught, 'pytest_rc': proc.returncode,
                            'failed_tests': n_failed, 'total_tests': len(results)})
            detail.extend(rows)
            if verbose:
                print(f"{mutant['id']}: {'CAUGHT' if caught else 'SURVIVED'} "
                      f"(rc={proc.returncode}, {n_failed}/{len(results)} tests failed)", flush=True)
                for r in rows:
                    print(f"    {r['status']:>8}  {r['catcher']}  ({r['n_tests']} tests)", flush=True)
    return summary, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('mutants', help='run the mutant acceptance gate')
    p.add_argument('--mutant', nargs='+', help='only these mutant ids (default: all)')
    p.add_argument('-m', dest='marker_expr', default='not slow', help='pytest marker expression (default: "not slow")')
    args = parser.parse_args()

    if args.command == 'mutants':
        summary, _ = mutants(args.mutant, args.marker_expr)
        sys.exit(0 if all(s['caught'] for s in summary) else 1)


if __name__ == '__main__':
    main()
