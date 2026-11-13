"""Shared helpers for the topaz test suite (imported by conftest.py and test modules).

Capability probes live here so that 2D/3D-parameterized tests skip 3D cases based on
what the installed topaz can actually do, never on a branch or version check. A probe
must be an explicit inspection of the code; never "try it and skip on exception".
"""
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
FIXTURES_DIR = TEST_DIR / 'fixtures'


def _option_strings(parser):
    """All option names (such as '--dims') that an argparse parser accepts."""
    return {opt for action in parser._actions for opt in action.option_strings}


def _probe_train_3d():
    """True if `topaz train` has a --dims option."""
    from topaz.commands import train
    return '--dims' in _option_strings(train.add_arguments())


def _probe_extract_3d():
    """True if `topaz extract --dims` accepts 3."""
    from topaz.commands import extract
    for action in extract.add_arguments()._actions:
        if '--dims' in action.option_strings:
            return action.choices is not None and 3 in action.choices
    return False


def _probe_model_3d():
    """True if the ResNet and BasicConv models take a `dims` argument."""
    from topaz.model.features.basic import BasicConv
    from topaz.model.features.resnet import ResNet
    return all('dims' in inspect.signature(cls.__init__).parameters for cls in (ResNet, BasicConv))


CAPABILITY_PROBES = {
    'train_3d': _probe_train_3d,
    'extract_3d': _probe_extract_3d,
    'model_3d': _probe_model_3d,
}
_capability_cache = {}


def has_capability(name):
    """Whether the installed topaz supports the named 3D feature (probed once, then cached)."""
    if name not in CAPABILITY_PROBES:
        raise KeyError(f'unknown capability {name!r}; known: {sorted(CAPABILITY_PROBES)}')
    if name not in _capability_cache:
        _capability_cache[name] = bool(CAPABILITY_PROBES[name]())
    return _capability_cache[name]


def require(capability, ndim):
    """Skip a 3D case when the installed topaz lacks `capability`. 2D cases always run."""
    import pytest
    if ndim == 3 and not has_capability(capability):
        pytest.skip(f'installed topaz lacks capability {capability!r}')


def topaz_executable():
    """Path of the `topaz` console script on PATH; raises if topaz isn't installed."""
    exe = shutil.which('topaz')
    if exe is None:
        raise RuntimeError('the `topaz` console script is not on PATH; install the package (pip install -e .[test])')
    return exe


def run_topaz(*args, cwd=None, timeout=600, check=True):
    """Run the installed `topaz` console script in a subprocess and return the CompletedProcess.

    A non-zero exit raises AssertionError showing stdout and stderr, unless check=False."""
    cmd = [topaz_executable()] + [str(a) for a in args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise AssertionError(f'{" ".join(cmd)} exited {proc.returncode}\n'
                             f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')
    return proc


def _console_script_interpreter(exe):
    """The python a console script runs under, read from its shebang."""
    with open(exe, 'rb') as f:
        first = f.readline().decode(errors='replace').strip()
    if first.startswith('#!') and 'env ' not in first:
        return first[2:].split()[0]
    return sys.executable


def console_script_topaz_file():
    """Where `import topaz` resolves for the `topaz` console script.

    Runs from an empty temp dir so the caller's cwd can't shadow the install, the same
    way the console script itself never has cwd on sys.path.
    """
    interp = _console_script_interpreter(topaz_executable())
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run([interp, '-c', 'import topaz; print(topaz.__file__)'],
                             cwd=tmp, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f'could not import topaz under {interp}:\n{out.stderr}')
    return Path(out.stdout.strip()).resolve()


def check_install_location():
    """Check that the tests import the same topaz the `topaz` CLI runs; return the folder it was imported from.

    The two differ when `python -m pytest` is run from the repo root with a non-editable install: the
    ./topaz source folder then shadows the installed package. With an editable install (pip install -e .)
    both are the source folder, so either command works. If TOPAZ_EXPECT_ROOT is set (the mutant gate
    sets it), topaz must also come from under that folder, which proves the mutant was the code tested.
    """
    import topaz
    if getattr(topaz, '__file__', None) is None:
        raise RuntimeError(f'`import topaz` resolved to a namespace package at {list(topaz.__path__)}, not the '
                           'installed package; a directory named topaz/ on sys.path is shadowing it')
    in_process = Path(topaz.__file__).resolve()
    cli = console_script_topaz_file()
    if in_process != cli:
        raise RuntimeError(f'tests import topaz from {in_process} but the `topaz` CLI imports it from {cli}. '
                           'This happens when `python -m pytest` is run from the repo root with a '
                           'non-editable install; run `pytest` instead.')
    expect_root = os.environ.get('TOPAZ_EXPECT_ROOT')
    if expect_root:
        expect_root = Path(expect_root).resolve()
        if expect_root not in in_process.parents:
            raise RuntimeError(f'TOPAZ_EXPECT_ROOT={expect_root} but topaz was imported from {in_process}')
    return str(in_process.parent)


def git_describe(path):
    """(branch, short sha, dirty) for the git checkout containing `path`, or None if it isn't one."""
    try:
        run = lambda *a: subprocess.run(['git', '-C', str(path)] + list(a), capture_output=True,
                                        text=True, timeout=10, check=True).stdout.strip()
        branch = run('rev-parse', '--abbrev-ref', 'HEAD')
        sha = run('rev-parse', '--short', 'HEAD')
        dirty = bool(run('status', '--porcelain', '--untracked-files=no'))
        return branch, sha, dirty
    except (OSError, subprocess.SubprocessError):
        return None


def array_image(array, targets, crop_size, split='pn', dims=2, mask_size=1):
    """A CroppableImage that reads from `array` instead of a file, so sampling tests need no image on disk."""
    from topaz.utils.data.memory_mapped_data import CroppableImage

    class ArrayImage(CroppableImage):
        def __init__(self):
            self.dims = dims
            self._load(None)
            super().__init__('<in-memory>', targets, crop_size, split, dims, mask_size)

        def _load(self, image_path):
            self.array = array
            self.shape = array.shape

    return ArrayImage()


def write_mrc(path, array):
    """Write a 2D or 3D float32 array as MRC with topaz's own writer."""
    from topaz import mrc
    import numpy as np
    array = np.asarray(array, dtype=np.float32)
    with open(path, 'wb') as f:
        mrc.write(f, array[np.newaxis] if array.ndim == 2 else array)


def run_topaz_inprocess(*args, cwd=None):
    """Run `topaz <args>` through topaz.main.main in this process (no interpreter startup cost).

    Exercises the real argument parsing and command code; use run_topaz for the installed entry point itself.
    Raises AssertionError on a non-zero exit.
    """
    import contextlib
    from unittest import mock
    from topaz import main as topaz_main
    argv = ['topaz'] + [str(a) for a in args]
    with mock.patch('sys.argv', argv), (contextlib.chdir(cwd) if cwd and hasattr(contextlib, 'chdir') else _chdir(cwd)):
        try:
            topaz_main.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise AssertionError(f'{" ".join(argv)} exited {e.code}')


class _chdir:
    """contextlib.chdir for Python < 3.11 (no-op when path is None)."""
    def __init__(self, path):
        """Remember the folder to switch to."""
        self.path = path

    def __enter__(self):
        """Switch to the folder, remembering the current one."""
        self.old = os.getcwd()
        if self.path:
            os.chdir(self.path)

    def __exit__(self, *exc):
        """Switch back to the original folder."""
        os.chdir(self.old)


_fixtures_verified = False


def verified_fixtures_dir():
    """test/fixtures, after checking every file against MANIFEST.json (once per session).

    Fixtures are frozen inputs written by v0.2.5 (see test/fixtures/generate.py); a changed,
    missing or extra file is an error, never silently accepted.
    """
    global _fixtures_verified
    if not _fixtures_verified:
        import hashlib
        import json
        manifest = json.loads((FIXTURES_DIR / 'MANIFEST.json').read_text())['files']
        present = {str(p.relative_to(FIXTURES_DIR)) for p in FIXTURES_DIR.rglob('*')
                   if p.is_file() and p.name not in ('generate.py', 'MANIFEST.json') and '__pycache__' not in p.parts}
        problems = [f'missing: {f}' for f in sorted(set(manifest) - present)]
        problems += [f'not in manifest: {f}' for f in sorted(present - set(manifest))]
        for f in sorted(set(manifest) & present):
            if hashlib.sha256((FIXTURES_DIR / f).read_bytes()).hexdigest() != manifest[f]:
                problems.append(f'modified: {f}')
        if problems:
            raise RuntimeError('test fixtures differ from test/fixtures/MANIFEST.json:\n  ' + '\n  '.join(problems))
        _fixtures_verified = True
    return FIXTURES_DIR
