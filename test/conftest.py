"""Shared pytest setup: checks which topaz is under test, applies category markers, and defines common fixtures."""
import importlib.util
import sys
from pathlib import Path

import pytest

# The suite runs with --import-mode=importlib, so test directories are never put on sys.path
# (a stray test/topaz/ directory would otherwise shadow the installed package). Shared helpers
# are registered as the module `_support` so test modules can `import _support`.
_spec = importlib.util.spec_from_file_location('_support', Path(__file__).resolve().parent / '_support.py')
_support = importlib.util.module_from_spec(_spec)
sys.modules['_support'] = _support
_spec.loader.exec_module(_support)

# test/<layer>/... gets the matching marker automatically
LAYER_MARKERS = {'smoke', 'unit', 'io', 'integration'}

_install_info = {}


def pytest_configure(config):
    """Before collecting anything, stop the run if the tests would import a different topaz than the CLI runs."""
    try:
        _install_info['topaz_dir'] = _support.check_install_location()
    except RuntimeError as e:
        raise pytest.UsageError(f'topaz install check failed: {e}')


def pytest_report_header(config):
    """Print the topaz version, location, git checkout and 3D capabilities at the top of every run."""
    import numpy
    import torch
    import topaz
    lines = [f'topaz {topaz.__version__} from {_install_info["topaz_dir"]}',
             f'torch {torch.__version__}, numpy {numpy.__version__}']
    described = _support.git_describe(_install_info['topaz_dir'])
    if described is not None:
        branch, sha, dirty = described
        lines.append(f'topaz checkout: {branch} @ {sha}{" (uncommitted changes)" if dirty else ""}')
    caps = ', '.join(f'{name}={_support.has_capability(name)}' for name in _support.CAPABILITY_PROBES)
    lines.append(f'capabilities: {caps}')
    return lines


def pytest_collection_modifyitems(config, items):
    """Mark each test with its folder's category, so `pytest -m unit` selects test/unit/."""
    test_dir = Path(__file__).resolve().parent
    for item in items:
        rel = Path(item.fspath).resolve().relative_to(test_dir)
        if len(rel.parts) > 1 and rel.parts[0] in LAYER_MARKERS:
            item.add_marker(getattr(pytest.mark, rel.parts[0]))


@pytest.fixture(params=[2, 3], ids=['2d', '3d'])
def ndim(request):
    """Run the test twice, with ndim=2 and ndim=3 (shown as [2d] and [3d] in test names)."""
    return request.param


@pytest.fixture
def topaz_cli(tmp_path):
    """A function that runs the installed `topaz` console script, by default in the test's temp folder."""
    def run(*args, **kwargs):
        kwargs.setdefault('cwd', tmp_path)
        return _support.run_topaz(*args, **kwargs)
    return run


@pytest.fixture(scope='session')
def fixtures_dir():
    """Path to test/fixtures, after checking that no fixture file has changed (see MANIFEST.json)."""
    return _support.verified_fixtures_dir()
