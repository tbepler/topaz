"""The installed `topaz` console script itself (subprocess, kept to a few calls: startup is slow)."""
from importlib.metadata import version

import pytest

import topaz

pytestmark = pytest.mark.cli


def test_version(topaz_cli):
    proc = topaz_cli('--version')
    assert proc.stdout.strip() == topaz.__version__ == version('topaz-em')


def test_subcommand_help(topaz_cli):
    proc = topaz_cli('train', '--help')
    assert '--train-images' in proc.stdout


def test_no_command_is_a_usage_error(topaz_cli):
    proc = topaz_cli(check=False)
    assert proc.returncode == 2
    assert 'required' in proc.stderr
