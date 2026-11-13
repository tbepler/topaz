"""Every command module imports, builds its parser, and answers --help through topaz.main."""
import argparse
import importlib
import pkgutil
import sys

import pytest

import topaz.commands
from topaz import main as topaz_main

COMMAND_MODULES = sorted(m.name for m in pkgutil.iter_modules(topaz.commands.__path__))


def test_command_modules_found():
    # guards against the discovery itself silently finding nothing
    assert len(COMMAND_MODULES) >= 20


@pytest.mark.parametrize('module_name', COMMAND_MODULES)
def test_module_interface(module_name):
    module = importlib.import_module(f'topaz.commands.{module_name}')
    for attr in ('name', 'help', 'add_arguments', 'main'):
        assert hasattr(module, attr), f'topaz.commands.{module_name} has no {attr!r}'
    assert isinstance(module.name, str) and module.name
    assert isinstance(module.help, str) and module.help
    # deprecated commands require a parser argument, so always pass one
    parser = module.add_arguments(argparse.ArgumentParser())
    assert isinstance(parser, argparse.ArgumentParser)


def test_top_level_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['topaz', '--help'])
    with pytest.raises(SystemExit) as exit_info:
        topaz_main.main()
    assert exit_info.value.code == 0
    assert 'commands:' in capsys.readouterr().out


@pytest.mark.parametrize('module_name', COMMAND_MODULES)
def test_subcommand_help(module_name, monkeypatch, capsys):
    # Runs through topaz.main, so a command module that isn't registered there fails here
    # with argparse's "invalid choice" (exit code 2).
    command = importlib.import_module(f'topaz.commands.{module_name}').name
    monkeypatch.setattr(sys, 'argv', ['topaz', command, '--help'])
    with pytest.raises(SystemExit) as exit_info:
        topaz_main.main()
    captured = capsys.readouterr()
    assert exit_info.value.code == 0, captured.err
    assert captured.out.startswith('usage:')
