"""The full option surface of every command matches the committed snapshot (cli_surface.json).

Any added, removed or changed option (flag, default, type, choices, nargs, required) shows up
as a failure. If the change is intended, regenerate the snapshot and review its diff:

    TOPAZ_UPDATE_SNAPSHOT=1 pytest test/smoke/test_cli_surface.py
"""
import argparse
import importlib
import json
import os
import pkgutil
from pathlib import Path

import pytest

import topaz.commands

SNAPSHOT = Path(__file__).resolve().parent / 'cli_surface.json'


def describe_action(action):
    t = action.type
    return {
        'dest': action.dest,
        'default': repr(action.default),
        'type': None if t is None else getattr(t, '__qualname__', repr(t)),
        'choices': None if action.choices is None else [repr(c) for c in action.choices],
        'nargs': action.nargs,
        # argparse derives `required` for a positional from its nargs, and Python versions disagree: for nargs='*'
        # it is True on 3.11 and False on 3.12.16 and 3.13.18. nargs is recorded above, so only options record it.
        'required': action.required if action.option_strings else None,
        'action': type(action).__name__,
    }


def command_surface():
    surface = {}
    for info in sorted(pkgutil.iter_modules(topaz.commands.__path__), key=lambda m: m.name):
        module = importlib.import_module(f'topaz.commands.{info.name}')
        parser = module.add_arguments(argparse.ArgumentParser())
        options = {}
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            key = '|'.join(action.option_strings) if action.option_strings else action.dest
            options[key] = describe_action(action)
        surface[module.name] = options
    return surface


def test_cli_surface_matches_snapshot():
    current = command_surface()
    if os.environ.get('TOPAZ_UPDATE_SNAPSHOT'):
        SNAPSHOT.write_text(json.dumps(current, indent=1, sort_keys=True) + '\n')
        pytest.skip(f'snapshot rewritten: {SNAPSHOT}')
    expected = json.loads(SNAPSHOT.read_text())

    problems = []
    for command in sorted(set(expected) | set(current)):
        if command not in current:
            problems.append(f'{command}: command removed')
            continue
        if command not in expected:
            problems.append(f'{command}: command added')
            continue
        exp, cur = expected[command], current[command]
        for opt in sorted(set(exp) | set(cur)):
            if opt not in cur:
                problems.append(f'{command} {opt}: removed')
            elif opt not in exp:
                problems.append(f'{command} {opt}: added')
            elif exp[opt] != cur[opt]:
                changed = {k: (exp[opt].get(k), cur[opt].get(k)) for k in set(exp[opt]) | set(cur[opt])
                           if exp[opt].get(k) != cur[opt].get(k)}
                problems.append(f'{command} {opt}: changed {changed}')
    assert not problems, 'CLI surface differs from snapshot:\n' + '\n'.join(problems)
