# topaz tests

The suite catches silent numerical regressions (topaz still runs but gives different answers) as well as
crashes. **[walkthrough.ipynb](walkthrough.ipynb)** explains what each part checks, why, and where its expected
values come from, with live examples. Start there.

## Running

```bash
pip install -e .[test]
pytest -m "not slow"   # what CI runs on every Python version (about a minute)
pytest -m unit         # one category: smoke, unit, io, integration
pytest -rs -rx         # also list skipped and expected-failure tests with reasons
```

A passing run ends with `2 xfailed`. Those are two cases of one test for a known bug in an NMS mode that no
command uses (see section 7 of the walkthrough).

Prefer `pytest` to `python -m pytest`. With an editable install (`pip install -e .`, as above) both work. With a
regular install, `python -m pytest` from the repo root imports the `./topaz` source folder instead of the
installed package, and `conftest.py` stops the run with an error rather than test the wrong code.

## Layout

| Folder | Contents |
|---|---|
| `smoke/` | commands import, parse and build; CLI option snapshot; models and pretrained weights load |
| `unit/` | numerics on small in-memory arrays, against brute-force, hand-computed or scikit-learn values |
| `io/` | image and coordinate file formats, against files written by topaz v0.2.5 |
| `integration/` | whole `topaz` commands on the fixtures |
| `fixtures/` | committed test data written by v0.2.5 (`generate.py`), checked against `MANIFEST.json` |
| `mutants/`, `release/checks.py` | the mutant gate: re-introduces past bugs and confirms the tests catch them (`python test/release/checks.py mutants`; run by hand before a release) |

`conftest.py` and `_support.py` hold the shared setup: category markers, the 2D/3D `ndim` fixture, 3D capability
checks, and the checks that the right topaz is under test and the fixtures are unchanged.
