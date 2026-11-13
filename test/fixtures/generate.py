"""Generate the committed test fixtures. Run ONCE, under topaz v0.2.5, then commit the outputs.

    PYTHONNOUSERSITE=1 /path/to/topaz-0.2.5-env/bin/python test/fixtures/generate.py

(the python of an environment with topaz v0.2.5 installed; PYTHONNOUSERSITE=1 keeps a ~/.local topaz from shadowing it)

Every MRC and coordinate file is written by v0.2.5 itself (its mrc writer and its `topaz convert`), so
v0.2.5 can read all of them, and the converted coordinate files double as v0.2.5 golden outputs.
The files are frozen inputs: tests never regenerate them, and conftest checks them against MANIFEST.json.
Re-running this script is only for deliberately replacing the fixtures (review the diff).

Contents:
  2d/train/*.mrc, 2d/test/*.mrc  10 synthetic 128x128 micrographs (7 train / 3 test), standardized
  2d/particles{,_train,_test}.txt ground-truth centres (tab-separated)
  2d/particles.{star,csv}, 2d/box/*.box, 2d/json/*.json   v0.2.5 `topaz convert` of particles.txt
  3d/tomo.mrc, 3d/particles.txt  one synthetic 64^3 volume with spheres
  scores/*.mrc, scores/peaks.txt log-likelihood-like maps with known peaks (2D 64^2, 3D 32^3)
  io/v025_*.mrc + .npy           v0.2.5-written MRCs and the arrays they hold
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import topaz
from topaz import mrc
from topaz.utils.data.loader import load_image

HERE = Path(__file__).resolve().parent
SEED = 20261001
PARTICLE_RADIUS = 6       # pixels (2D), soft-edged disc
PARTICLE_RADIUS_3D = 4
BOXSIZE = 24


def write_mrc(path, array):
    array = np.asarray(array, dtype=np.float32)
    with open(path, 'wb') as f:
        mrc.write(f, array[np.newaxis] if array.ndim == 2 else array)


def place(rng, n, size, margin, min_sep, ndim):
    """Rejection-sample n integer centres at least min_sep apart and margin from every edge."""
    centres = []
    while len(centres) < n:
        c = rng.integers(margin, size - margin, size=ndim)
        if all(np.sum((c - o) ** 2) >= min_sep ** 2 for o in centres):
            centres.append(c)
    return np.array(centres)


def soft_particles(shape, centres, radius, amplitude):
    """Sum of soft-edged discs/spheres; centres in array index order."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1).astype(np.float32)
    image = np.zeros(shape, dtype=np.float32)
    for c in centres:
        d = np.sqrt(((grid - c) ** 2).sum(axis=-1))
        image += amplitude / (1 + np.exp((d - radius) / 0.75))
    return image


def standardize(x):
    return ((x - x.mean()) / x.std()).astype(np.float32)


def make_2d(rng):
    out = HERE / '2d'
    rows = []
    for i in range(10):
        name = f'mic{i:02d}'
        split = 'train' if i < 7 else 'test'
        (out / split).mkdir(parents=True, exist_ok=True)
        centres = place(rng, 15, 128, margin=10, min_sep=16, ndim=2)  # (y, x)
        image = soft_particles((128, 128), centres, PARTICLE_RADIUS, 1.5) + rng.standard_normal((128, 128))
        write_mrc(out / split / f'{name}.mrc', standardize(image))
        rows += [{'image_name': name, 'x_coord': int(x), 'y_coord': int(y), 'split': split} for y, x in centres]
    table = pd.DataFrame(rows)
    cols = ['image_name', 'x_coord', 'y_coord']
    table[cols].to_csv(out / 'particles.txt', sep='\t', index=False)
    table.loc[table.split == 'train', cols].to_csv(out / 'particles_train.txt', sep='\t', index=False)
    table.loc[table.split == 'test', cols].to_csv(out / 'particles_test.txt', sep='\t', index=False)

    # v0.2.5's own conversions of the ground truth (golden files for the coordinate formats)
    convert = [str(Path(sys.executable).parent / 'topaz'), 'convert']
    src = str(out / 'particles.txt')
    for args in (['-o', str(out / 'particles.star'), src],
                 ['-o', str(out / 'particles.csv'), src],
                 ['--to', 'box', '--boxsize', str(BOXSIZE), '-o', str(out / 'box'), src],
                 ['--to', 'json', '-o', str(out / 'json'), src]):
        if args[0] == '--to':
            (out / args[-2]).mkdir(exist_ok=True)
        subprocess.run(convert + args, check=True)


def make_3d(rng):
    out = HERE / '3d'
    out.mkdir(exist_ok=True)
    centres = place(rng, 10, 64, margin=8, min_sep=12, ndim=3)  # (z, y, x)
    volume = soft_particles((64, 64, 64), centres, PARTICLE_RADIUS_3D, 1.5) + rng.standard_normal((64, 64, 64))
    write_mrc(out / 'tomo.mrc', standardize(volume))
    pd.DataFrame([{'image_name': 'tomo', 'x_coord': int(x), 'y_coord': int(y), 'z_coord': int(z)} for z, y, x in centres]) \
        .to_csv(out / 'particles.txt', sep='\t', index=False)


def make_scores():
    """Background -10 with Gaussian bumps (sigma 1.5) whose peaks are the known maxima. Pairs of peaks 6 voxels
    apart make the result depend on the extraction radius: both peaks are kept at r <= 5, and at r >= 6 the weaker
    one is suppressed. Its shoulder pixel (score about 0.41) is then picked instead unless the threshold is above it."""
    out = HERE / 'scores'
    out.mkdir(exist_ok=True)
    rows = []
    specs = {
        'scores2d': ((64, 64), [(12, 12, 4.0), (12, 18, 3.0), (40, 20, 2.5), (30, 50, 1.0), (52, 44, 0.5)]),
        'scores3d': ((32, 32, 32), [(10, 10, 10, 4.0), (10, 10, 16, 3.0), (22, 20, 8, 2.0), (20, 24, 24, 1.0)]),
    }
    for name, (shape, peaks) in specs.items():
        grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1).astype(np.float64)
        field = np.full(shape, -10.0)
        for *pos, height in peaks:
            bump = (height + 10.0) * np.exp(-((grid - np.array(pos)) ** 2).sum(axis=-1) / (2 * 1.5 ** 2)) - 10.0
            field = np.maximum(field, bump)
        write_mrc(out / f'{name}.mrc', field)
        for *pos, height in peaks:
            xyz = list(reversed(pos))  # index order (z,) y, x -> x, y(, z)
            rows.append({'image_name': name, 'x_coord': xyz[0], 'y_coord': xyz[1],
                         'z_coord': xyz[2] if len(xyz) == 3 else np.nan, 'score': height})
    pd.DataFrame(rows).to_csv(out / 'peaks.txt', sep='\t', index=False)


def make_io(rng):
    out = HERE / 'io'
    out.mkdir(exist_ok=True)
    arrays = {'v025_float32_2d': rng.standard_normal((37, 53)).astype(np.float32),
              'v025_float32_3d': rng.standard_normal((5, 19, 23)).astype(np.float32)}
    for name, array in arrays.items():
        write_mrc(out / f'{name}.mrc', array)
        np.save(out / f'{name}.npy', array)


def verify_readable():
    """Everything written must load back under v0.2.5: 2D through its image loader (which is 2D-only),
    3D through its MRC parser."""
    for path in sorted(HERE.rglob('*.mrc')):
        with open(path, 'rb') as f:
            array, _, _ = mrc.parse(f.read())
        assert array.size > 0, path
        if array.squeeze().ndim == 2:
            image = load_image(str(path))
            assert np.asarray(image).shape == array.squeeze().shape, path
    for name in ('v025_float32_2d', 'v025_float32_3d'):
        with open(HERE / 'io' / f'{name}.mrc', 'rb') as f:
            array, _, _ = mrc.parse(f.read())
        np.testing.assert_array_equal(array.squeeze(), np.load(HERE / 'io' / f'{name}.npy'))


def write_manifest():
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in ('generate.py', 'MANIFEST.json')
                   and '__pycache__' not in p.parts)
    manifest = {
        'generator': {'topaz': topaz.__version__, 'python': platform.python_version(), 'numpy': np.__version__,
                      'pandas': pd.__version__, 'seed': SEED},
        'files': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return len(files)


def main():
    if not topaz.__version__.startswith('0.2.5'):
        sys.exit(f'run this under topaz v0.2.5 (found {topaz.__version__} at {topaz.__file__})')
    for sub in ('2d', '3d', 'scores', 'io'):
        shutil.rmtree(HERE / sub, ignore_errors=True)
    rng = np.random.default_rng(SEED)
    make_2d(rng)
    make_3d(rng)
    make_scores()
    make_io(rng)
    verify_readable()
    n = write_manifest()
    print(f'wrote {n} fixture files with topaz {topaz.__version__} from {os.path.dirname(topaz.__file__)}')


if __name__ == '__main__':
    main()
