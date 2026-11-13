"""Parse realistic (tutorial-style) command lines and check the resulting argument values.

Ported from the old test_commands_simple.py, which only checked that parsing didn't raise.
"""
import importlib
import math

import pytest

CASES = {
    'train': (
        ['-n', '400', '--num-workers=8',
         '--train-images', 'data/EMPIAR-10025/processed/micrographs/',
         '--train-targets', 'data/EMPIAR-10025/processed/particles.txt',
         '--save-prefix=saved_models/EMPIAR-10025/model',
         '-o', 'saved_models/EMPIAR-10025/model_training.txt'],
        {'num_particles': 400.0, 'num_workers': 8, 'pi': None, 'radius': 3, 'method': 'GE-binomial',
         'train_images': 'data/EMPIAR-10025/processed/micrographs/',
         'train_targets': 'data/EMPIAR-10025/processed/particles.txt',
         'save_prefix': 'saved_models/EMPIAR-10025/model',
         'output': 'saved_models/EMPIAR-10025/model_training.txt',
         'model': 'resnet8', 'units': 32, 'pretrained': True, 'device': 0, 'minibatch_size': 256,
         'minibatch_balance': 0.0625, 'epoch_size': 1000, 'num_epochs': 10, 'learning_rate': 0.0002,
         'preload': 'auto', 'format_': 'auto'},
    ),
    'extract': (
        ['-r', '14', '-x', '8', '-m', 'saved_models/EMPIAR-10025/model_epoch10.sav',
         '-o', 'data/EMPIAR-10025/topaz/predicted_particles_all_upsampled.txt',
         'data/EMPIAR-10025/processed/micrographs/*.mrc'],
        {'radius': 14, 'up_scale': 8.0, 'down_scale': 1, 'threshold': -6,
         'model': 'saved_models/EMPIAR-10025/model_epoch10.sav',
         'output': 'data/EMPIAR-10025/topaz/predicted_particles_all_upsampled.txt',
         'paths': ['data/EMPIAR-10025/processed/micrographs/*.mrc'], 'format': 'coord', 'dims': 2},
    ),
    'precision_recall_curve': (
        ['-r', '14', '--predicted', 'data/EMPIAR-10025/topaz/predicted_particles_all.txt',
         '--targets', 'data/EMPIAR-10025/processed/particles_test.txt'],
        {'assignment_radius': 14, 'images': 'target',
         'predicted': 'data/EMPIAR-10025/topaz/predicted_particles_all.txt',
         'targets': 'data/EMPIAR-10025/processed/particles_test.txt'},
    ),
    'preprocess': (
        ['-v', '-s', '8', '-o', 'data/EMPIAR-10025/processed/micrographs/',
         'data/EMPIAR-10025/rawdata/micrographs/*.mrc'],
        {'verbose': True, 'scale': 8, 'destdir': 'data/EMPIAR-10025/processed/micrographs/',
         'files': ['data/EMPIAR-10025/rawdata/micrographs/*.mrc'], 'affine': False, 'sample': 10,
         'niters': 100, 'alpha': 900, 'beta': 1, 'format_': 'mrc'},
    ),
    'denoise': (
        ['--patch-size', '1024', '-o', 'data/EMPIAR-10025/denoised/',
         'data/EMPIAR-10025/rawdata/micrographs/*.mrc'],
        {'patch_size': 1024, 'patch_padding': 500, 'output': 'data/EMPIAR-10025/denoised/',
         'micrographs': ['data/EMPIAR-10025/rawdata/micrographs/*.mrc'], 'model': ['unet'],
         'format_': 'mrc', 'normalize': False},
    ),
    'convert': (
        ['-s', '8', '-o', 'data/EMPIAR-10025/processed/particles.txt', 'data/EMPIAR-10025/rawdata/particles.txt'],
        {'down_scale': 8.0, 'up_scale': 1, 'output': 'data/EMPIAR-10025/processed/particles.txt',
         'files': ['data/EMPIAR-10025/rawdata/particles.txt'], '_from': 'auto', 'to': 'auto',
         'threshold': -math.inf, 'invert_y': False, 'boxsize': 0},
    ),
    'particle_stack': (
        ['--image-root=data/EMPIAR-10025/rawdata/micrographs/', '--size=400', '--threshold=0',
         '--metadata=data/EMPIAR-10025/rawdata/ctf_params.star',
         '-o', 'data/EMPIAR-10025/topaz/predicted_particle_stack.mrcs',
         'data/EMPIAR-10025/topaz/predicted_particles_all_upsampled.txt'],
        {'image_root': 'data/EMPIAR-10025/rawdata/micrographs/', 'size': 400, 'threshold': 0.0,
         'metadata': 'data/EMPIAR-10025/rawdata/ctf_params.star',
         'output': 'data/EMPIAR-10025/topaz/predicted_particle_stack.mrcs',
         'file': 'data/EMPIAR-10025/topaz/predicted_particles_all_upsampled.txt', 'image_ext': '.mrc'},
    ),
    'train_test_split': (
        ['-n', '10', '--image-dir', 'data/EMPIAR-10025/processed/micrographs/',
         'data/EMPIAR-10025/processed/particles.txt'],
        {'number': 10, 'image_dir': 'data/EMPIAR-10025/processed/micrographs/',
         'file': 'data/EMPIAR-10025/processed/particles.txt', 'seed': 0, 'format_': 'auto'},
    ),
}


@pytest.mark.parametrize('command', sorted(CASES))
def test_parse_tutorial_command_line(command):
    argv, expected = CASES[command]
    module = importlib.import_module(f'topaz.commands.{command}')
    args = vars(module.add_arguments().parse_args(argv))
    actual = {key: args[key] for key in expected}
    assert actual == expected
