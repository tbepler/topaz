"""Model construction per architecture and dimensionality, and loading of the packaged pretrained weights."""
import importlib.resources
import sys

import pytest
import torch

from _support import require
from topaz.denoising import models as denoise_models
from topaz.model.classifier import LinearClassifier
from topaz.model.factory import get_feature_extractor, load_model
from topaz.model.utils import predict_in_patches

ARCHITECTURES = ['resnet6', 'resnet8', 'resnet16', 'conv31', 'conv63', 'conv127']


def build(arch, ndim):
    torch.manual_seed(0)
    features = get_feature_extractor(arch, units=4, dims=ndim, bn=False)
    return LinearClassifier(features, dims=ndim).eval()


@pytest.mark.parametrize('arch', ARCHITECTURES)
def test_receptive_field_matches_width(arch):
    """The input region that influences one output pixel is exactly `width` pixels across."""
    model = build(arch, 2)
    n = 2 * model.width + 1
    x = torch.randn(1, 1, n, n, requires_grad=True)
    y = model(x)
    y[0, 0, y.shape[-2] // 2, y.shape[-1] // 2].backward()
    nonzero = (x.grad[0, 0] != 0).nonzero()
    extent = (nonzero.max(0).values - nonzero.min(0).values + 1).tolist()
    assert extent == [model.width, model.width]


@pytest.mark.parametrize('arch', ARCHITECTURES)
def test_width_sized_input_gives_one_output(arch, ndim):
    require('model_3d', ndim)
    model = build(arch, ndim)
    with torch.no_grad():
        out = model(torch.zeros((1, 1) + (model.width,) * ndim))
    assert tuple(out.shape) == (1, 1) + (1,) * ndim


@pytest.mark.parametrize('patch_size', [24, 50, 100])
def test_patched_scores_match_whole_image(patch_size):
    """Scoring in patches gives the same scores as scoring the whole image at once, for patches smaller and
    larger than the receptive field (71) and an image they don't divide evenly. The image has a blank
    quarter: all-zero patches used to be dropped, which crashed the reassembly."""
    model = build('resnet8', 2)
    model.fill()
    image = torch.randn(1, 1, 160, 128)
    image[..., :80, :64] = 0
    with torch.no_grad():
        whole = model(image)[0, 0].numpy()
    patched = predict_in_patches(model, image, patch_size=patch_size)[0, 0]
    assert patched == pytest.approx(whole, abs=1e-5)


def test_non_positive_patch_size_is_an_error():
    model = build('resnet8', 2)
    model.fill()
    with pytest.raises(ValueError, match='must be positive'):
        predict_in_patches(model, torch.randn(1, 1, 96, 96), patch_size=0)


# parameter counts of the packaged pretrained weights
DETECTORS = {'resnet8_u32': (328801, 71), 'resnet8_u64': (1310913, 71),
             'resnet16_u32': (458145, 91), 'resnet16_u64': (1827649, 91)}
DENOISERS = {'affine': 962, 'fcnn': 511233, 'unet': 1105809, 'unet-small': 649137, 'unet-v0.2.1': 969745,
             'unet-3d-10a': 2916337, 'unet-3d-20a': 2916337}


@pytest.fixture
def modern_resources_only(monkeypatch):
    """Block the deprecated pkg_resources fallback so weights must load via importlib.resources.

    pkg_resources ships with setuptools, which fresh Python 3.12+ environments don't have; a
    resource path that only the fallback can resolve (e.g. a module name or a '..' path) must
    fail here rather than on a user's machine. Python 3.8 has no importlib.resources.files, so
    the fallback is the only route there and stays available.
    """
    if hasattr(importlib.resources, 'files'):
        monkeypatch.setitem(sys.modules, 'pkg_resources', None)


@pytest.mark.parametrize('name', sorted(DETECTORS))
def test_pretrained_detector_loads(name, modern_resources_only):
    model = load_model(name)
    n_params, width = DETECTORS[name]
    assert sum(p.numel() for p in model.parameters()) == n_params
    assert model.width == width


@pytest.mark.parametrize('name', sorted(DENOISERS))
def test_pretrained_denoiser_loads(name, modern_resources_only):
    model = denoise_models.load_model(name)
    assert sum(p.numel() for p in model.parameters()) == DENOISERS[name]
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_all_packaged_denoisers_covered():
    # every alias topaz exposes is exercised above (unet-3d is an alias of unet-3d-10a)
    assert set(denoise_models.model_name_dict) - {'unet-3d'} == set(DENOISERS)
