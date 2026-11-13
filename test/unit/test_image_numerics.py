"""Normalization (deterministic parts only), Fourier downsampling and 8-bit quantization."""
import numpy as np
import pytest

from topaz.stats import norm_fit, normalize
from topaz.utils.image import downsample, quantize, unquantize


def noise_image(seed=0, shape=(48, 64)):
    rng = np.random.default_rng(seed)
    return (3.0 + 2.0 * rng.standard_normal(shape)).astype(np.float32)


def test_affine_normalize_is_exact():
    x = noise_image()
    y, metadata = normalize(x, method='affine')
    assert y.dtype == np.float32
    assert metadata == {'mu': pytest.approx(float(x.mean())), 'std': pytest.approx(float(x.std())), 'pi': 1}
    np.testing.assert_allclose(y, (x - x.mean()) / x.std(), rtol=1e-5, atol=1e-6)


def test_norm_fit_initialization_and_selection():
    """The GMM fit itself is statistical; what's pinned here is its fixed structure: the initial pi grid,
    the single-component (pi = 1) entry, and selecting the parameters with the highest log-probability."""
    x = noise_image()
    mu, std, pi, logp, mus, stds, pis, logps = norm_fit(x, num_iters=20)
    assert len(mus) == len(stds) == len(pis) == len(logps) == 12
    # the last candidate is the single-component model: sample mean and (unbiased, as torch.var) std
    assert mus[-1] == pytest.approx(float(x.mean()), rel=1e-5)
    assert stds[-1] == pytest.approx(float(np.sqrt(x.var(ddof=1))), rel=1e-4)
    assert pis[-1] == 1
    i = int(np.argmax(logps))
    assert (mu, std, pi, logp) == (mus[i], stds[i], pis[i], logps[i])


def test_gmm_normalize_applies_selected_parameters():
    x = noise_image()
    y, metadata = normalize(x, method='gmm', num_iters=20)
    assert y.dtype == np.float32
    assert {'mu', 'std', 'pi', 'logp', 'mus', 'stds', 'pis', 'logps', 'alpha', 'beta', 'sample'} <= set(metadata)
    np.testing.assert_allclose(y, ((x - metadata['mu']) / metadata['std']).astype(np.float32), rtol=1e-6)


def test_downsample_factor_one_is_identity():
    x = noise_image()
    np.testing.assert_allclose(downsample(x, 1), x, atol=1e-5)


@pytest.mark.parametrize('factor', [2, 4])
def test_downsample_preserves_band_limited_signal(factor):
    """A cosine below the new Nyquist frequency survives Fourier-cropping with the same amplitude and phase."""
    m, n = 64, 96
    ky, kx = 3, 5  # cycles across the image, well below min(m, n) / (2 * factor)
    yy, xx = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    x = (1.5 + np.cos(2 * np.pi * (ky * yy / m + kx * xx / n))).astype(np.float32)

    small = downsample(x, factor)
    assert small.shape == (m // factor, n // factor) and small.dtype == np.float32
    sy, sx = np.meshgrid(np.arange(m // factor), np.arange(n // factor), indexing='ij')
    expected = 1.5 + np.cos(2 * np.pi * (ky * sy / (m // factor) + kx * sx / (n // factor)))
    np.testing.assert_allclose(small, expected, atol=1e-4)


def test_quantize_roundtrip_exact_on_uint8():
    q = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(quantize(unquantize(q)), q)


def test_unquantize_error_within_half_step():
    x = np.linspace(-3, 3, 10001, dtype=np.float32)
    err = np.abs(unquantize(quantize(x)) - x)
    assert err.max() <= (6 / 255) / 2 + 1e-6


def test_quantize_clips_out_of_range():
    np.testing.assert_array_equal(quantize(np.array([-10.0, 10.0], dtype=np.float32)), [0, 255])
