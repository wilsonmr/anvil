from math import sqrt

import torch

from anvil.api import API
from anvil.distributions import Gaussian, PhiFourScalar
from anvil.free_scalar import FreeScalar
from anvil.geometry import Geometry2D
from anvil.sample import metropolis_hastings


def test_eigenvalues():
    """Test correct eigenvalues and indexing (negative momenta are left/top)"""
    geometry = Geometry2D(4)
    fs = FreeScalar(geometry, m_sq=1)
    expected = torch.tensor(
        [
            [5, 3, 5, 7],
            [3, 1, 3, 5],
            [5, 3, 5, 7],
            [7, 5, 7, 9],
        ]
    )
    assert torch.all(fs.eigenvalues == expected)


def test_variances():
    """Test correct variances and indexing"""
    geometry = Geometry2D(4)
    fs = FreeScalar(geometry, m_sq=1)
    # Vol / eigevals
    expected = 16 / torch.tensor(
        [
            [5, 3, 5, 7],
            [3, 1, 3, 5],
            [5, 3, 5, 7],
            [7, 5, 7, 9],
        ]
    )
    expected /= 2  # complex components - variance shared between real/imag
    expected[1, 1] *= 2
    expected[1, -1] *= 2
    expected[-1, 1] *= 2
    expected[-1, -1] *= 2
    assert torch.all(fs.variances == expected)


def test_sample_variances_fourier():
    """Test that the generated momentum-space fields have expected variance"""
    geometry = Geometry2D(4)
    fs = FreeScalar(geometry, m_sq=1)
    n = 100000
    eigenmodes = fs.rvs_eigenmodes(n)
    sample_var = eigenmodes.real.var(dim=0)  # just take real
    expected = torch.roll(fs.variances, (-1, -1), (-2, -1))  # roll so k=0 is at [0,0]
    # SE on variance: see https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
    n_sigmas = 4
    assert torch.allclose(
        sample_var, expected, atol=0, rtol=n_sigmas * sqrt(2 / (n - 1))
    )


def test_sample_variance_real():
    """Test that the real-space fields have expected variance"""
    geometry = Geometry2D(4)
    fs = FreeScalar(geometry, m_sq=1)
    n = 100000
    sample, _ = fs(n)
    result = sample.var()  # all dof have same variance
    expected = torch.reciprocal(fs.eigenvalues).mean()
    n_sigmas = 4
    tol = expected * sqrt(2 / (n * 4 ** 2 - 1)) * n_sigmas
    assert abs(result - expected) < tol


def test_change_of_variables():
    """Check that the log density is unchanged, other than an overall shift that
    is the same for all field configurations (if I could be bothered to put it in
    at all), when we do the inverse Fourier transform."""
    L = 4
    gaussian = Gaussian(L ** 2)
    model = API.explicit_model(model={"layer": "gauss_to_free"}, lattice_length=L)
    z, gaussian_log_density = gaussian(1000)
    phi, model_log_density = model(z, gaussian_log_density)
    offset = gaussian_log_density[0] - model_log_density[0]
    assert torch.allclose(gaussian_log_density - offset, model_log_density)


def test_action_is_gaussian():
    """Check that the action is equal to the log-density of a Gaussian up to a
    shift that is the same for all fields."""
    L = 4
    geometry = Geometry2D(L)
    gaussian = Gaussian(L ** 2)
    fs = FreeScalar(geometry, m_sq=1)
    model = API.explicit_model(model={"layer": "gauss_to_free"}, lattice_length=L)
    z, gaussian_log_density = gaussian(1000)
    phi, _ = model(z, gaussian_log_density)
    action = fs.action(phi)
    offset = gaussian_log_density[0] + action[0]
    assert torch.allclose(gaussian_log_density - offset, -action)


def test_actions_agree():
    """Check that the Phi Four action with g=0 and the Free Scalar action agree"""
    geometry = Geometry2D(4)
    base = FreeScalar(geometry, m_sq=1)
    target = PhiFourScalar.from_standard(geometry, m_sq=1, g=0)
    phi, _ = base(100)
    assert torch.allclose(base.action(phi), target.action(phi))


def test_exact_sampling():
    """Check that acceptance is 100% when using FreeScalar as a base and PhiFourScalar
    with g=0 as a target."""
    geometry = Geometry2D(4)
    base = FreeScalar(geometry, m_sq=1)
    target = PhiFourScalar.from_standard(geometry, m_sq=1, g=0)

    model = lambda z, log_density, *args: (z, log_density)  # just a placeholder

    _, _, acceptance = metropolis_hastings(
        model, base, target, sample_size=1000, thermalization=None, sample_interval=1
    )
    assert acceptance == 1
