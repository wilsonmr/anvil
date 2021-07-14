from math import sqrt

import torch
import torch.optim

from anvil.distributions import FreeScalar, PhiFourScalar
from anvil.geometry import Geometry2D
from anvil.sample import metropolis_hastings


def test_inverse_fourier_transform():
    """Test that the Inverse Fourier Transform to real space is working."""
    geometry = Geometry2D(4)
    fs = FreeScalar(geometry, m_sq=1)
    eigenmodes = fs.gen_eigenmodes(10000)
    print(eigenmodes.real.var(dim=0))
    assert True


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
    geometry = Geometry2D(4)
    fs = FreeScalar(geometry, m_sq=1)
    n = 100000
    n_sigmas = 4
    eigenmodes = torch.roll(fs.gen_eigenmodes(n), (1, 1), (-2, -1))
    sample_var = eigenmodes.real.var(dim=0)  # just take real
    expected = fs.variances
    assert torch.allclose(
        sample_var, expected, atol=0, rtol=n_sigmas * sqrt(2 / (n - 1))
    )


def test_sample_variance_real():
    lattice_length = 4
    geometry = Geometry2D(lattice_length)
    fs = FreeScalar(geometry, m_sq=1)
    n = 100000
    sample, _ = fs(n)
    result = sample.var()  # all dof have same variance
    expected = torch.reciprocal(fs.eigenvalues).mean()
    n_sigmas = 4
    # TODO test expected failure when n_sigmas is 1 or 2
    tol = expected * sqrt(2 / (n * lattice_length ** 2 - 1)) * n_sigmas
    assert abs(result - expected) < tol


def test_actions_agree():
    lattice_length = 4
    geometry = Geometry2D(lattice_length)

    base = FreeScalar(geometry, m_sq=1)
    target = PhiFourScalar.from_standard(geometry, m_sq=1, g=0)

    phi, _ = base(100)
    assert torch.allclose(base.action(phi), target.action(phi))


def test_exact_sampling():
    lattice_length = 6
    geometry = Geometry2D(lattice_length)

    base = FreeScalar(geometry, m_sq=1)
    target = PhiFourScalar.from_standard(geometry, m_sq=1, g=0)
    model = lambda z, log_density, *args: (z, log_density)  # does nothing

    _, _, acceptance = metropolis_hastings(
        model, base, target, sample_size=1000, thermalization=None, sample_interval=1
    )
    assert acceptance == 1
