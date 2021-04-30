# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
Test distributions module

"""
from math import log, pi, sqrt

import numpy as np
from scipy.stats import norm
import torch

from anvil.geometry import Geometry2D
from anvil.distributions import Gaussian, PhiFourScalar

MEAN = 0
SIGMA = 1
N_SAMPLE = 10000
# allow for 5 std dev. tolerance - probably will never see a fail
# (unless there's a problem)
TOL = 5*SIGMA/sqrt(N_SAMPLE)


@torch.no_grad()
def test_normal_distribution():
    """Test that normal distribution generates a sample whose mean is within
    5 sigma of expected value.

    """
    lattice_size = 5
    generator = Gaussian(lattice_size, mean=MEAN, sigma=SIGMA)
    sample_pt, log_density = generator(N_SAMPLE)
    sample_np = sample_pt.numpy()
    np.testing.assert_allclose(
        sample_np.mean(axis=0),
        np.zeros(lattice_size),
        atol=TOL
    )
    # check scipy agrees
    scipy_log_density = norm.logpdf(
        sample_np, loc=MEAN, scale=SIGMA).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        scipy_log_density,
        log_density.numpy(),
        atol=TOL
    )


@torch.no_grad()
def test_phi_four_action():
    """Basic test that phi four action initialises and can handle expected
    states.

    """
    lattice_size = 9
    generator = Gaussian(lattice_size, mean=MEAN, sigma=SIGMA)
    geom = Geometry2D(3)
    target = PhiFourScalar.from_standard(geom, m_sq=1, g=1)
    sample, _ = generator(100)
    # check action passes with batch
    target.action(sample)
    sample, _ = generator(1)
    # check action passes with single state
    target.action(sample)


def test_phi_four_uniform_limit():
    """Test phi four action in uniform limit"""
    lattice_size = 9
    generator = Gaussian(lattice_size, mean=MEAN, sigma=SIGMA)
    geom = Geometry2D(3)
    sample, _ = generator(100)

    uniform_target = PhiFourScalar(geom, 0, 0, 0)
    const_log_density = uniform_target.action(sample).numpy()
    # log prob should be constant.
    assert np.all(const_log_density[0] == const_log_density)


def test_phi_four_gaussian_limit():
    """Test phi four action in gaussian limit"""
    lattice_size = 9
    generator = Gaussian(lattice_size, mean=MEAN, sigma=SIGMA)
    geom = Geometry2D(3)
    # gaussian with unit variance
    gaussian_target = PhiFourScalar(geom, 0, 0.5, 0)

    # log prob doesn't feature normalisation for phi four.
    log_normalisation = lattice_size * log(sqrt(2 * pi))
    sample, _ = generator(100)
    np.testing.assert_allclose(
        generator.log_density(sample).numpy(),
        gaussian_target.log_density(sample).numpy() - log_normalisation,
        rtol=1e-05, # use allclose not testing.allclose tolerance
        atol=1e-08,
    )
