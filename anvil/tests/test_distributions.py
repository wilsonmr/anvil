"""
Test distributions module

"""
from math import sqrt

import numpy as np
from anvil.distributions import NormalDist

MEAN = 0
SIGMA = 1
N_SAMPLE = 10000
# allow for 5 std dev. tolerance - probably will never see a fail
# (unless there's a problem)
TOL = 5*SIGMA/sqrt(N_SAMPLE)

def test_normal_distribution():
    """Test that normal distribution generates a sample whose mean is within
    5 sigma of expected value.

    """
    lattice_size = 5
    generator = NormalDist(lattice_size, sigma=SIGMA, mean=MEAN)
    sample_pt, _ = generator(N_SAMPLE)
    sample_np = sample_pt.detach().numpy()
    np.testing.assert_allclose(
        sample_np.mean(axis=0),
        np.zeros(lattice_size),
        atol=TOL
    )
