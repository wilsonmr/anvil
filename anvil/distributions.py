# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
distributions.py

Module containing classes corresponding to different probability distributions.
"""
import torch
import torch.nn as nn

from math import log, sqrt, pi

class Gaussian:
    """
    Class which handles the generation of a sample of latent Gaussian variables.

    Inputs:
    -------
    lattice_size: int
        Number of nodes on the lattice.
    sigma: float
        Standard deviation for the distribution.
    mean: float
        Mean for the distribution.
    """

    def __init__(self, size_out, *, sigma=1, mean=0):
        self.size_out = size_out
        self.sigma = sigma
        self.mean = mean
        
        # Pre-calculate normalisation for log density
        self.exp_coeff = 1 / (2 * self.sigma ** 2)
        self.log_normalisation = self.size_out * log(sqrt(2 * pi) * self.sigma)

    def __call__(self, sample_size):
        """Return a sample of variables drawn from the normal distribution,
        with dimensions (sample_size, lattice_size)."""
        sample = torch.empty(sample_size, self.size_out).normal_(
            mean=self.mean, std=self.sigma
        )
        
        return sample, self.log_density(sample)

    
    def log_density(self, sample):
        """Logarithm of the pdf, calculated for a given sample. Dimensions (sample_size, 1)."""
        exponent = -self.exp_coeff * torch.sum(
            (sample - self.mean).pow(2), dim=1, keepdim=True
        )
        return exponent - self.log_normalisation


class PhiFourScalar:
    """Return the phi^4 action given either a single state size
    (1, length * length) or a stack of N states (N, length * length).
    See Notes about action definition.

    The forward pass returns the corresponding log density (unnormalised) which
    is equal to -S

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    parameterisation:
        which parameterisation to use. See below for options.
    couplings: dict
        dictionary with two entries that are the couplings of the theory.
        See below.


        parameterisation            couplings
        -------------------------------------
        standard                    m_sq, g
        albergo2019                 m_sq, lam
        nicoli2020                  kappa, lam
        bosetti2015                 beta, lam

    Notes
    -----
    The general form of the action is

        S(\phi) = \sum_{x \in \Lambda} [

            C_ising * \sum_{\mu = 1}^d \phi(x + e_\mu) \phi(x)

          + C_quadratic * \phi(x)^2

          + C_quartic * \phi(x)^4

        ]

    where C_ising, C_quadratic and C_quartic are coefficients built from the two couplings
    provided in the constructor, \Lambda is the lattice, d is the number of space-time
    dimensions and and e_\mu is a unit vector in the \mu-th dimensions.


    Examples
    --------
    Consider the toy example of this class acting on a random state

    >>> geom = Geometry2D(2)
    >>> action = PhiFourAction(geom, "standard", {"m_sq": 4, "g": 0})
    >>> state = torch.rand((1, 2*2))
    >>> action(state)
    tensor([[-2.3838]])
    >>> state = torch.rand((5, 2*2))
    >>> action(state)
    tensor([[-3.9087],
            [-2.2697],
            [-2.3940],
            [-2.3499],
            [-1.9730]])
    """

    def __init__(
        self,
        geometry,
        ising_coefficient,
        quadratic_coefficient,
        quartic_coefficient,
    ):
        self.shift = geometry.get_shift()
        self.c_ising = ising_coefficient
        self.c2 = quadratic_coefficient
        self.c4 = quartic_coefficient

    @classmethod
    def from_standard(cls, geometry, *, m_sq, g):
        return cls(geometry, -1, (4 + m_sq) / 2, g / 24)

    @classmethod
    def from_bosetti2015(cls, geometry, *, beta, lam):
        return cls(geometry, -beta, 1 - 2 * lam, lam)

    @classmethod
    def from_albergo2019(cls, geometry, *, m_sq, lam):
        return cls(geometry, -2, 4 + m_sq, lam)

    @classmethod
    def from_nicoli2020(cls, geometry, *, kappa, lam):
        return cls.from_bosetti2015(geometry, beta=2 * kappa, lam=lam)

    def action(self, phi):
        """Action computed for a sample of field configurations."""
        return (
            self.c_ising * (phi[:, self.shift] * phi.unsqueeze(dim=1)).sum(dim=1)
            + self.c2 * phi.pow(2)
            + self.c4 * phi.pow(4)
        ).sum(dim=1, keepdim=True)

    def log_density(self, phi):
        """Logarithm of the un-normalized probability density, i.e. negative action,
        for a sample of field configurations."""
        return -self.action(phi)

def gaussian(lattice_size, sigma=1):
    return Gaussian(lattice_size, sigma=sigma)

def phi_four(geometry, parameterisation, couplings):
    constructor = getattr(PhiFourScalar, f"from_{parameterisation}")
    return constructor(geometry, **couplings)

BASE_OPTIONS = {
    "gaussian": gaussian,
}
TARGET_OPTIONS = {
    "phi_four": phi_four,
}
