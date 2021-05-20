# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
distributions.py

Module containing classes corresponding to different probability distributions.
"""
from torch.distributions import Normal


class Gaussian(Normal):
    """
    Class which handles the generation of a sample of latent Gaussian variables.

    Parameters
    ----------
    lattice_size: int
        Number of nodes on the lattice.
    loc: float, default=0
        Mean for the distribution.
    scale: float, default=1
        Standard deviation for the distribution.
    """

    def __init__(self, size_out, *, loc=0, scale=1):
        super().__init__(loc, scale)
        self.size_out = size_out


    def __call__(self, sample_size):
        """Return a sample of variables drawn from the normal distribution,
        with dimensions (sample_size, lattice_size).
        """
        sample = self.sample((sample_size, self.size_out))
        return sample, self.log_density(sample)


    def log_density(self, sample):
        """Returns the log probability for each configuration.

        Parameters
        ----------
        sample: torch.tensor
            input sample size (n_batch, self.size_out)

        Returns
        -------
        log_density: torch.tensor
            density evaluated for each input configuration, number of dimensions
            is retained: size (n_batch, 1).

        """
        return self.log_prob(sample).sum(dim=1, keepdim=True)


class PhiFourScalar:
    r"""Class associated with the action for a scalar field theory with
    :math:`\phi^4` interaction.

    methods to evaluate either the action or shifted log density on either a
    single state - torch tensor, size (1, length * length) - or a stack of N
    states - torch tensor, size (N, length * length).
    See Notes about action definition.

    The parameters required differ depending on the parameterisation you're
    using:

    ================  =============
    parameterisation  couplings
    ================  =============
    standard          m_sq, g
    albergo2019       m_sq, lam
    nicoli2020        kappa, lam
    bosetti2015       beta, lam
    ================  =============

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

    Notes
    -----
    The general form of the action is

    .. math::

        S(\phi) = \sum_{x \in \Lambda} \left[
            C_{\rm ising} * \sum_{\mu = 1}^d \phi(x + e_\mu) \phi(x) +
            C_{\rm quadratic} * \phi(x)^2 +
            C_{\rm quartic} * \phi(x)^4
        \right]

    where :math:`C_{\rm ising}`, :math:`C_{\rm quadratic}` and
    :math:`C_{\rm quartic}` are coefficients built from the two couplings
    provided in the constructor, :math:`\Lambda` is the space-time lattice
    (the sum over the lattice is a sum over the lattice sites),
    d is the number of space-time dimensions and
    :math:`e_\mu` is a unit vector in the :math:`\mu^{th}` dimension.

    Examples
    --------
    Consider the toy example of this class acting on a random state

    >>> from anvil.geometry import Geometry2D
    >>> from anvil.distributions import PhiFourScalar
    >>> import torch
    >>> geom = Geometry2D(2)
    >>> target = PhiFourScalar.from_standard(geom, **{"m_sq": 4, "g": 0})
    >>> state = torch.rand((1, 2*2)) # 2-D so lattice cardinality is 4
    >>> target.log_density(state)
    tensor([[-2.3838]])
    >>> state = torch.rand((5, 2*2))
    >>> target.log_density(state)
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
        self.c_quadratic = quadratic_coefficient
        self.c_quartic = quartic_coefficient

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
            + self.c_quadratic * phi.pow(2)
            + self.c_quartic * phi.pow(4)
        ).sum(dim=1, keepdim=True)

    def log_density(self, phi):
        """Logarithm of the un-normalized probability density, i.e. negative action,
        for a sample of field configurations."""
        return -self.action(phi)

def gaussian(lattice_size, loc=0, sigma=1):
    return Gaussian(lattice_size, loc=loc, scale=sigma)

def phi_four(geometry, parameterisation, couplings):
    constructor = getattr(PhiFourScalar, f"from_{parameterisation}")
    return constructor(geometry, **couplings)

BASE_OPTIONS = {
    "gaussian": gaussian,
}
TARGET_OPTIONS = {
    "phi_four": phi_four,
}
