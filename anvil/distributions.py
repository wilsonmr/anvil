# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
distributions.py

Module containing classes corresponding to different probability distributions.
"""
import torch.distributions


class Gaussian(torch.distributions.Normal):
    """
    Class which handles the generation of a sample of latent Gaussian variables.

    Parameters
    ----------
    size_out
        Number of (independent) Gaussian numbers making up a 'latent configuration'.
    loc
        Mean of the Gaussian distribution.
    scale
        Standard deviation of the Gaussian distribution.

    Attributes
    ----------
    size_out
        Number of (independent) Gaussian numbers making up a 'latent configuration'.
    """

    def __init__(self, size_out: int, *, loc: float = 0, scale: float = 1):
        super().__init__(loc, scale)
        self.size_out = size_out

    def __call__(self, sample_size: int):
        """Return a sample of variables drawn from the normal distribution,
        dimensions ``(sample_size, self.size_out)``.

        Parameters
        ----------
        sample_size
            Number of latent configurations, each containing ``self.size_out``
            independent Gaussian numbers, in the sample

        Returns
        -------
        tuple
            Sample drawn from Gaussian distribution, dimensions
            ``(sample_size, self.size_out)``
            Tensor containing logarithm of the probability density evaluated for
            each latent configuration, dimensions ``(sample_size, 1)``
        """
        sample = self.sample((sample_size, self.size_out))
        return sample, self.log_density(sample)

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the log probability for each latent configuration.

        Parameters
        ----------
        sample
            Sample of Gaussian variables, dimensions ``(sample_size, self.size_out)``

        Returns
        -------
        torch.Tensor
            Tensor containing logarithm of the probability density evaluated for
            each latent configuration, dimensions ``(sample_size, 1)``

        """
        return self.log_prob(sample).sum(dim=1, keepdim=True)


class PhiFourScalar:
    r"""Class associated with the action for a scalar field theory with
    :math:`\phi^4` interaction.

    methods to evaluate either the action or shifted log density on either a
    single state - torch tensor, size ``(1, length * length)`` - or a stack of
    ``N`` states - torch tensor, size ``(N, length * length)``.
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
    geometry
        defines the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    parameterisation
        which parameterisation to use. See below for options.
    couplings
        dictionary with two entries that are the couplings of the theory.
        See below.

    Notes
    -----
    The general form of the action is

    .. math::

        S(\phi) = \sum_{x \in \Lambda} \left[
            c_{\rm ising} * \sum_{\mu = 1}^d \phi(x + e_\mu) \phi(x) +
            c_{\rm quadratic} * \phi(x)^2 +
            c_{\rm quartic} * \phi(x)^4
        \right]

    where :math:`c_{\rm ising}`, :math:`c_{\rm quadratic}` and
    :math:`c_{\rm quartic}` are coefficients built from the two couplings
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
        ising_coefficient: float,
        quadratic_coefficient: float,
        quartic_coefficient: float,
    ):
        self.shift = geometry.get_shift()
        self.c_ising = ising_coefficient
        self.c_quadratic = quadratic_coefficient
        self.c_quartic = quartic_coefficient

    @classmethod
    def from_standard(cls, geometry, *, m_sq: float, g: float):
        """
        Standard parameterisation.

        Parameters
        ----------
        m_sq
            Bare mass squared
        g
            Quartic coupling constant
        """
        return cls(geometry, -1, (4 + m_sq) / 2, g / 24)

    @classmethod
    def from_bosetti2015(cls, geometry, *, beta: float, lam: float):
        """
        Parameterisation used in Bosetti et al. (2015),
        https://arxiv.org/abs/1506.08587

        Parameters
        ----------
        beta
            inverse temperature
        lam
            Quartic coupling constant
        """
        return cls(geometry, -beta, 1 - 2 * lam, lam)

    @classmethod
    def from_albergo2019(cls, geometry, *, m_sq: float, lam: float):
        """
        Parameterisation used in Albergo et al. (2019),
        https://arxiv.org/abs/1904.12072

        Parameters
        ----------
        m_sq
            Bare mass squared
        lam
            Quartic coupling constant
        """
        return cls(geometry, -2, 4 + m_sq, lam)

    @classmethod
    def from_nicoli2020(cls, geometry, *, kappa: float, lam: float):
        """
        Parameterisation used in Nicoli et al. (2020),
        https://arxiv.org/abs/2007.07115

        Parameters
        ----------
        m_sq
            Bare mass squared
        lam
            Quartic coupling constant
        """
        return cls.from_bosetti2015(geometry, beta=2 * kappa, lam=lam)

    def action(self, phi: torch.Tensor) -> torch.Tensor:
        """Action computed for a sample of field configurations.

        Parameters
        ----------
        phi
            Tensor containing sample of configurations, dimensions
            ``(sample_size, lattice_size)``

        Returns
        -------
        torch.Tensor
            The computed action for each configuration in the sample, dimensions
            ``(sample_size, 1)``
        """
        return (
            self.c_ising * (phi[:, self.shift] * phi.unsqueeze(dim=1)).sum(dim=1)
            + self.c_quadratic * phi.pow(2)
            + self.c_quartic * phi.pow(4)
        ).sum(dim=1, keepdim=True)

    def log_density(self, phi: torch.Tensor) -> torch.Tensor:
        """The negative action for a sample of field configurations.

        This is equal to the logarithm of the probability density up to an constant
        arising from unknown normalisation (the partition function).

        See :py:mod:`anvil.distributions.PhiFourScalar.action`
        """
        return -self.action(phi)
