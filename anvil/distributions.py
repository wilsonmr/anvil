# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
distributions.py

Module containing classes corresponding to different probability distributions.
"""
from functools import cached_property
from math import pi, sqrt

import torch
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


class FreeScalar:
    r"""Class representing a non-interacting scalar field.

    The action for the theory of a free scalar on a lattice is

    .. math::

        S(\phi) = \frac{1}{2} \sum_x \sum_y \phi(x) K(x, y) \phi(y)

    The eigenmodes of the matrix

    .. math:
        K(x, y) = \box(x, y) + m^2 \delta(x - y)

    (which is referred to here as the kinetic operator) are the momentum
    states :math:`\tilde\phi(p)`, and the associated eigenvalues in d=2 are

    .. math:

        \lambda_k = m^2 + 4 \sin^2(k1 / 2) + 4 \sin^2(k2 / 2)

    where (k1, k2) are the two components of the momentum.

    A Fourier transform diagonalises the kinetic operator, which lets us write
    the action in Fourier space as

    .. math:

        S(\tilde\phi) = \frac{1}{2V} \lambda_k |\tilde\phi_k|^2

    and hence the partition function is a product of Gaussian distributions
    for the variables :math:`|\tilde\phi(p)|`, with variances

    .. math:

        \sigma^2_k = V / \lambda_k

    This means we can sample from this probability distribution in Fourier
    space by simply generating Gaussian random numbers.
    """

    def __init__(self, geometry, m_sq=None):
        self.geometry = geometry
        self.shift = self.geometry.get_shift()
        self.size_out = geometry.volume

        if m_sq is not None:
            self.m_sq = m_sq
        else:
            self.m_sq = 16 / geometry.length ** 2

        self.i0 = self.geometry.length // 2 - 1  # index for zero mom k=0
        self.imax = -1  # index for maximum momentum

    @cached_property
    def _real_mode_mask(self):
        r"""Returns a boolean array which selects purely real eigenmodes.

        These eigenmodes have momenta:
            (0, 0)
            (0, k_max)
            (k_max, 0)
            (k_max, k_max)

        where :math:`k_{max} = \frac{2 \pi}{L} \frac{L}{2} = \pi` is the Nyquist
        frequency for a set of :math:`L` samples with unit spacing.
        """
        mask = torch.zeros((self.geometry.length, self.geometry.length), dtype=bool)
        mask[self.i0, self.i0] = True  # (0, 0)
        mask[self.i0, self.imax] = True  # (0, kmax)
        mask[self.imax, self.i0] = True  # (kmax, 0)
        mask[self.imax, self.imax] = True  # (kmax, kmax)
        return mask

    @property
    def eigenvalues(self) -> torch.Tensor:
        r"""Returns 2d tensor containing eigenvalues of the kinetic operator.

        The eigenvalues are given by

        .. math:

            \lambda_k = 4 \sin^2(k_1 / 2) + 4 \sin^2(k_2 / 2) + m_0^2

        where :math:`m_0` is the bare mass and :math:`(k_1, k_2)` are momenta.
        """
        momenta = (2 * pi / self.geometry.length) * torch.arange(
            -self.geometry.length // 2 + 1, self.geometry.length // 2 + 1
        )

        k1, k2 = torch.meshgrid(momenta, momenta)
        # sin^2(x/2) = (1 - cos(x))/2
        return 4 - 2 * (torch.cos(k1) + torch.cos(k2)) + self.m_sq

    @cached_property  # cache since used during training and sampling
    def variances(self) -> torch.Tensor:
        r"""Returns 2d tensor containing variances of the real and imaginary
        components of the eignemodes.

        The form of the Gaussian distribution implies that the eigenvalues are
        related to variances of the degrees of freedom :math:`|\tilde\phi_k|`,

        .. math:

            \sigma^2_k = V \lambda_k^{-1}

        However, in practice we generate both real and imaginary components of
        the eigenmodes, whose variance will each be **half** the variance of
        :math:`tilde\phi_k`, with the exception of the four purely real modes.
        """
        variances_abs = self.geometry.volume * torch.reciprocal(self.eigenvalues)

        variances = variances_abs

        # Complex modes - real and imag parts have half the variance of |\tilde\phi|
        variances[~self._real_mode_mask] /= 2

        return variances

    def gen_eigenmodes(self, sample_size):

        i0 = self.i0  # just easier on the eye

        # Start with L x L complex zeros
        eigenmodes = torch.complex(
            torch.zeros((sample_size, self.geometry.length, self.geometry.length)),
            torch.zeros((sample_size, self.geometry.length, self.geometry.length)),
        )

        # Generate Gaussian numbers for bottom right square (+, +)
        # NOTE: var(real + imag) = 1, thus var(real) = var(imag) = 1/2
        eigenmodes[:, i0:, i0:] = torch.randn_like(eigenmodes[:, i0:, i0:])
        eigenmodes.imag[:, self._real_mode_mask] = 0  # four of these are real

        # Generate top right square (-, +)
        eigenmodes[:, :i0, i0 + 1 : -1] = torch.randn_like(
            eigenmodes[:, :i0, i0 + 1 : -1]
        )

        # Reflect bottom right (+, +) to top left (-, -)
        eigenmodes[:, : i0 + 1, : i0 + 1] = torch.flip(
            eigenmodes[:, i0:-1, i0:-1].conj(), dims=(-2, -1)
        )

        # Reflect top right (+, -) to bottom left (-, +)
        eigenmodes[:, i0 + 1 : -1, :i0] = torch.flip(
            eigenmodes[:, :i0, i0 + 1 : -1].conj(), dims=(-2, -1)
        )

        # Reflect row/col with k1 = kmax / k2 = kmax
        eigenmodes[:, :i0, -1] = torch.flip(
            eigenmodes[:, i0 + 1 : -1, -1].conj(), dims=(-1,)
        )
        eigenmodes[:, -1, :i0] = torch.flip(
            eigenmodes[:, -1, i0 + 1 : -1].conj(), dims=(-1,)
        )

        # Let everything have variance 1, not 1/2
        eigenmodes *= sqrt(2)

        # Multiply by standard deviations
        eigenmodes *= self.variances.sqrt().unsqueeze(dim=0)

        return torch.roll(eigenmodes, shifts=(-self.i0, -self.i0), dims=(-2, -1))

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
            -(phi[:, self.shift] * phi.unsqueeze(dim=1)).sum(dim=1)
            + (4 + self.m_sq) / 2 * phi.pow(2)
        ).sum(dim=1, keepdim=True)

    def log_density(self, phi: torch.Tensor) -> torch.Tensor:
        """The negative action for a sample of field configurations.

        This is equal to the logarithm of the probability density up to an constant
        arising from unknown normalisation (the partition function).

        See :py:mod:`anvil.distributions.PhiFourScalar.action`
        """
        return -self.action(phi)

    def __call__(self, sample_size):
        """hi"""
        eigenmodes = self.gen_eigenmodes(sample_size)

        real_space_configs = torch.fft.ifft2(eigenmodes, norm="backward")

        assert torch.all(real_space_configs.imag.abs() < 1e-9)

        phi = real_space_configs.real

        # The action takes a split representation, so need to convert!
        phi_split = phi.view(-1, self.geometry.volume)[:, self.geometry.lexisplit]

        return phi_split, self.log_density(phi_split)


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
