# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
free_scalar.py

module containing the FreeScalar class used to compare with model
trained to free scalar theory

"""
from functools import cached_property
from math import pi, sqrt

import torch


class FreeScalar:
    r"""Class representing a non-interacting scalar field in momentum space.

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

    def rvs_eigenmodes(self, sample_size: int) -> torch.Tensor:
        r"""Generates a sample of field configurations in momentum space.

        Given ``sample_size``, generates a sample of complex, Hermitean configurations
        that are distributed according to the Fourier transform of the action of a
        free scalar theory, that is, a product of uncorrelated one-dimensional
        Gaussian distributions with variances determined by the ``variances`` method.

        Each configuration returned is a 2D tensor. Going from low to high indices in
        the two dimensions corresponds to running through integers :math:`n` labeling
        momentum sates

            .. math:

                k_n = \frac{2\pi}{L} n

        where

            .. math:

                n = 0, 1, ..., L/2-1, -L/2, -L/2+1, ..., -1

        Parameters
        ----------
        sample_size
            Number of configurations to generate

        Returns
        -------
        torch.Tensor
            Complex tensor with shape ``(sample_size, L, L)``. Zero momentum component
            found at the (0, 0) position.
        """

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

        # Roll so that [0,0] indexes the zero momentum component
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

    def __call__(self, sample_size: int):
        """Returns a tuple of field configurations and their associated actions.

        The returned configurations are in *real* space, and have been arranged
        according to the flat-split representation set out in
        :py:class:`anvil.geometry.Geometry2D` so that actions can be computed in
        the usual way.

        This allows the class to be used as a 'base' distribution, similarly to
        :py:class:`anvil.distributions.Gaussian`.

        Parameters
        ----------
        sample_size
            Number of free field configurations to generate.

        Returns
        -------
        tuple
            Sample of field configurations, dimensions ``(sample_size, lattice_size)``
            Tensor containing the negative actions, dimensions ``(sample_size, 1)``
        """
        # Generate Hermitean configs representing configs in Fourier space
        eigenmodes = self.rvs_eigenmodes(sample_size)

        # Inverse Fourier transform, including normalization by 1/|\Lambda|
        real_space_configs = torch.fft.ifft2(eigenmodes, norm="backward")

        assert torch.all(real_space_configs.imag.abs() < 1e-9)

        # Convert to real tensor
        phi = real_space_configs.real

        # The action takes a split representation, so need to convert!
        phi_split = phi.view(-1, self.geometry.volume)[:, self.geometry.lexisplit]

        return phi_split, self.log_density(phi_split)
