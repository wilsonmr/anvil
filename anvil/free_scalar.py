# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
free_scalar.py

module containing the FreeScalarMomentumSpace class used to compare with model
trained to free scalar theory

"""
from functools import cached_property
from math import pi

import numpy as np
import torch
    

class FreeScalarMomentumSpace:
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

