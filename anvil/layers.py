# -*- coding: utf-8 -*-
r"""
layers.py

Contains nn.Modules which implement transformations of an input vector whilst computing the
Jacobian determinant of the inverse transformation.

Each transformation layers may contain several neural networks or learnable parameters.

A normalising flow, f, can be constructed from multiple layers using function composition:

        f(\phi) = g_1( g_2( ... ( g_n( \phi ) ) ... ) )

which is implemented using the architecture provided by torch.nn


All layers in this module contain a `forward` method which takes two torch.tensor objects
as inputs:

    - a batch of input vectors x, dimensions (N_batch, D) where D is the total number of
      units being transformed (i.e. number of lattice sites multiplied by field dimension).

    - a batch of scalar values for the logarithm of the 'current' probability density, at
      this stage in the normalising flow.

and returns two torch.tensor objects:

    - a batch of vectors \phi which have been transformed according to the *inverse*
    transformation g_i^{-1}, with the same dimensions as the input.

    - the updated logarithm of the probability density, including the contribution from
      this transformation, | \det \partial g_i / \partial \phi |.
"""
import torch
import torch.nn as nn
from torchsearchsorted import searchsorted
from math import pi

from anvil.core import FullyConnectedNeuralNetwork

import numpy as np


class CouplingLayer(nn.Module):
    """
    Base class for coupling layers.

    A generic coupling transformation takes the form

        x_a = \phi_a
        x_b = C( \phi_b ; {N(\phi_a)} )

    where the D-dimensional \phi vector has been split into two D/2-dimensional vectors
    \phi_a and \phi_b, and {N(\phi_a)} is a set of functions, possible neural networks,
    which take \phi_a as parameters.

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.


    Attributes
    ----------
    a_ind: slice (protected)
        Slice object which can be used to access the passive partition.
    b_ind: slice (protected)
        Slice object which can be used to access the partition that gets transformed.
    join_func: function (protected)
        Function which returns the concatenation of the two partitions in the
        appropriate order.
    """

    def __init__(self, size_half: int, even_sites: bool):
        super().__init__()

        if even_sites:
            # a is first half of input vector
            self._a_ind = slice(0, size_half)
            self._b_ind = slice(size_half, 2 * size_half)
            self._join_func = torch.cat
        else:
            # a is second half of input vector
            self._a_ind = slice(size_half, 2 * size_half)
            self._b_ind = slice(0, size_half)
            self._join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )


class AdditiveLayer(CouplingLayer):
    def __init__(
        self,
        size_half: int,
        *,
        hidden_shape: list,
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.t_network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )

    def forward(self, x_input, log_density, *unused) -> torch.Tensor:
        r"""Forward pass of affine transformation."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        x_a_stand = (x_a - x_a.mean()) / x_a.std()  # reduce numerical instability

        t_out = self.t_network(x_a_stand)

        phi_b = x_b - t_out

        phi_out = self._join_func([x_a, phi_b], dim=1)

        return phi_out, log_density


class AffineLayer(CouplingLayer):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    An affine transformation, x = g_i(\phi), is defined as:

        x_a = \phi_a
        x_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural networks.
    hidden_shape: list
        list containing hidden vector sizes for the neural networks.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the networks.
    s_final_activation: str
        string which is a key for an activation function, which the output of the s
        network will be passed through.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural networks.

    Attributes
    ----------
    s_network: torch.nn.Module
        the dense layers of network s, values are intialised as per the default
        initialisation of `nn.Linear`
    t_network: torch.nn.Module
        the dense layers of network t, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        size_half: int,
        *,
        hidden_shape: list,
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)

        self.s_network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )
        self.t_network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )
        self.z2_equivar = z2_equivar

    def forward(self, x_input, log_density, *unused) -> torch.Tensor:
        r"""Forward pass of affine transformation."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        x_a_stand = (x_a - x_a.mean()) / x_a.std()

        s_out = self.s_network(x_a_stand)
        t_out = self.t_network(x_a_stand)

        if self.z2_equivar:
            s_out = torch.abs(s_out)

        phi_b = (x_b - t_out) * torch.exp(-s_out)

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density += s_out.sum(dim=1, keepdim=True)

        return phi_out, log_density


class RationalQuadraticSplineLayer(CouplingLayer):
    r"""A coupling transformation from a finite interval to itself based on a piecewise
    rational quadratic spline function.

    The interval is divided into K segments (bins) with widths w_k and heights h_k. The
    'knot points' (\phi_k, x_k) are the cumulative sum of (h_k, w_k), starting at (-B, -B)
    and ending at (B, B).

    In addition to the w_k and h_k, the derivatives d_k at the internal knot points are
    generated by a neural network. d_0 and d_K are set to 1.

    Defing the slopes s_k = h_k / w_k and fractional position within a bin

            alpha(x) = (x - x_{k-1}) / w_k

    the coupling transformation is defined piecewise by

            \phi = C^{-1}(x, {h_k, s_k, d_k})
                 = \phi_{k-1}
                 + ( h_k(s_k * \alpha^2 + d_k * \alpha * (1 - \alpha)) )
                 / ( s_k + (d_{k+1} + d_k - 2s_k) * \alpha * (1 - \alpha) )

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural network.
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural network.

    Attributes
    ----------
    network: torch.nn.Module
        the dense layers of the network, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        size_half: int,
        interval: int,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.size_half = size_half
        self.n_segments = n_segments

        self.network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half * (3 * n_segments - 1),
            hidden_shape=hidden_shape,
            activation=activation,
            bias=True,  # biases very useful
        )

        self.norm_func = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()

        self.B = interval
        self.eps = 1e-6

        self.z2_equivar = z2_equivar

    def forward(self, x_input, log_density, negative_mag):
        """Forward pass of the rational quadratic spline layer."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        x_a_stand = (x_a - x_a.mean()) / x_a.std()  # reduce numerical instability

        # Naively enforce \phi \to -\phi symmetry
        if self.z2_equivar:
            x_a_stand[negative_mag] = -x_a_stand[negative_mag]

        phi_b = torch.zeros_like(x_b)
        grad = torch.ones_like(x_b).unsqueeze(dim=-1)

        # Apply mask for linear tails
        inside_mask = abs(x_b) <= self.B
        x_b_in = x_b[inside_mask]
        phi_b[~inside_mask] = x_b[~inside_mask]

        h_raw, w_raw, d_raw = (
            self.network(x_a_stand)
            .view(-1, self.size_half, 3 * self.n_segments - 1)
            .split(
                (self.n_segments, self.n_segments, self.n_segments - 1),
                dim=2,
            )
        )

        if self.z2_equivar:
            h_raw[negative_mag] = torch.flip(h_raw[negative_mag], dims=(2,))
            w_raw[negative_mag] = torch.flip(w_raw[negative_mag], dims=(2,))
            d_raw[negative_mag] = torch.flip(d_raw[negative_mag], dims=(2,))

        h_norm = self.norm_func(h_raw[inside_mask]) * 2 * self.B
        w_norm = self.norm_func(w_raw[inside_mask]) * 2 * self.B
        d_pad = nn.functional.pad(
            self.softplus(d_raw)[inside_mask], (1, 1), "constant", 1
        )

        x_knot_points = (
            torch.cat(
                (
                    torch.zeros(w_norm.shape[0], 1) - self.eps,
                    torch.cumsum(w_norm, dim=1),
                ),
                dim=1,
            )
            - self.B
        )
        phi_knot_points = (
            torch.cat(
                (
                    torch.zeros(h_norm.shape[0], 1),
                    torch.cumsum(h_norm, dim=1),
                ),
                dim=1,
            )
            - self.B
        )

        k_ind = (
            searchsorted(
                x_knot_points.contiguous(),
                x_b_in.contiguous().view(-1, 1),
            )
            - 1
        ).clamp(0, self.n_segments - 1)

        w_k = torch.gather(w_norm, 1, k_ind)
        h_k = torch.gather(h_norm, 1, k_ind)
        s_k = h_k / w_k
        d_k = torch.gather(d_pad, 1, k_ind)
        d_kp1 = torch.gather(d_pad, 1, k_ind + 1)

        x_km1 = torch.gather(x_knot_points, 1, k_ind)
        phi_km1 = torch.gather(phi_knot_points, 1, k_ind)

        alpha = (x_b_in.unsqueeze(dim=-1) - x_km1) / w_k

        phi_b[inside_mask] = (
            phi_km1
            + (h_k * (s_k * alpha.pow(2) + d_k * alpha * (1 - alpha)))
            / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha))
        ).squeeze()

        grad[inside_mask] = (
            s_k.pow(2)
            * (
                d_kp1 * alpha.pow(2)
                + 2 * s_k * alpha * (1 - alpha)
                + d_k * (1 - alpha).pow(2)
            )
        ) / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha)).pow(2)

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density -= torch.log(grad).sum(dim=1)
        
        return phi_out, log_density

# TODO not necessary to define a nn.module for this now I've taken out learnable gamma
class BatchNormLayer(nn.Module):
    """Performs batch normalisation on the input vector.

    Parameters
    ----------
    scale: int
        An additional scale factor to be applied after batch normalisation.
    """

    def __init__(self, scale=1):
        super().__init__()
        self.gamma = scale

    def forward(self, v_in, log_density, *unused):
        """Forward pass of the batch normalisation transformation."""

        v_out = self.gamma * (v_in - v_in.mean()) / torch.std(v_in)

        return (
            v_out,
            log_density,
        )  # don't need to update log dens - nothing to optimise

class GlobalRescaling(nn.Module):
    def __init__(self, initial=1):
        super().__init__()
        
        self.scale = nn.Parameter(torch.Tensor([initial]))

    def forward(self, v_in, log_density, *unused):
        v_out = self.scale * v_in
        log_density -= v_out.shape[-1] * torch.log(self.scale)
        return v_out, log_density

