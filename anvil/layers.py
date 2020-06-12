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
from math import pi

from anvil.core import NeuralNetwork


class AffineLayer(nn.Module):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    Affine transformation, x = g_i(\phi), defined as:

        x_a = \phi_a
        x_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    Parameters
    ----------
    size_in: int
        Size of the input tensor at dimension 1, which is the size of the input vector
        for the neural networks.
    hidden_shape: list
        list containing hidden vector sizes the neural networks.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the networks.
    s_final_activation: str
        string which is a key for an activation function, which the output of the s
        network will be passed through.
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the neural
        networks.

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
    forward(x_in, x_passive, log_density)
    """

    def __init__(
        self,
        size_in: int,
        *,
        hidden_shape: list,
        activation: str,
        s_final_activation: str,
        batch_normalise: bool,
    ):
        super().__init__()
        self.s_network = NeuralNetwork(
            size_in=size_in,
            size_out=size_in,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=s_final_activation,
            batch_normalise=batch_normalise,
        )
        self.t_network = NeuralNetwork(
            size_in=size_in,
            size_out=size_in,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            batch_normalise=batch_normalise,
        )

    def forward(self, x_in, x_passive, log_density):
        """Forward pass of affine transformation."""
        # standardise to reduce numerical instability
        x_for_net = (x_passive - x_passive.mean()) / x_passive.std()

        s_out = self.s_network(x_for_net)
        t_out = self.t_network(x_for_net)

        phi_out = (x_in - t_out) * torch.exp(-s_out)
        log_density += s_out.sum(dim=1, keepdim=True)

        return phi_out, log_density


class ProjectionLayer(nn.Module):
    r"""Applies the stereographic projection map S1 - {0} -> R1 to the entire
    input vector.

    The projection map is defined as

        \phi = \tan( (x - \pi) / 2 )

    And the gradient of its inverse is

        dx / d\phi = 2 / (1 + \phi^2)

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_input, log_density):
        """Forward pass of the projection transformation."""
        phi_out = torch.tan(0.5 * (x_input - pi))
        log_density -= torch.log1p(phi_out ** 2).sum(dim=2)
        return phi_out, log_density


class InverseProjectionLayer(nn.Module):
    r"""Applies the inverse stereographic projection map R1 -> S1 - {0} to the
    entire input vector.

    The inverse projection map is defined as

        \phi = 2 \arctan(x) + \pi

    And the gradient of its inverse is

        dx / d\phi = 1/2 * sec^2( (\phi - \pi) / 2 )

    Attributes
    ----------
    phase_shift: nn.Parameter
        A learnable phase shift.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self):
        super().__init__()
        self.phase_shift = nn.Parameter(torch.rand(1))

    def forward(self, x_input, log_density):
        """Forward pass of the inverse projection transformation."""
        phi_out = 2 * torch.atan(x_input) + pi
        log_density -= 2 * torch.log(torch.cos(0.5 * (phi_out - pi))).sum(dim=2)
        return (phi_out + self.phase_shift) % (2 * pi), log_density


class ProjectionLayer2D(nn.Module):
    r"""Applies the stereographic projection map S2 - {0} -> R2 to the entire
    input vector.

    The projection map is defined as

        \phi_1 = \tan(x_1 / 2) \cos(x_2)
        \phi_2 = \tan(x_1 / 2) \sin(x_2)

    where ( R = \tan(x_1 / 2), x_2 ), ( \phi_1, \phi_2 ) are, respectively, polar and
    Euclidean coordinates on R2.

    The Jacobian determinant of the inverse is

        | \det J | = 2 / ( R(1 + R^2) )

    where R is the radial coordinate defined above.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_input, log_density):
        """Forward pass of the projection transformation."""
        polar, azimuth = x_input.split(1, dim=1)
        rad = torch.tan(0.5 * polar)  # radial coordinate

        # -1 factor because actually want azimuth - pi, but -1 is faster than shift by pi
        phi_out = -rad * torch.cat((torch.cos(azimuth), torch.sin(azimuth)), dim=1)
        log_density -= torch.log(rad + rad.pow(3)).sum(dim=2)

        return phi_out, log_density


class InverseProjectionLayer2D(nn.Module):
    r"""Applies the inverse stereographic projection map R2 -> S2 - {0} to the
    entire input vector.

    The inverse projection map is defined, with R = \sqrt(x_1^2 + x_2^2), as

        \phi_1 = 2 \arctan(R)
        \phi_2 = \arctan2(x_1, x_2)

    where \arctan2 is a variant of \arctan than maps inputs to (-pi, pi).

    The Jacobian determinant of the projection map is

        | \det J | = 1/2 \sec^2(\phi_1 / 2) \tan(\phi_1 / 2)

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_input, log_density):
        """Forward pass of the inverse projection transformation."""
        proj_x, proj_y = x_input.split(1, dim=1)

        polar = 2 * torch.atan(torch.sqrt(proj_x.pow(2) + proj_y.pow(2)))
        azimuth = torch.atan2(proj_x, proj_y) + pi

        phi_out = torch.cat((polar, azimuth), dim=1)
        log_density += (
            torch.log(torch.sin(0.5 * polar)) - 3 * torch.log(torch.cos(0.5 * polar))
        ).sum(dim=2)

        return phi_out, log_density
