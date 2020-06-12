# -*- coding: utf-8 -*-
r"""
layers.py

Contains nn.Modules which implement transformations of an input vector whilst computing the
Jacobian determinant of the inverse transformation.

Each transformation layers may contain several neural networks or learnable parameters.

A normalising flow, f, can be constructed from multiple layers using function composition:

        f(\phi) = g_1( g_2( ... ( g_n( \phi ) ) ... ) )

which is implemented using the architecture provided by torch.nn

Layers can be divided into two classes.

1. Coupling layers: (x_in, x_passive, log_density) -> (phi_out, log_density)

    Takes two tensors `x_in`, `x_passive` of dimensions (n_batch, D) and one tensor `log_density`
    of dimensions (n_batch, 1).

    The `x_in` is transformed by a bijective function whose parameters are neural networks
    which themselves take `x_passive` as an input vector.

    The current value of the logarithm of the probability density is updated by adding the
    Jacobian determinant of the *inverse* of the coupling transformation.

2. Full layers: (x_in, log_density) -> (phi_out, log_density)

    Takes a single input tensor and transforms the entire tensor, with no coupling occuring
    between lattice sites.

    These layers may be 'cascaded' using anvil.core.Sequential provided `x_in` is the full
    dataset.
"""
import torch
import torch.nn as nn
from math import pi

from anvil.core import NeuralNetwork

# ----------------------------------------------------------------------------------------- #
#                                                                                           #
#                                   Coupling layers                                         #
#                                                                                           #
# ----------------------------------------------------------------------------------------- #


class AffineLayer(nn.Module):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    An affine transformation, x = g_i(\phi), is defined as:

        x_r = \phi_r
        x_b = \phi_b * exp(s_i(\phi_r)) + t_i(\phi_r)

    Parameters
    ----------
    size_in: int
        Size of the input tensor at dimension 1, which is the size of the input vector
        for the neural networks.
    hidden_shape: list
        list containing hidden vector sizes for the neural networks.
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
        Performs an affine transformation of the `x_in` tensor, using the `x_passive`
        tensor as parameters for the neural networks. Returns the transformed tensor
        along with the updated log density.
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


class NCPLayer(nn.Module):
    r"""A coupling layer which performs a transformation from (0, 2\pi) -> (0, 2\pi),
    which is an affine transformation X -> \alpha * X + \beta on the Euclidean vector
    obtained by stereographic projection of the input data.

    The transformation x = g(\phi) is defined as

        x_r = \phi_r
        x_b = (
                2 \arctan( \alpha(\phi_r) + \tan((\phi_b - \pi) / 2) + \beta(\phi_r))
                + \pi + \theta
              )

    where \alpha and \beta are neural networks, and \theta is a learnable global phase shift.

    The Jacobian determinant can be computed analytically for the full transformation,
    including projection and inverse. 
        
        | \det J | = \prod_n (
            (1 + \beta ** 2) / \alpha * \sin^2(x / 2) + \beta 
            + \alpha * \cos^2(x / 2)
            - \beta * \sin(x_b)
            )
    
    Parameters
    ----------
    size_in: int
        Size of the input tensor at dimension 1, which is the size of the input vector
        for the neural networks.
    hidden_shape: list
        list containing hidden vector sizes for the neural networks.
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
    s_network: NeuralNetwork
        When exponentiated, this is the linear transformation \alpha.
    t_network: NeuralNetwork
        The shift part (\beta) of the affine transformation.
    phase_shift: nn.Parameter
        A learnable global phase shift.

    Notes
    -----
    The Jacobian determinant is computed using the gradient of the inverse transformation,
    which is the reciprocal of the gradient of the forward transformation.

    This can be thought of as a lightweight version of real_nvp_circle, with just one affine
    transformation.
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
        self.phase_shift = nn.Parameter(torch.rand(1))

    def forward(self, x_in, x_passive, log_density):
        """Forward pass of the project-affine-inverse transformation."""
        # standardise to reduce numerical instability
        x_for_net = (x_passive - x_passive.mean()) / x_passive.std()

        alpha = torch.exp(self.s_network(x_for_net))
        beta = self.t_network(x_for_net)

        phi_out = (
            2 * torch.atan(alpha * torch.tan((x_in - pi) / 2) + beta)
            + pi
            + self.phase_shift
        ) % (2 * pi)

        log_density += torch.log(
            (1 + beta ** 2) / alpha * torch.sin(x_in / 2) ** 2
            + alpha * torch.cos(x_in / 2) ** 2
            - beta * torch.sin(x_in)
        ).sum(dim=1, keepdim=True)

        return phi_out, log_density


# ----------------------------------------------------------------------------------------- #
#                                                                                           #
#                                       Full layers                                         #
#                                                                                           #
# ----------------------------------------------------------------------------------------- #


class ProjectionLayer(nn.Module):
    r"""Applies the stereographic projection map S1 - {0} -> R1 to the entire
    input vector.

    The projection map is defined as

        \phi = \tan( (x - \pi) / 2 )

    And the gradient of its inverse is

        dx / d\phi = 2 / (1 + \phi^2)

    Methods
    -------
    forward(x_in, log_density)
    """

    def forward(self, x_in, log_density):
        """Forward pass of the projection transformation."""
        phi_out = torch.tan(0.5 * (x_in - pi))
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
    forward(x_in, log_density)
        see docstring for anvil.layers
    """

    def __init__(self):
        super().__init__()
        self.phase_shift = nn.Parameter(torch.rand(1))

    def forward(self, x_in, log_density):
        """Forward pass of the inverse projection transformation."""
        phi_out = 2 * torch.atan(x_in) + pi
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
    forward(x_in, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_in, log_density):
        """Forward pass of the projection transformation."""
        polar, azimuth = x_in.split(1, dim=1)
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
    forward(x_in, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_in, log_density):
        """Forward pass of the inverse projection transformation."""
        proj_x, proj_y = x_in.split(1, dim=1)

        polar = 2 * torch.atan(torch.sqrt(proj_x.pow(2) + proj_y.pow(2)))
        azimuth = torch.atan2(proj_x, proj_y) + pi

        phi_out = torch.cat((polar, azimuth), dim=1)
        log_density += (
            torch.log(torch.sin(0.5 * polar)) - 3 * torch.log(torch.cos(0.5 * polar))
        ).sum(dim=2)

        return phi_out, log_density
