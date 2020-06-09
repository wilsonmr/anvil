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

from anvil.core import NeuralNetwork


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


class AffineLayer(CouplingLayer):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    Affine transformation, x = g_i(\phi), defined as:

        x_a = \phi_a
        x_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural networks.
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
        s_final_activation: str,
        batch_normalise: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.s_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=s_final_activation,
            batch_normalise=batch_normalise,
        )
        self.t_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            batch_normalise=batch_normalise,
        )
        # NOTE: Could potentially have non-default inputs for s and t networks
        # by adding dictionary of overrides - e.g. s_options = {}

    def forward(self, x_input, log_density) -> torch.Tensor:
        r"""Forward pass of affine transformation."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        x_a_stand = (x_a - x_a.mean()) / x_a.std()  # reduce numerical instability
        s_out = self.s_network(x_a_stand)
        t_out = self.t_network(x_a_stand)
        phi_b = (x_b - t_out) * torch.exp(-s_out)

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density += s_out.sum(dim=1, keepdim=True)

        return phi_out, log_density


class LinearSplineLayer(CouplingLayer):
    r"""A coupling transformation from [0, 1] -> [0, 1] based on a piecewise linear function.

    The interval is divided into K equal-width (w) segments (bins), with K+1 knot points
    (bin boundaries). The coupling transformation is defined piecewise by the unique
    polynomials whose end-points are the knot points.

    A neural network generates the K values for the y-positions (heights) at the knot points.
    The coupling transformation is then defined as the cumulative distribution function
    associated with the probability masses given by the heights.

    The inverse coupling transformation is

        \phi = C^{-1}(x, {h_k}) = \phi_{k-1} + \alpha h_k

    where x_{k-1} = \sum_{k'=1}^{k-1} h_{k'} is the (k-1)-th knot point, and \alpha is the
    fractional position of x in the k-th bin, which is (x - (k-1) * w) / w.

    The gradient of the forward transformation is simply
        
        dx / d\phi = w / h_k

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
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the neural
        network.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural network.

    Attributes
    ----------
    h_network: torch.nn.Module
        the dense layers of network h, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        size_half: int,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        batch_normalise: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)

        self.size_half = size_half
        self.n_segments = n_segments
        self.width = 1 / n_segments

        eps = 1e-6  # prevent rounding error which causes sorting into -1th bin
        self.x_knot_points = torch.linspace(-eps, 1 + eps, n_segments + 1).view(1, -1)

        self.h_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half * n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=activation,
            batch_normalise=batch_normalise,
        )

        self.norm_func = nn.Softmax(dim=2)

    def forward(self, x_input, log_density):
        """Forward pass of the linear spline layer."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]

        net_out = self.norm_func(
            self.h_network(x_a - 0.5).view(-1, self.size_half, self.n_segments)
        )
        phi_knot_points = torch.cat(
            (
                torch.zeros(net_out.shape[0], self.size_half, 1),
                torch.cumsum(net_out, dim=2),
            ),
            dim=2,
        )

        # Sort x_b into the appropriate bin
        # NOTE: need to make x_b contiguous, otherwise searchsorted returns nonsense
        k_ind = searchsorted(self.x_knot_points, x_b.contiguous()) - 1
        k_ind.unsqueeze_(dim=-1)

        h_k = torch.gather(net_out, 2, k_ind)
        phi_km1 = torch.gather(phi_knot_points, 2, k_ind)
        alpha = (x_b.unsqueeze(dim=-1) - k_ind * self.width) / self.width
        phi_b = (phi_km1 + alpha * h_k).squeeze()

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density -= torch.log(h_k).sum(dim=1)

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
        log_density -= torch.log1p(phi_out ** 2).sum(dim=1, keepdim=True)
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
        log_density -= 2 * torch.log(torch.cos(0.5 * (phi_out - pi))).sum(
            dim=1, keepdim=True
        )
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

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, size_half: int):
        super().__init__()
        self.size_half = size_half
        self.size_out = 2 * size_half

    def forward(self, x_input, log_density):
        """Forward pass of the projection transformation."""
        polar, azimuth = x_input.view(-1, self.size_half, 2).split(1, dim=2)
        rad = torch.tan(0.5 * polar)  # radial coordinate

        # -1 factor because actually want azimuth - pi, but -1 is faster than shift by pi
        phi_out = -rad * torch.cat((torch.cos(azimuth), torch.sin(azimuth)), dim=2)
        log_density -= torch.log(rad + rad.pow(3)).sum(dim=1)

        return phi_out.view(-1, self.size_out), log_density


class InverseProjectionLayer2D(nn.Module):
    r"""Applies the inverse stereographic projection map R2 -> S2 - {0} to the
    entire input vector.

    The inverse projection map is defined, with R = \sqrt(x_1^2 + x_2^2), as

        \phi_1 = 2 \arctan(R)
        \phi_2 = \arctan2(x_1, x_2)

    where \arctan2 is a variant of \arctan than maps inputs to (-pi, pi).

    The Jacobian determinant of the projection map is

        | \det J | = 1/2 \sec^2(\phi_1 / 2) \tan(\phi_1 / 2)
    
    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, size_half: int):
        super().__init__()
        self.size_half = size_half
        self.size_out = 2 * size_half

    def forward(self, x_input, log_density):
        """Forward pass of the inverse projection transformation."""
        proj_x, proj_y = x_input.view(-1, self.size_half, 2).split(1, dim=2)

        polar = 2 * torch.atan(torch.sqrt(proj_x.pow(2) + proj_y.pow(2)))
        azimuth = torch.atan2(proj_x, proj_y) + pi

        phi_out = torch.cat((polar, azimuth), dim=2)
        log_density += (
            torch.log(torch.sin(0.5 * polar)) - 3 * torch.log(torch.cos(0.5 * polar))
        ).sum(dim=1)

        return phi_out.view(-1, self.size_out), log_density


class GlobalAffineLayer(nn.Module):
    r"""Applies an affine transformation to every data point using a given scale and shift,
    which are *not* learnable. Useful to shift the domain of a learned distribution. This is
    done at the cost of a constant term in the logarithm of the Jacobian determinant, which
    is ignored.

    Parameters
    ----------
    scale: (int, float)
        Every data point will be multiplied by this factor.
    shift: (int, float)
        Every scaled data point will be shifted by this factor.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, scale, shift):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x_input, log_density):
        """Forward pass of the global affine transformation."""
        return self.scale * x_input + self.shift, log_density
