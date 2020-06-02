# -*- coding: utf-8 -*-
r"""
layers.py
"""
import torch
import torch.nn as nn

from math import pi
from functools import partial

from reportengine import collect

from anvil.core import Sequential, NeuralNetwork


class CouplingLayer(nn.Module):
    """
    Base class for coupling layers
    """

    def __init__(self, i_layer: int, size_half: int, even_sites: bool):
        super().__init__()

        if even_sites:
            # a is first half of input vector
            self._a_ind = slice(0, size_half)
            self._b_ind = slice(size_half, 2 * size_half)
            self.join_func = torch.cat
            partition = "even"
        else:
            # a is second half of input vector
            self._a_ind = slice(size_half, 2 * size_half)
            self._b_ind = slice(0, size_half)
            self.join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )
            partition = "odd"

        self.label = f"Layer: {i_layer}, partition: {partition}"

    def __str__(self):
        return f"{self.label}\n------------\n{self}"


class AffineLayer(CouplingLayer):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    Affine transformation, x = g_i(\phi), defined as:

        x_a = \phi_a
        x_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    where D-dimensional phi has been split into two D/2-dimensional pieces
    \phi_a and \phi_b. s_i and t_i are neural networks, whose parameters are
    torch.nn parameters of this class.

    In the case of this class the partitions are expected to
    be the first and second part of the input data, an additional transformation
    will therefore need to be applied to the output/input of this network if
    the desired partition is to have a different pattern, for example a
    checkerboard transformation.

    Input and output data are flat vectors stacked in the first dimension
    (batch dimension).

    Parameters
    ----------
    size_in: int
        number of dimensions, D, of input/output data. Data should be fed to
        affine layer in shape (N_states, size_in).
    s_network: NeuralNetwork
        The 's' network, see the `NeuralNetwork` class for details.
    t_network: NeuralNetwork
        As above, for the 't' network
    i_affine: int
        index of this affine layer in full set of affine transformations,
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.

    Attributes
    ----------
    s_network: torch.nn.Module
        the dense layers of network s, values are intialised as per the
        default initialisation of `nn.Linear`
    t_network: torch.nn.Module
        the dense layers of network t, values are intialised as per the
        default initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input):
        performs the transformation of the *inverse* coupling layer, denoted
        g_i^{-1}(x). Returns the output vector along with the contribution
        to the determinant of the jacobian of the *forward* transformation.
        = \frac{\partial g(\phi)}{\partial \phi}

    """

    def __init__(
        self,
        i_layer: int,
        size_half: int,
        *,
        s_hidden_shape: list = [24,],
        t_hidden_shape: list = [24,],
        activation: str = "leaky_relu",
        batch_normalise: bool = False,
        even_sites: bool,
    ):
        super().__init__(i_layer, size_half, even_sites)

        # Construct networks
        self.s_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=s_hidden_shape,
            activation=activation,
            final_activation=activation,
            batch_normalise=batch_normalise,
            label=f"({self.label}) 's' network",
        )
        self.t_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=t_hidden_shape,
            activation=activation,
            final_activation=activation,
            batch_normalise=batch_normalise,
            label=f"({self.label}) 't' network",
        )

    def forward(self, x_input, log_density) -> torch.Tensor:
        r"""performs the transformation of the inverse coupling layer, denoted
        g_i^{-1}(x)

        inverse transformation, \phi = g_i^{-1}(x), defined as:

        \phi_a = x_a
        \phi_b = (x_b - t_i(x_a)) * exp(-s_i(x_a))

        see eq. (10) of https://arxiv.org/pdf/1904.12072.pdf

        Also computes the logarithm of the jacobian determinant for the
        forward transformation (inverse of the above), which is equal to
        the logarithm of

        \frac{\partial g(\phi)}{\partial \phi} = prod_j exp(s_i(\phi)_j)

        Parameters
        ----------
        x_input: torch.Tensor
            stack of vectors x, shape (N_states, D)
        log_density: torch.Tensor
            current value for the logarithm of the map density

        Returns
        -------
            out: torch.Tensor
                stack of transformed vectors phi, with same shape as input
            log_density: torch.Tensor
                updated log density for the map, with the addition of the logarithm
                of the jacobian determinant for the inverse of the transformation
                applied here.
        """
        x_a = x_input[..., self._a_ind]
        x_b = x_input[..., self._b_ind]
        s_out = self.s_network(x_a)
        t_out = self.t_network(x_a)
        phi_b = (x_b - t_out) * torch.exp(-s_out)

        phi_out = self.join_func([x_a, phi_b], dim=-1)
        log_density += s_out.sum(dim=tuple(range(1, len(s_out.shape)))).view(-1, 1)

        return phi_out, log_density


def affine_transformation(i_layer, size_half, layer_spec):
    print(layer_spec)
    coupling_transformation = partial(AffineLayer, i_layer, size_half, **layer_spec)
    return Sequential(
        coupling_transformation(even_sites=True),
        coupling_transformation(even_sites=False),
    )


LAYER_OPTIONS = {"real_nvp": affine_transformation}
