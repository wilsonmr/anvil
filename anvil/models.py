# -*- coding: utf-8 -*-
r"""
models.py

Module containing the base classes for affine layers and full normalising flow
models used to transform a simple distribution into an estimate of a target
distribution as per https://arxiv.org/abs/1904.12072

Classes
-------
RealNVP: nn.Module
    Model which performs a real-valued non-volume preserving (real NVP)
    transformation, which maps a simple distribution z^n to a complicated
    distribution \phi^n, where n refers to the dimensionality of the data.

"""
from math import sqrt, pi, log

import torch
import torch.nn as nn

ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
    None: nn.Identity,
}


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        size_in: int,
        size_out: int,
        hidden_shape: list = [24,],
        activation: str = "leaky_relu",
        final_activation: str = "identity",
        do_batch_norm: bool = False,
        name: str = "net",
    ):
        super(NeuralNetwork, self).__init__()

        self.name = name

        self.size_in = size_in
        self.size_out = size_out
        self.hidden_shape = hidden_shape

        if do_batch_norm:
            self.batch_norm = nn.BatchNorm1d
        else:
            self.batch_norm = nn.Identity

        self.activation_func = ACTIVATION_LAYERS[activation]
        self.final_activation_func = ACTIVATION_LAYERS[final_activation]

        self.network = self._construct_network()

    def __str__(self):
        output = ""
        print(f"Network: {self.name}")
        print("-----------")
        print(f"Shape: ", self.size_in, *self.hidden_shape, self.size_out)
        print(f"Activation function: {self.activation_func}")
        return output


    def _block(self, size_in, size_out, activation_func):
        return [
            nn.Linear(size_in, size_out),
            self.batch_norm(size_out),
            activation_func(),
        ]

    def _construct_network(self):
        layers = self._block(self.size_in, self.hidden_shape[0], self.activation_func)
        for f_in, f_out in zip(self.hidden_shape[:-1], self.hidden_shape[1:]):
            layers += self._block(f_in, f_out, self.activation_func)
        layers += self._block(
                self.hidden_shape[-1], self.size_out, self.final_activation_func
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AffineLayer(nn.Module):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    Affine transformation, z = g_i(\phi), defined as:

        z_a = \phi_a
        z_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

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
        network in shape (N_states, size_in).
    s_hidden_shape: tuple
        tuple which gives the number of nodes in the hidden layers of neural
        network s_i, can be a single layer network with 16 nodes e.g (16,)
    t_hidden_shape: tuple
        tuple which gives the number of nodes in the hidden layers of neural
        network t_i.
    i_affine: int
        index of this affine layer in full set of affine transformations,
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.

    Attributes
    ----------
    s_layers: torch.nn.ModuleList
        the dense layers of network s, values are intialised as per the
        default initialisation of `nn.Linear`
    t_layers: torch.nn.ModuleList
        the dense layers of network t, values are intialised as per the
        default initialisation of `nn.Linear`

    Methods
    -------
    coupling_layer(phi_input):
        performs the transformation of a single coupling layer, denoted
        g_i(\phi)
    inverse_coupling_layer(z_input):
        performs the transformation of the inverse coupling layer, denoted
        g_i^{-1}(z)
    det_jacobian(phi_input):
        returns the contribution to the determinant of the jacobian
        = \frac{\partial g(\phi)}{\partial \phi}

    """

    def __init__(
        self,
        i_affine: int,
        size_in: int,
        s_network_spec: dict,
        t_network_spec: dict,
        standardise_inputs: bool = True,
    ):
        super(AffineLayer, self).__init__()
        size_half = int(size_in / 2)

        self.s_network = NeuralNetwork(size_half, size_half, **s_network_spec, name=f"s{i_affine}")
        self.t_network = NeuralNetwork(size_half, size_half, **t_network_spec, name=f"t{i_affine}")

        #print(self.s_network)

        if (i_affine % 2) == 0:  # starts at zero
            # a is first half of input vector
            self._a_ind = slice(0, int(size_half))
            self._b_ind = slice(int(size_half), size_in)
            self.join_func = torch.cat
        else:
            # a is second half of input vector
            self._a_ind = slice(int(size_half), size_in)
            self._b_ind = slice(0, int(size_half))
            self.join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )

        if standardise_inputs == True:
            self.standardise = lambda x: (x - x.mean()) / x.std()
        else:
            self.standardise = lambda x: x

    def forward(self, x_input) -> torch.Tensor:
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

        Returns
        -------
            out: torch.Tensor
                stack of transformed vectors phi, with same shape as input
            log_det_jacobian: torch.Tensor
                logarithm of the jacobian determinant for the inverse of the
                transformation applied here.
        """
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        x_a_stand = self.standardise(x_a)
        s_out = self.s_network(x_a_stand)
        t_out = self.t_network(x_a_stand)
        phi_b = (x_b - t_out) * torch.exp(-s_out)
        return self.join_func([x_a, phi_b], dim=1), s_out.sum(dim=1, keepdim=True)


class RealNVP(nn.Module):
    r"""Extension to nn.Module which is built up of multiple `AffineLayer`s
    as per eq. (12) of https://arxiv.org/abs/1904.12072.

    Each affine layer transforms half of the input vector and the half of the
    input vector which is transformed is alternated. It is therefore recommended
    to have an even number of affine layers. Each affine layer has it's own
    pair of neural networks, s and t, which have seperate sets of parameters
    from other affine layers.

    For now each affine layer has networks with the same architecture
    for simplicity however this could be extended if required.

    Parameters
    ----------
    size_in: int
        number of units defining a single field configuration. The size of
        the second dimension of the input data.
    n_affine: int
        number of affine layers, it is recommended to choose an even number
    affine_hidden_shape: tuple
        tuple defining the number of nodes in the hidden layers of s and t.

    Attributes
    ----------
    affine_layers: torch.nn.ModuleList
        list of affine layers that form the full transformation

    """

    def __init__(
        self,
        *,
        size_in,
        n_affine: int,
        network_spec: dict,
        standardise_inputs: bool = True
    ):
        super(RealNVP, self).__init__()

        s_network_spec = network_spec["s"]
        t_network_spec = network_spec["t"]

        self.affine_layers = nn.ModuleList(
            [
                AffineLayer(
                    i, size_in, s_network_spec, t_network_spec, standardise_inputs
                )
                for i in range(n_affine)
            ]
        )

    def map(self, x_input: torch.Tensor):
        """Function that maps field configuration to simple distribution"""
        raise NotImplementedError

    def forward(self, z_input: torch.Tensor) -> torch.Tensor:
        r"""Function which maps simple distribution, z, to target distribution
        \phi, and at the same time calculates the density of the output
        distribution using the change of variables formula, according to 
        eq. (8) of https://arxiv.org/pdf/1904.12072.pdf.

        Parameters
        ----------
        z_input: torch.Tensor
            stack of simple distribution state vectors, shape (N_states, D)

        Returns
        -------
        phi_out: torch.Tensor
            stack of transformed states, which are drawn from an approximation
            of the target distribution, same shape as input.
        log_density: torch.Tensor
            logarithm of the probability density of the output distribution,
            with shape (n_states, 1)
        """
        log_density = torch.zeros((z_input.shape[0], 1))
        phi_out = z_input

        for layer in reversed(self.affine_layers):  # reverse layers!
            phi_out, log_det_jacob = layer(phi_out)
            log_density += log_det_jacob
            # TODO: make this yield, then make a yield from wrapper?

        return phi_out, log_density


def real_nvp(lattice_size, n_affine, network_spec, standardise_inputs):
    return RealNVP(
        size_in=lattice_size,
        n_affine=n_affine,
        network_spec=network_spec,
        standardise_inputs=standardise_inputs,
    )


if __name__ == "__main__":
    pass
