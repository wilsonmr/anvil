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
import torch
import torch.nn as nn

from math import pi

ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    None: nn.Identity,
}


class NeuralNetwork(nn.Module):
    """Generic class for neural networks used in coupling layers.

    Networks consist of 'blocks' of
        - Dense (linear) layer
        - Batch normalisation layer
        - Activation function

    Parameters
    ----------
    size_in: int
        Number of nodes in the input layer
    size_out: int
        Number of nodes in the output layer
    hidden_shape: list
        List specifying the number of nodes in the intermediate layers
    activation: (str, None)
        Key representing the activation function used for each layer
        except the final one.
    final_activation: (str, None)
        Key representing the activation function used on the final
        layer.
    do_batch_norm: bool
        Flag dictating whether batch normalisation should be performed
        before the activation function.
    name: str
        A label for the neural network, used for diagnostics.

    Methods
    -------
    forward:
        The forward pass of the network, mapping a batch of input vectors
        with 'size_in' nodes to a batch of output vectors of 'size_out'
        nodes.
    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        hidden_shape: list = [24,],
        activation: (str, None) = "leaky_relu",
        final_activation: (str, None) = None,
        do_batch_norm: bool = False,
        name: str = "network",
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

        # nn.Sequential object containing the network layers
        self.network = self._construct_network()

    def __str__(self):
        return f"Network: {self.name}\n------------\n{self.network}"

    def _block(self, f_in, f_out, activation_func):
        """Constructs a single 'dense block' which maps 'f_in' inputs to
        'f_out' output features. Returns a list with three elements:
            - Dense (linear) layer
            - Batch normalisation (or identity if this is switched off)
            - Activation function
        """
        return [
            nn.Linear(f_in, f_out),
            self.batch_norm(f_out),
            activation_func(),
        ]

    def _construct_network(self):
        """Constructs the neural network from multiple calls to _block.
        Returns a torch.nn.Sequential object which has the 'forward' method built in.
        """
        layers = self._block(self.size_in, self.hidden_shape[0], self.activation_func)
        for f_in, f_out in zip(self.hidden_shape[:-1], self.hidden_shape[1:]):
            layers += self._block(f_in, f_out, self.activation_func)
        layers += self._block(
            self.hidden_shape[-1], self.size_out, self.final_activation_func
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        """Forward pass of the network.
        
        Takes a tensor of shape (n_batch, size_in) and returns a new tensor of
        shape (n_batch, size_out)
        """
        return self.network(x)


class AffineLayer(nn.Module):
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
        i_affine: int,
        size_in: int,
        s_network,
        t_network,
        standardise_inputs: bool,
    ):
        super(AffineLayer, self).__init__()
        size_half = size_in // 2
        self.s_network = s_network
        self.t_network = t_network

        if (i_affine % 2) == 0:  # starts at zero
            # a is first half of input vector
            self._a_ind = slice(0, size_half)
            self._b_ind = slice(size_half, size_in)
            self.join_func = torch.cat
        else:
            # a is second half of input vector
            self._a_ind = slice(int(size_half), size_in)
            self._b_ind = slice(0, int(size_half))
            self.join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )

        if standardise_inputs:
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
        x_a = x_input[..., self._a_ind]
        x_b = x_input[..., self._b_ind]
        x_a_stand = self.standardise(x_a)
        s_out = self.s_network(x_a_stand)
        t_out = self.t_network(x_a_stand)
        phi_b = (x_b - t_out) * torch.exp(-s_out)
        return (
            self.join_func([x_a, phi_b], dim=-1),
            s_out.sum(dim=tuple(range(1, len(s_out.shape)))).view(-1, 1)
        )


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
        Number of units defining a single field configuration. The size of
        the second dimension of the input data.
    n_affine: int
        Number of affine layers, it is recommended to choose an even number
    s_networks
        List of s neural networks for each affine layer
    t_networks
        List of t neural networks for each affine layer
    standardise_inputs: bool
        Flag dictating whether or not input vectors are standardised (i.e.
        zero mean, unit variance) before being passed to a neural network.

    Attributes
    ----------
    affine_layers: torch.nn.ModuleList
        list of affine layers that form the full transformation

    """

    def __init__(
        self, *, size_in: int, s_networks, t_networks, standardise_inputs: bool,
    ):
        super(RealNVP, self).__init__()

        self.affine_layers = nn.ModuleList(
            [
                AffineLayer(i, size_in, s_network, t_network, standardise_inputs)
                for i, (s_network, t_network) in enumerate(zip(s_networks, t_networks))
            ]
        )

    def map(self, phi_input: torch.Tensor):
        """Function that maps field configuration to simple distribution"""
        raise NotImplementedError

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        r"""Function which maps simple distribution, x ~ r, to target distribution
        \phi ~ p, and at the same time calculates the density of the output
        distribution using the change of variables formula, according to
        eq. (8) of https://arxiv.org/pdf/1904.12072.pdf.

        Parameters
        ----------
        x_input: torch.Tensor
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
        phi_out = x_input
        rev_layers = reversed(self.affine_layers) # reverse layers!
        phi_out, log_density = next(rev_layers)(phi_out)
        for layer in rev_layers:
            phi_out, log_det_jacob = layer(phi_out)
            log_density += log_det_jacob
        return phi_out, log_density


class ProjectCircle(nn.Module):
    def __init__(self, inner_flow):
        super().__init__()
        self.inner_flow = inner_flow

    def forward(self, z_input: torch.Tensor) -> torch.Tensor:

        # Projection
        phi_out = torch.tan(0.5 * (z_input - pi))
        log_density_proj = (-torch.log1p(phi_out ** 2)).sum(dim=1, keepdim=True)

        # Inner flow on real line e.g. RealNVP
        phi_out, log_density_inner = self.inner_flow(phi_out)

        # Inverse projection
        phi_out = 2 * torch.atan(phi_out) + pi
        log_density_proj += (-2 * torch.log(torch.cos(0.5 * (phi_out - pi)))).sum(
            dim=1, keepdim=True
        )

        return phi_out, log_density_proj + log_density_inner


class ProjectSphere(nn.Module):
    def __init__(self, inner_flow, size_in):
        super().__init__()
        self.inner_flow = inner_flow
        self.size_in = size_in

    def forward(self, z_input: torch.Tensor) -> torch.Tensor:
        polar, azimuth = z_input.view(-1, self.size_in // 2, 2).split(1, dim=2)

        # Projection
        # -1 factor because coordinate = azimuth - pi (*-1 is faster than shift)
        x_coords = -torch.tan(0.5 * polar) * torch.cat(  # radial coordinate
            (torch.cos(azimuth), torch.sin(azimuth)), dim=2
        )
        rad_sq = x_coords.pow(2).sum(dim=2)
        log_density_proj = (-0.5 * torch.log(rad_sq) - torch.log1p(rad_sq)).sum(
            dim=1, keepdim=True
        )

        # Inner flow on real plane e.g. RealNVP
        x_coords, log_density_inner = self.inner_flow(x_coords.view(-1, self.size_in))
        x_1, x_2 = x_coords.view(-1, self.size_in // 2, 2).split(1, dim=2)

        # Inverse projection
        polar = 2 * torch.atan(torch.sqrt(x_1.pow(2) + x_2.pow(2)))
        phi_out = torch.cat((polar, torch.atan2(x_2, x_1) + pi,), dim=2)  # azimuth
        log_density_proj += (
            torch.log(torch.sin(0.5 * polar)) - 3 * torch.log(torch.cos(0.5 * polar))
        ).sum(dim=1)

        return phi_out.view(-1, self.size_in), log_density_proj + log_density_inner


def real_nvp(config_size, s_networks, t_networks, standardise_inputs=False):
    """Returns an instance of the RealNVP class."""
    return RealNVP(
        size_in=config_size,
        s_networks=s_networks,
        t_networks=t_networks,
        standardise_inputs=standardise_inputs,
    )


def stereographic_projection(
    target, config_size, s_networks, t_networks, standardise_inputs=False
):
    """Returns an instance of either ProjectCircle or ProjectSphere, depending on the
    dimensionality of the fields."""
    inner_flow = RealNVP(
        size_in=config_size,
        s_networks=s_networks,
        t_networks=t_networks,
        standardise_inputs=standardise_inputs,
    )
    if target == "o2":
        return ProjectCircle(inner_flow)
    elif target == "o3":
        return ProjectSphere(inner_flow, size_in=config_size)
    # Should raise config error.
    return


MODEL_OPTIONS = {
    "real_nvp": real_nvp,
    "projection": stereographic_projection,
}
