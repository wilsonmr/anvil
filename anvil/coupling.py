# -*- coding: utf-8 -*-
r"""
coupling.py
"""
import torch
import torch.nn as nn

from math import pi

from reportengine import collect

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
        hidden_shape: list,
        activation: (str, None),
        final_activation: (str, None) = None,
        do_batch_norm: bool = False,
        label: str = "network",
    ):
        super(NeuralNetwork, self).__init__()
        self.label = label
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
        return f"Network: {self.label}\n------------\n{self.network}"

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


class CouplingLayer(nn.Module):
    """
    Base class for coupling layers
    """

    def __init__(self, i_couple: int, size_in: int):
        super().__init__()
        self.size_in = size_in
        self.size_half = size_in // 2

        if (i_couple % 2) == 0:  # starts at zero
            # a is first half of input vector
            self._a_ind = slice(0, self.size_half)
            self._b_ind = slice(self.size_half, self.size_in)
            self.join_func = torch.cat
        else:
            # a is second half of input vector
            self._a_ind = slice(int(self.size_half), self.size_in)
            self._b_ind = slice(0, int(self.size_half))
            self.join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )


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
        i_couple: int,
        size_in: int,
        hidden_shape: list = [24,],
        activation_func: str = "leaky_relu",
        batch_normalise: bool = False,
        *,
        t_hidden_shape: list = None,  # optionally different from s
    ):
        super().__init__(i_couple, size_in)

        if t_hidden_shape is None:
            t_hidden_shape = hidden_shape

        # Construct networks
        self.s_network = NeuralNetwork(
            size_in=self.size_half,
            size_out=self.size_half,
            hidden_shape=hidden_shape,
            activation=activation_func,
            final_activation=activation_func,
            do_batch_norm=batch_normalise,
            label="Affine layer {i_couple}: s network",
        )
        self.t_network = NeuralNetwork(
            size_in=self.size_half,
            size_out=self.size_half,
            hidden_shape=t_hidden_shape,
            activation=activation_func,
            final_activation=None,
            do_batch_norm=batch_normalise,
            label="Affine layer {i_couple}: t network",
        )

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
        s_out = self.s_network(x_a)
        t_out = self.t_network(x_a)
        phi_b = (x_b - t_out) * torch.exp(-s_out)
        return (
            self.join_func([x_a, phi_b], dim=-1),
            s_out.sum(dim=tuple(range(1, len(s_out.shape)))).view(-1, 1),
        )


LAYER_OPTIONS = {"affine": AffineLayer}


def coupling_layer(
    config_size, layer_type, hidden_shape, activation, batch_normalise, extra_args={}
):
    layer_class = LAYER_OPTIONS[layer_type]

    red_layer = layer_class(
        0, config_size, hidden_shape, activation, batch_normalise, **extra_args,
    )
    black_layer = layer_class(
        1, config_size, hidden_shape, activation, batch_normalise, **extra_args,
    )
    return [red_layer, black_layer]


coupling_layers = collect("coupling_layer", ("layer_spec",))
