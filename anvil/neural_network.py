# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
neural_network.py

Module containing neural networks which are used as part of transformation
layers, found in :py:mod:`anvil.layers`.

"""
import torch
import torch.nn as nn


ACTIVATION_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    None: nn.Identity,
}


class DenseNeuralNetwork(nn.Module):
    """Dense neural networks used in coupling layers.

    Parameters
    ----------
    size_in
        Number of nodes in the input layer
    size_out
        Number of nodes in the output layer
    hidden_shape
        Tuple specifying the number of nodes in the intermediate layers
    activation
        Key representing the activation function used for each layer
        except the final one. Valid options can be found in
        ``ACTIVATION_LAYERS``.
    bias
        Whether to use biases in networks.

    Attributes
    ----------
    network : torch.nn.Sequential
        A PyTorch Module object containing the layers of the neural network.

    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        hidden_shape: (tuple, list),
        activation: str,
        bias: bool = True,
    ):
        super().__init__()
        network_shape = [size_in, *hidden_shape, size_out]

        activation = ACTIVATION_LAYERS[activation]
        activations = [activation for _ in hidden_shape]
        activations.append(nn.Identity)  # don't pass final output through activation

        # Construct network
        layers = []
        for f_in, f_out, act in zip(network_shape[:-1], network_shape[1:], activations):
            layers.append(nn.Linear(f_in, f_out, bias=bias))
            layers.append(act())

        self.network = nn.Sequential(*layers)

    def forward(self, v_in: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Takes a tensor of shape ``(batch_size, size_in)`` and returns a new tensor of
        shape ``(batch_size, size_out)`` .
        """
        return self.network(v_in)
