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
    "none": nn.Identity,
}


class DenseNeuralNetwork(nn.Module):
    """Dense neural networks used in coupling layers.

    Parameters
    ----------
    size_in
        Number of nodes in the input layer
    hidden_shape
        Tuple specifying the number of nodes in the intermediate layers
    activation
        Key representing the activation function used for each layer
        except the final one. Valid options can be found in
        ``ACTIVATION_LAYERS``.
    final_activation
        Activation function for the output layer of the network. None
        by default.
    use_bias
        Whether to use biases in networks.
    out_channels
        Number of output channels. The output will have dimensions
        ``(n_batch, out_channels, size_out // out_channels)``.
    size_out
        Number of nodes in the output layer. If not provided, defaults
        to ``size_in * out_channels``.

    Attributes
    ----------
    network : torch.nn.Sequential
        A PyTorch Module object containing the layers of the neural network.

    """

    def __init__(
        self,
        size_in: int,
        hidden_shape: (tuple, list),
        activation: str,
        final_activation: str = "none",
        use_bias: bool = True,
        out_channels: int = 1,
        size_out: int = None,
    ):
        super().__init__()
        if size_out is None:
            size_out = size_in * out_channels
        assert (
            size_out % out_channels == 0
        ), f"incompatible values given for size_out: {size_out} and out_channels {out_channels}"

        network_shape = [size_in, *hidden_shape, size_out]

        activation = ACTIVATION_LAYERS[activation]
        final_activation = ACTIVATION_LAYERS[final_activation]
        activations = [activation for _ in hidden_shape]
        activations.append(final_activation)

        # Construct network
        layers = []
        for f_in, f_out, act in zip(network_shape[:-1], network_shape[1:], activations):
            layers.append(nn.Linear(f_in, f_out, bias=use_bias))
            layers.append(act())

        self.network = nn.Sequential(*layers)
        self.shape_out = (out_channels, size_out // out_channels)

    def forward(self, v_in: torch.Tensor) -> torch.Tensor:
        """Takes an input tensor with dims ``(n_batch, size_in)`` and returns the
        output of the network with dims  ``(n_batch, out_channels, size_out // out_channels)``.
        """
        return self.network(v_in).view(-1, *self.shape_out)


class ConvolutionalNeuralNetwork(nn.Module):
    """Convolutional neural networks used in coupling layers.

    Parameters
    ----------
    out_channels
        Number of output channels.
    hidden_shape
        Tuple specifying number of channels in the intermdiate layers.
    activation
        Key representing the activation function used for each layer
        except the final one. Valid options can be found in
        ``ACTIVATION_LAYERS``.
    final_activation
        Activation function for the output layer of the network. None
        by default.
    use_bias
        Whether to use biases in networks.
    kernel_size
        Side length of (square) convolutional kernel.

    Attributes
    ----------
    network : torch.nn.Sequential
        A PyTorch Module object containing the layers of the neural network.

    """

    def __init__(
        self,
        out_channels: int,
        hidden_shape: (tuple, list),
        activation: str,
        final_activation: str = "none",
        use_bias: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()
        network_shape = [1, *hidden_shape, out_channels]

        assert kernel_size % 2 == 1, "kernel size must be odd"
        padding_size = kernel_size // 2

        activation = ACTIVATION_LAYERS[activation]
        final_activation = ACTIVATION_LAYERS[final_activation]
        activations = [activation for _ in hidden_shape]
        activations.append(final_activation)

        # Construct network
        layers = []
        for f_in, f_out, act in zip(network_shape[:-1], network_shape[1:], activations):
            layers.append(
                nn.Conv2d(
                    f_in,
                    f_out,
                    kernel_size,
                    padding=padding_size,
                    bias=use_bias,
                    stride=1,
                    padding_mode="circular",
                )
            )
            layers.append(act())

        self.network = nn.Sequential(*layers)

    def forward(self, v_in: torch.Tensor) -> torch.Tensor:
        """Takes an input tensor of dims ``(n_batch, L_x, L_y)`` and returns the
        output of the network with dims ``(n_batch, out_channels, L_x, L_y)``"""
        return self.network(v_in.unsqueeze(dim=1))
