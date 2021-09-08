"""
Tests of the classes in :py:mod:`anvil.neural_networks`

"""
import torch

from anvil.neural_network import DenseNeuralNetwork, ConvolutionalNeuralNetwork


def test_dense_net_dims():
    net = DenseNeuralNetwork(
        size_in=18,
        hidden_shape=(36, 36),
        activation="tanh",
        out_channels=4,
    )
    with torch.no_grad():
        v_in = torch.rand(100, 18)
        v_out = net(v_in)
        assert tuple(v_out.shape) == (100, 4, 18)


def test_conv_net_dims():
    net = ConvolutionalNeuralNetwork(
        out_channels=4,
        hidden_shape=(6, 6),
        activation="tanh",
    )
    with torch.no_grad():
        v_in = torch.rand(100, 6, 6)
        v_out = net(v_in)
        assert tuple(v_out.shape) == (100, 4, 6, 6)


def test_dense_net_zeros():
    net = DenseNeuralNetwork(
        size_in=18,
        hidden_shape=(36, 36),
        activation="tanh",
        out_channels=1,
        use_bias=False,
    )
    with torch.no_grad():
        v_in = torch.zeros(10, 18)
        v_out = net(v_in)
        assert torch.allclose(v_in, v_out)


def test_conv_net_zeros():
    net = ConvolutionalNeuralNetwork(
        out_channels=1,
        hidden_shape=(6, 6),
        activation="tanh",
        use_bias=False,
    )
    with torch.no_grad():
        v_in = torch.zeros(10, 6, 6)
        v_out = net(v_in).squeeze()
        assert torch.allclose(v_in, v_out)


def test_conv_net_symm():
    net = ConvolutionalNeuralNetwork(
        out_channels=1,
        hidden_shape=(6, 6),
        activation="tanh",
        use_bias=False,
    )
    with torch.no_grad():
        v_in = torch.ones(10, 6, 6)
        v_out = net(v_in).squeeze()
        f = v_out.flatten()[0]
        assert torch.allclose(v_in, v_out / f)


def test_dense_net_numel():
    net_shape = [10, 100, 100, 10]
    net = DenseNeuralNetwork(
        size_in=net_shape[0],
        size_out=net_shape[-1],
        hidden_shape=net_shape[1:-1],
        activation="tanh",
        use_bias=True,
    )
    expec = sum([a * b + b for a, b in zip(net_shape[:-1], net_shape[1:])])
    got = sum([tensor.numel() for tensor in net.parameters()])
    assert expec == got


def test_conv_net_numel():
    channels = [1, 10, 10, 2]
    k = 3
    net = ConvolutionalNeuralNetwork(
        out_channels=channels[-1],
        hidden_shape=channels[1:-1],
        activation="tanh",
        use_bias=True,
        kernel_size=k,
    )
    expec = sum([k ** 2 * a * b + b for a, b in zip(channels[:-1], channels[1:])])
    got = sum([tensor.numel() for tensor in net.parameters()])
    assert expec == got
