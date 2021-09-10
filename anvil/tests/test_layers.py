"""
Tests of the base classes in :py:mod:`anvil.layers`

"""
import numpy as np
import pytest
import torch

from anvil.geometry import Geometry2D
import anvil.layers as layers
from anvil.distributions import Gaussian

N_BATCH = 100
LENGTH = 6
SIZE = LENGTH ** 2
HIDDEN_SHAPE = [18]
ACTIVATION = "tanh"

MASK = Geometry2D(LENGTH).checkerboard


def basic_layer_test(layer, input_states, input_log_density, *args):
    """Basic check that layer transforms input states properly.

    In practice we check:

        - field variables and log densities are valid real numbers
        - output states are correct shape
        - outputs are correct typing

    """
    output_states, output_log_density = layer(input_states, input_log_density, *args)
    # all numbers
    any_nan = torch.any(torch.isnan(output_states)) or torch.any(
        torch.isnan(output_log_density)
    )
    assert not any_nan
    # correct shape
    assert input_states.shape == output_states.shape

    assert isinstance(output_states, torch.Tensor)
    assert isinstance(output_log_density, torch.Tensor)


@pytest.fixture()
@torch.no_grad()
def gaussian_input():
    """Basic input states for testing"""
    latent_distribution = Gaussian(SIZE)  # use default standard normal
    return latent_distribution(N_BATCH)


@pytest.mark.parametrize("layer_class", [layers.AdditiveLayer, layers.AffineLayer])
@pytest.mark.parametrize("z2_equivar", [True, False])
@pytest.mark.parametrize("use_convnet", [True, False])
@torch.no_grad()
def test_basic(gaussian_input, layer_class, z2_equivar, use_convnet):
    """Apply :py:func:`basic_layer_test` to
    :py:class:`anvil.layers.AffineLayer` and
    :py:class:`anvil.layers.AdditiveLayer`.
    """
    layer = layer_class(
        mask=MASK,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        final_activation=ACTIVATION,
        z2_equivar=z2_equivar,
        use_convnet=use_convnet,
    )
    basic_layer_test(layer, *gaussian_input)


@pytest.mark.parametrize("z2_equivar", [True, False])
def test_legacy_affine_basic(gaussian_input, z2_equivar):
    """Apply :py:func:`basic_layer_test` to
    :py:class:`anvil.layers.LegacyAffineLayer`.
    """
    layer = layers.LegacyAffineLayer(
        mask=MASK,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        final_activation=ACTIVATION,
        z2_equivar=z2_equivar,
    )
    basic_layer_test(layer, *gaussian_input)


@pytest.mark.parametrize("use_convnet", [True, False])
def test_rqs_basic(gaussian_input, use_convnet):
    """Apply :py:func:`basic_layer_test` to
    :py:class:`anvil.layers.RationalQuadraticSplineLayer`.
    """
    layer = layers.RationalQuadraticSplineLayer(
        mask=MASK,
        interval=5,
        n_segments=4,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        final_activation=ACTIVATION,
        use_convnet=use_convnet,
    )
    basic_layer_test(layer, *gaussian_input)


def test_legacy_equivariant_rqs_basic(gaussian_input):
    """Apply :py:func:`basic_layer_test` to
    :py:class:`anvil.layers.LegacyEquivariantSplineLayer`.
    """
    layer = layers.LegacyEquivariantSplineLayer(
        mask=MASK,
        interval=5,
        n_segments=4,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        final_activation=ACTIVATION,
    )
    basic_layer_test(layer, *gaussian_input)



@pytest.mark.parametrize(
    "layer_class",
    [layers.GlobalRescaling, layers.BatchNormLayer, layers.GlobalAffineLayer],
)
def test_scaling_layer_basic(gaussian_input, layer_class):
    """Apply :py:func:`basic_layer_test` to
    :py:class:`anvil.layers.GlobalRescaling`,
    :py:class:`anvil.layers.BatchNormLayer`,
    :py:class:`anvil.layers.GlobalAffineLayer`,
    """
    if layer_class is layers.GlobalAffineLayer:
        layer = layer_class(1, 0)
    elif layer_class is layers.GlobalRescaling:
        layer = layer_class(scale=1.0, learnable=False)
    else:
        layer = layer_class()
    basic_layer_test(layer, *gaussian_input)


@torch.no_grad()
def test_sequential_basic(gaussian_input):
    """Apply :py:func:`basic_layer_test` to
    :py:clas:`anvil.layers.Sequential`
    """
    inner_layers = [
        layers.AffineLayer(
            mask=MASK,
            hidden_shape=HIDDEN_SHAPE,
            activation=ACTIVATION,
            final_activation=ACTIVATION,
            z2_equivar=False,
            use_convnet=False,
        )
        for i in range(4)
    ]
    layer = layers.Sequential(*inner_layers)
    basic_layer_test(layer, *gaussian_input)

    # check application of sequetion matches output of applying each layer.
    output_states, output_density = inner_layers[0](*gaussian_input)
    for el in inner_layers[1:]:
        output_states, output_density = el(output_states, output_density)

    seq_output_states, seq_output_density = layer(*gaussian_input)

    assert torch.allclose(seq_output_states, output_states)
    assert torch.allclose(seq_output_density, output_density)


@pytest.mark.parametrize("Layer", (layers.AdditiveLayer, layers.AffineLayer))
@pytest.mark.parametrize("use_convnet", (True, False))
def test_equivariant_layers_input_zeros(Layer, use_convnet):
    """Given zeroes as input to the equivariant affine and additive layers, the
    output should also be zeroes and the density should be unchanged."""
    model = Layer(
        mask=MASK,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        final_activation=ACTIVATION,
        z2_equivar=True,  # no biases implies input == output
        use_convnet=use_convnet,
    )
    input_tensor = torch.zeros((N_BATCH, SIZE))
    input_density = torch.rand((N_BATCH, 1))
    with torch.no_grad():
        output_tensor, output_density = model(input_tensor, input_density)

    assert torch.allclose(input_density, output_density)
    assert torch.allclose(input_tensor, output_tensor)
