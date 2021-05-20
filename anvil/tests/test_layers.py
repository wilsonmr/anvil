"""
Tests of the base classes in :py:mod:`anvil.layers`

"""
from hypothesis import given
from hypothesis.strategies import integers, booleans
import numpy as np
import pytest
import torch

import anvil.layers as layers
from anvil.distributions import Gaussian

N_BATCH = 100
SIZE = 36
SIZE_HALF = SIZE // 2
HIDDEN_SHAPE = [36]
ACTIVATION = "tanh"

@given(integers(min_value=0, max_value=2**16), booleans())
def test_coupling_init(size_half, even_sites):
    """Hypothesis test the initialisation of the base class in layers"""
    layers.CouplingLayer(size_half, even_sites)


def test_additive_layers():
    equivar_additive = layers.AdditiveLayer(
        size_half=SIZE_HALF,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        z2_equivar=True,
        even_sites=True
    )
    input_tensor = torch.zeros((N_BATCH, SIZE))
    with torch.no_grad():
        output_tensor, output_density = equivar_additive(input_tensor, 0)

    assert output_density == 0
    np.testing.assert_allclose(input_tensor.numpy(), output_tensor.numpy())


def basic_layer_test(layer, input_states, input_log_density, *args):
    """Basic check that layer transforms input states properly.

    In practice we check:

        - field variables and log densities are valid real numbers
        - output states are correct shape
        - outputs are correct typing

    """
    output_states, output_log_density = layer(input_states, input_log_density, *args)
    # all numbers
    any_nan = (
        torch.any(torch.isnan(output_states)) or
        torch.any(torch.isnan(output_log_density))
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
    latent_distribution = Gaussian(SIZE) # use default standard normal
    return latent_distribution(N_BATCH)

@pytest.mark.parametrize("layer_class", [layers.AdditiveLayer, layers.AffineLayer])
@pytest.mark.parametrize("z2_equivar", [True, False])
@pytest.mark.parametrize("even_sites", [True, False])
@torch.no_grad()
def test_affine_like_basic(gaussian_input, layer_class, z2_equivar, even_sites):
    """Apply :py:func:`basic_layer_test` to layers with same initialisation
    parameters as :py:class:`anvil.layers.AffineLayer`.

    """
    layer = layer_class(
        size_half=SIZE_HALF,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        z2_equivar=z2_equivar,
        even_sites=even_sites,
    )
    basic_layer_test(layer, *gaussian_input)

@pytest.mark.parametrize("z2_equivar", [True, False])
@pytest.mark.parametrize("even_sites", [True, False])
@torch.no_grad()
def test_rqs_basic(gaussian_input, z2_equivar, even_sites):
    """Apply :py:func:`basic_layer_test` to
    :py:class:`anvil.layers.RationalQuadraticSplineLayer`.
    """
    layer = layers.RationalQuadraticSplineLayer(
        size_half=SIZE_HALF,
        interval=5,
        n_segments=4,
        hidden_shape=HIDDEN_SHAPE,
        activation=ACTIVATION,
        z2_equivar=z2_equivar,
        even_sites=even_sites,
    )
    negative_mag = gaussian_input[0].sum(dim=1) < 0
    basic_layer_test(layer, *gaussian_input, negative_mag)

@pytest.mark.parametrize(
    "layer_class",
    [layers.GlobalRescaling, layers.BatchNormLayer, layers.GlobalAffineLayer]
)
@torch.no_grad()
def test_scaling_layer_basic(gaussian_input, layer_class):
    if layer_class is layers.GlobalAffineLayer:
        layer = layer_class(1, 0)
    elif layer_class is layers.GlobalRescaling:
        layer = layer_class(scale=1.0, learnable=False)
    else:
        layer = layer_class()
    basic_layer_test(layer, *gaussian_input)

@torch.no_grad()
def test_sequential_basic(gaussian_input):
    inner_layers = [
        layers.AffineLayer(
            size_half=SIZE_HALF,
            hidden_shape=HIDDEN_SHAPE,
            activation=ACTIVATION,
            z2_equivar=False,
            even_sites=bool(i % 2),
        ) for i in range(8)]
    layer = layers.Sequential(*inner_layers)
    basic_layer_test(layer, *gaussian_input)

    # check application of sequetion matches output of applying each layer.
    output_states, output_density = inner_layers[0](*gaussian_input)
    for el in inner_layers[1:]:
        output_states, output_density = el(output_states, output_density)

    seq_output_states, seq_output_density = layer(*gaussian_input)

    np.testing.assert_allclose(seq_output_states.numpy(), output_states.numpy())
    np.testing.assert_allclose(seq_output_density.numpy(), output_density.numpy())
