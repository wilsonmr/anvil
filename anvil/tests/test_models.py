"""
Test higher level model construction from :py:mod:`anvil.models`.

"""
from hypothesis import given
from hypothesis.strategies import integers, lists
import pytest
import torch
from copy import deepcopy

from anvil.api import API
from anvil.models import LAYER_OPTIONS

LAYERS = list(LAYER_OPTIONS.keys())

PARAMS = {
    "hidden_shape": (32,),
    "n_blocks": 2,
    "n_segments": 4,
    "lattice_length": 6,
    "lattice_dimension": 2,
    "scale": 1.0,
}


@pytest.mark.parametrize("layer_action", LAYERS)
def test_layer_actions(layer_action):
    """Call the API on each of the layer actions, using mainly default arguments"""
    getattr(API, layer_action)(**PARAMS)
    return


# put limits on these so not to crash your computer.
@given(
    lists(integers(min_value=0, max_value=2), min_size=1, max_size=3),
    integers(min_value=1, max_value=4),
    integers(min_value=1, max_value=8),
    lists(integers(min_value=1, max_value=2 ** 6), min_size=1, max_size=3),
)
def test_model_construction(layer_idx, n_blocks, lattice_length_half, hidden_shape):
    """Hypothesis test the model construction"""
    # require even lattice sites.
    model = [{"layer": LAYERS[idx]} for idx in layer_idx]
    lattice_length = 2 * lattice_length_half
    params = {
        "model": model,
        "n_blocks": n_blocks,
        "lattice_length": lattice_length,
        "hidden_shape": hidden_shape,
        "lattice_dimension": 2,
        # for some reason the RQS defaults get missed?
        "interval": 5,
        "n_segments": 4,
    }
    # might help with memory.
    with torch.no_grad():
        API.model_to_load(**params)


def layer_independence_test(model_spec):
    """Check that each layer's parameters are updated independently."""

    # Collect over these layers
    model = API.model_to_load(**model_spec)
    layer1, layer2 = [layer for layer in model]

    layer2_copy = deepcopy(layer2)

    # Update parameters in first layer
    valid_key, valid_tensor = next(iter(layer1.state_dict().items()))
    update = {valid_key: torch.rand_like(valid_tensor)}
    layer1.load_state_dict(update, strict=False)

    # Check that second layer is unchanged
    # NOTE: may be safer to iterate over shared keys
    for original, copy in zip(layer2.parameters(), layer2_copy.parameters()):
        assert torch.allclose(original, copy)


# TODO: extend to other layers... @pytest.mark.parametrize("layer_action", LAYERS)
@torch.no_grad()
def test_layer_independence_global_rescaling():
    # Build a model with two identical sets of layers
    working_example = {  # This is OK
        "model": [
            {"layer": "global_rescaling", "scale": 1.0},
            {"layer": "global_rescaling", "scale": 1.0},
        ]
    }
    layer_independence_test(working_example)

    breaking_example = {  # This is NOT ok!
        "model": [
            {"layer": "global_rescaling"},
            {"layer": "global_rescaling"},
        ],
        "scale": 1.0,
    }
    layer_independence_test(breaking_example)
