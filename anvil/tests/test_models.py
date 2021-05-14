"""
Test higher level model construction from :py:mod:`anvil.models`.

"""
from hypothesis import given
from hypothesis.strategies import integers, lists
import pytest
import torch

from anvil.api import API
from anvil.models import LAYER_OPTIONS

LAYERS = list(LAYER_OPTIONS.keys())

PARAMS = {
    "hidden_shape": (32,),
    "n_blocks": 3,
    "n_segments": 4,
    "lattice_length": 6,
    "lattice_dimension": 2,
    "scale": 1.0,
}

@pytest.mark.parametrize("layer_action", LAYERS)
def test_layer_actions(layer_action):
    """Call the API on each of the layer actions, using mainly default arguments
    """
    getattr(API, layer_action)(**PARAMS)
    return

# put limits on these so not to crash your computer.
@given(
    lists(integers(min_value=0, max_value=2), min_size=1, max_size=3),
    integers(min_value=1, max_value=4),
    integers(min_value=1, max_value=8),
    lists(integers(min_value=1, max_value=2 ** 6), min_size=1, max_size=3)
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
