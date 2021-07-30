from copy import deepcopy

import pytest
import torch
import torch.optim

from anvil.api import API
from anvil.distributions import Gaussian, FreeScalar
from anvil.geometry import Geometry2D
import anvil.train

L = 8
PARAMS = {"lattice_length": L, "lattice_dimension": 2}
GEOMETRY = Geometry2D(L)
GAUSSIAN = Gaussian(GEOMETRY.volume)
FREE_SCALAR = FreeScalar(GEOMETRY)
ETA = 0.01  # learning rate for optimizer


def train(base, target, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=ETA)
    for _ in range(1000):
        z, base_log_density = base(1000)
        phi, model_log_density = model(z, base_log_density)
        target_log_density = target.log_density(phi)
        kl = anvil.train.reverse_kl(model_log_density, target_log_density)

        optimizer.zero_grad()
        kl.backward()
        optimizer.step()


@pytest.mark.parametrize("dist", [GAUSSIAN, FREE_SCALAR])
def test_rescaling(dist):
    base = dist
    target = deepcopy(base)

    model_spec = (
        {
            "layer": "global_rescaling",
            "scale": 1,
            "learnable": True,
        },
    )
    model = API.explicit_model(model=model_spec, **PARAMS)

    train(base, target, model)

    scale = next(iter(model.parameters()))

    assert abs(scale - 1) < 5 * ETA


@pytest.mark.parametrize("dist", [GAUSSIAN, FREE_SCALAR])
def test_zero_kl(dist):
    base = dist
    target = deepcopy(base)

    phi, base_log_density = base(1000)
    kl = anvil.train.reverse_kl(base_log_density, target.log_density(phi))

    assert kl == 0

def test_preprocessing():
    base = GAUSSIAN
    target = FREE_SCALAR
    model_spec = (
            {"layer": "inverse_fourier"},
            {"layer": "global_rescaling", "scale": 1, "learnable": True}
    )
    model = API.explicit_model(model=model_spec, **PARAMS)

    train(base, target, model)

    scale = next(iter(model.parameters()))

    assert abs(scale - 1) < 5 * ETA
