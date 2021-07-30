import torch

from anvil.api import API
from anvil.distributions import Gaussian, FreeScalar
from anvil.geometry import Geometry2D
from anvil.models import inverse_fourier


def test_change_of_variables():
    L = 4
    geometry = Geometry2D(L)
    gaussian = Gaussian(L ** 2)
    free_scalar = FreeScalar(geometry, m_sq=1)

    model = API.explicit_model(model={"layer": "inverse_fourier"}, lattice_length=L)

    # Transform a sample using the inverse FT
    z, base_log_density = gaussian(1000)
    phi, model_log_density = model(z, base_log_density)

    # Compare with action of free scalar
    target_log_density = free_scalar.log_density(phi)

    # There will be an offset since target_log_density is un-normalized
    offset = model_log_density[0] - target_log_density[0]

    assert torch.allclose(model_log_density - offset, target_log_density)
