import torch
from math import pi

from anvil.geometry import Geometry2D
from anvil.fields import ClassicalSpinField

L = 2
TESTING_GEOMETRY = Geometry2D(L)


def assert_unit_norm(field):
    assert torch.allclose(torch.abs(field.pow(2).sum(dim=-1) - 1), torch.zeros(L ** 2), atol=1e-5)

def test_rotor_unit_norm():
    input_state = torch.rand(1, L ** 2) * 2 * pi
    rotors = ClassicalSpinField(input_state, TESTING_GEOMETRY).spins
    assert_unit_norm(rotors)

def test_spin_unit_norm():
    input_state = torch.rand(1, 2 * L**2) * pi
    input_state[1::2] *= 2  # azimuthal angle [0, 2pi)
    spins = ClassicalSpinField(input_state, TESTING_GEOMETRY).spins
    assert_unit_norm(spins)
