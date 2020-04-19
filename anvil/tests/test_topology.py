import torch
from math import pi, sqrt
from itertools import product
from random import randint

from anvil.geometry import Geometry2D
from anvil.fields import HeisenbergField

L = 4  # must be 4 or greater
TESTING_GEOMETRY = Geometry2D(L)
SPLITCART = TESTING_GEOMETRY._splitcart()


def add_minimal_hedgehog(state, loc: tuple, positive: bool = True):
    """Takes a single configuration in the cartesian representation and adds
    a minimal hedgehog Skyrmion centred on "loc".

    Parameters
    ----------
    state: torch.Tensor
        The state to which the hedgehog skyrmion is to be added. Should have
        shape (L, L, 2), with the two components in the last dimension being
        the polar and azimuthal angles, in that order.
    loc: tuple
        The coordinates for the centre of the skyrmion.
    positive: bool
        True means that the added skyrmion will have a topological charge of
        +1 if placed in a background configuration with all polar angles
        being pi and azimuthal angles being 0. False is the reverse case: a
        charge of -1 in a background configuration with all angles being 0.

    Returns
    -------
    state: torch.Tensor
        The input state with the 9 elements centered on "loc" changed to
        a minimal hedgehog skyrmion configuration.

    Notes
    -----
    Totally overkill for these tests. More useful if we wanted to run tests
    with multiple skyrmions, but I can't yet see a need.
    """
    x0, y0 = loc
    if positive:
        sign = 1
    else:
        sign = -1

    polar = torch.zeros((3, 3))
    azimuth = torch.zeros((3, 3))

    polar[1, 1] = pi * (1 - sign) / 2
    polar[0, 1] = polar[1, 0] = polar[2, 1] = polar[1, 2] = pi / 2
    polar[0, 0] = polar[0, 2] = polar[2, 0] = polar[2, 2] = pi * (
        (1 - sign) / 2 + sign / sqrt(2)
    )

    azimuth[0, 2] = pi / 4
    azimuth[0, 1] = pi / 2
    azimuth[0, 0] = 3 * pi / 4
    azimuth[1, 0] = pi
    azimuth[2, 0] = 5 * pi / 4
    azimuth[2, 1] = 3 * pi / 2
    azimuth[2, 2] = 7 * pi / 4

    # Ensures this works even when loc is on a boundary
    state = state.roll((1 - x0, 1 - y0), (0, 1))
    state[:3, :3, 0] = polar
    state[:3, :3, 1] = azimuth
    state = state.roll((x0 - 1, y0 - 1), (0, 1))

    return state


def cart_to_split(state):
    """Takes a tensor in cartesian representation and returns the 
    same elements in the split representation.

    The input tensor should have shape (L, L, 2).
    """
    state_split = torch.empty((L ** 2, 2))
    for i, j in product(range(L), range(L)):
        state_split[SPLITCART[i, j], :] = state[i, j, :]
    return state_split


def test_uncharged():
    state = torch.rand((1, 2 * L ** 2))
    field = HeisenbergField(state, TESTING_GEOMETRY)
    assert abs(float(field._topological_charge)) < 1e-7


def test_minimal_hedgehog_pos():
    state = torch.stack((torch.ones((L, L)) * pi, torch.zeros((L, L))), dim=-1)
    state = add_minimal_hedgehog(
        state, (randint(0, L - 1), randint(0, L - 1)), positive=True
    )

    state_split = cart_to_split(state).view(1, -1)

    field = HeisenbergField(state_split, TESTING_GEOMETRY)
    assert abs(float(field._topological_charge) - 1) < 1e-7


def test_minimal_hedgehog_neg():
    state = torch.zeros((L, L, 2))
    state = add_minimal_hedgehog(
        state, (randint(0, L - 1), randint(0, L - 1)), positive=False
    )

    state_split = cart_to_split(state).view(1, -1)

    field = HeisenbergField(state_split, TESTING_GEOMETRY)
    assert abs(float(field._topological_charge) + 1) < 1e-7
