#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
test_geometry.py

Unit test suite for geometry.py

"""
import torch

from anvil.geometry import Geometry2D

L = 2  # very small system

TESTING_GEOMETRY = Geometry2D(L)


def test_checkerboard():
    # TODO: update once mask is flexible
    assert torch.equal(
        TESTING_GEOMETRY.checkerboard, torch.tensor([[1, 0], [0, 1]], dtype=bool)
    )


def test_indexing():
    """Tests that the indexing example of get_shift is reproduced."""
    state_2d = torch.arange(4).view(2, 2)
    phi = state_2d.flatten()
    shift = TESTING_GEOMETRY.get_shift()
    assert torch.allclose(phi[shift], torch.tensor([[2, 3, 0, 1], [1, 0, 3, 2]]))
    # multiple simultaneous shifts
    shift = TESTING_GEOMETRY.get_shift(shifts=((1, 1),), dims=((0, 1),))
    assert torch.allclose(phi[shift], torch.tensor([[3, 2, 1, 0]]))
