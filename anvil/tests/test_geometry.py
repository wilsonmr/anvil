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


def test_splitcart():
    assert torch.allclose(
        TESTING_GEOMETRY.splitcart, torch.tensor([[0, 2], [3, 1]])
    )


def test_splitlexi():
    assert torch.allclose(TESTING_GEOMETRY.lexisplit, torch.tensor([[0, 3, 1, 2]]))


def test_indexing():
    """Tests that the indexing example of get_shift is reproduced.
    """
    phi = torch.tensor([0, 3, 1, 2])
    shift = TESTING_GEOMETRY.get_shift()
    assert torch.allclose(phi[shift], torch.tensor([[2, 1, 3, 0], [1, 2, 0, 3]]))
    # multiple simultaneous shifts
    shift = TESTING_GEOMETRY.get_shift(shifts=((1, 1),), dims=((0, 1),))
    assert torch.allclose(phi[shift], torch.tensor([[3, 0, 2, 1]]))
