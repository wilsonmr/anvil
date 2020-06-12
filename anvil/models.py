"""
models.py

Module containing reportengine actions which return callable objects that execute
normalising flows constructed from multiple layers via function composition.
"""
from functools import partial
import torch
import torch.nn as nn

from anvil.core import Sequential, RedBlackLayers
import anvil.layers as layers


def real_nvp(
    n_affine,
    n_lattice,
    n_components,
    hidden_shape=[24,],
    activation="leaky_relu",
    s_final_activation="leaky_relu",
    batch_normalise=False,
):
    """Action that returns a callable object that performs a sequence of `n_affine`
    affine coupling transformations on both partitions of the input vector."""
    return RedBlackLayers(
        layers.AffineLayer,
        n_affine,
        n_lattice,
        n_components,
        hidden_shape=hidden_shape,
        activation=activation,
        s_final_activation=s_final_activation,
        batch_normalise=batch_normalise,
    )


def real_nvp_circle(real_nvp):
    """Action that returns a callable object that projects an input vector from 
    (0, 2\pi)->R1, performs a sequence of affine transformations, then does the
    inverse projection back to (0, 2\pi)"""
    return Sequential(
        layers.ProjectionLayer(), real_nvp, layers.InverseProjectionLayer()
    )


def real_nvp_sphere(real_nvp):
    """Action that returns a callable object that projects an input vector from 
    S2 - {0} -> R2, performs a sequence of affine transformations, then does the
    inverse projection back to S2 - {0}"""
    return Sequential(
        layers.ProjectionLayer2D(), real_nvp, layers.InverseProjectionLayer2D(),
    )


def ncp_circle(
    n_lattice,
    n_pairs=1,  # unlikely that function composition is beneficial
    hidden_shape=[24,],
    activation="leaky_relu",
    s_final_activation=None,
    batch_normalise=False,
):
    """Action that returns a callable object that performs a sequence of transformations
    from (0, 2\pi) -> (0, 2\pi), each of which are the composition of a stereographic
    projection transformation, an affine transformation, and the inverse projection."""
    return RedBlackLayers(
        layers.NCPLayer,
        n_pairs,
        n_lattice,
        n_components=1,
        hidden_shape=hidden_shape,
        activation=activation,
        s_final_activation=s_final_activation,
        batch_normalise=batch_normalise,
    )


MODEL_OPTIONS = {
    "real_nvp": real_nvp,
    "real_nvp_circle": real_nvp_circle,
    "real_nvp_sphere": real_nvp_sphere,
    "ncp_circle": ncp_circle,
}
