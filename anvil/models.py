"""
models.py

Module containing reportengine actions which return callable objects that execute
normalising flows constructed from multiple layers via function composition.
"""
from functools import partial

from anvil.core import Sequential

import anvil.layers as layers


def coupling_block(coupling_layer, size_half, **layer_spec):
    """Helper function which returns a callable object that performs a coupling
    transformation on both even and odd lattice sites."""
    coupling_transformation = partial(coupling_layer, size_half, **layer_spec)
    return Sequential(
        coupling_transformation(even_sites=True),
        coupling_transformation(even_sites=False),
    )


def nice(
    size_half,
    n_additive,
    hidden_shape,
    activation,
    z2_equivar=False,
):
    """Action that returns a callable object that performs a sequence of `n_affine`
    affine coupling transformations on both partitions of the input vector."""
    blocks = [
        coupling_block(
            layers.AdditiveLayer,
            size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            z2_equivar=z2_equivar,
        )
        for _ in range(n_additive)
    ]
    return Sequential(*blocks)


def real_nvp(
    size_half,
    n_affine,
    hidden_shape,
    activation,
    z2_equivar=False,
):
    """Action that returns a callable object that performs a sequence of `n_affine`
    affine coupling transformations on both partitions of the input vector."""
    blocks = [
        coupling_block(
            layers.AffineLayer,
            size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            z2_equivar=z2_equivar,
        )
        for _ in range(n_affine)
    ]
    return Sequential(*blocks)


# TODO: either use uniform prior or need to shift and normalise before splien
def linear_spline(
    size_half,
    n_segments,
    interval,
    hidden_shape,
    activation,
):
    """Action that returns a callable object that performs a pair of linear spline
    transformations, one on each half of the input vector."""
    return Sequential(
        coupling_block(
            layers.LinearSplineLayer,
            size_half,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
        ),
        layers.GlobalAffineLayer(scale=2 * interval, shift=-interval),
    )

# TODO: same as above
def quadratic_spline(
    size_half,
    n_segments,
    interval,
    hidden_shape,
    activation,
):
    """Action that returns a callable object that performs a pair of linear spline
    transformations, one on each half of the input vector."""
    return Sequential(
        coupling_block(
            layers.QuadraticSplineLayer,
            size_half,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
        ),
        layers.GlobalAffineLayer(scale=2 * interval, shift=-interval),
    )


def rational_quadratic_spline(
    size_half,
    n_spline,
    interval,
    n_segments,
    hidden_shape,
    activation,
    spline_z2_equivar=False,
    scale_before_spline=1,
):
    """Action that returns a callable object that performs a pair of circular spline
    transformations, one on each half of the input vector."""
    return Sequential(
        layers.BatchNormLayer(scale=scale_before_spline),
        *[
            coupling_block(
                layers.RationalQuadraticSplineLayer,
                size_half,
                interval=interval,
                n_segments=n_segments,
                hidden_shape=hidden_shape,
                activation=activation,
                z2_equivar=spline_z2_equivar,
            )
            for _ in range(n_spline)
        ],
    )


MODEL_OPTIONS = {
    "nice": nice,
    "real_nvp": real_nvp,
    #"linear_spline": linear_spline,
    #"quadratic_spline": quadratic_spline,
    "rational_quadratic_spline": rational_quadratic_spline,
}
