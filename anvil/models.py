# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
models.py

Module containing reportengine actions which return normalising flow models.
Generally this involves piecing together components from :py:mod:`anvil.layers`
and :py:mod:`anvil.core` to produce sequences of transformations.

"""
from functools import partial

from reportengine import collect

from anvil.core import Sequential
import anvil.layers as layers


def _coupling_pair(coupling_layer, **kwargs):
    """Helper function which wraps a pair of coupling layers from
    :py:mod:`anvil.layers` in the module container
    :py:class`anvil.core.Sequential`. The first transformation layer acts on
    the even sites and the second transformation acts on the odd sites, so one
    of these blocks ensures all sites are transformed as part of an
    active partition.

    """
    coupling_transformation = partial(coupling_layer, **kwargs)
    return Sequential(
        coupling_transformation(even_sites=True),
        coupling_transformation(even_sites=False),
    )


def _real_nvp(
    size_half,
    n_blocks,
    hidden_shape,
    activation="tanh",
    z2_equivar=True,
):
    r"""Action which returns a sequence of ``n_blocks`` pairs of
    :py:class:`anvil.layers.AffineLayer` s, followed by a single
    :py:class:`anvil.layers.GlobalRescaling` all wrapped in the module container
    :py:class`anvil.core.Sequential`.

    The first ``n_blocks`` elements of the outer ``Sequential``
    are ``Sequential`` s containing a pair of ``AffineLayer`` s which
    act on the even and odd sites respectively.

    Parameters
    ----------
    size_half: int
        Inferred from ``lattice_size``, the size of the active/passive
        partitions (which are equal size, `lattice_size / 2`).
    n_blocks: int
        The number of pairs of :py:class:`anvil.layers.AffineLayer`
        transformations.
    hidden_shape: list[int]
        the shape of the neural networks used in the AffineLayer. The visible
        layers are defined by the ``lattice_size``. Typically we have found
        a single hidden layer neural network is effective, which can be
        specified by passing a list of length 1, i.e. ``[72]`` would
        be a single hidden layered network with 72 nodes in the hidden layer.
    activation: str, default="tanh"
        The activation function to use for each hidden layer. The output layer
        of the network is linear (has no activation function).
    z2_equivar: bool, default=True
        Whether or not to impose z2 equivariance. This changes the transformation
        such that the neural networks have no bias term and s(-v) = s(v) which
        imposes a :math:`\mathbb{Z}_2` symmetry.

    Returns
    -------
    real_nvp: anvil.core.Sequential
        A sequence of affine transformations, which we refer to as a real NVP
        (Non-volume preserving) flow.

    See Also
    --------
    :py:mod:`anvil.core` contains the fully connected neural network class
    as well as valid choices for activation functions.

    """
    blocks = [
        _coupling_pair(
            layers.AffineLayer,
            size_half=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            z2_equivar=z2_equivar,
        )
        for i in range(n_blocks)
    ]
    return Sequential(*blocks, layers.GlobalRescaling())


def _nice(
    size_half,
    n_blocks,
    hidden_shape,
    activation="tanh",
    z2_equivar=True,
):
    """Similar to :py:func:`real_nvp`, excepts instead wraps pairs of
    :py:class:`layers.AdditiveLayer` s followed by a single
    :py:class:`layers.GlobalRescaling`. The pairs of ``AdditiveLayer`` s
    act on the even and odd sites respectively.

    Parameters
    ----------
    size_half: int
        Inferred from ``lattice_size``, the size of the active/passive
        partitions (which are equal size, `lattice_size / 2`).
    n_blocks: int
        The number of pairs of :py:class:`anvil.layers.AffineLayer`
        transformations.
    hidden_shape: list[int]
        the shape of the neural networks used in the each layer. The visible
        layers are defined by the ``lattice_size``.
    activation: str, default="tanh"
        The activation function to use for each hidden layer. The output layer
        of the network is linear (has no activation function).
    z2_equivar: bool, default=True
        Whether or not to impose z2 equivariance. This changes the transformation
        such that the neural networks have no bias term and s(-v) = s(v) which
        imposes a :math:`\mathbb{Z}_2` symmetry.

    Returns
    -------
    nice: anvil.core.Sequential
        A sequence of additive transformations, which we refer to as a
        nice flow.

    """
    blocks = [
        _coupling_pair(
            layers.AdditiveLayer,
            size_half=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            z2_equivar=z2_equivar,
        )
        for i in range(n_blocks)
    ]
    return Sequential(*blocks, layers.GlobalRescaling())


def _rational_quadratic_spline(
    size_half,
    hidden_shape,
    interval=5,
    n_blocks=1,
    n_segments=4,
    activation="tanh",
    z2_equivar=False,
):
    """Similar to :py:func:`real_nvp`, excepts instead wraps pairs of
    :py:class:`layers.RationalQuadraticSplineLayer` s followed by a single
    :py:class:`layers.GlobalRescaling`. The pairs of RQS's
    act on the even and odd sites respectively.

    Parameters
    ----------
    size_half: int
        inferred from ``lattice_size``, the size of the active/passive
        partitions (which are equal size, `lattice_size / 2`).
    hidden_shape: list[int]
        the shape of the neural networks used in the each layer. The visible
        layers are defined by the ``lattice_size``.
    interval: int, default=5
        the interval within which the RQS applies the transformation, at present
        if a field variable is outside of this region it is mapped to itself
        (i.e the gradient of the transformation is 1 outside of the interval).
    n_blocks: int, default=1
        The number of pairs of :py:class:`anvil.layers.AffineLayer`
        transformations. For RQS this is set to 1.
    n_segments: int, default=4
        The number of segments to use in the RQS transformation.
    activation: str, default="tanh"
        The activation function to use for each hidden layer. The output layer
        of the network is linear (has no activation function).
    z2_equivar: bool, default=False
        Whether or not to impose z2 equivariance. This is only done crudely
        by splitting the sites according to the sign of the sum across lattice
        sites.

    """
    blocks = [
        _coupling_pair(
            layers.RationalQuadraticSplineLayer,
            size_half=size_half,
            interval=interval,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            z2_equivar=z2_equivar,
        )
        for _ in range(n_blocks)
    ]
    return Sequential(
        #layers.BatchNormLayer(),
        *blocks,
        layers.GlobalRescaling(),
    )

_normalising_flow = collect("layer_action", ("model_params",))

def model_to_load(_normalising_flow):
    """action which wraps a list of layers in
    :py:class:`anvil.core.Sequential`. This allows the user to specify an
    arbitrary combination of layers as the model.

    For more information
    on valid choices for layers, see :py:var:`LAYER_OPTIONS` or the various
    functions in :py:mod:`anvil.models` which produce sequences of the layers
    found in :py:mod:`anvil.layers`.

    """
    return Sequential(*_normalising_flow)

LAYER_OPTIONS = {
    "nice": _nice,
    "real_nvp": _real_nvp,
    "rational_quadratic_spline": _rational_quadratic_spline,
}
