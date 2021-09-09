# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
models.py

Module containing reportengine actions which return normalising flow models.
Generally this involves piecing together components from :py:mod:`anvil.layers`
to produce sequences of transformations.

"""
from reportengine import collect

import anvil.layers as layers


def real_nvp(
    mask,
    n_blocks: int,
    hidden_shape: (tuple, list),
    activation: str = "tanh",
    final_activation: (str, type(None)) = None,
    z2_equivar: bool = True,
    use_convnet: bool = False,
) -> layers.Sequential:
    r"""Action which returns a sequence of ``n_blocks`` pairs of
    :py:class:`anvil.layers.AffineLayer` s, wrapped in the module container
    :py:class`anvil.layers.Sequential`.

    The first ``n_blocks`` elements of the outer ``Sequential``
    are ``Sequential`` s containing a pair of ``AffineLayer`` s which
    act on the even and odd sites respectively.

    Parameters
    ----------
    mask
        Boolean mask which differentiates the two partitions as required by
        the coupling layers.
    n_blocks
        The number of pairs of :py:class:`anvil.layers.AffineLayer`
        transformations.
    hidden_shape
        the shape of the neural networks used in the AffineLayer. The visible
        layers are defined by the ``lattice_size``. Typically we have found
        a single hidden layer neural network is effective, which can be
        specified by passing a list of length 1, i.e. ``[72]`` would
        be a single hidden layered network with 72 nodes in the hidden layer.
    activation
        The activation function to use for each hidden layer.
    final_activation
        The activation function to use for the output layer.
    z2_equivar
        Whether or not to impose z2 equivariance. This changes the transformation
        such that the neural networks have no bias term and s(-v) = s(v) which
        imposes a :math:`\mathbb{Z}_2` symmetry.
    use_convnet
        If true, use convolutional networks. Otherwise, use fully-connected.

    Returns
    -------
    anvil.layers.Sequential
        A sequence of affine transformations, which we refer to as a real NVP
        (Non-volume preserving) flow.

    See Also
    --------
    :py:mod:`anvil.neural_network`
    """
    blocks = [
        layers.AffineLayer(
            mask=mask,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=final_activation,
            z2_equivar=z2_equivar,
            use_convnet=use_convnet,
        )
        for _ in range(n_blocks)
    ]
    return layers.Sequential(*blocks)


def legacy_real_nvp(
    mask,
    n_blocks: int,
    hidden_shape: (tuple, list),
    activation: str = "tanh",
    final_activation: str = "none",
    z2_equivar: bool = True,
    use_convnet: bool = False,
) -> layers.Sequential:
    """Legacy version of affine layers, where each coupling layer has two
    neural networks, rather than one."""
    assert (
        use_convnet is False
    ), "Convolutional networks are not supported by the legacy version of affine coupling layers"
    blocks = [
        layers.LegacyAffineLayer(
            mask=mask,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=final_activation,
            z2_equivar=z2_equivar,
        )
        for _ in range(n_blocks)
    ]
    return layers.Sequential(*blocks)


def nice(
    mask,
    n_blocks: int,
    hidden_shape: (tuple, list),
    activation: str = "tanh",
    final_activation: (str, type(None)) = None,
    z2_equivar: bool = True,
    use_convnet: bool = False,
) -> layers.Sequential:
    r"""Similar to :py:func:`real_nvp`, excepts instead wraps pairs of
    :py:class:`anvil.layers.AdditiveLayer` .
    The pairs of ``AdditiveLayer`` s act on the even and odd sites respectively.

    Parameters
    ----------
    mask
        Boolean mask which differentiates the two partitions as required by
        the coupling layers.
    n_blocks
        The number of pairs of :py:class:`anvil.layers.AffineLayer`
        transformations.
    hidden_shape
        the shape of the neural networks used in the each layer. The visible
        layers are defined by the ``lattice_size``.
    activation
        The activation function to use for each hidden layer.
    final_activation
        The activation function to use for the output layer.
    z2_equivar
        Whether or not to impose z2 equivariance. This changes the transformation
        such that the neural networks have no bias term and s(-v) = s(v) which
        imposes a :math:`\mathbb{Z}_2` symmetry.
    use_convnet
        If true, use convolutional networks. Otherwise, use fully-connected.

    Returns
    -------
    anvil.layers.Sequential
        A sequence of additive transformations, which we refer to as a
        nice flow.

    """
    blocks = [
        layers.AdditiveLayer(
            mask=mask,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=final_activation,
            z2_equivar=z2_equivar,
            use_convnet=use_convnet,
        )
        for _ in range(n_blocks)
    ]
    return layers.Sequential(*blocks)


def rational_quadratic_spline(
    mask,
    n_blocks: int,
    hidden_shape: (tuple, list),
    n_segments: int,
    interval: (int, float) = 5,
    activation: str = "tanh",
    final_activation: (str, type(None)) = None,
    use_convnet: bool = False,
) -> layers.Sequential:
    """Similar to :py:func:`real_nvp`, excepts instead wraps pairs of
    :py:class:`anvil.layers.RationalQuadraticSplineLayer` s.
    The pairs of RQS's act on the even and odd sites respectively.

    Parameters
    ----------
    mask
        Boolean mask which differentiates the two partitions as required by
        the coupling layers.
    n_blocks
        The number of pairs of :py:class:`anvil.layers.AffineLayer`
        transformations. For RQS this is set to 1.
    hidden_shape
        the shape of the neural networks used in the each layer. The visible
        layers are defined by the ``lattice_size``.
    n_segments
        The number of segments to use in the RQS transformation.
    interval
        an integer :math:`a` denoting a symmetric interval :math:`[-a, a]`
        within which the RQS applies the transformation. At present, if a
        field variable is outside of this region it is mapped to itself
        (i.e the gradient of the transformation is 1 outside of the interval).
    activation
        The activation function to use for each hidden layer.
    final_activation
        The activation function to use for the output layer.
    use_convnet
        If true, use convolutional networks. Otherwise, use fully-connected.
    """

    blocks = [
        layers.RationalQuadraticSplineLayer(
            mask=mask,
            interval=interval,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=final_activation,
            use_convnet=use_convnet,
        )
        for _ in range(n_blocks)
    ]
    return layers.Sequential(*blocks)


def legacy_equivariant_spline(
    mask,
    n_blocks: int,
    hidden_shape: (tuple, list),
    n_segments: int,
    interval: (int, float) = 5,
    activation: str = "tanh",
    final_activation: (str, type(None)) = None,
) -> layers.Sequential:
    """Similar to :py:func:`rational_quadratic_spline`, excepts instead wraps
    pairs of :py:class:`anvil.layers.LegacyEquivariantSplineLayer` s.

    **HEALTH WARNING:** Unfortunately, this results in transformations that are not
    necessarily continuous which means density estimation is not guaranteed to be
    correct. We did find that these layers improved performance in the broken symmetry
    phase of phi^4 theory, but due to the problem just stated we DO NOT recommend
    using these layers.

    Parameters
    ----------
    mask
        Boolean mask which differentiates the two partitions as required by
        the coupling layers.
    n_blocks
        The number of pairs of :py:class:`anvil.layers.AffineLayer`
        transformations. For RQS this is set to 1.
    hidden_shape
        the shape of the neural networks used in the each layer. The visible
        layers are defined by the ``lattice_size``.
    n_segments
        The number of segments to use in the RQS transformation.
    interval
        an integer :math:`a` denoting a symmetric interval :math:`[-a, a]`
        within which the RQS applies the transformation. At present, if a
        field variable is outside of this region it is mapped to itself
        (i.e the gradient of the transformation is 1 outside of the interval).
    activation
        The activation function to use for each hidden layer.
    final_activation
        The activation function to use for the output layer.
    """

    blocks = [
        layers.LegacyEquivariantSplineLayer(
            mask=mask,
            interval=interval,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=final_activation,
        )
        for _ in range(n_blocks)
    ]
    return layers.Sequential(*blocks)


def batch_norm(scale: (int, float) = 1.0) -> layers.Sequential:
    r"""Action which returns an instance of :py:class:`anvil.layers.BatchNormLayer`.

    Parameters
    ----------
    scale
        The multiplicative factor applied to the standardised data.

    Returns
    -------
    anvil.layers.Sequential
        An instance of :py:class:`anvil.layers.BatchNormLayer` wrapped by
        :py:class:`anvil.layers.Sequential` , which is simply there to make
        iterating over layers easier and has no effect on the transformation
        applied to the layer inputs.
    """
    return layers.Sequential(layers.BatchNormLayer(scale=scale))


def global_rescaling(scale: (int, float), learnable: bool = True) -> layers.Sequential:
    r"""Action which returns and instance of :py:class:`anvil.layers.GlobalRescaling`.

    Parameters
    ----------
    scale
        The multiplicative factor applied to the inputs.
    learnable
        If True, ``scale`` will be optimised during the training.

    Returns
    -------
    anvil.layers.Sequential
        An instance of :py:class:`anvil.layers.GlobalRescaling` wrapped by
        :py:class:`anvil.layers.Sequential` , which is simply there to make
        iterating over layers easier and has no effect on the transformation
        applied to the layer inputs.
    """
    return layers.Sequential(layers.GlobalRescaling(scale=scale, learnable=learnable))


def gauss_to_free(geometry, m_sq=None):
    """Action which returns an instance of :py:class:`anvil.layers.GaussToFreeField`.

    Parameters
    ----------
    geometry
        The :py:class:`anvil.geometry.Geometry2D` object representing the lattice
    m_sq
        The bare mass squared of the theory

    Returns
    -------
    anvil.layers.Sequential
    """
    return layers.Sequential(layers.GaussToFreeField(geometry, m_sq))


# collect layers from copied runcard
_normalizing_flow = collect(
    "layer_action",
    (
        "training_context",
        "model",
    ),
)


def model_to_load(_normalizing_flow) -> layers.Sequential:
    """action which wraps a list of layers in
    :py:class:`anvil.layers.Sequential`. This allows the user to specify an
    arbitrary combination of layers as the model.

    For more information on valid choices for layers, see
    ``anvil.models.LAYER_OPTIONS`` or the various
    functions in :py:mod:`anvil.models` which produce sequences of the layers
    found in :py:mod:`anvil.layers`.

    At present, available transformations are:

        - ``nice``
        - ``real_nvp``
        - ``rational_quadratic_spline``
        - ``batch_norm``
        - ``global_rescaling``
        - ``gauss_to_free``

    You can see their dependencies using the ``anvil`` provider help, e.g.
    for ``real_nvp``:

    .. code::

        $ anvil-sample --help real_nvp
        ...
        < action docstring - poorly formatted>
        ...
        The following resources are read from the configuration:

            lattice_length(int):
        [Used by lattice_size]

            lattice_dimension(int): Parse lattice dimension from runcard
        [Used by lattice_size]

        The following additionl arguments can be used to control the
        behaviour. They are set by default to sensible values:

        n_blocks
        hidden_shape
        activation = tanh
        z2_equivar = True

    ``anvil-train`` will also provide the same information.

    """
    # assume that _normalizing_flow is a list of layers, each layer
    # is a sequential of blocks, each block is a pair of transformations
    # which transforms the entire input state - flatten this out, so output
    # is Sequential of blocks
    flow_flat = [block for layer in _normalizing_flow for block in layer]
    return layers.Sequential(*flow_flat)


# annoyingly, the api may not have a training output. In which case
# load model from explicitly declared params.
_api_normalizing_flow = collect("layer_action", ("model",))


def explicit_model(_api_normalizing_flow):
    """Action to be called from the API. Build model from an explicit
    specification, with same input as a training runcard config.

    Examples
    --------

    >>> from anvil.api import API
    >>> model_spec = {
    ...     "model": [
    ...         {"layer": "global_rescaling", "scale": 1.0},
    ...         {"layer": "global_rescaling", "scale": 1.0},
    ...     ]
    ... }
    >>> API.explicit_model(**model_spec)
    Sequential(
      (0): GlobalRescaling()
      (1): GlobalRescaling()
    )

    """
    # Note: use same action as train/sample apps, so that tests can cover these.
    return model_to_load(_api_normalizing_flow)


# Update docstring above if you add to this!
LAYER_OPTIONS = {
    "nice": nice,
    "real_nvp": real_nvp,
    "rational_quadratic_spline": rational_quadratic_spline,
    "batch_norm": batch_norm,
    "global_rescaling": global_rescaling,
    "gauss_to_free": gauss_to_free,
    "legacy_real_nvp": legacy_real_nvp,
    "legacy_equivariant_spline": legacy_equivariant_spline,
}
