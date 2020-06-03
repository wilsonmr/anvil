from functools import partial

from anvil.core import Sequential

import anvil.layers as layers


def coupling_pair(coupling_layer, size_half, **layer_spec):
    """Helper function which returns a callable object that performs a coupling
    transformation on both even and odd lattice sites."""
    coupling_transformation = partial(coupling_layer, size_half, **layer_spec)
    return Sequential(
        coupling_transformation(even_sites=True),
        coupling_transformation(even_sites=False),
    )


def real_nvp(
    size_half,
    n_affine,
    hidden_shape=[24,],
    activation="leaky_relu",
    batch_normalise=False,
):
    affine_pairs = [
        coupling_pair(
            layers.AffineLayer,
            size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            batch_normalise=batch_normalise,
            i_layer=i,
        )
        for i in range(n_affine)
    ]
    return Sequential(*affine_pairs)


def real_nvp_circle(size_half, real_nvp):
    return Sequential(layers.ProjectCircle(), real_nvp, layers.ProjectCircleInverse())


MODEL_OPTIONS = {
    "real_nvp": real_nvp,
    "real_nvp_circle": real_nvp_circle,
}
