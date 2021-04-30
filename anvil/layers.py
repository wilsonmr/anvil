# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
r"""
layers.py

Contains the transformations or "layers" which are the building blocks of
normalising flows. The layers are implemented using the PyTorch library, which
in practice means they subclass :py:class:`torch.nn.Module`. For more
information, check out the PyTorch
`Module docs <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module>`_.

The basic idea is of a flow is to generate a latent variable, in our framework
this would be using a class in :py:mod:`anvil.distributions`. The latent
variables are then transformed by sequentially applying the transformation
layers. The key feature of the transformations is the ability to easily calculate
the Jacobian determinant. If the base density function is known, then we can
evaluate the model density exactly.

The bottom line is that we enforce a convention to the ``forward`` method
of each layer (a special method of :py:class:`torch.nn.Module` subclasses).
All layers in this module should contain a ``forward`` method which takes two
:py:class:`torch.Tensor` objects as inputs:

    - a batch of input configurations, dimensions ``(batch size, lattice size)``.
    - a batch of scalars, dimensions ``(batch size, 1)``, that are the logarithm of the
      'current' probability density, at this stage in the normalising flow.

Each transformation layers may contain several neural networks or learnable
parameters.

A full normalising flow, f, can be constructed from multiple layers using
function composition:

.. math::

        f(z) = g_{N_layers}( \ldots ( g_2( g_1( z ) ) ) \ldots )

As a matter of convenience we provide a subclass of
:py:class:`torch.nn.Sequential`, which is initialised by passing multiple layers
as arguments (in the order in which the layers are applied). The main feature
of our version, :py:class:`Sequential`, is that it conforms to our ``forward``
convention. From the perspective of the user :py:class:`Sequential` appears
as a single subclass of :py:class:`torch.nn.Module` which performs the
full normalising flow transformation :math:`f(z)`.

"""
import torch
import torch.nn as nn

from anvil.core import FullyConnectedNeuralNetwork


class CouplingLayer(nn.Module):
    """
    Base class for coupling layers.

    A generic coupling layer takes the form

        v^P <- v^P                  passive partition
        v^A <- C( v^A ; {N(v^P)} )  active partition

    where the |\Lambda|-dimensional input configuration or 'vector' v has been split
    into two partitions, labelled by A and P (active and passive). Here, the paritions
    are split according to a checkerboard (even/odd) scheme.

    {N(v^P)} is a set of functions of the passive partition (neural networks) that
    parameterise the coupling layer.

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.

    Attributes
    ----------
    _passive_ind: slice
        Slice object which can be used to access the passive partition.
    _active_ind: slice
        Slice object which can be used to access the partition that gets transformed.
    _join_func: function
        Function which returns the concatenation of the two partitions in the
        appropriate order.
    """

    def __init__(self, size_half: int, even_sites: bool):
        super().__init__()

        if even_sites:
            # a is first half of input vector
            self._passive_ind = slice(0, size_half)
            self._active_ind = slice(size_half, 2 * size_half)
            self._join_func = torch.cat
        else:
            # a is second half of input vector
            self._passive_ind = slice(size_half, 2 * size_half)
            self._active_ind = slice(0, size_half)
            self._join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )


class AdditiveLayer(CouplingLayer):
    r"""Extension to `nn.Module` for an additive coupling layer.

    The additive transformation is given by

        C( v^A ; t(v^P) ) = v^A - t(v^P)

    The Jacobian determinant is

        \log \det J = 0
    """

    def __init__(
        self,
        size_half: int,
        *,
        hidden_shape: list,
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)

        self.t_network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )

    def forward(self, v_in, log_density, *unused) -> torch.Tensor:
        r"""Forward pass of affine transformation."""
        v_in_passive = v_in[:, self._passive_ind]
        v_in_active = v_in[:, self._active_ind]

        t_out = self.t_network(
            (v_in_passive - v_in_passive.mean()) / v_in_passive.std()
        )

        v_out = self._join_func([v_in_passive, v_in_active - t_out], dim=1)

        return v_out, log_density


class AffineLayer(CouplingLayer):
    r"""Extension to `nn.Module` for an affine coupling layer.

    The affine transformation is given by

        C( v^A ; s(v^P), t(v^P) ) = ( v^A - t(v^P) ) * \exp( -s(v^P) )

    The Jacobian determinant is

        \log \det J = \sum_x s_x(v^P)

    where x are the lattice sites in the active partition.

    """

    def __init__(
        self,
        size_half: int,
        *,
        hidden_shape: list,
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)

        self.s_network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )
        self.t_network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )
        self.z2_equivar = z2_equivar

    def forward(self, v_in, log_density, *unused) -> torch.Tensor:
        r"""Forward pass of affine transformation."""
        v_in_passive = v_in[:, self._passive_ind]
        v_in_active = v_in[:, self._active_ind]
        v_for_net = (v_in_passive - v_in_passive.mean()) / v_in_passive.std()

        s_out = self.s_network(v_for_net)
        t_out = self.t_network(v_for_net)

        # If enforcing s(-v) = s(v), we want to use |s(v)| in affine transf.
        if self.z2_equivar:
            s_out = torch.abs(s_out)

        v_out = self._join_func(
            [v_in_passive, (v_in_active - t_out) * torch.exp(-s_out)], dim=1
        )
        log_density += s_out.sum(dim=1, keepdim=True)

        return v_out, log_density


class RationalQuadraticSplineLayer(CouplingLayer):
    r"""A coupling transformation from a finite interval to itself based on a piecewise
    rational quadratic spline function.

    The interval is divided into K segments (bins) with widths w_k and heights h_k. The
    'knot points' (\phi_k, x_k) are the cumulative sum of (h_k, w_k), starting at (-B, -B)
    and ending at (B, B).

    In addition to the w_k and h_k, the derivatives d_k at the internal knot points are
    generated by a neural network. d_0 and d_K are set to 1.

    Defing the slopes s_k = h_k / w_k and fractional position within a bin

            alpha(x) = (x - x_{k-1}) / w_k

    the coupling transformation is defined piecewise by

            C(v^A, {h_k, s_k, d_k | k = 1, ..., K})
                 = \phi_{k-1}
                 + ( h_k(s_k * \alpha^2 + d_k * \alpha * (1 - \alpha)) )
                 / ( s_k + (d_{k+1} + d_k - 2s_k) * \alpha * (1 - \alpha) )
    """
    # TODO sort out indices

    def __init__(
        self,
        size_half: int,
        interval: int,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.size_half = size_half
        self.n_segments = n_segments

        self.network = FullyConnectedNeuralNetwork(
            size_in=size_half,
            size_out=size_half * (3 * n_segments - 1),
            hidden_shape=hidden_shape,
            activation=activation,
            bias=True,
        )

        self.norm_func = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()

        self.B = interval

        self.z2_equivar = z2_equivar

    def forward(self, v_in, log_density, negative_mag):
        """Forward pass of the rational quadratic spline layer."""
        v_in_passive = v_in[:, self._passive_ind]
        v_in_active = v_in[:, self._active_ind]
        v_for_net = (
            v_in_passive - v_in_passive.mean()
        ) / v_in_passive.std()  # reduce numerical instability

        # Naively enforce C(-v) = C(v)
        if self.z2_equivar:
            v_for_net[negative_mag] = -v_for_net[negative_mag]

        v_out_b = torch.zeros_like(v_in_active)
        gradient = torch.ones_like(v_in_active).unsqueeze(dim=-1)

        # Apply mask for linear tails
        # NOTE potentially a waste of time since we NEVER want to map out <- Id(in)
        inside_interval_mask = torch.abs(v_in_active) <= self.B
        v_in_b_inside_interval = v_in_active[inside_interval_mask]
        v_out_b[~inside_interval_mask] = v_in_active[~inside_interval_mask]

        h_net, w_net, d_net = (
            self.network(v_for_net)
            .view(-1, self.size_half, 3 * self.n_segments - 1)
            .split(
                (self.n_segments, self.n_segments, self.n_segments - 1),
                dim=2,
            )
        )

        if self.z2_equivar:
            h_net[negative_mag] = torch.flip(h_net[negative_mag], dims=(2,))
            w_net[negative_mag] = torch.flip(w_net[negative_mag], dims=(2,))
            d_net[negative_mag] = torch.flip(d_net[negative_mag], dims=(2,))

        h_norm = self.norm_func(h_net[inside_interval_mask]) * 2 * self.B
        w_norm = self.norm_func(w_net[inside_interval_mask]) * 2 * self.B
        d_pad = nn.functional.pad(
            self.softplus(d_net)[inside_interval_mask], (1, 1), "constant", 1
        )

        knots_xcoords = (
            torch.cat(
                (
                    torch.zeros(w_norm.shape[0], 1),
                    torch.cumsum(w_norm, dim=1),
                ),
                dim=1,
            )
            - self.B
        )
        knots_ycoords = (
            torch.cat(
                (
                    torch.zeros(h_norm.shape[0], 1),
                    torch.cumsum(h_norm, dim=1),
                ),
                dim=1,
            )
            - self.B
        )
        k_this_segment = (
            torch.searchsorted(
                knots_xcoords,
                v_in_b_inside_interval.view(-1, 1),
            )
            - 1
        ).clamp(0, self.n_segments - 1)

        w_at_lower_knot = torch.gather(w_norm, 1, k_this_segment)
        h_at_lower_knot = torch.gather(h_norm, 1, k_this_segment)
        s_at_lower_knot = h_at_lower_knot / w_at_lower_knot
        d_at_lower_knot = torch.gather(d_pad, 1, k_this_segment)
        d_at_upper_knot = torch.gather(d_pad, 1, k_this_segment + 1)

        v_in_at_lower_knot = torch.gather(knots_xcoords, 1, k_this_segment)
        v_out_at_lower_knot = torch.gather(knots_ycoords, 1, k_this_segment)

        alpha = (
            v_in_b_inside_interval.unsqueeze(dim=-1) - v_in_at_lower_knot
        ) / w_at_lower_knot

        v_out_b[inside_interval_mask] = (
            v_out_at_lower_knot
            + (
                h_at_lower_knot
                * (
                    s_at_lower_knot * alpha.pow(2)
                    + d_at_lower_knot * alpha * (1 - alpha)
                )
            )
            / (
                s_at_lower_knot
                + (d_at_upper_knot + d_at_lower_knot - 2 * s_at_lower_knot)
                * alpha
                * (1 - alpha)
            )
        ).squeeze()

        gradient[inside_interval_mask] = (
            s_at_lower_knot.pow(2)
            * (
                d_at_upper_knot * alpha.pow(2)
                + 2 * s_at_lower_knot * alpha * (1 - alpha)
                + d_at_lower_knot * (1 - alpha).pow(2)
            )
        ) / (
            s_at_lower_knot
            + (d_at_upper_knot + d_at_lower_knot - 2 * s_at_lower_knot)
            * alpha
            * (1 - alpha)
        ).pow(
            2
        )

        v_out = self._join_func([v_in_passive, v_out_b], dim=1)
        log_density -= torch.log(gradient).sum(dim=1)

        return v_out, log_density


class GlobalAffineLayer(nn.Module):
    r"""Applies an affine transformation to every data point using a given scale and shift,
    which are *not* learnable. Useful to shift the domain of a learned distribution. This is
    done at the cost of a constant term in the logarithm of the Jacobian determinant, which
    is ignored.

    Parameters
    ----------
    scale: (int, float)
        Every data point will be multiplied by this factor.
    shift: (int, float)
        Every scaled data point will be shifted by this factor.
    """

    def __init__(self, scale, shift):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, v_in, log_density):
        """Forward pass of the global affine transformation."""
        return self.scale * v_in + self.shift, log_density


# TODO not necessary to define a nn.module for this now I've taken out learnable gamma
class BatchNormLayer(nn.Module):
    """Performs batch normalisation on the input vector.

    Parameters
    ----------
    scale: int
        An additional scale factor to be applied after batch normalisation.
    """

    def __init__(self, scale=1):
        super().__init__()
        self.gamma = scale

    def forward(self, v_in, log_density, *unused):
        """Forward pass of the batch normalisation transformation."""
        mult = self.gamma / torch.std(v_in)
        v_out = mult * (v_in - v_in.mean())
        log_density -= mult * v_out.shape[1]
        return (v_out, log_density)


class GlobalRescaling(nn.Module):
    def __init__(self, initial=1):
        super().__init__()

        self.scale = nn.Parameter(torch.Tensor([initial]))

    def forward(self, v_in, log_density, *unused):
        v_out = self.scale * v_in
        log_density -= v_out.shape[-1] * torch.log(self.scale)
        return v_out, log_density


class Sequential(nn.Sequential):
    """Similar to :py:class:`torch.nn.Sequential` except conforms to our
    ``forward`` convention.

    """

    def forward(self, v, log_density, *args):
        for module in self:
            v, log_density = module(v, log_density, *args)
        return v, log_density
