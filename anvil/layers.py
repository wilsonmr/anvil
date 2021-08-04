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

        f(z) = g_{N_{\rm layers}}( \ldots ( g_2( g_1( z ) ) ) \ldots )

As a matter of convenience we provide a subclass of
:py:class:`torch.nn.Sequential`, which is initialised by passing multiple layers
as arguments (in the order in which the layers are applied). The main feature
of our version, :py:class:`Sequential`, is that it conforms to our ``forward``
convention. From the perspective of the user :py:class:`Sequential` appears
as a single subclass of :py:class:`torch.nn.Module` which performs the
full normalising flow transformation :math:`f(z)`.

"""
import itertools
from functools import cached_property

import torch
import torch.nn as nn

from anvil.neural_network import DenseNeuralNetwork
from anvil.free_scalar import FreeScalar


class CouplingLayer(nn.Module):
    r"""
    Base class for coupling layers, inheriting from py:class:`torch.nn.Module` but
    redefining the ``forward`` method so that the Jacobian determinant of the layers
    are accumulated alongside the activations.

    A generic coupling layer takes the form
    
    .. math::

        v^P & \leftarrow v^P \\
        v^A & \leftarrow C \left( v^A ; \{\mathbf{N}(v^P)\} \right)

    where the :math:`|\Lambda|`-dimensional input configuration :math:`v` has been split
    into two partitions, labelled by :math:`A` and :math:`P` (active and passive).

    Here, the paritions are split according to a checkerboard (even/odd) scheme, as
    defined in :py:class:`anvil.geometry.Geometry2D`.

    :math:`\{\mathbf{N}(v^P)\}` is a set of functions of the passive partition that parameterise
    the coupling layer. These functions are to be modelled by neural networks.

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.
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
    r"""Class implementing additive coupling layers.

    The additive transformation is given by

    .. math::

        C( v^A ; \mathbf{t}(v^P) ) = v^A - \mathbf{t}(v^P)

    where :math:`\mathbf{t}` is a neural network.

    The transformation is volume-preserving, i.e.

    .. math::

        \log \det J = 0

    Reference: https://arxiv.org/abs/1410.8516
    """

    def __init__(
        self,
        size_half: int,
        *,
        hidden_shape: (tuple, list),
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)

        self.t_network = DenseNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )

    def forward(self, v_in: torch.Tensor, log_density: torch.Tensor, *args):
        r"""Forward pass of affine transformation."""
        v_in_passive = v_in[:, self._passive_ind]
        v_in_active = v_in[:, self._active_ind]
        v_for_net = v_in_passive / torch.sqrt(v_in_passive.var() + 1e-6)

        t_out = self.t_network(v_for_net)

        v_out = self._join_func([v_in_passive, v_in_active - t_out], dim=1)

        return v_out, log_density


class AffineLayer(CouplingLayer):
    r"""Class implementing affine coupling layers.

    The affine transformation is given by

    .. math::

        C( v^A ; \mathbf{s}(v^P), \mathbf{t}(v^P) )
        = ( v^A - \mathbf{t}(v^P) ) * \exp( -\mathbf{s}(v^P) )

    The Jacobian determinant is

    .. math::

        \log \det J = - \sum_{x\in\Lambda^A} \mathbf{s}_x(v^P)

    Reference: https://arxiv.org/abs/1605.08803
    """

    def __init__(
        self,
        size_half: int,
        *,
        hidden_shape: (tuple, list),
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)

        self.s_network = DenseNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )
        self.t_network = DenseNeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            bias=not z2_equivar,
        )
        self.z2_equivar = z2_equivar

    def forward(self, v_in: torch.Tensor, log_density: torch.Tensor, *args):
        r"""Forward pass of affine transformation."""
        v_in_passive = v_in[:, self._passive_ind]
        v_in_active = v_in[:, self._active_ind]
        v_for_net = v_in_passive / torch.sqrt(v_in_passive.var() + 1e-6)

        s_out = self.s_network(v_for_net)
        t_out = self.t_network(v_for_net)

        # If enforcing C(-v) = -C(v), we want to use |s(v)| in affine transf.
        if self.z2_equivar:
            s_out = torch.abs(s_out)

        v_out = self._join_func(
            [v_in_passive, (v_in_active - t_out) * torch.exp(-s_out)], dim=1
        )
        log_density += s_out.sum(dim=1, keepdim=True)

        return v_out, log_density


class RationalQuadraticSplineLayer(CouplingLayer):
    r"""Class implementing rational quadratic spline coupling layers.

    The transformation maps a finite interval :math:`[-a, a]` to itself and is defined
    by a piecewise rational quadratic spline function, with the pieces joined up at 'knots'.

    The interval is divided into :math:`K` bins with widths :math:`w^k` and heights
    :math:`\mathbf{h}^k`. In addition to the widths and heights, the derivatives
    :math:`\mathbf{d}^k` at the internal knots are generated by a neural network.
    :math:`\mathbf{d}^0` and :math:`\mathbf{d}^K` are set to unity.

    Define the slopes connecting knots

    .. math ::

        \mathbf{s}_i^k = \frac{\mathbf{h}_i^k}{\mathbf{w}_i^k}

    and fractional position of the input :math:`v_{i,x}` in the :math:`\ell` -th bin

    .. math::

        \frac{(v_{i,x} - v_{i,x}^{\ell-1})}{\mathbf{w}_{i,x}^\ell}
        \equiv \alpha_{i,x} \in [0, 1]

    Then, the transformation is given by

    .. math::

        C_{i,x} ( v_{i, x} ; \mathbf{N}_{i, x}) = -a + \sum_{k=1}^{\ell-1} \mathbf{h}_{i,x}^k
        + \frac{ \mathbf{h}_{i, x}^\ell
        \left[ \mathbf{s}_{i,x}^\ell \alpha_{i,x}^2
        + \mathbf{d}_{i,x}^{\ell-1} \alpha_{i,x} (1 - \alpha_{i,x}) \right] }
        {\mathbf{s}_{i,x}^\ell + (\mathbf{d}_{i,x}^{\ell-1}
        + \mathbf{d}_{i,x}^\ell - 2 \mathbf{s}_{i,x}^\ell) \alpha_{i,x}(1 - \alpha_{i,x})}

    The gradient is

    .. math::

        \frac{1}{\mathbf{w}_{i,x}^\ell} \frac{d C_{i, x}}{d\alpha_{i,x}}
        = \frac{ (\mathbf{s}_{i,x}^\ell)^2 \left[
        \mathbf{d}_{i,x}^\ell \alpha_{i,x}^2 + 2\mathbf{s}_{i,x}^\ell
        \alpha_{i,x}(1 - \alpha_{i,x})
        + \mathbf{d}_{i,x}^{\ell-1} (1 - \alpha_{i,x})^2 \right]}
        {\left[ \mathbf{s}_{i,x}^\ell
        + (\mathbf{d}_{i,x}^{\ell-1} + \mathbf{d}_{i,x}^\ell - 2\mathbf{s}_{i,x}^\ell
        \alpha_{i,x} (1 - \alpha_{i,x}) \right]^2}

    To obtain the logarithm of the Jacobian determinant we first take the logarithm and
    then sum over :math:`x\in\Lambda^A` .

    Reference: https://arxiv.org/abs/1906.04032
    """

    def __init__(
        self,
        size_half: int,
        interval: int,
        n_segments: int,
        hidden_shape: (tuple, list),
        activation: str,
        z2_equivar: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.size_half = size_half
        self.n_segments = n_segments

        self.network = DenseNeuralNetwork(
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

    def forward(
        self, v_in: torch.Tensor, log_density: torch.Tensor, negative_mag: torch.Tensor
    ):
        """Forward pass of the rational quadratic spline layer."""
        v_in_passive = v_in[:, self._passive_ind]
        v_in_active = v_in[:, self._active_ind]
        v_for_net = v_in_passive / torch.sqrt(v_in_passive.var() + 1e-6)

        # Naively enforce C(-v) = -C(v)
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

    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, v_in: torch.Tensor, log_density: torch.Tensor, *args):
        """Forward pass of the global affine transformation."""
        return self.scale * v_in + self.shift, log_density


# NOTE: not necessary to define a nn.module for this now gamma is no longer learnable
class BatchNormLayer(nn.Module):
    r"""Performs batch normalisation on the inputs, conforming to our ``forward``
    convention.

    Inputs are standardised over all tensor dimensions such that the resulting sample
    has null mean and unit variance, after which a rescaling factor is applied.

    .. math::

            v_{\rm out} = \gamma
                \frac{v_{\rm in} - \mathbb{E}[ v_{\rm in} ]}
                {\sqrt{\mathrm{var}( v_{\rm in} ) + \epsilon}}

    Parameters
    ----------
    scale
        The multiplicative factor, :math:`\gamma`, applied to the standardised data.

    Notes
    -----
    Applying batch normalisation before the first spline layer can be helpful for
    ensuring that the inputs remain within the transformation interval. However,
    this layer adds undesirable stochasticity which can impede optimisation. One
    might consider replacing it with :py:class:`anvil.layers.GlobalRescaling` using
    a static scale parameter.
    """

    def __init__(self, scale: float = 1):
        super().__init__()
        self.gamma = scale

    def forward(self, v_in: torch.Tensor, log_density: torch.Tensor, *args):
        """Forward pass of the batch normalisation transformation."""
        mult = self.gamma / torch.sqrt(v_in.var() + 1e-6)  # for stability
        v_out = mult * (v_in - v_in.mean())
        log_density -= torch.log(mult) * v_out.shape[1]
        return (v_out, log_density)


class GlobalRescaling(nn.Module):
    r"""Performs a global rescaling of the inputs via a (potentially learnable)
    multiplicative factor, conforming to our ``forward`` convention.

    Parameters
    ----------
    scale:
        The multiplicative factor applied to the inputs.
    learnable:
        If True, ``scale`` will be optimised during the training.

    Notes
    -----
    Applying a rescaling layer with a learnable ``scale`` to the final layer of a
    normalizing flow can be useful since it avoids the need to tune earlier layers
    to match the width of the target density. However, for best performance one
    should generally use a static ``scale`` to reduce stochasticity in the
    optimisation.

    """

    def __init__(self, scale: float = 1, learnable: bool = True):
        super().__init__()

        self.scale = torch.Tensor([scale])
        if learnable:
            self.scale = nn.Parameter(self.scale)

    def forward(self, v_in: torch.Tensor, log_density: torch.Tensor, *args):
        """Forward pass of the global rescaling layer."""
        v_out = self.scale * v_in
        log_density -= v_out.shape[-1] * torch.log(self.scale)
        return v_out, log_density


class Sequential(nn.Sequential):
    """Similar to :py:class:`torch.nn.Sequential` except conforms to our
    ``forward`` convention.
    """

    def forward(self, v: torch.Tensor, log_density: torch.Tensor, *args):
        """overrides the base class ``forward`` method to conform to our
        conventioned for expected inputs/outputs of ``forward`` methods.
        """
        for module in self:
            v, log_density = module(v, log_density, *args)
        return v, log_density


class GaussToFreeField(nn.Module):
    r"""Transform Gaussian latent variables into non-interacting field configurations.

    The inputs of the transformation are assumed to be vectors of Gaussian variates,
    i.e. drawn from a univariate Gaussian distribution with null mean and unit variance.
    These variates represent the :math:`|\Lambda| = L^2` degrees of freedom in a real
    scalar field configuration.

    The degrees of freedom are first **assigned** to the real and imaginary parts of
    a complex, Hermitean field configuration. Since the inputs are all drawn from the
    same distribution, this assignment is arbitrary. The Hermitean condition is

    .. math:

        \tilde\phi(-k) = \tilde\phi(k)^*

    The degrees of freedom of the Hermitean field configuration are then rescaled such
    that they correspond to momentum states of a free theory with bare mass
    :math:`m^2`. Through this rescaling we effectively impose a geometry on the
    configurations by assigning momenta to the degrees of freedom.

    Finally, the Hermitean configuration is inverse-Fourier transformed to produce a
    field configuration in real space. This two-dimensional field is flattened and
    split by the red-black checkerboard partitioning.

    See Also
    --------
    :py:mod:`anvil.free_scalar`
    """

    def __init__(self, geometry, m_sq=None):
        super().__init__()
        self.geometry = geometry
        self.length = geometry.length
        self.volume = geometry.volume

        self.scale = torch.roll(
            (FreeScalar(geometry, m_sq).variances.sqrt().unsqueeze(dim=0)),
            (-geometry.length // 2 + 1, -self.geometry.length // 2 + 1),
            (-2, -1),
        )

    @cached_property
    def _rth_select(self):
        """Returns two tensors whose role is to select degrees of freedom from the
        input tensor of univariate Gaussians and assign them to the real and imaginary
        parts of a two-dimensional Hermitean field configuration.

        Let 0 be the index corresponding to zero momentum, and L/2 label the largest
        magnitude momentum (Nyquist frequency). Then, a[i, j] = a[-i, -j] implies
        that the real parts of phi satisfy the Hermitean condition, and b[i, j] =
        -b[-i, -j], along with b[0, 0] = b[0, L/2] = b[L/2, 0] = b[L/2, L/2] = 0
        takes care of the imaginary
        """
        n = self.length // 2  # nyquist term

        init_val = -1
        select_real = torch.full(
            size=(self.length, self.length), fill_value=init_val, dtype=torch.long
        )
        select_imag = torch.full(
            size=(self.length, self.length), fill_value=init_val, dtype=torch.long
        )

        counter = 0  # counts the input degrees of freedom
        for i, j in itertools.product(range(self.length), range(self.length)):
            if select_real[i, j] == init_val:
                select_real[i, j] = counter
                select_real[-i, -j] = counter
                counter += 1

        for i, j in itertools.product(range(self.length), range(self.length)):
            if select_imag[i, j] == init_val:
                # Skip 0 and nyquist terms for imag
                if (i, j) not in ((0, 0), (0, n), (n, 0), (n, n)):
                    select_imag[i, j] = counter
                    select_imag[-i, -j] = counter  # must negate these elements!
                    counter += 1

        assert counter == self.length ** 2

        return select_real, select_imag

    @cached_property
    def _negate_imag(self):
        """Completes the mapping of the input Gaussian variates to a Hermitean
        field configuration by negating the imaginary components with negative
        momentum, so as to satisfy b[i, j] = -b[-i, -j]."""
        n = self.length // 2  # nyquist term
        neg = torch.ones((self.length, self.length)).to(torch.long)
        neg[n + 1 :, :] = -1
        neg[0, n + 1 :] = -1
        neg[n, n + 1 :] = -1
        return neg.unsqueeze(dim=0)

    @cached_property
    def _real_mode_mask(self):
        """Boolean mask that selects the purely real components of the Hermitean
        field configuration, which are assumed to be indexed by 0 and L/2."""
        n = self.length // 2  # nyquist term
        mask = torch.zeros((self.length, self.length)).bool()
        mask[0, 0] = mask[0, n] = mask[n, 0] = mask[n, n] = True
        return mask

    def real_to_hermitean(self, v: torch.Tensor):
        """Takes a sample of Gaussian inputs and maps them to complex-valued Hermitean
        states representing real scalar field configurations in momentum space.

        Parameters
        ----------
        v: torch.Tensor
            Sample of univariate inputs distributed according to a Gaussian with
            null mean and unit variance.

        Returns
        -------
        torch.Tensor
            Complex tensor containing the field configurations in momentum space,
            shape ``(sample_size, L, L)`` where ``L`` is the lattice length.
        """
        select_real, select_imag = self._rth_select
        real_part = v[:, select_real]
        imag_part = v[:, select_imag] * self._negate_imag
        imag_part[:, self._real_mode_mask] = 0
        return torch.complex(real_part, imag_part)

    def forward(self, v: torch.Tensor, log_density: torch.Tensor, *args):
        """Maps a sample of Gaussian inputs to configurations of real scalar fields
        in real space.

        The log-density is merely shifted due to the rescaling. The Fourier kernel
        is unitary and thus volume-preserving.

        Parameters
        ----------
        v: torch.Tensor
            Sample of univariate inputs distributed according to a Gaussian with
            null mean and unit variance.
        log_density: torch.Tensor
            Logarithm of the probability density corresponding to the Gaussian inputs.

        Returns
        -------
        tuple
            Tuple of two tensors. The first is the sample of field configurations in
            in the flat-split representation, shape ``(sample_size, lattice_size)``.
            The second is the log-density, which should match the action of the
            configurations up to a constant shift that is the same for all configs.
        """

        # Map to Hermitean matrix representing momenta of real scalar field
        v_herm = self.real_to_hermitean(v)

        # Rescale according to the mass-squared of the theory
        v_herm *= self.scale

        # NOTE: uncomment this if commenting out the line above to 'turn off' the
        # rescaling as an experiment. The variance of real/imag components of complex
        # modes still must be 1/2 that of real to get consistency in tests.
        # v_herm[:, ~self._real_mode_mask] /= sqrt(2)  # from math import sqrt...

        # Inverse Fourier transform. Result should be real
        ift = torch.fft.ifft2(v_herm, norm="backward")

        assert torch.all(ift.imag.abs() < 1e-9)

        # Convert to split representation so action makes sense!
        ift_split = ift.real.view(-1, self.volume)[:, self.geometry.lexisplit]

        return ift_split, log_density
