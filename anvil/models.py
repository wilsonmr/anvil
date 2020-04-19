# -*- coding: utf-8 -*-
r"""
models.py

Module containing the base classes for affine layers and full normalising flow
models used to transform a simple distribution into an estimate of a target
distribution as per https://arxiv.org/abs/1904.12072

Classes
-------
RealNVP: nn.Module
    Model which performs a real-valued non-volume preserving (real NVP)
    transformation, which maps a simple distribution z^n to a complicated
    distribution \phi^n, where n refers to the dimensionality of the data.

"""
from math import sqrt, pi, log

import torch
import torch.nn as nn

import numpy as np


class AffineLayer(nn.Module):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    Affine transformation, z = g_i(\phi), defined as:

        z_a = \phi_a
        z_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    where D-dimensional phi has been split into two D/2-dimensional pieces
    \phi_a and \phi_b. s_i and t_i are neural networks, whose parameters are
    torch.nn parameters of this class.

    In the case of this class the partitions are expected to
    be the first and second part of the input data, an additional transformation
    will therefore need to be applied to the output/input of this network if
    the desired partition is to have a different pattern, for example a
    checkerboard transformation.

    Input and output data are flat vectors stacked in the first dimension
    (batch dimension).

    Parameters
    ----------
    size_in: int
        number of dimensions, D, of input/output data. Data should be fed to
        network in shape (sample_size, size_in).
    s_hidden_shape: tuple
        tuple which gives the number of nodes in the hidden layers of neural
        network s_i, can be a single layer network with 16 nodes e.g (16,)
    t_hidden_shape: tuple
        tuple which gives the number of nodes in the hidden layers of neural
        network t_i.
    i_affine: int
        index of this affine layer in full set of affine transformations,
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.

    Attributes
    ----------
    s_layers: torch.nn.ModuleList
        the dense layers of network s, values are intialised as per the
        default initialisation of `nn.Linear`
    t_layers: torch.nn.ModuleList
        the dense layers of network t, values are intialised as per the
        default initialisation of `nn.Linear`

    Methods
    -------
    coupling_layer(phi_input):
        performs the transformation of a single coupling layer, denoted
        g_i(\phi)
    inverse_coupling_layer(z_input):
        performs the transformation of the inverse coupling layer, denoted
        g_i^{-1}(z)
    det_jacobian(phi_input):
        returns the contribution to the determinant of the jacobian
        = \frac{\partial g(\phi)}{\partial \phi}

    """

    def __init__(
        self, size_in: int, s_hidden_shape: tuple, t_hidden_shape: tuple, i_affine: int
    ):
        super(AffineLayer, self).__init__()
        size_half = int(size_in / 2)
        s_shape = [size_half, *s_hidden_shape, size_half]
        t_shape = [size_half, *t_hidden_shape, size_half]

        self.s_layers = nn.ModuleList(
            [
                self._block(s_in, s_out)
                for s_in, s_out in zip(s_shape[:-2], s_shape[1:-1])
            ]
        )
        self.t_layers = nn.ModuleList(
            [
                self._block(t_in, t_out)
                for t_in, t_out in zip(t_shape[:-2], t_shape[1:-1])
            ]
        )
        # No ReLU on final layers: need to be able to scale data by
        # 0 < s, not 1 < s, and enact both +/- shifts
        self.s_layers += [nn.Linear(s_shape[-2], s_shape[-1]), nn.Tanh()]
        self.t_layers += [
            nn.Linear(t_shape[-2], t_shape[-1]),
        ]

        if (i_affine % 2) == 0:  # starts at zero
            # a is first half of input vector
            self._a_ind = slice(0, int(size_half))
            self._b_ind = slice(int(size_half), size_in)
            self.join_func = torch.cat
        else:
            # a is second half of input vector
            self._a_ind = slice(int(size_half), size_in)
            self._b_ind = slice(0, int(size_half))
            self.join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )

    def _block(self, f_in, f_out):
        """Defines a single block within the neural networks.

        Currently hard coded to be a dense layed followed by a leaky ReLU,
        but could potentially specify in runcard.
        """
        return nn.Sequential(nn.Linear(f_in, f_out), nn.Tanh(),)

    def _s_forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Internal method which performs the forward pass of the network
        s.

        Input data x_input should be a torch tensor of size D, with the
        appropriate mask_mat applied such that elements corresponding to
        partition b are set to zero

        """
        for s_layer in self.s_layers:
            x_input = s_layer(x_input)
        return x_input

    def _t_forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Internal method which performs the forward pass of the network
        t.

        Input data x_input should be a torch tensor of size D, with the
        appropriate mask_mat applied such that elements corresponding to
        partition b are set to zero

        """
        for t_layer in self.t_layers:
            x_input = t_layer(x_input)
        return x_input

    def forward(self, z_input) -> torch.Tensor:
        r"""performs the transformation of the inverse coupling layer, denoted
        g_i^{-1}(z)

        inverse transformation, \phi = g_i^{-1}(z), defined as:

        \phi_a = z_a
        \phi_b = (z_b - t_i(z_a)) * exp(-s_i(z_a))

        see eq. (10) of https://arxiv.org/pdf/1904.12072.pdf
            
        Also computes the logarithm of the jacobian determinant for the
        forward transformation (inverse of the above), which is equal to
        the logarithm of 

        \frac{\partial g(\phi)}{\partial \phi} = prod_j exp(s_i(\phi)_j)

        Parameters
        ----------
        z_input: torch.Tensor
            stack of vectors z, shape (sample_size, D)

        Returns
        -------
            out: torch.Tensor
                stack of transformed vectors phi, with same shape as input
            log_det_jacobian: torch.Tensor
                logarithm of the jacobian determinant for the inverse of the
                transformation applied here.
        """
        z_a = z_input[:, self._a_ind]
        z_b = z_input[:, self._b_ind]
        s_out = self._s_forward(z_a)
        t_out = self._t_forward(z_a)
        phi_b = (z_b - t_out) * torch.exp(-s_out)
        return self.join_func([z_a, phi_b], dim=1), s_out.sum(dim=1, keepdim=True)


class RealNVP(nn.Module):
    r"""Extension to nn.Module which is built up of multiple `AffineLayer`s
    as per eq. (12) of https://arxiv.org/abs/1904.12072.

    Each affine layer transforms half of the input vector and the half of the
    input vector which is transformed is alternated. It is therefore recommended
    to have an even number of affine layers. Each affine layer has it's own
    pair of neural networks, s and t, which have seperate sets of parameters
    from other affine layers.

    For now each affine layer has networks with the same architecture
    for simplicity however this could be extended if required.

    Parameters
    ----------
    size_in: int
        number of units defining a single field configuration. The size of
        the second dimension of the input data.
    n_affine: int
        number of affine layers, it is recommended to choose an even number
    affine_hidden_shape: tuple
        tuple defining the number of nodes in the hidden layers of s and t.

    Attributes
    ----------
    affine_layers: torch.nn.ModuleList
        list of affine layers that form the full transformation

    """

    def __init__(
        self, *, size_in, n_affine: int = 2, affine_hidden_shape: tuple = (16,)
    ):
        super(RealNVP, self).__init__()

        self.affine_layers = nn.ModuleList(
            [
                AffineLayer(size_in, affine_hidden_shape, affine_hidden_shape, i)
                for i in range(n_affine)
            ]
        )

    def map(self, x_input: torch.Tensor):
        """Function that maps field configuration to simple distribution"""
        raise NotImplementedError

    def forward(self, z_input: torch.Tensor) -> torch.Tensor:
        r"""Function which maps simple distribution, z, to target distribution
        \phi, and at the same time calculates the density of the output
        distribution using the change of variables formula, according to 
        eq. (8) of https://arxiv.org/pdf/1904.12072.pdf.

        Parameters
        ----------
        z_input: torch.Tensor
            stack of simple distribution state vectors, shape (sample_size, D)

        Returns
        -------
        phi_out: torch.Tensor
            stack of transformed states, which are drawn from an approximation
            of the target distribution, same shape as input.
        log_density: torch.Tensor
            logarithm of the probability density of the output distribution,
            with shape (sample_size, 1)
        """
        log_density = torch.zeros((z_input.shape[0], 1))
        phi_out = z_input

        for i, layer in enumerate(reversed(self.affine_layers)):  # reverse layers!
            phi_out, log_det_jacob = layer(phi_out)
            log_density += log_det_jacob
            # TODO: make this yield, then make a yield from wrapper?
            #if not phi_out.requires_grad:
            #    np.savetxt(f"layer_{i}.txt", phi_out)

        return phi_out, log_density


class ProjectCircle(nn.Module):
    def __init__(self, *, inner_flow, size_in):
        super().__init__()
        self.inner_flow = inner_flow
        self.size_in = size_in

    def forward(self, z_input: torch.Tensor) -> torch.Tensor:

        # Projection
        phi_out = torch.tan(0.5 * (z_input - pi))
        log_density_proj = (-torch.log1p(phi_out ** 2)).sum(dim=1, keepdim=True)

        # Inner flow on real line e.g. RealNVP
        phi_out, log_density_inner = self.inner_flow(phi_out)

        # Inverse projection
        phi_out = 2 * torch.atan(phi_out) + pi
        log_density_proj += (-2 * torch.log(torch.cos(0.5 * (phi_out - pi)))).sum(
            dim=1, keepdim=True
        )

        return phi_out, log_density_proj + log_density_inner


class ProjectSphere(nn.Module):
    def __init__(self, *, inner_flow, size_in):
        super().__init__()
        self.inner_flow = inner_flow
        self.size_in = size_in

    def forward(self, z_input: torch.Tensor) -> torch.Tensor:
        polar, azimuth = z_input.view(-1, self.size_in // 2, 2).split(1, dim=2)

        # Projection
        # -1 factor because polar coordinate = azimuth - pi (*-1 is faster than shift)
        x_coords = -torch.tan(0.5 * polar) * torch.cat(  # radial coordinate
            (torch.cos(azimuth), torch.sin(azimuth)), dim=2
        )
        rad_sq = x_coords.pow(2).sum(dim=2)
        log_density_proj = (-0.5 * torch.log(rad_sq) - torch.log1p(rad_sq)).sum(
            dim=1, keepdim=True
        )

        # Inner flow on real plane e.g. RealNVP
        x_coords, log_density_inner = self.inner_flow(x_coords.view(-1, self.size_in))
        x_1, x_2 = x_coords.view(-1, self.size_in // 2, 2).split(1, dim=2)

        # Inverse projection
        polar = 2 * torch.atan(torch.sqrt(x_1.pow(2) + x_2.pow(2)))
        phi_out = torch.cat((polar, torch.atan2(x_2, x_1) + pi,), dim=2)  # azimuth
        log_density_proj += (
            torch.log(torch.sin(0.5 * polar)) - 3 * torch.log(torch.cos(0.5 * polar))
        ).sum(dim=1)

        return phi_out.view(-1, self.size_in), log_density_proj + log_density_inner


def real_nvp(config_size, n_affine, network_kwargs):
    return RealNVP(size_in=config_size, n_affine=n_affine, **network_kwargs)


def project_circle(real_nvp, config_size):
    return ProjectCircle(inner_flow=real_nvp, size_in=config_size)


def project_sphere(real_nvp, config_size):
    return ProjectSphere(inner_flow=real_nvp, size_in=config_size)


if __name__ == "__main__":
    pass
