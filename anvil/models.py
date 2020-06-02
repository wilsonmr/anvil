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

ProjectCircle: nn.Module
    Model which wraps around Real NVP to enable learning maps between distributions
    defined on the unit circle.

ProjectSphere: nn.Module
    Model which wraps around Real NVP to enable learning maps between distributions
    defined on the unit sphere.
"""
import torch
import torch.nn as nn

from math import pi

class NormalisingFlow(nn.Module):
    def __init__(self, label, layers):
        super().__init__()
        self.label = label
        self.layers = nn.ModuleList([ for layer in layers])

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        r"""Function which maps simple distribution, x ~ r, to target distribution
        \phi ~ p, and at the same time calculates the density of the output
        distribution using the change of variables formula.

        Parameters
        ----------
        x_input: torch.Tensor
            stack of simple distribution state vectors, shape (N_states, config_size)

        Returns
        -------
        phi_out: torch.Tensor
            stack of transformed states, which are drawn from an approximation
            of the target distribution, same shape as input.
        log_density: torch.Tensor
            logarithm of the probability density of the output distribution,
            with shape (n_states, 1)
        """
        phi_out = x_input
        rev_layers = reversed(self.affine_layers)  # reverse layers!
        phi_out, log_density = next(rev_layers)(phi_out)
        for layer in rev_layers:
            phi_out, log_det_jacob = layer(phi_out)
            log_density += log_det_jacob
        return phi_out, log_density


def normalising_flow(i_mixture, config_size, coupling_layers):
    return NormalisingFlow(config_size, coupling_layers)



def stereographic_projection(
    target, config_size, s_networks, t_networks, standardise_inputs=False
):
    """Returns an instance of either ProjectCircle or ProjectSphere, depending on the
    dimensionality of the fields."""
    inner_flow = RealNVP(
        size_in=config_size,
        s_networks=s_networks,
        t_networks=t_networks,
        standardise_inputs=standardise_inputs,
    )
    if target == "o2":
        return ProjectCircle(inner_flow)
    elif target == "o3":
        return ProjectSphere(inner_flow, size_in=config_size)
    # Should raise config error.
    return

