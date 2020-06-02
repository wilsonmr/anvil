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

