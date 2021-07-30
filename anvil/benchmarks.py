# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
benchmarks.py

Module containing benchmarking functions which compare a NVP trained on free
theory to theoretical values. Largely used to check that the anvil machinery
is working correctly.
    
Notes
-----
See the docstring for anvil.free_scalar.FreeScalarEigenmodes for an explanation
of the theoretical predictions and how to match them to quantities derived from
a sample of generated field configurations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from reportengine.figure import figure
from reportengine.table import table

import anvil.free_scalar
from anvil.checks import check_trained_with_free_theory


@check_trained_with_free_theory
def free_scalar_theory(
    training_target_dist, training_geometry
) -> anvil.free_scalar.FreeScalarMomentumSpace:
    """Returns instance of FreeScalarMomentumSpace with specific mass and lattice size."""
    # load target and extract m_sq from target
    m_sq = training_target_dist.c_quadratic * 2 - 4
    return anvil.free_scalar.FreeScalarMomentumSpace(
        m_sq=m_sq, geometry=training_geometry
    )


def fourier_transform(configs: torch.Tensor, training_geometry) -> torch.Tensor:
    """Takes the Fourier transform of a sample of field configurations.

    Parameters
    ----------
    configs
        A (hopefully decorrelated) sample of field configurations in the
        split representation. Shape: (sample_size, lattice_size)
    training_geometry
        The geometry object corresponding to the lattice.

    Returns
    -------
    torch.Tensor
        The Fourier transform of the sample in the Cartesian representation.
        Defined such that the momenta increase monotonically with the index
        on each axis.
    """
    L = training_geometry.length

    x, y = torch.meshgrid(torch.arange(L), torch.arange(L))
    x_split = torch.cat(
        (x[training_geometry.checkerboard], x[~training_geometry.checkerboard])
    )
    y_split = torch.cat(
        (y[training_geometry.checkerboard], y[~training_geometry.checkerboard])
    )

    # Put the sample back in Cartesian form
    phi = torch.empty_like(configs).view(-1, L, L)
    phi[:, x_split, y_split] = configs

    phi_tilde = torch.fft.fft2(phi).roll(
        (L // 2 - 1, L // 2 - 1), (1, 2)
    )  # so we have monotonically increasing momenta on each axis

    return phi_tilde


def eigvals_from_sample(
    fourier_transform: torch.Tensor, training_geometry
) -> np.ndarray:
    """Returns a prediction for the eigenvalues of the kinetic operator.

    The prediction is based on the sample variance of the field configurations in
    Fourier space.

    Parameters
    ----------
    fourier_transform
        Sample of field configurations in Fourier space.
    training_geometry
        Geometry object corresponding to the lattice.

    Returns
    -------
    numpy.ndarray
        An array of dimensions (L, L) containing the eigenvalues.
    """
    variance = fourier_transform.real.var(dim=0) + fourier_transform.imag.var(dim=0)
    eigvals = training_geometry.length ** 2 * torch.reciprocal(variance)
    return eigvals


@table
def table_real_space_variance(configs, free_scalar_theory):
    """Compare the sample variance of the generated configurations with the
    theoretical prediction based on the free scalar theory.

    Due to translational invariance, the field at each lattice site follows the
    same Gaussian distribution. We therefore compare the lattice-average of the
    sample variance with the theory prediction.
    """
    predic = np.reciprocal(free_scalar_theory.eigenvalues).mean()
    sample_var = configs.var(dim=0)
    pc_diff = (sample_var.mean() - predic) / predic * 100
    data = [
        [
            float(predic),
            rf"{float(sample_var.mean()):.4g} $\pm$ {float(sample_var.std()):.1g}",
            float(pc_diff),
        ]
    ]
    df = pd.DataFrame(
        data, columns=["Theory prediction", "Sample variance", "Percent deviation"]
    )
    return df


@table
def table_kinetic_eigenvalues(eigvals_from_sample, free_scalar_theory):
    """Compare the eigenvalues of the kinetic operator inferrered from the
    sample of generated configurations with the theoretical predictions based
    on the free scalar theory.
    """
    pc_diff = (
        (eigvals_from_sample - free_scalar_theory.eigenvalues)
        / free_scalar_theory.eigenvalues
        * 100
    )
    data = [
        [float(vt), float(vs), float(vd)]
        for vt, vs, vd in zip(
            free_scalar_theory.eigenvalues.flatten(),
            eigvals_from_sample.flatten(),
            pc_diff.flatten(),
        )
    ]
    df = pd.DataFrame(data, columns=["Theory", "Sample", "Percent deviation"])
    return df


@figure
def plot_kinetic_eigenvalues(eigvals_from_sample, free_scalar_theory):
    """Plot the eigenvalues of the kinetic operator inferred from the sample
    of generated field configurations with the theoretical predictions based
    on the free scalar theory.

    The plot is a two-dimensional heatmap with the momentum monotonically
    increasing, from maximum negative to maximum positive, from the top
    left to the bottom right corners.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax1.set_title(r"$\lambda$ estimate from sample")
    ax2.set_title("Percent deviation from theory")
    ax1.set_xlabel("$p_1$")
    ax1.set_ylabel("$p_2$")
    ax2.set_xlabel("$p_1$")
    ax2.set_ylabel("$p_2$")

    pc_diff = (
        (eigvals_from_sample - free_scalar_theory.eigenvalues)
        / free_scalar_theory.eigenvalues
        * 100
    )

    im1 = ax1.imshow(eigvals_from_sample)
    im2 = ax2.imshow(pc_diff)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    return fig
