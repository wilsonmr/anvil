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

from reportengine import collect
from reportengine.figure import figure
from reportengine.table import table

from anvil.free_scalar import FreeScalarEigenmodes
from anvil.checks import check_trained_with_free_theory


def free_scalar_theory(couplings, lattice_length):
    """Returns instance of FreeScalarEigenmodes class with specific
    mass and lattice size.
    """
    # TODO: load target and extract m_sq from target.c2 - indep or parameterisation?
    m_sq = couplings["m_sq"]
    return FreeScalarEigenmodes(m_sq=m_sq, lattice_length=lattice_length)


def fourier_transform(sample_training_output, training_geometry):
    """Takes the Fourier transform of a sample of field configurations.

    Inputs
    ------
    sample_training_output: torch.tensor
        A (hopefully decorrelated) sample of field configurations in the
        split representation. Shape: (sample_size, lattice_size)
    training_geometry: geometry object

    Returns
    -------
    phi_tilde: torch.tensor
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
    phi = torch.empty_like(sample_training_output).view(-1, L, L)
    phi[:, x_split, y_split] = sample_training_output

    # TODO: update to new PyTorch version with torch.fft.rfft2
    phi_tilde = torch.rfft(phi, signal_ndim=2, onesided=False).roll(
        (L // 2 - 1, L // 2 - 1), (1, 2)
    )  # so we have monotonically increasing momenta on each axis

    return phi_tilde


def eigvals_from_sample(fourier_transform, training_geometry):
    """Returns a prediction for the eigenvalues of the kinetic operator
    for the free theory, based on the sample variance of the fourier
    transformed fields.

    The output is converted to an (L x L) numpy.ndarray.
    """
    variance = torch.var(fourier_transform, dim=0).sum(dim=-1)  # sum real + imag
    eigvals = training_geometry.length ** 2 * torch.reciprocal(variance)
    return eigvals.numpy()


free_theory_from_training_ = collect("free_scalar_theory", ("training_context",))

# TODO: work out way to not have to do this.. However it allows us to use check
#@check_trained_with_free_theory
def free_theory_from_training(free_theory_from_training_, training_context):
    """Returns free_scalar_theory but with m_sq and lattice_length extracted
    from a training config.

    """
    (res,) = free_theory_from_training_
    return res


@table
def table_real_space_variance(sample_training_output, free_theory_from_training):
    """Compare the sample variance of the generated configurations with the
    theoretical prediction based on the free scalar theory.

    Due to translational invariance, the field at each lattice site follows the
    same Gaussian distribution. We therefore compare the lattice-average of the
    sample variance with the theory prediction.
    """
    predic = np.reciprocal(free_theory_from_training.eigenvalues).mean()
    sample_var = sample_training_output.var(dim=0)
    pc_diff = (sample_var.mean() - predic) / predic * 100
    data = [
        [
            float(predic),
            f"{float(sample_var.mean()):.4g} $\pm$ {float(sample_var.std()):.1g}",
            float(pc_diff),
        ]
    ]
    df = pd.DataFrame(
        data, columns=["Theory prediction", "Sample variance", "Percent deviation"]
    )
    return df


@table
def table_kinetic_eigenvalues(eigvals_from_sample, free_theory_from_training):
    """Compare the eigenvalues of the kinetic operator inferrered from the
    sample of generated configurations with the theoretical predictions based 
    on the free scalar theory.
    """
    pc_diff = (
        (eigvals_from_sample - free_theory_from_training.eigenvalues)
        / free_theory_from_training.eigenvalues
        * 100
    )
    data = [
        [float(vt), float(vs), float(vd)]
        for vt, vs, vd in zip(
            free_theory_from_training.eigenvalues.flatten(),
            eigvals_from_sample.flatten(),
            pc_diff.flatten(),
        )
    ]
    df = pd.DataFrame(data, columns=["Theory", "Sample", "Percent deviation"])
    return df


@figure
def plot_kinetic_eigenvalues(eigvals_from_sample, free_theory_from_training):
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
        (eigvals_from_sample - free_theory_from_training.eigenvalues)
        / free_theory_from_training.eigenvalues
        * 100
    )
    extent = [
        free_theory_from_training.momenta[0],
        free_theory_from_training.momenta[-1],
        free_theory_from_training.momenta[-1],
        free_theory_from_training.momenta[0],
    ]

    im1 = ax1.imshow(eigvals_from_sample, extent=extent)
    im2 = ax2.imshow(pc_diff, extent=extent)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    return fig
