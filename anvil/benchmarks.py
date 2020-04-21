"""
benchmarks.py

Module containing benchmarking functions which compare a NVP trained on free
theory to theoretical values. Largely used to check that the anvil machinery
is working correctly.

"""

import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from reportengine import collect
from reportengine.figure import figure
from reportengine.table import table

from anvil.free_theory import FreeScalarEigenmodes
from anvil.checks import check_trained_with_free_theory


def free_scalar_theory(m_sq, lattice_length):
    """Returns instance of FreeScalarEigenmodes class with specific
    mass and lattice size.
    """
    return FreeScalarEigenmodes(m_sq=m_sq, lattice_length=lattice_length)


def fourier_transform(sample_training_output, training_geometry):
    """Takes a decorrelated sample of field configurations and performs
    a Fourier transform.
    """
    L = training_geometry.length

    x, y = torch.meshgrid(torch.arange(L), torch.arange(L))
    x_split = torch.cat(
        (x[training_geometry.checkerboard], x[~training_geometry.checkerboard])
    )
    y_split = torch.cat(
        (y[training_geometry.checkerboard], y[~training_geometry.checkerboard])
    )

    phi = torch.empty_like(sample_training_output).view(-1, L, L)
    phi[:, x_split, y_split] = sample_training_output

    phi_tilde = torch.rfft(phi, signal_ndim=2, onesided=False).roll(
        (L // 2 - 1, L // 2 - 1), (1, 2)
    )  # so we have monotonically increasing momenta on each axis

    return phi_tilde


def eigvals_from_sample(fourier_transform, training_geometry):
    """Returns a prediction for the eigenvalues of the kinetic operator
    for the free theory, based on the sample variance of the fourier
    transformed fields.

    Converts output to numpy.ndarray.
    """
    variance = torch.var(fourier_transform, dim=0).sum(dim=-1)  # sum real + imag
    eigvals = training_geometry.length ** 2 * torch.reciprocal(variance)
    return eigvals.numpy()


free_theory_from_training_ = collect("free_scalar_theory", ("training_context",))

# TODO: work out way to not have to do this.. However it allows us to use check
@check_trained_with_free_theory
def free_theory_from_training(free_theory_from_training_, training_context):
    """Returns free_scalar_theory but with m_sq and lattice_length extracted
    from a training config.

    """
    res, = free_theory_from_training_
    return res


@table
def table_variance(sample_training_output, free_theory_from_training):
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
def table_eigenvalues(eigvals_from_sample, free_theory_from_training):
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
def plot_eigenvalues(eigvals_from_sample, free_theory_from_training):

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
