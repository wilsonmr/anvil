import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from reportengine.figure import figure, figuregen
from reportengine.table import table

from free_theory import FreeScalarEigenmodes


def theory(training_context, training_geometry):
    """Returns instance of FreeScalarEigenmodes class with specific
    mass and lattice size.
    """
    m_sq = training_context["action"].m_sq
    return FreeScalarEigenmodes(m_sq=m_sq, lattice_length=training_geometry.length)


def fourier_transform(sample_training_output, training_geometry):
    """Takes a decorrelated samples of field configurations and performs
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


@table
def table_variance(sample_training_output, theory):
    predic = np.reciprocal(theory.eigenvalues).mean()
    sample_var = sample_training_output.var(dim=0)
    pc_diff = (sample_var.mean() - predic) / predic * 100
    table = [
        [
            float(predic),
            f"{float(sample_var.mean()):.4g} $\pm$ {float(sample_var.std()):.1g}",
            float(pc_diff),
        ],
    ]
    df = pd.DataFrame(
        table, columns=["Theory prediction", "Sample variance", "Percent deviation"]
    )
    return df


@table
def table_eigenvalues(eigvals_from_sample, theory):
    pc_diff = (eigvals_from_sample - theory.eigenvalues) / theory.eigenvalues * 100
    table = [
        [float(vt), float(vs), float(vd)]
        for vt, vs, vd in zip(
            theory.eigenvalues.flatten(),
            eigvals_from_sample.flatten(),
            pc_diff.flatten(),
        )
    ]
    df = pd.DataFrame(table, columns=["Theory", "Sample", "Percent deviation"])
    return df


@figure
def plot_eigenvalues(eigvals_from_sample, theory):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax1.set_title(r"$\lambda$ estimate from sample")
    ax2.set_title("Percent deviation from theory")
    ax1.set_xlabel("$p_1$")
    ax1.set_ylabel("$p_2$")
    ax2.set_xlabel("$p_1$")
    ax2.set_ylabel("$p_2$")

    pc_diff = (eigvals_from_sample - theory.eigenvalues) / theory.eigenvalues * 100
    extent = [
        theory.momenta[0],
        theory.momenta[-1],
        theory.momenta[-1],
        theory.momenta[0],
    ]

    im1 = ax1.imshow(eigvals_from_sample, extent=extent)
    im2 = ax2.imshow(pc_diff, extent=extent)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    return fig
