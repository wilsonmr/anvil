# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
plot.py

module containing all actions for plotting observables

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties

from reportengine.figure import figure, figuregen
from reportengine import collect

from anvil.observables import cosh_shift


def field_component(i, x_base, phi_model, base_neg, model_neg):
    fig, ax = plt.subplots()

    ax.hist(x_base, bins=50, density=True, histtype="step", label="base")
    ax.hist(phi_model, bins=50, density=True, histtype="step", label="model, full")
    ax.hist(
        base_neg, bins=50, density=True, histtype="step", label="model, $M_{base} < 0$"
    )
    ax.hist(
        model_neg, bins=50, density=True, histtype="step", label="model, $M_{mod} < 0$"
    )
    ax.set_title(f"Coordinate {i}")
    ax.legend()
    fig.tight_layout()
    return fig


def field_components(loaded_model, base_dist, lattice_size):
    """Plot the distributions of base coordinates 'x' and output coordinates 'phi' and,
    if known, plot the pdf of the target distribution."""
    sample_size = 10000

    # Generate a large sample from the base distribution and pass it through the trained model
    with torch.no_grad():
        x_base, _ = base_dist(sample_size)
        sign = x_base.sum(dim=1).sign()
        neg = (sign < 0).nonzero().squeeze()
        phi_model, model_log_density = loaded_model(x_base, 0, neg)

    base_neg = phi_model[neg]

    sign = phi_model.sum(dim=1).sign()
    neg = (sign < 0).nonzero().squeeze()
    model_neg = phi_model[neg]

    # Convert to shape (n_coords, sample_size * lattice_size)
    # NOTE: this is all pointless for the 1-component scalar
    x_base = x_base.reshape(sample_size * lattice_size, -1).transpose(0, 1)
    phi_model = phi_model.reshape(sample_size * lattice_size, -1).transpose(0, 1)

    base_neg = base_neg.reshape(1, -1)
    model_neg = model_neg.reshape(1, -1)

    for i in range(x_base.shape[0]):
        yield field_component(i, x_base[i], phi_model[i], base_neg[i], model_neg[i])


_plot_field_components = collect("field_components", ("training_context",))


def example_configs(loaded_model, base_dist, training_geometry):
    sample_size = 10

    # Generate a large sample from the base distribution and pass it through the trained model
    with torch.no_grad():
        x_base, _ = base_dist(sample_size)
        sign = x_base.sum(dim=1).sign()
        neg = (sign < 0).nonzero().squeeze()
        phi_model, model_log_density = loaded_model(x_base, 0, neg)

    L = int(np.sqrt(phi_model.shape[1]))

    phi_true = np.zeros((4, L, L))
    phi_true[:, training_geometry.checkerboard] = phi_model[:4, : L ** 2 // 2]
    phi_true[:, ~training_geometry.checkerboard] = phi_model[:4, L ** 2 // 2 :]

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        conf = ax.imshow(phi_true[i])
        fig.colorbar(conf, ax=ax)

    fig.suptitle("Example configurations")

    return fig


_plot_example_configs = collect("example_configs", ("training_context",))


@figure
def plot_example_configs(_plot_example_configs):
    return _plot_example_configs[0]


@figuregen
def plot_field_components(_plot_field_components):
    yield from _plot_field_components[0]


@figure
def plot_zero_momentum_correlator(
    zero_momentum_correlator,
    training_geometry,
    fit_zero_momentum_correlator,
):
    """Plot zero_momentum_2pf as a function of t. Points are means across bootstrap
    sample and errorbars are standard deviations across boostrap samples
    """
    L = training_geometry.length

    fig, ax = plt.subplots()

    ax.errorbar(
        x=np.arange(L),
        y=zero_momentum_correlator.mean(axis=-1),
        yerr=zero_momentum_correlator.std(axis=-1),
        linestyle="",
        label="sample statistics",
    )

    if fit_zero_momentum_correlator is not None:
        popt, pcov, x0 = fit_zero_momentum_correlator

        x_2 = np.linspace(x0, L - x0, 100)
        ax.plot(
            x_2,
            cosh_shift(x_2 - L // 2, *popt),
            marker="",
            label=r"fit: $A \cosh(-(x_2 - L/2) / \xi) + c$",
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sum_{x_1=0}^{L-1} G(x_1, x_2)$")
    ax.set_xlabel("$x_2$")
    ax.set_title("Correlation of 1-dimensional slices")
    ax.legend()
    return fig


@figure
def plot_effective_pole_mass(training_geometry, effective_pole_mass):
    """Plot effective pole mass as a function of x_2. The points are means
    across bootstrap samples and the errorbars are standard deviation across
    bootstrap.
    """
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(1, training_geometry.length - 1),
        y=effective_pole_mass.mean(axis=-1),
        yerr=effective_pole_mass.std(axis=-1),
    )
    ax.set_ylabel("$m_p^\mathrm{eff}$")
    ax.set_xlabel("$x_2$")
    ax.set_title("Effective pole mass")
    return fig


@figure
def plot_correlation_length(
    effective_pole_mass,
    low_momentum_correlation_length,
    correlation_length_from_fit,
):
    """Plots three estimates of correlation length on the same figure.

    These are:
        1. Estimate from fitting a cosh function to the correlation between
            1-dimensional slices, using py:func:`correlation_length_from_fit`
        2. Reciprocal of the effective pole mass estimator, using
            :py:func:`effective_pole_mass`
        3. Low momentum estimate, using :py:func:`low_momentum_correlation_length`

    2. is evaluated at a specific value of the separation, x_2.
    """
    xi_arcosh = np.reciprocal(effective_pole_mass)

    fig, ax = plt.subplots()

    arcosh_points = ax.errorbar(
        x=range(1, xi_arcosh.shape[0] + 1),
        y=xi_arcosh.mean(axis=-1),
        yerr=xi_arcosh.std(axis=-1),
        zorder=3,
    )

    xi_lm = low_momentum_correlation_length.mean()
    e_lm = low_momentum_correlation_length.std()
    lm_hline = ax.axhline(xi_lm, linestyle="-", marker="", color="grey", zorder=2)
    lm_fill = ax.fill_between(
        ax.get_xlim(),
        xi_lm + e_lm,
        xi_lm - e_lm,
        color="grey",
        alpha=0.3,
        zorder=1,
    )

    xi_fit = correlation_length_from_fit[0]  # .mean(  # TODO: update when bootstrapped)
    e_fit = correlation_length_from_fit[1]  # .std()
    fit_hline = ax.axhline(xi_fit, linestyle="-", marker="", color="orange", zorder=2)
    fit_fill = ax.fill_between(
        ax.get_xlim(),
        xi_fit + e_fit,
        xi_fit - e_fit,
        color="orange",
        alpha=0.3,
        zorder=1,
    )

    ax.legend(
        handles=[arcosh_points, (lm_fill, lm_hline), (fit_fill, fit_hline)],
        labels=["Estimate using arcosh", "Low momentum estimate", "Estimate from fit"],
    )
    return fig


@figure
def plot_two_point_correlator(two_point_correlator):
    """Represent the two point correlator as a heatmap. The data shown is the mean
    of the bootstrap sample of correlation functions, and is normalised so that
    G(0, 0) = 1. The colour axis is scaled using a symmetric log scale, with a linear
    region spanning [-0.01, 0.01].
    """
    corr = two_point_correlator.mean(axis=-1)
    corr /= corr[0, 0]

    L = corr.shape[0]
    corr = np.roll(corr, (-L // 2 - 1, -L // 2 - 1), (0, 1))

    fig, ax = plt.subplots()
    ax.set_title("$G(x)$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    tick_positions = [0, L // 2 - 1, L - 1]
    tick_labels = [r"$-\frac{L + 1}{2}$", 0, r"$\frac{L}{2}$"]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    norm = mpl.colors.SymLogNorm(linthresh=0.01, base=10)
    im = ax.imshow(corr, norm=norm)
    fig.colorbar(im, ax=ax, pad=0.01)

    return fig


@figure
def plot_two_point_correlator_error(two_point_correlator):
    """Heatmap of the error in the two point correlator for each separation (x_1, x_2).
    The error is computed as the standard deviation over the bootstrap sample. The data
    shown is this error divided by the mean of the bootstrap sample, i.e. the
    fractional error.
    """
    corr = two_point_correlator.mean(axis=-1)
    error = two_point_correlator.std(axis=-1)

    L = corr.shape[0]
    corr = np.roll(corr, (-L // 2 - 1, -L // 2 - 1), (0, 1))
    error = np.roll(error, (-L // 2 - 1, -L // 2 - 1), (0, 1))

    fig, ax = plt.subplots()
    ax.set_title(r"$| \sigma_G(x) / G(x) |$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    tick_positions = [0, L // 2 - 1, L - 1]
    tick_labels = [r"$-\frac{L + 1}{2}$", 0, r"$\frac{L}{2}$"]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    im = ax.imshow(np.abs(error / corr))
    fig.colorbar(im, ax=ax, pad=0.01)

    return fig


@figure
def plot_magnetization(magnetization_series):
    fig, ax = plt.subplots()
    ax.set_title("Magnetization")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("$M(t)$")
    ax.hist(magnetization_series, histtype="stepfilled", edgecolor="black")
    return fig


@figure
def plot_magnetization_series(magnetization_series, sample_interval):
    n_rows = 5
    if magnetization_series.size % n_rows != 0:
        magnetization_series = np.pad(
            magnetization_series,
            (0, magnetization_series.size % n_rows),
            "empty",
        )
    magnetization_series = magnetization_series.reshape(n_rows, -1)
    t = (np.arange(magnetization_series.size) * sample_interval).reshape(n_rows, -1)

    fig, axes = plt.subplots(n_rows, 1, sharey=True)
    for ax, x, y in zip(axes, t, magnetization_series):
        ax.plot(x, y, linestyle="-", marker="")
        ax.margins(0, 0)

    axes[0].set_title("Magnetization")
    axes[n_rows // 2].set_ylabel("$M(t)$")
    axes[-1].set_xlabel("$t$")
    fig.tight_layout()
    return fig


@figure
def plot_magnetization_autocorr(
    magnetization_autocorr, magnetization_optimal_window, sample_interval
):
    cut = max(10, 2 * magnetization_optimal_window)
    chain_indices = np.arange(cut) * sample_interval

    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation of magnetization")
    ax.set_ylabel(r"$\Gamma_M(\delta t)$")
    ax.set_xlabel("$\delta t$")

    ax.plot(chain_indices, magnetization_autocorr[:cut])

    ax.set_xlim(left=0)
    ax.set_ylim(top=1)

    ax.axvline(
        magnetization_optimal_window * sample_interval,
        linestyle="-",
        marker="",
        color="k",
        label="Optimal window size",
        zorder=1,
    )
    ax.fill_betweenx(
        ax.get_ylim(),
        magnetization_optimal_window * sample_interval,
        ax.get_xlim()[1],
        color="grey",
        alpha=0.5,
        label="Truncated",
        zorder=0,
    )
    ax.legend()
    return fig


@figure
def plot_magnetization_integrated_autocorr(
    magnetization_integrated_autocorr,
    magnetization_optimal_window,
    sample_interval,
):
    cut = max(10, 2 * np.max(magnetization_optimal_window))
    chain_indices = np.arange(cut) * sample_interval
    tau = magnetization_integrated_autocorr[magnetization_optimal_window]

    fig, ax = plt.subplots()
    ax.set_title("Integrated autocorrelation of magnetization")
    ax.set_ylabel(r"$\tau_\mathrm{int,M}(W)$")
    ax.set_xlabel("$W$")

    ax.plot(chain_indices, magnetization_integrated_autocorr[:cut], zorder=2)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.5)

    scalebar = AnchoredSizeBar(
        ax.transData,
        sample_interval,
        f"sample interval: {sample_interval}",
        "upper left",
        pad=0.6,
        frameon=False,
        sep=4,
        label_top=True,
        fontproperties=FontProperties(size="x-large"),
    )
    ax.add_artist(scalebar)
    ax.axhline(
        tau,
        linestyle="-",
        marker="",
        color="r",
        label=r"$\tau_\mathrm{int,M}(W_\mathrm{opt})$",
        zorder=1,
    )
    ax.axvline(
        magnetization_optimal_window * sample_interval,
        linestyle="-",
        marker="",
        color="k",
        label="Optimal window size",
        zorder=1,
    )
    ax.fill_betweenx(
        ax.get_ylim(),
        magnetization_optimal_window * sample_interval,
        ax.get_xlim()[1],
        color="grey",
        alpha=0.5,
        label="Truncated",
        zorder=0,
    )
    ax.legend()
    return fig


def plot_bootstrap_single_number(observable, label):
    """Given a 1 dimensional tensor of observables, plot a histogram of the
    distribution
    """
    fig, ax = plt.subplots()
    ax.hist(
        observable,
        label=f"mean: {observable.mean()}, std dev.: {observable.std()}",
    )
    title = "Bootstrap distribution: " + label
    ax.set_xlabel(label)
    ax.set_ylabel("Counts")
    ax.legend()
    ax.set_title(title)
    return fig


def plot_bootstrap_multiple_numbers(observable, labels):
    """Given a 2D tensor with boostrap sample on the final axis, yield a figure
    with a histogram for each element's bootstrap sample

    """
    for i in range(observable.shape[0]):
        yield plot_bootstrap_single_number(observable[i], labels[i])


@figure
def plot_bootstrap_susceptibility(susceptibility):
    """plot a bootstrap distribution of the susceptibility"""
    return plot_bootstrap_single_number(susceptibility, r"$\chi$")


@figure
def plot_bootstrap_ising_energy(ising_energy):
    """plot a bootstrap distribution of the ising_energy"""
    return plot_bootstrap_single_number(ising_energy, r"Ising $E$")


@figuregen
def plot_bootstrap_zero_momentum_2pf(zero_momentum_correlator):
    """For each value of t, plot a boostrap distribution of
    zero_momentum_correlator[t]

    """
    labels = [
        r"$\tilde{G}$" + f"$(0,{t})$" for t in range(zero_momentum_correlator.shape[0])
    ]
    yield from plot_bootstrap_multiple_numbers(zero_momentum_correlator, labels)


@figuregen
def plot_bootstrap_effective_pole_mass(effective_pole_mass):
    """For each value of t from 1 to L-2, plot a bootstrap distribution of
    effective_pole_mass[t]

    """
    labels = [r"$m_p^{eff}$" + f"$({t})$" for t in range(effective_pole_mass.shape[0])]
    yield from plot_bootstrap_multiple_numbers(effective_pole_mass, labels)
