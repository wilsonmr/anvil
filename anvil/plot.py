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


@figure
def plot_zero_momentum_correlator(
    zero_momentum_correlator,
    training_geometry,
    fit_zero_momentum_correlator,
    cosh_fit_window: slice,
    plot_cosh_fit: bool = True,
):
    r"""Plots the correlation function for pairs of one-dimensional 'slices', otherwise
    referred to as the two point correlator at zero spatial momentum, as a function of
    time.

    Points and errorbars are means and standard deviations across a boostrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Optionally plots a :math:`1\sigma` confidence interval for a pure-exponential (cosh)
    fit performed for each member of the bootstrap sample in
    :py:func:`fit_zero_momentum_correlator`.

    Parameters
    ---------
    zero_momentum_correlator
        Array containing bootstrapped correlation function
    training_geometry
        Geometry object defining the lattice
    fit_zero_momentum_correlator
        The parameters resulting from a least-squares fit of a cosh function to the
        correlator
    cosh_fit_window
        Slice object which indexes the lattice separations that were used to perform
        the cosh fit to the correlation function
    plot_cosh_fit
        If False, only plot the correlation function, and not the result of the fit

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :py:func:`anvil.table.table_zero_momentum_correlator`
    """
    fig, ax = plt.subplots()

    ax.errorbar(
        x=np.arange(training_geometry.length),
        y=zero_momentum_correlator.mean(axis=-1),
        yerr=zero_momentum_correlator.std(axis=-1),
        linestyle="",
        zorder=2,
        label="sample statistics",
    )

    if plot_cosh_fit:
        t = np.arange(training_geometry.length)[cosh_fit_window]
        fit = []
        for xi, A, c in zip(*fit_zero_momentum_correlator):
            fit.append(
                cosh_shift(t - training_geometry.length // 2, xi, A, c),
            )
        fit = np.array(fit).T  # (n_points, n_boot)

        ax.fill_between(
            t,
            fit.mean(axis=1) - fit.std(axis=1),
            fit.mean(axis=1) + fit.std(axis=1),
            color="orange",
            alpha=0.3,
            zorder=1,
            label=r"fit: $A \cosh(-(x_2 - L/2) / \xi) + c$"
            + "\n"
            + r"($1\sigma$ confidence)",
        )

    ax.set_yscale("log")
    ax.set_ylabel(r"$\sum_{x_1=0}^{L-1} G(x_1, x_2)$")
    ax.set_xlabel("$x_2$")
    ax.set_title("Correlation of 1-dimensional slices")
    ax.legend()
    return fig


@figure
def plot_effective_pole_mass(training_geometry, effective_pole_mass):
    r"""Plots the (effective) pole mass as a function of 'time' separation.
    
    Points and errorbars are means and standard deviations across a boostrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Parameters
    ----------
    training_geometry
        Geometry object defining the lattice.
    effective_pole_mass
        Array containing bootstrap ensemble of effective pole mass, for each
        separation :math:`t = 1, \ldots, T - 1`

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :py:func:`anvil.table.table_effective_pole_mass`
    """
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(1, training_geometry.length - 1),
        y=effective_pole_mass.mean(axis=-1),
        yerr=effective_pole_mass.std(axis=-1),
    )
    ax.set_ylabel(r"$m_p^\mathrm{eff}$")
    ax.set_xlabel("$x_2$")
    ax.set_title("Effective pole mass")
    return fig


@figure
def plot_correlation_length(
    effective_pole_mass,
    low_momentum_correlation_length,
    correlation_length_from_fit,
):
    r"""Plots three estimates of correlation length.
    
    These are:
        1. Estimate from fitting a cosh function to the correlation between
            1-dimensional slices, using py:func:`correlation_length_from_fit`
        2. Reciprocal of the effective pole mass estimator, using
            :py:func:`effective_pole_mass` (evaluated at each separation, :math:`x_2`. 
        3. Low momentum estimate, using :py:func:`low_momentum_correlation_length`

    Points and errorbars are means and standard deviations across a boostrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.
    
    Parameters
    ----------
    effective_pole_mass
        Array containing estimate of the effective pole mass, for each separation
        and each member of the bootstrap ensemble
    low_momentum_correlation_length
        Array containing a low-momentum estimate of the correlation length for
        each member of the bootstrap ensemble.
    correlation_length_from_fit
        Array containing an estimate of the correlation length from a cosh fit
        to the correlation function, for each member of the bootstrap
        ensemble.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :py:func:`anvil.observables.fit_zero_momentum_correlator`
    :py:func:`anvil.table.table_correlation_length`
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

    xi_fit = correlation_length_from_fit.mean()
    e_fit = correlation_length_from_fit.std()
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
    r"""Represents the two point correlator as a heatmap.

    The data shown is the mean of a bootstrap sample of correlation functions, and is
    normalised so that :math:`G(0, 0) = 1`. The colour axis is scaled using a symmetric
    log scale, with a linear region spanning :math:`[-0.01, 0.01]`.
    
    The bootstrap dimension is assumed to be the last (``-1``) dimension of input arrays.
    
    Parameters
    ----------
    two_point_correlator
        Array containing two point correlation function for each two-dimensional
        separation :math:`(x_1, x_2)`, for each member of a bootstrap ensemble.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :py:func:`anvil.plot.plot_two_point_correlator_error`
    :py:func:`anvil.table.table_two_point_correlator`
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
    r"""Heatmap of the error in the two point correlator for each separation.

    The error is computed as the standard deviation over the bootstrap sample. The
    data shown is this error divided by the mean of the bootstrap sample, i.e. the
    fractional error.
    
    The bootstrap dimension is assumed to be the last (``-1``) dimension of input arrays.
    
    Parameters
    ----------
    two_point_correlator
        Array containing two point correlation function for each two-dimensional
        separation :math:`(x_1, x_2)`, for each member of a bootstrap ensemble.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    :py:func:`anvil.plot.plot_two_point_correlator`
    :py:func:`anvil.table.table_two_point_correlator`
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
    r"""Plots a histogram of the magnetization of each configuration in the Markov
    chain resulting from the Metropolis-Hastings sampling phase.

    Parameters
    ----------
    magnetization_series
        Array containing the magnetization for each configuration in the output
        sample from the Metropolis-Hastings sampling phase.

    Returns
    -------
    matplotlib.figure.Figure

    See also
    --------
    :py:func:`anvil.plot.plot_magnetization_series`
    :py:func:`anvil.table.table_magnetization`
    """
    fig, ax = plt.subplots()
    ax.set_title("Magnetization")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("$M(t)$")
    ax.hist(magnetization_series, histtype="stepfilled", edgecolor="black")
    return fig


@figure
def plot_magnetization_series(magnetization_series, sample_interval):
    r"""Plots the magnetization of each configuration in the Markov chain over the
    course of the Metropolis-Hastings sampling phase.
    
    Parameters
    ----------
    magnetization_series
        Array containing the magnetization for each configuration in the output
        sample from the Metropolis-Hastings sampling phase.
    sample_interval
        The number of Metropolis updates which were discarded between each
        configuration appearing in the input series.

    Returns
    -------
    matplotlib.figure.Figure

    See also
    --------
    :py:func:`anvil.plot.plot_magnetization`.
    :py:func:`anvil.table.table_magnetization`
    """
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
        magnetization_autocorr, magnetization_optimal_window: int, sample_interval: int
):
    r"""Plots the autocorrelation function for the magnetization of the sequence of
    configurations generated in the Metropolis-Hastings sampling phase.

    The x-axis corresponds to a number of steps separating pairs of configurations
    in the sequence.
    
    Parameters
    ----------
    magnetization_autocorr
        Array containing the autocorrelation function of the magnetization for each
        configuration in the output sample from the Metropolis-Hastings sampling phase.
    magnetization_optimal_window
        The size of the window in which the integrated autocorrelation time is to be
        computed such that the total error is minimized.
    sample_interval
        The number of Metropolis updates which were discarded between each
        configuration appearing in the input series.

    Returns
    -------
    matplotlib.figure.Figure

    See also
    --------
    :py:func:`anvil.observables.optimal_window`
    :py:func:`anvil.plot.plot_magnetization_integrated_autocorr`.
    """
    cut = max(10, 2 * magnetization_optimal_window)
    chain_indices = np.arange(cut) * sample_interval

    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation of magnetization")
    ax.set_ylabel(r"$\Gamma_M(\delta t)$")
    ax.set_xlabel(r"$\delta t$")

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
    magnetization_optimal_window: int,
    sample_interval: int,
):
    r"""Plots the integrated autocorrelation function for the magnetization of the
    sequence of configurations generated in the Metropolis-Hastings sampling phase.
    
    The x axis represents the size of the 'window' in which the summation is performed,
    i.e. the point at which the autocorrelation function is truncated.
    
    Parameters
    ----------
    magnetization_integrated_autocorr
        Array containing the cumulative sum of the autocorrelation function of the
        magnetization for each configuration in the output sample from the Metropolis-
        Hastings sampling phase.
    magnetization_optimal_window
        The size of the window in which the integrated autocorrelation time is to be
        computed such that the total error is minimized.
    sample_interval
        The number of Metropolis updates which were discarded between each
        configuration appearing in the input series.

    Returns
    -------
    matplotlib.figure.Figure

    See also
    --------
    :py:func:`anvil.observables.optimal_window`
    :py:func:`anvil.plot.plot_magnetization_autocorr`.
    """
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
        "center right",
        pad=0.6,
        frameon=False,
        sep=4,
        label_top=False,
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
