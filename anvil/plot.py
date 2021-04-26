"""
plot.py

module containing all actions for plotting observables

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

from reportengine.figure import figure, figuregen
from reportengine import collect

from anvil.observables import cosh_shift


# TODO: subplots for different neural networks
def plot_layer_weights(model_weights):
    for weights in model_weights:
        fig, ax = plt.subplots()
        labels = list(weights.keys())
        data = weights.values()
        ax.hist(data, bins=50, stacked=True, label=labels)
        fig.legend()
        yield fig


@figuregen
def plot_layerwise_weights(plot_layer_weights):
    yield from plot_layer_weights


@figure
def plot_layer_histogram(configs):
    v = configs.numpy()
    v_pos = v[v.sum(axis=1) > 0].flatten()
    v_neg = v[v.sum(axis=1) < 0].flatten()
    fig, ax = plt.subplots()
    ax.hist([v_pos, v_neg], bins=50, density=True, histtype="step")
    return fig


@figure
def plot_correlation_length(table_correlation_length):
    fig, ax = plt.subplots()
    ax.errorbar(
        x=table_correlation_length.index,
        y=table_correlation_length.value,
        yerr=table_correlation_length.error,
        linestyle="",
        marker="o",
    )
    ax.set_xticklabels(table_correlation_length.index, rotation=45)
    return fig


@figure
def plot_zero_momentum_correlator(
    zero_momentum_correlator,
    training_geometry,
    fit_zero_momentum_correlator,
):
    """Plot zero_momentum_2pf as a function of t. Points are means across bootstrap
    sample and errorbars are standard deviations across boostrap samples
    """
    T = training_geometry.length
    shift = 0

    fig, ax = plt.subplots()

    if fit_zero_momentum_correlator is not None:
        popt, pcov, t0 = fit_zero_momentum_correlator
        xi, A, shift = popt

        t = np.linspace(t0, T - t0, 100)
        ax.plot(
            t,
            cosh_shift(t - T // 2, *popt) - shift,
            "r--",
            label=r"fit $A \cosh(-(t - T/2) / \xi) + c$" + "\n" + fr"$\xi = ${xi:.2f}",
        )
    ax.errorbar(
        x=np.arange(T),
        y=zero_momentum_correlator.mean(axis=-1) - shift,
        yerr=zero_momentum_correlator.std(axis=-1),
        fmt="bo",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel("$t$")
    ax.set_title("Zero momentum two point function")
    ax.legend()
    return fig


@figure
def plot_effective_pole_mass(training_geometry, effective_pole_mass):
    """Plot effective pole mass as a function of t. The points are means
    across bootstrap samples and the errorbars are standard deviation across
    bootstrap.
    """
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(1, training_geometry.length - 1),
        y=effective_pole_mass.mean(axis=-1),
        yerr=effective_pole_mass.std(axis=-1),
        fmt="-b",
        label=f"L = {training_geometry.length}",
    )
    ax.set_ylabel("$m_p$")
    ax.set_xlabel("$t$")
    ax.set_title("Effective pole mass")
    return fig


@figure
def plot_two_point_correlator(two_point_correlator):
    """Represent the two point function and it's error in x and t as heatmaps
    of the respective matrices. Returns a figure with two plots, the left plot
    is the mean two point function across bootstrap and the right plot is the
    standard deviation divide by the mean (fractional error)

    """
    corr = two_point_correlator.mean(axis=-1)
    error = two_point_correlator.std(axis=-1)
    fractional_error = np.abs(error / corr)
    L = corr.shape[0]

    corr = np.roll(corr, (-L // 2 - 1, -L // 2 - 1), (0, 1))
    fractional_error = np.roll(fractional_error, (-L // 2 - 1, -L // 2 - 1), (0, 1))

    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax_std.set_title(r"$\sigma_G / G$")
    ax_mean.set_title("$G(x, t)$")
    ax_mean.set_xlabel("$x$")
    ax_std.set_xlabel("$x$")
    ax_mean.set_ylabel("$t$")
    norm = mpl.colors.LogNorm()

    im1 = ax_mean.imshow(corr, norm=norm)
    im2 = ax_std.imshow(fractional_error, norm=norm)

    ax_mean.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_mean.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_std.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_std.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.colorbar(im1, ax=ax_mean)
    fig.colorbar(im2, ax=ax_std)

    return fig


@figure
def plot_magnetization_series(magnetization_series, sample_interval):
    chain_indices = np.arange(magnetization_series.shape[-1]) * sample_interval
    fig, ax = plt.subplots()
    ax.set_title("Magnetization")
    ax.set_ylabel("$m(t)$")
    ax.set_xlabel("$t$")
    ax.plot(
        chain_indices,
        magnetization_series,
        linestyle="-",
        linewidth=0.5,
    )
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

    ax.plot(
        chain_indices,
        magnetization_autocorr[:cut],
        linestyle="--",
        linewidth=0.5,
    )
    ax.axvline(
        magnetization_optimal_window * sample_interval,
        linestyle="-",
        color="r",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(top=1)
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
    ax.set_ylabel(r"$\sum \Gamma_M(\delta t)$")
    ax.set_xlabel("$\delta t$")

    ax.plot(
        chain_indices,
        magnetization_integrated_autocorr[:cut],
        linestyle="--",
        linewidth=0.5,
    )
    ax.axvline(
        magnetization_optimal_window * sample_interval,
        linestyle="-",
        color="r",
    )
    ax.annotate(
        fr"$\tau_M$ / {sample_interval} = {tau:.2g}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.5)
    return fig


def plot_bootstrap_single_number(observable, label):
    """Given a 1 dimensional tensor of observables, plot a histogram of the
    distribution
    """
    fig, ax = plt.subplots()
    ax.hist(
        observable,
        bins=30,
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
def plot_bootstrap_two_point(two_point_correlator):
    """Plot the distribution of G(0, 0)"""
    x = t = 0
    data_to_plot = two_point_correlator(x, t)
    return plot_bootstrap_single_number(data_to_plot, rf"$G$({x},{t})")


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
