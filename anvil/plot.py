"""
plot.py

module containing all actions for plotting observables

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from reportengine.figure import figure, figuregen


@figure
def plot_zero_momentum_two_point(zero_momentum_two_point, training_geometry):
    """Plot zero_momentum_2pf as a function of t. Points are means across bootstrap
    sample and errorbars are standard deviations across boostrap samples
    """
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(training_geometry.length),
        y=zero_momentum_two_point.mean(dim=-1),
        yerr=zero_momentum_two_point.std(dim=-1),
        fmt="-r",
        label=f"L = {training_geometry.length}",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel("$t$")
    ax.set_title("Zero momentum two point function")
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
        y=effective_pole_mass.mean(dim=-1),
        yerr=effective_pole_mass.std(dim=-1),
        fmt="-b",
        label=f"L = {training_geometry.length}",
    )
    ax.set_ylabel("$m_p^{eff}$")
    ax.set_xlabel("$t$")
    ax.set_title("Effective pole mass")
    return fig


@figure
def plot_two_point_function(two_point_function):
    """Represent the two point function and it's error in x and t as heatmaps
    of the respective matrices. Returns a figure with two plots, the left plot
    is the mean two point function across bootstrap and the right plot is the
    standard deviation divide by the mean (fractional error)

    """
    corr = two_point_function.mean(dim=-1)
    std = two_point_function.std(dim=-1)

    fractional_std = std / abs(corr)

    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax_std.set_title(r"$\sigma_G / G$")
    ax_mean.set_title("$G(x, t)$")
    ax_mean.set_xlabel("$x$")
    ax_std.set_xlabel("$x$")
    ax_mean.set_ylabel("$t$")

    im1 = ax_mean.imshow(corr)
    im2 = ax_std.imshow(fractional_std)

    ax_mean.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_mean.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_std.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_std.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.colorbar(im1, ax=ax_mean)
    fig.colorbar(im2, ax=ax_std)

    return fig


@figure
def plot_volume_averaged_two_point(volume_avg_two_point_function):
    """Plot the volumn averaged two point function for the shift (0, 1)
    """
    # TODO: do we want to plot this for all shifts?
    fig, ax = plt.subplots()
    ax.set_title("Volume-averaged two point function")
    ax.set_ylabel("$G_k(0,1)$")
    ax.set_xlabel("$t$")
    ax.plot(volume_avg_two_point_function[0, 1, :], "-")
    return fig


@figure
def plot_autocorr_two_point(autocorr_two_point, optimal_window):
    """Plot autocorrelation as a function of Monte Carlo time for 4 x the optimal
    window estimated by optimal_window. Mark on the optimal window as a verticle
    line
    """
    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation of volume-averaged two point function")
    ax.set_ylabel(r"$\Gamma_{G(s)}(t)$")
    ax.set_xlabel("$t$")
    ax.plot(autocorr_two_point[: 4 * optimal_window])
    ax.axvline(optimal_window + 1, linestyle="-", color="r", label="Optimal window")
    ax.legend()
    return fig


@figure
def plot_integrated_autocorr_two_point(integrated_autocorr_two_point, optimal_window):
    """plot integrated_autocorr_two_point as a function of w, up until 4 x the
    optimal window estimated by optimal_window. Mark on the optimal window as a
    verticle line

    """
    tau_int = integrated_autocorr_two_point[: 4 * optimal_window]
    windows = np.arange(1, tau_int.size + 1)
    fig, ax = plt.subplots()
    # Integrated autocorrelation time
    ax.set_title("Integrated autocorrelation time")
    ax.set_ylabel(r"$\tau_{int}(W)$")
    ax.plot(windows, tau_int)
    ax.axvline(optimal_window, linestyle="-", color="r", label="Optimal window")
    ax.legend()
    return fig


@figure
def plot_exp_autocorr_two_point(exp_autocorr_two_point, optimal_window):
    """plot exp_autocorr_two_point as a function of w, up until 4 x the
    optimal window estimated by optimal_window. Mark on the optimal window as a
    verticle line

    """
    fig, ax = plt.subplots()
    tau_exp = exp_autocorr_two_point[: 4 * optimal_window]
    windows = np.arange(1, tau_exp.size + 1)
    ax.set_title("Exponential autocorrelation time")
    ax.set_ylabel(r"$\tau_{exp}(W)$")
    ax.plot(windows, tau_exp)
    ax.axvline(optimal_window, linestyle="-", color="r", label="Optimal window")
    ax.legend()
    return fig


@figure
def plot_automatic_windowing_function(automatic_windowing_function, optimal_window):
    """plot automatic_windowing_function as a function of w, up until 4 x the
    optimal window estimated by optimal_window. Mark on the optimal window as a
    verticle line

    """
    fig, ax = plt.subplots()
    g_func = automatic_windowing_function[: 4 * optimal_window]
    windows = np.arange(1, g_func.size + 1)
    ax.set_title("g")
    ax.set_ylabel("$g$")
    ax.set_xlabel("$W$")
    ax.plot(windows, g_func)
    ax.axvline(optimal_window, linestyle="-", color="r", label="Optimal window")
    ax.legend()
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
def plot_bootstrap_two_point(two_point_function):
    """Plot the distribution of G(0, 0)"""
    x = t = 0
    data_to_plot = two_point_function(x, t)
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
def plot_bootstrap_zero_momentum_2pf(zero_momentum_two_point):
    """For each value of t, plot a boostrap distribution of
    zero_momentum_two_point[t]

    """
    labels = [
        r"$\tilde{G}$" + f"$(0,{t})$" for t in range(zero_momentum_two_point.shape[0])
    ]
    yield from plot_bootstrap_multiple_numbers(zero_momentum_two_point, labels)


@figuregen
def plot_bootstrap_effective_pole_mass(effective_pole_mass):
    """For each value of t from 1 to L-2, plot a bootstrap distribution of
    effective_pole_mass[t]

    """
    labels = [r"$m_p^{eff}$" + f"$({t})$" for t in range(effective_pole_mass.shape[0])]
    yield from plot_bootstrap_multiple_numbers(effective_pole_mass, labels)
