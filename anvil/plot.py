from math import ceil, floor, log10, fabs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import correlate
import torch
from tqdm import tqdm

from reportengine.table import table
from reportengine.figure import figure


def print_to_precision(val, err):
    """Given a value and associated error, returns two strings - the value and error rounded to
    a precision dictated by the first nonzero"""
    val = float(val)
    err = float(err)
    prec = floor(log10(abs(err)))
    if prec < 0:
        err_rnd = round(err, -prec)
        prec = floor(log10(abs(err_rnd)))
        val_str = np.format_float_positional(
            val, -prec, unique=False, fractional=True, pad_right=1
        )
        err_str = np.format_float_positional(err, -prec, fractional=True, pad_left=1)
    else:
        err_rnd = round(err, prec)
        prec = floor(log10(abs(err_rnd)))
        int_prec = ceil(log10(abs(float(val))))
        val_str = np.format_float_positional(
            val, int_prec - prec, fractional=False, pad_right=1
        ).replace(".", "")
        err_str = np.format_float_positional(
            err, 1, fractional=False, pad_left=1
        ).replace(".", "")
    return val_str, err_str


@table
def ising_observables_table(Ising_Energy, Susceptibility, bootstrap, training_output):
    IE, IE_std = print_to_precision(Ising_Energy[0], bootstrap(Ising_Energy))
    S, S_std = print_to_precision(Susceptibility[0], bootstrap(Susceptibility))
    res = [[IE, IE_std], [S, S_std]]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Ising energy", "susceptibility"],
    )
    return df


@figure
def plot_zero_momentum_2pf(Zero_Momentum_2pf, training_geometry, bootstrap):
    print("Computing zero-momentum two point function...")
    error = bootstrap(Zero_Momentum_2pf)
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(training_geometry.length),
        y=Zero_Momentum_2pf[0],
        yerr=error,
        fmt="-r",
        label=f"L = {training_geometry.length}",
    )
    ax.set_yscale("log")
    ax.set_ylabel("$\hat{G}(0, t)$")
    ax.set_xlabel("$t$")
    ax.set_title("Zero momentum two point function")
    return fig


@figure
def plot_effective_pole_mass(training_geometry, Effective_Pole_Mass, bootstrap):
    print("Computing effective pole mass...")
    error = bootstrap(Effective_Pole_Mass)
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(1, training_geometry.length - 1),
        y=Effective_Pole_Mass[0],
        yerr=error,
        fmt="-b",
        label=f"L = {training_geometry.length}",
    )
    ax.set_ylabel("$m_p^{eff}$")
    ax.set_xlabel("$t$")
    ax.set_title("Effective pole mass")
    return fig


@figure
def plot_2pf(training_geometry, Two_Point_Function, bootstrap):
    print("Computing two point function and error...")
    corr = np.empty((training_geometry.length, training_geometry.length))
    std = np.empty((training_geometry.length, training_geometry.length))
    pbar = tqdm(total=training_geometry.length ** 2, desc="(x,t)")
    for t in range(training_geometry.length):
        for x in range(training_geometry.length):
            corr[x, t] = float(Two_Point_Function(t, x)[0])
            std[x, t] = float(bootstrap(Two_Point_Function(t, x)))
            pbar.update(1)
    pbar.close()

    fractional_std = std / corr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax2.set_title(r"$\sigma_G / G$")
    ax1.set_title("$G(x, t)$")
    ax1.set_xlabel("$x$")
    ax2.set_xlabel("$x$")
    ax1.set_ylabel("$t$")
    im1 = ax1.pcolor(corr)
    im2 = ax2.pcolor(fractional_std)

    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    return fig


#################################
###     Time-series plots     ###
#################################
@figure
def plot_volume_averaged_2pf(Volume_Averaged_2pf):
    print("Computing volume-averaged two point function for each step...")
    fig, ax = plt.subplots()
    ax.set_title("Volume-averaged two point function")
    ax.set_ylabel("$G_k(0,0)$")
    ax.set_xlabel("$t$")
    ax.plot(Volume_Averaged_2pf(0, 0), "-")
    return fig


@figure
def plot_autocorrelation_2pf(Autocorrelation_2pf):
    """Plot autocorrelation as a function of Monte Carlo time,  and
    integrated autocorrelation, exponential autocorrelation, and the "g" function
    whose minimum corresponds to a window size where errors in the integrated
    autocorrelation are minimised."""
    print("Computing autocorrelation...")
    autocorrelation, tau_int_W, tau_exp_W, g_W, W_opt = Autocorrelation_2pf
    W = np.arange(1, tau_int_W.size + 1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(6, 10))
    # Autocorrelation
    ax1.set_title("Autocorrelation of volume-averaged two point function")
    ax1.set_ylabel("$\Gamma_{G(s)}(t)$")
    ax1.set_xlabel("$t$")
    ax1.plot(autocorrelation)
    ax1.plot([W_opt + 1,] * 2, [autocorrelation.min(), autocorrelation.max()], "r-")
    # Integrated autocorrelation time
    ax2.set_title("Integrated autocorrelation time")
    ax2.set_ylabel(r"$\tau_{int}(W)$")
    ax2.plot(W, tau_int_W)
    ax2.plot([W_opt,] * 2, [tau_int_W.min(), tau_int_W.max()], "r-")
    # Exponential autocorrelation time
    ax3.set_title("Exponential autocorrelation time")
    ax3.set_ylabel(r"$\tau_{exp}(W)$")
    ax3.plot(W, tau_exp_W)
    ax3.plot([W_opt,] * 2, [tau_exp_W.min(), tau_exp_W.max()], "r-")
    # "g" function
    ax4.set_title("g")
    ax4.set_ylabel("$g$")
    ax4.set_xlabel("$W$")
    ax4.plot(W, g_W)
    ax4.plot([W_opt,] * 2, [g_W.min(), g_W.max()], "r-", label=r"$W_{opt}$")
    ax4.legend()

    return fig


#######################################
###     Bootstrap distributions     ###
#######################################
def plot_bootstrap_dist(bootstrap, observable, title):
    """Plot the distribution of some observable calculated using many bootstrap samples"""

    def do_plot(ax, full_data, bootstrap_data, std):
        hist, bins = np.histogram(bootstrap_data, bins=50)
        w = bins[1] - bins[0]
        c = (bins[:-1] + bins[1:]) / 2
        ax.bar(c, hist, width=w, color="w", edgecolor="k")
        m = np.max(hist)
        ax.plot([full_data,] * 2, [0, m], "r-", lw=2, label="full sample")
        ax.plot([full_data - std,] * 2, [0, m], "m-", lw=2)
        ax.plot([full_data + std,] * 2, [0, m], "m-", lw=2, label=r"$\pm 1 \sigma$")
        bmean = torch.mean(bootstrap_data)
        ax.plot([bmean,] * 2, [0, m], "b-", lw=2, label=r"bootstrap mean")
        return ax

    obs_full = observable[0]
    obs_bootstrap = observable[1:]
    std = bootstrap(observable)
    title = "Bootstrap distribution: " + title

    if len(obs_bootstrap.shape) == 2:
        nax = obs_bootstrap.shape[1]
        fig, all_ax = plt.subplots(nax // 2, 2)  # assume L even
        all_ax_1d = [ax for tup in all_ax for ax in tup]
        for i, ax in enumerate(all_ax_1d):
            ax = do_plot(ax, obs_full[i], obs_bootstrap[:, i], std[i])
            if i == 0:
                ax.set_title(title)
            elif i == 1:
                ax.legend()

    else:
        fig, ax = plt.subplots()
        ax = do_plot(ax, obs_full, obs_bootstrap, std)
        ax.legend()
        ax.set_title(title)

    return fig


@figure
def plot_bootstrap_2pf(bootstrap, Two_Point_Function):
    x = t = 0
    data_to_plot = Two_Point_Function(x, t)
    return plot_bootstrap_dist(bootstrap, data_to_plot, rf"$G$({x},{t})")


@figure
def plot_bootstrap_susceptibility(bootstrap, Susceptibility):
    return plot_bootstrap_dist(bootstrap, Susceptibility, r"$\chi$")


@figure
def plot_bootstrap_ising_energy(bootstrap, Ising_Energy):
    return plot_bootstrap_dist(bootstrap, Ising_Energy, r"Ising $E$")


@figure
def plot_bootstrap_zero_momentum_2pf(bootstrap, Zero_Momentum_2pf):
    return plot_bootstrap_dist(bootstrap, Zero_Momentum_2pf, r"$\tilde G(0,t)$")


@figure
def plot_bootstrap_effective_pole_mass(bootstrap, Effective_Pole_Mass):
    return plot_bootstrap_dist(bootstrap, Effective_Pole_Mass, r"$m_p^{eff}$")
