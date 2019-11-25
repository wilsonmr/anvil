from math import ceil, floor, log10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import correlate
import torch
from tqdm import tqdm

from reportengine.table import table
from reportengine.figure import figure


def print_format(val, err):
    """Given a value and associated error, returns two strings - the value and error rounded to
    a precision dictated by the first nonzero"""
    prec = floor(log10(abs(round(err, 1))))
    if prec < 0:
        val_str = np.format_float_positional(
            val, -prec, unique=False, fractional=True, pad_right=1
        )
        err_str = np.format_float_positional(err, -prec, fractional=True, pad_left=1)
    else:
        int_prec = ceil(log10(abs(float(val))))
        val_str = np.format_float_positional(
            val, int_prec - prec, fractional=False, pad_right=1
        ).strip(".")
        err_str = np.format_float_positional(
            err, 1, fractional=False, pad_left=1
        ).strip(".")
    return val_str, err_str


@table
def ising_observables_table(
    ising_energy, susceptibility, bootstrap_std, training_output
):
    IE, IE_std = print_format(ising_energy[0], bootstrap_std(ising_energy))
    S, S_std = print_format(susceptibility[0], bootstrap_std(susceptibility))
    res = [[IE, IE_std], [S, S_std]]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Ising energy", "susceptibility"],
    )
    return df


@figure
def plot_zero_momentum_2pf(zero_momentum_2pf, training_geometry, bootstrap_std):
    print("Computing zero-momentum two point function...")
    error = bootstrap_std(zero_momentum_2pf)
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(training_geometry.length),
        y=zero_momentum_2pf[0],
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
def plot_effective_pole_mass(training_geometry, effective_pole_mass, bootstrap_std):
    print("Computing effective pole mass...")
    error = bootstrap_std(effective_pole_mass)
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(1, training_geometry.length - 1),
        y=effective_pole_mass[0],
        yerr=error,
        fmt="-b",
        label=f"L = {training_geometry.length}",
    )
    ax.set_ylabel("$m_p^{eff}$")
    ax.set_xlabel("$t$")
    ax.set_title("Effective pole mass")
    return fig


@figure
def plot_2pf(training_geometry, two_point_function, bootstrap_std):
    print("Computing two point function and error...")
    corr = np.empty((training_geometry.length, training_geometry.length))
    std = np.empty((training_geometry.length, training_geometry.length))
    t_pbar = tqdm(range(training_geometry.length), desc="t")
    for t in t_pbar:
        for x in range(training_geometry.length):
            corr[x, t] = float(two_point_function(t, x)[0])
            std[x, t] = float(bootstrap_std(two_point_function(t, x)))

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


@figure
def plot_volume_averaged_2pf(volume_averaged_2pf):
    print("Computing volume-averaged two point function for each step...")
    fig, ax = plt.subplots()
    ax.set_title("Volume-averaged two point function")
    ax.set_ylabel("$G_k$")
    ax.set_xlabel("$t$")
    ax.plot(volume_averaged_2pf(0, 0), "-")
    return fig


@figure
def plot_autocorrelation_2pf(autocorrelation_2pf):
    print("Computing autocorrelation...")
    autocorrelation, integrated_autocorrelation = autocorrelation_2pf
    autocorrelation = autocorrelation
    fig, ax = plt.subplots()
    # ax.set_yscale("log")
    ax.set_title("Autocorrelation of volume-averaged two point function")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\Gamma_{G(s)}(t)$")
    ax.plot(autocorrelation, "-")
    x = 0.8 * (1 + len(autocorrelation))
    y = 0.8
    ax.text(
        x, y, r"$\tau_{int} = $ %.3g" % integrated_autocorrelation, fontsize="large"
    )
    return fig
