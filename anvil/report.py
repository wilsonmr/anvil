from math import acosh, sqrt, fabs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import correlate
import torch

from reportengine.table import table
from reportengine.figure import figure

@table
def ising_observables_table(ising_energy, susceptibility, training_output): 
    ising_energy = ising_energy[0]
    susceptibility = susceptibility[0]
    res = [[rf"{ising_energy[0]:.5g} $\pm$ {ising_energy[1]:.1g}"],
           [rf"{susceptibility[0]:.5g} $\pm$ {susceptibility[1]:.1g}"]]
    df = pd.DataFrame(
        res, columns=[training_output.name], index=["Ising energy", "susceptibility"]
    )
    return df

@figure
def plot_zero_momentum_2pf(zero_momentum_2pf, training_geometry):
    zero_momentum_2pf = zero_momentum_2pf[0]
    fig, ax = plt.subplots()
    ax.errorbar(
            x = range(len(zero_momentum_2pf['values'])),
            y = zero_momentum_2pf['values'],
            yerr = zero_momentum_2pf['errors'],
            fmt = "-r",
            label=f"L = {training_geometry.length}"
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel(r"$t$")
    ax.set_title("Zero momentum two point function")
    return fig

@figure
def plot_effective_pole_mass(training_geometry, effective_pole_mass):
    effective_pole_mass = effective_pole_mass[0]
    Npoints = len(effective_pole_mass['values'])
    fig, ax = plt.subplots()
    ax.errorbar(
        x = range(1, Npoints + 1),
        y = effective_pole_mass['values'],
        yerr = effective_pole_mass['errors'],
        fmt = "-b",
        label = f"L = {training_geometry.length}"
    )
    ax.set_ylabel(r"$m_p^{eff}$")
    ax.set_xlabel(r"$t$")
    ax.set_title("Effective pole mass")
    return fig

'''
@figure
def plot_2pf(training_geometry, two_point_function):

    corr = np.empty( (training_geometry.length, training_geometry.length) )
    error = np.empty( (training_geometry.length, training_geometry.length) )
    for t in range(training_geometry.length):
        for x in range(training_geometry.length):
            corr[x,t] = float(two_point_green_function(t, x))
            error = float(two_point_green_function(t, x, error=True))
    
    fractional_error = error / corr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6), sharey=True)
    ax2.set_title(r"Fractional error in $G(x, t)$")
    ax1.set_title(r"$G(x, t)$")
    ax1.set_xlabel(r"$x$")
    ax2.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")
    im1 = ax1.pcolor(corr)
    im2 = ax2.pcolor(fractional_error)

    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    return fig
'''

@figure
def plot_autocorrelation_2pf(autocorrelation_2pf):
    autocorrelation, integrated_autocorrelation = autocorrelation_2pf[0]
    autocorrelation = autocorrelation[:100]

    fig, ax = plt.subplots()
    #ax.set_yscale("log")
    ax.set_title(r"Autocorrelation of volume-averaged two point function")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Auto")
    ax.plot(autocorrelation, '-', label="Autocorrelation")

    x = 0.8 * (1 + len(autocorrelation))
    y = 0.8
    ax.text(x, y, r"$\tau_{int} = $ %.3g" %integrated_autocorrelation, fontsize='large')
    ax.legend()
    return fig

'''
@figure
def plot_G_series(two_point_green_function):
    series = two_point_green_function(0, 0, sequence=True)
    fig, ax = plt.subplots()
    ax.set_title("Series")
    ax.set_ylabel(r"G")
    ax.set_xlabel(r"$t$")
    ax.plot(series[:1000], '-')
    return fig
'''
