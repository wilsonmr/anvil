"""
observables.py

functions for calculating observables on a stack of states.

Functions
---------
print_plot_observables:
    can be considered the main function which in turn calculates the rest
    of the observables defined in this module.}

Notes
-----
Check the definitions of functions, most are defined according the the arxiv
version: https://arxiv.org/pdf/1904.12072.pdf

"""
from math import acosh

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

from reportengine.table import table
from reportengine.figure import figure

class GreenFunction:
    def __init__(self, states, geometry):
        self.geometry = geometry
        self.sample = states

    def __call__(self, x_0: int, x_1: int):
        r"""Calculates the two point connected green function given a set of
        states G(x) where x = (x_0, x_1) refers to a shift applied to the fields
        \phi

        Parameters
        ----------
        x_0: int
            shift of dimension 0
        x_1: int
            shift of dimension 1

        Returns
        -------
        g_func: torch.Tensor
            scalar (torch.Tensor with single element) value of green function G(x)

        """
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(
            -1
        )  # make 1d

        g_func = (self.sample[:, shift] * self.sample).mean(
            dim=0
        ) - self.sample[  # mean over states
            :, shift
        ].mean(
            dim=0
        ) * self.sample.mean(
            dim=0
        )
        return g_func.mean()  # integrate over y and divide by volume


class VolumeAveraged2pf:
    def __init__(self, states, geometry):
        self.sample = states
        self.geometry = geometry

    def __call__(self, x_0: int, x_1: int):
        """
        Return torch Tensor of volume-averaged two point functions, i.e.
        where <\phi(x)> is a mean over points within a single configuration.
        
        Parameters
        ----------
        x_0: int
            shift of dimension 0
        x_1: int
            shift of dimension 1
        Returns
        -------
        va_2pf: torch.Tensor
            A 1d Tensor containing the volume-averaged two point function
            for each state in the sample
        """
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(-1)

        va_2pf = (self.sample[:, shift] * self.sample).mean(dim=1) - self.sample.mean(
            dim=1
        ).pow(2)
        return va_2pf


def two_point_green_function(sample_training_output, training_geometry):
    r"""Return instance of GreenFunction which can be used to calculate the
    two point green function for a given seperation
    """
    return GreenFunction(sample_training_output[0], training_geometry)

def volume_averaged_2pf(sample_training_output, training_geometry):
    return VolumeAveraged2pf(sample_training_output[0], training_geometry)

def zero_momentum_green_function(training_geometry, two_point_green_function):
    r"""Calculate the zero momentum green function as a function of t
    \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

        \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

    Returns
    -------
    g_func_zeromom: list
        zero momentum green function as function of t, where t runs from 0 to
        length - 1

    Notes
    -----
    This is \tilde{G}(t, 0) as defined in eq. (23) of
    https://arxiv.org/pdf/1904.12072.pdf (defined as mean instead of sum over
    spacial directions) and with momentum explicitly set to zero.

    """
    g_func_zeromom = []
    for t in range(training_geometry.length):
        g_tilde_t = 0
        for x in range(training_geometry.length):
            # not sure if factors are correct here should we account for
            # forward-backward in green function?
            g_tilde_t += float(two_point_green_function(t, x))
        g_func_zeromom.append(g_tilde_t / training_geometry.length)
    return g_func_zeromom


def effective_pole_mass(zero_momentum_green_function):
    r"""Calculate the effective pole mass m^eff(t) defined as

        m^eff(t) = arccosh(
            (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
        )

    from t = 1 to t = L-2, where L is the length of lattice side

    Returns
    -------
    m_t: list
        effective pole mass as a function of t

    Notes
    -----
    This is m^eff(t) as defined in eq. (28) of
    https://arxiv.org/pdf/1904.12072.pdf

    """
    g_func_zeromom = zero_momentum_green_function
    m_t = []
    for i, g_0_t in enumerate(g_func_zeromom[1:-1], start=1):
        m_t.append(acosh((g_func_zeromom[i - 1] + g_func_zeromom[i + 1]) / (2 * g_0_t)))
    return m_t


def susceptibility(zero_momentum_green_function):
    r"""Calculate the susceptibility, which is the sum of two point connected
    green functions over all seperations

        \chi = sum_x G(x)

    Parameters
    ----------
    g_func_zeromom: list
        zero momentum green function as a function of t

    Returns
    -------
    chi: float
        value for the susceptibility

    Notes
    -----
    as defined in eq. (25) of https://arxiv.org/pdf/1904.12072.pdf

    """
    g_func_zeromom = zero_momentum_green_function
    # TODO: we can write this in a more efficient and clearer way.
    partial_sum = [len(g_func_zeromom) * el for el in g_func_zeromom]  # undo the mean
    return sum(partial_sum)


def ising_energy(two_point_green_function):
    r"""Ising energy defined as

        E = 1/d sum_{\mu} G(\mu)

    where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
    in 2D

    Returns
    -------
    E: float
        value for the Ising energy

    Notes
    -----
    as defined in eq. (26) of https://arxiv.org/pdf/1904.12072.pdf

    """
    E = (
        two_point_green_function(1, 0) + two_point_green_function(0, 1)
    ) / 2  # am I missing a factor of 2?
    return float(E)

def autocorrelation_2pf(training_geometry, volume_averaged_2pf):
    r"""Compute the autocorrelation of the volume-averaged two point function.
    Autocorrelation is defined by
    
        \Gamma(t) = <G(s)G(s+t)> - <G(s)><G(t)>
    where G(s) is the volume-averaged two point function at Monte Carlo timestep s.
    
    Return 
    """
    x = t = 0
    # Should really look at more than one separation
    # Could go through all x, t and look for largest integrated autocorrelation time,
    # though this would not be computationally cheap.

    G_series = volume_averaged_2pf(x, t)
    G_series -= G_series.mean()
    autocorrelation = correlate(G_series, G_series, mode="same")  # converts to numpy array
    c = np.argmax(autocorrelation)
    autocorrelation = autocorrelation[c:] / autocorrelation[c]
    
    # This gives the same results, but is much slower
    """
    Nstates = len(G_series)
    autocorr2 = torch.zeros(Nstates)
    autocorr2[0] = G_series.pow(2).mean() - G_series.mean() ** 2
    for t in range(1, Nstates):
        term1 = torch.mean(G_series[:-t] * G_series[t:])
        term2 = torch.mean(G_series[:-t]) * torch.mean(G_series[t:])
        autocorr2[t] = term1 - term2
    autocorr2 = autocorr2 / autocorr2[0]
    """
    
    integrated_autocorrelation = 0.5 + np.sum(autocorrelation[1:])

    return autocorrelation, integrated_autocorrelation

def print_plot_observables(self):
    """Given a sample of states, calculate the relevant observables and either
    print them to terminal or create a figure and save to the cwd

    Output
    -----
        saves greenfunc.png and mass.png to cwd and prints Ising energy and
        Susceptibility.

    """
    g_func_zeromom = self.zero_momentum_green_function()
    m_t = self.effective_pole_mass()
    susc = self.susceptibility()
    E = self.ising_energy()
    print(f"Ising energy: {E}")
    print(f"Susceptibility: {susc}")

    fig1, ax = plt.subplots()
    ax.plot(g_func_zeromom, "-r", label=f"L = {self.geometry.length}")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel("t")
    ax.set_title("Zero momentum Green function")
    fig1.tight_layout()
    fig1.savefig(f"{self.outpath}/greenfunc.png")

    fig2, ax = plt.subplots()
    ax.plot(m_t, "-r", label=f"L = {self.geometry.length}")
    ax.set_ylabel(r"$m^{\rm eff}_p(t)$")
    ax.set_xlabel("t")
    ax.set_title("Effective Pole Mass")
    fig2.tight_layout()
    fig2.savefig(f"{self.outpath}/mass.png")
    return fig1, fig2


@table
def ising_observables_table(ising_energy, susceptibility, training_output):
    res = [[ising_energy], [susceptibility]]
    df = pd.DataFrame(
        res, columns=[training_output.name], index=["Ising energy", "susceptibility"]
    )
    return df

@figure
def plot_zero_momentum_green_function(zero_momentum_green_function, training_geometry):
    fig, ax = plt.subplots()
    ax.plot(zero_momentum_green_function, "-r", label=f"L = {training_geometry.length}")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel("t")
    ax.set_title("Zero momentum Green function")
    return fig

@figure
def plot_volume_averaged_2pf(volume_averaged_2pf):
    fig, ax = plt.subplots()
    ax.set_title("Volume-averaged two point function")
    ax.set_ylabel(r"$G_V$")
    ax.set_xlabel(r"$t$")
    ax.plot(volume_averaged_2pf(0, 0), '-')
    return fig

@figure
def plot_autocorrelation_2pf(autocorrelation_2pf):
    autocorrelation, integrated_autocorrelation = autocorrelation_2pf
    autocorrelation = autocorrelation
    fig, ax = plt.subplots()
    #ax.set_yscale("log")
    ax.set_title(r"Autocorrelation of volume-averaged two point function")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("$\Gamma_{G_V}(t)$")
    ax.plot(autocorrelation, '-')
    x = 0.8 * (1 + len(autocorrelation))
    y = 0.8
    ax.text(x, y, r"$\tau_{int} = $ %.3g" %integrated_autocorrelation, fontsize='large')
    return fig
