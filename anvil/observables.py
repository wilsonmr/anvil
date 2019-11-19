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
from math import acosh, sqrt, fabs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import correlate
import torch

from reportengine.table import table
from reportengine.figure import figure
from reportengine import collect


class TwoPointFunction:
    def __init__(self, states, geometry):
        self.geometry = geometry
        self.sample = states

    def __call__(self, x_0: int, x_1: int, error=False):
        r"""Calculates the two point connected green function given a set of
        states G(x) where x = (x_0, x_1) refers to a shift applied to the fields
        \phi

        Also calculates statistical uncertainty in G(x), based on the assumption
        that the \phi fields are statistically independent, so that the error on
        the mean is the standard error, i.e. the standard deviation divided by
        the square root of the number of fields being averaged over.

        Standard errors in <\phi(x)>, <\phi(x+shift)>, <\phi(x+shift)\phi(x)> are
        propagated to get an error for G(x) using the functional approach.
        See Hughes & Hase, Measurements and their uncertainties for details.

        Parameters
        ----------
        x_0: int
            shift of dimension 0
        x_1: int
            shift of dimension 1
        error: bool
            if True, computes and returns the standard error

        Returns
        -------
        g_func: torch.Tensor
            scalar (torch.Tensor with single element) value of green function G(x)

        """
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(
            -1
        )  # make 1d

        phi = self.sample
        phi_shift = self.sample[:, shift]

        #  Average over stack of states
        phi_mean = phi.mean(dim=0)
        phi_shift_mean = phi_shift.mean(dim=0)
        phi_shift_phi_mean = (phi_shift * phi).mean(dim=0)

        if error == True:
            phi_var = phi.var(dim=0)
            phi_shift_var = phi_shift.var(dim=0)
            phi_shift_phi_var = (phi_shift * phi).var(dim=0)
            Nstates = len(phi[:, 0])
            Npoints = len(phi[0, :])

            g_func_error = (
                phi_shift_phi_var / Nstates  # first term error squared
                + (  # add squared errors from first and second term
                    phi_shift_mean * phi_mean
                )
                ** 2
                * (
                    phi_shift_var / (Nstates * phi_shift_mean ** 2)
                    + phi_var / (Nstates * phi_mean ** 2)
                )  # second term: add fractional errors in quadrature
            ).sum().sqrt() / Npoints  # sum over coordinates, sqrt, scale

            return g_func_error
        else:
            g_func = phi_shift_phi_mean - phi_shift_mean * phi_mean
            return g_func.mean()  # average over coordinates


class VolumeAveragedTwoPointFunction:
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
        vag_func: torch.Tensor
            A 1d Tensor containing the volume-averaged two point function
            for each state in the sample
        """
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(-1)

        vag_func = (self.sample[:, shift] * self.sample).mean(dim=1) - self.sample.mean(
            dim=1
        ).pow(2)
        return vag_func


def two_point_function(sample_training_output, training_geometry):
    r"""Return instance of TwoPointFunction which can be used to calculate the
    two point green function for a given seperation
    """
    return TwoPointFunction(sample_training_output[0], training_geometry)


def volume_averaged_2pf(sample_training_output, training_geometry):
    r"""Return instance of VolumeAveragedTwoPointFunction"""
    return VolumeAveragedTwoPointFunction(sample_training_output[0], training_geometry)


def compute_2pf_autocorrelation(training_geometry, volume_averaged_2pf):
    r"""Compute the autocorrelation of the volume-averaged two point function.

    Autocorrelation is defined by
    
        \Gamma(t) = <G(s)G(s+t)> - <G(s)><G(t)>

    where G(s) is the volume-averaged two point function at Monte Carlo timestep s.
    
    Return 
    """
    x = t = 0  # Should really look at more than one separation
    G_series = volume_averaged_2pf(x, t)
    G_series -= G_series.mean()
    autocorrelation = correlate(G_series, G_series, mode="same")  # converts in numpy array
    c = len(autocorrelation) // 2
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


def compute_zero_momentum_2pf(training_geometry, two_point_function):
    r"""Calculate the zero momentum green function as a function of t
    \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

        \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

    Returns
    -------
    g_func_zeromom: dict of lists
        'values': zero momentum green function as function of t, where t runs from 0 to
            length - 1
        'errors': uncertainty propagated from uncertainty in G(x) using
            the functional approach

    Notes
    -----
    This is \tilde{G}(t, 0) as defined in eq. (23) of
    https://arxiv.org/pdf/1904.12072.pdf (defined as mean instead of sum over
    spacial directions) and with momentum explicitly set to zero.

    """
    g_func_zeromom = {"values": [], "errors": []}
    for t in range(training_geometry.length):
        g_tilde_t = 0
        error_sq = 0
        for x in range(training_geometry.length):
            # not sure if factors are correct here should we account for
            # forward-backward in green function?
            g_tilde_t += float(two_point_function(t, x))
            error_sq += float(two_point_function(t, x, error=True)) ** 2

        g_func_zeromom["values"].append(g_tilde_t / training_geometry.length)
        g_func_zeromom["errors"].append(sqrt(error_sq) / training_geometry.length)

    return g_func_zeromom


def compute_effective_pole_mass(compute_zero_momentum_2pf):
    r"""Calculate the effective pole mass m^eff(t) defined as

        m^eff(t) = arccosh(
            (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
        )

    from t = 1 to t = L-2, where L is the length of lattice side

    Returns
    -------
    m_t: dict of lists
        'values': effective pole mass as a function of t
        'errors': uncertainty propagated from uncertainty in \tilde{G}(t, 0)

    Notes
    -----
    This is m^eff(t) as defined in eq. (28) of
    https://arxiv.org/pdf/1904.12072.pdf

    """
    g_func_zeromom = compute_zero_momentum_2pf
    m_t = {"values": [], "errors": []}
    for i, g_0_t in enumerate(g_func_zeromom["values"][1:-1], start=1):

        numerator = g_func_zeromom["values"][i - 1] + g_func_zeromom["values"][i + 1]
        denominator = 2 * g_0_t
        argument = numerator / denominator
        m_t["values"].append(acosh(argument))
        # m_t['values'].append(0)

        # Error calculation 1: functional approach based on function m_t of three
        # independent variables, whose errors are added in quadrature
        """error1_sq = fabs(acosh((numerator + g_func_zeromom['errors'][i - 1]) / denominator)
                        - acosh(argument))**2
        error2_sq = fabs(acosh((numerator + g_func_zeromom['errors'][i + 1]) / denominator)
                        - acosh(argument))**2
        error3_sq = fabs(acosh(numerator / (denominator + 2 * g_func_zeromom['errors'][i]))
                        - acosh(argument))**2
        error_v1 = sqrt(error1_sq + error2_sq + error3_sq)
        #print("v1: ", error_v1)"""

        # Error calculation 2: two-level functional approach based on function m_t
        # of one variable, which itself is a function of three variables.
        # The two calculations should agree

        error_numerator_sq = (
            g_func_zeromom["errors"][i - 1] ** 2 + g_func_zeromom["errors"][i + 1] ** 2
        )
        error_denominator_sq = (2 * g_func_zeromom["errors"][i]) ** 2
        error_argument = (numerator / denominator) * sqrt(
            error_numerator_sq / numerator ** 2
            + error_denominator_sq / denominator ** 2
        )
        error_v2 = fabs(acosh(argument + error_argument) - acosh(argument))

        # error_v2 = 0
        m_t["errors"].append(error_v2)

    return m_t


def compute_susceptibility(training_geometry, two_point_function):
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
    error: float
        uncertainty in the susceptibility, based on uncertainty in G(x)

    Notes
    -----
    as defined in eq. (25) of https://arxiv.org/pdf/1904.12072.pdf

    """
    chi = 0
    error_sq = 0
    for t in range(training_geometry.length):
        for x in range(training_geometry.length):
            chi += float(two_point_function(t, x))
            error_sq += (
                float(two_point_function(t, x, error=True)) ** 2
            )  # sum -> add errors in quadrature

    return chi, sqrt(error_sq)


def compute_ising_energy(two_point_function):
    r"""Ising energy defined as

        E = 1/d sum_{\mu} G(\mu)

    where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
    in 2D

    Returns
    -------
    E: float
        value for the Ising energy
    error: float
        uncertainty based on uncertainty in G(x)

    Notes
    -----
    as defined in eq. (26) of https://arxiv.org/pdf/1904.12072.pdf

    """
    E = (
        two_point_function(1, 0) + two_point_function(0, 1)
    ) / 2  # am I missing a factor of 2?
    error = (
        sqrt(
            two_point_function(1, 0, error=True) ** 2
            + two_point_function(0, 1, error=True) ** 2
        )
        / 2
    )
    return float(E), error


autocorrelation_2pf = collect("compute_2pf_autocorrelation", ("training_context",))
zero_momentum_2pf = collect("compute_zero_momentum_2pf", ("training_context",))
effective_pole_mass = collect("compute_effective_pole_mass", ("training_context",))
ising_energy = collect("compute_ising_energy", ("training_context",))
susceptibility = collect("compute_susceptibility", ("training_context",))
