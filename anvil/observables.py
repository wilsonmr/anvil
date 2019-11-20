"""
observables.py

functions for calculating observables on a stack of states.

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

from tqdm import tqdm

from reportengine import collect


class TwoPointFunction:
    def __init__(self, states, geometry):
        self.geometry = geometry
        self.sample = states

    def __call__(self, x_0: int, x_1: int, sample_size=None):
        r"""Calculates the two point connected green function given a set of
        states G(x) where x = (x_0, x_1) refers to a shift applied to the fields
        \phi

        Parameters
        ----------
        x_0: int
            shift of dimension 0
        x_1: int
            shift of dimension 1
        sample_size: int
            compute the 2pf from a subsample of states, where the indices of the
            sample_size states are chosen randomly, with replacement

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

        if sample_size is not None:
            Nstates = len(phi[:, 0])
            sample_indices = np.random.choice(Nstates, sample_size, replace=True)
            phi = phi[sample_indices, :]
            phi_shift = phi_shift[sample_indices, :]

        #  Average over stack of states
        phi_mean = phi.mean(dim=0)
        phi_shift_mean = phi_shift.mean(dim=0)
        phi_shift_phi_mean = (phi_shift * phi).mean(dim=0)

        g_func = torch.mean(phi_shift_phi_mean - phi_shift_mean * phi_mean)

        return g_func


class TwoPointFunctionError:
    def __init__(self, states, geometry):
        self.sample = states
        self.geometry = geometry

    def __call__(self, x_0, x_1):
        """
        Calculates the statistical uncertainty in G(x), based on the assumption
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

        Returns
        -------
        g_func_error: float
        """
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(
            -1
        )  # make 1d

        phi = self.sample
        phi_shift = self.sample[:, shift]

        phi_mean = phi.mean(dim=0)
        phi_shift_mean = phi_shift.mean(dim=0)
        phi_shift_phi_mean = (phi_shift * phi).mean(dim=0)

        phi_var = phi.var(dim=0)
        phi_shift_var = phi_shift.var(dim=0)
        phi_shift_phi_var = (phi_shift * phi).var(dim=0)

        Nstates = len(phi[:, 0])
        Npoints = len(phi[0, :])

        g_func_error = (
            phi_shift_phi_var / Nstates  # first error term squared
            + (  # add squared errors from first and second term
                phi_shift_mean * phi_mean
            )
            ** 2
            * (
                phi_shift_var / (Nstates * phi_shift_mean ** 2)
                + phi_var / (Nstates * phi_mean ** 2)
            )  # second error term squared
        ).sum().sqrt() / sqrt(
            Npoints
        )  # sum over coords, sqrt, scale by rootN

        return g_func_error


class VolumeAveraged2pf(TwoPointFunction):
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


class ZeroMomentum2pf:
    def __init__(self, training_geometry, two_point_function):
        self.training_geometry = training_geometry
        self.two_point_function = two_point_function

    def __call__(self, sample_size=None):
        r"""Calculate the zero momentum green function as a function of t
        \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

            \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

        Parameters
        ----------
        sample_size: int
            calculation done based on a subsample of states. See TwoPointFunction

        Returns
        -------
        g_func_zeromom: list
            'values': zero momentum green function as function of t, where t runs from 0 to
                length - 1

        Notes
        -----
        This is \tilde{G}(t, 0) as defined in eq. (23) of
        https://arxiv.org/pdf/1904.12072.pdf (defined as mean instead of sum over
        spacial directions) and with momentum explicitly set to zero.

        """
        g_func_zeromom = []
        for t in range(self.training_geometry.length):
            g_tilde_t = 0
            for x in range(self.training_geometry.length):
                # not sure if factors are correct here should we account for
                # forward-backward in green function?
                g_tilde_t += float(self.two_point_function(t, x, sample_size))

            g_func_zeromom.append(g_tilde_t / self.training_geometry.length)
        return g_func_zeromom


class EffectivePoleMass:
    def __init__(self, zero_momentum_2pf):
        self.zero_momentum_2pf = zero_momentum_2pf

    def __call__(self, sample_size=None):
        r"""Calculate the effective pole mass m^eff(t) defined as

            m^eff(t) = arccosh(
                (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
            )

        from t = 1 to t = L-2, where L is the length of lattice side

        Parameters
        ----------
        sample_size: int
            calculation done based on a subsample of states. See TwoPointFunction

        Returns
        -------
        m_t: list
            effective pole mass as a function of t

        Notes
        -----
        This is m^eff(t) as defined in eq. (28) of
        https://arxiv.org/pdf/1904.12072.pdf

        """
        g_func_zeromom = self.zero_momentum_2pf(sample_size)
        m_t = []
        for i, g_0_t in enumerate(g_func_zeromom[1:-1], start=1):

            numerator = g_func_zeromom[i - 1] + g_func_zeromom[i + 1]
            denominator = 2 * g_0_t
            argument = numerator / denominator
            m_t.append(acosh(argument))

        return m_t


class Susceptibility:
    def __init__(self, training_geometry, two_point_function):
        self.training_geometry = training_geometry
        self.two_point_function = two_point_function

    def __call__(self, sample_size=None):
        r"""Calculate the susceptibility, which is the sum of two point connected
        green functions over all seperations

            \chi = sum_x G(x)

        Parameters
        ----------
        sample_size: int
            calculation done based on a subsample of states. See TwoPointFunction

        Returns
        -------
        chi: float
            value for the susceptibility

        Notes
        -----
        as defined in eq. (25) of https://arxiv.org/pdf/1904.12072.pdf

        """
        chi = 0
        for t in range(self.training_geometry.length):
            for x in range(self.training_geometry.length):
                chi += float(self.two_point_function(t, x, sample_size))

        return chi


class IsingEnergy:
    def __init__(self, two_point_function):
        self.two_point_function = two_point_function

    def __call__(self, sample_size=None):
        r"""Ising energy defined as

            E = 1/d sum_{\mu} G(\mu)

        where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
        in 2D

        Parameters
        ----------
        sample_size: int
            calculation done based on a subsample of states. See TwoPointFunction

        Returns
        -------
        E: float
            value for the Ising energy

        Notes
        -----
        as defined in eq. (26) of https://arxiv.org/pdf/1904.12072.pdf

        """
        E = (
            self.two_point_function(1, 0, sample_size)
            + self.two_point_function(0, 1, sample_size)
        ) / 2
        return float(E)


##############################################################################


def autocorrelation_2pf(training_geometry, volume_averaged_2pf):
    r"""Compute the autocorrelation of the volume-averaged two point function.

    Autocorrelation is defined by

        \Gamma(t) = <G(s)G(s+t)> - <G(s)><G(t)>

    where G(s) is the volume-averaged two point function at Monte Carlo timestep s.

    Integrated autocorrelation is defined by

        \tau = 0.5 + sum_t \Gamma(t)

    Returns
    -------
    autocorrelation: numpy.array
    integrated_autocorrelation: float
    """
    x = t = 0  # Should really look at more than one separation
    G_series = volume_averaged_2pf(x, t)
    G_series -= G_series.mean()
    autocorrelation = correlate(
        G_series, G_series, mode="same"
    )  # converts in numpy array
    c = np.argmax(autocorrelation)
    autocorrelation = autocorrelation[c:] / autocorrelation[c]

    integrated_autocorrelation = 0.5 + np.sum(autocorrelation[1:])

    return autocorrelation, integrated_autocorrelation


def bootstrap(observable, sample_size):
    mean = observable()
    bootstrap_results = []
    Nb = 100  # number of bootstrap samples
    pbar = tqdm(range(Nb), desc="bootstrap sample")
    for sample in pbar:
        bootstrap_results.append(observable(sample_size))
    bootstrap_results = np.array(bootstrap_results)

    error_sq = np.sum((bootstrap_results - mean) ** 2, axis=0) / Nb

    return mean, np.sqrt(error_sq)


def two_point_function(sample_training_output, training_geometry):
    r"""Return instance of TwoPointFunction which can be used to calculate the
    two point green function for a given seperation
    """
    return TwoPointFunction(sample_training_output, training_geometry)


def two_point_function_error(sample_training_output, training_geometry):
    return TwoPointFunctionError(sample_training_output, training_geometry)


def volume_averaged_2pf(sample_training_output, training_geometry):
    return VolumeAveraged2pf(sample_training_output, training_geometry)


def zero_momentum_2pf(training_geometry, two_point_function):
    return ZeroMomentum2pf(training_geometry, two_point_function)


############################################################################


def zero_momentum_2pf_out(training_geometry, two_point_function, target_length):
    print("Computing zero-momentum two point function")
    return bootstrap(
        ZeroMomentum2pf(training_geometry, two_point_function), target_length // 10
    )


def effective_pole_mass(zero_momentum_2pf, target_length):
    print("Computing effective pole mass")
    return bootstrap(EffectivePoleMass(zero_momentum_2pf), target_length // 10)


def susceptibility(training_geometry, two_point_function, target_length):
    print("Computing susceptibility")
    return bootstrap(
        Susceptibility(training_geometry, two_point_function), target_length // 10
    )


def ising_energy(two_point_function, target_length):
    print("Computing Ising energy")
    return bootstrap(IsingEnergy(two_point_function), target_length // 10)


##############################################################################
# Currently not used
# Will need these if we start looping over different namespaces!
ising_energy_output = collect("ising_energy", ("training_context",))
susceptibility_output = collect("susceptibility", ("training_contect",))
zero_momentum_2pf_output = collect("zero_momentum_2pf_out", ("training_context",))
effective_pole_mass_output = collect("effective_pole_mass", ("training_context",))
