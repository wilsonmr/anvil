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

from reportengine.table import table
from reportengine.figure import figure

class GreenFunction:
    def __init__(self, states, geometry):
        self.geometry = geometry
        self.sample = states

    def __call__(self, x_0: int, x_1: int, error=False, sequence=False):
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
        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(-1)  # make 1d

        phi = self.sample
        phi_shift = self.sample[:, shift]
        
        if sequence == True:
            term1 = (phi_shift * phi).mean(dim=1)  # average over coordinates
            term2 = phi_shift.mean(dim=1) * phi.mean(dim=1)
            term1var = (phi_shift * phi).var(dim=1)
            term2var = phi_shift.var(dim=1) + phi.var(dim=1)
            e = np.sqrt(term1var + term2var)
            g_func_stack = term1 - term2
            return g_func_stack
            #return g_func_stack / (e * self.geometry.length)

        #  Average over stack of states
        phi_mean = phi.mean(dim=0)
        phi_shift_mean = phi_shift.mean(dim=0)
        phi_shift_phi_mean = (phi_shift * phi).mean(dim=0)
        
        if error == True:
            phi_var = phi.var(dim=0)
            phi_shift_var = phi_shift.var(dim=0)
            phi_shift_phi_var = (phi_shift * phi).var(dim=0)
            Nstates = len(phi[:,0])
            Npoints = len(phi[0,:])

            g_func_error = (
                phi_shift_phi_var / Nstates  # first term error squared
                +  # add squared errors from first and second term
                (phi_shift_mean * phi_mean)**2 * (
                    phi_shift_var / (Nstates * phi_shift_mean**2)
                    + phi_var / (Nstates * phi_mean**2)
                )  # second term: add fractional errors in quadrature
            ).sum().sqrt() / Npoints  # sum over coordinates, sqrt, scale 
            
            return g_func_error
        else:
            g_func = (phi_shift_phi_mean - phi_shift_mean * phi_mean)
            return g_func.mean()  # average over coordinates


def two_point_green_function(sample_training_output, training_geometry):
    r"""Return instance of GreenFunction which can be used to calculate the
    two point green function for a given seperation
    """
    return GreenFunction(sample_training_output[0], training_geometry)

def green_function_autocorrelation(training_geometry, two_point_green_function):
    """Autocorr"""
    output = []
    for coord in ((0, 0), (3, 3)):
        G_series = two_point_green_function(coord[0], coord[1], sequence=True)
        G_series -= G_series.mean()
        autocorr = correlate(G_series, G_series, mode="full")
        c = np.argmax(autocorr)
        output.append(autocorr[c:c+100])
    return output

def zero_momentum_green_function(training_geometry, two_point_green_function):
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
    g_func_zeromom = {'values': [], 'errors': []}
    for t in range(training_geometry.length):
        g_tilde_t = 0
        error_sq = 0
        for x in range(training_geometry.length):
            # not sure if factors are correct here should we account for
            # forward-backward in green function?
            g_tilde_t += float(two_point_green_function(t, x))
            error_sq += float(two_point_green_function(t, x, error=True))**2
        
        g_func_zeromom['values'].append(g_tilde_t / training_geometry.length)
        g_func_zeromom['errors'].append(sqrt(error_sq) / training_geometry.length)

    return g_func_zeromom


def effective_pole_mass(zero_momentum_green_function):
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
    g_func_zeromom = zero_momentum_green_function
    m_t = {'values': [], 'errors': []}
    for i, g_0_t in enumerate(g_func_zeromom['values'][1:-1], start=1):

        numerator = g_func_zeromom['values'][i - 1] + g_func_zeromom['values'][i + 1]
        denominator = 2 * g_0_t
        argument = numerator / denominator
        #m_t['values'].append(acosh(argument))
        m_t['values'].append(0)
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
                g_func_zeromom['errors'][i - 1]**2 + g_func_zeromom['errors'][i + 1]**2
        )
        error_denominator_sq = (2 * g_func_zeromom['errors'][i])**2
        error_argument = (numerator / denominator) * sqrt(
                error_numerator_sq / numerator**2 + error_denominator_sq / denominator**2
        )
        #error_v2 = fabs(acosh(argument + error_argument) - acosh(argument))
        #print("v2: ", error_v2)
        error_v2 = 0
        m_t['errors'].append(error_v2)

    return m_t


def susceptibility(training_geometry, two_point_green_function):
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
            chi += float(two_point_green_function(t, x))
            error_sq += float(
                    two_point_green_function(t, x, error=True)
            )**2  # sum -> add errors in quadrature

    return chi, sqrt(error_sq)


def ising_energy(two_point_green_function):
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
        two_point_green_function(1, 0) + two_point_green_function(0, 1)
    ) / 2  # am I missing a factor of 2?
    error = sqrt(
        two_point_green_function(1, 0, error=True)**2 +
        two_point_green_function(0, 1, error=True)**2
    ) / 2
    return float(E), error


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
    res = [[rf"{ising_energy[0]:.3g} $\pm$ {ising_energy[1]:.1g}"],
           [rf"{susceptibility[0]:.3g} $\pm$ {susceptibility[1]:.1g}"]]
    df = pd.DataFrame(
        res, columns=[training_output.name], index=["Ising energy", "susceptibility"]
    )
    return df


@figure
def plot_zero_momentum_green_function(zero_momentum_green_function, training_geometry):
    fig, ax = plt.subplots()
    ax.errorbar(
            x = range(len(zero_momentum_green_function['values'])),
            y = zero_momentum_green_function['values'],
            yerr = zero_momentum_green_function['errors'],
            fmt = "-r",
            label=f"L = {training_geometry.length}"
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel(r"$t$")
    ax.set_title("Zero momentum Green function")
    return fig

@figure
def plot_effective_pole_mass(training_geometry, effective_pole_mass):
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

@figure
def plot_G(training_geometry, two_point_green_function):

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

@figure
def plot_G_autocorr(green_function_autocorrelation):
    data = green_function_autocorrelation
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_title(r"Autocorrelation of Green function")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Auto")
    ax.plot(data[0], label="G(0,0)")
    ax.plot(data[1], label="G(3,3)")
    ax.legend()
    return fig

@figure
def plot_G_series(two_point_green_function):
    series = two_point_green_function(0, 0, sequence=True)
    fig, ax = plt.subplots()
    ax.set_title("Series")
    ax.set_ylabel(r"G")
    ax.set_xlabel(r"$t$")
    ax.plot(series[:500], '-')
    return fig
