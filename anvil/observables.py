"""
observables.py

functions for calculating observables on a stack of states.

Notes
-----
Check the definitions of functions, most are defined according the the arxiv
version: https://arxiv.org/pdf/1904.12072.pdf

"""
import numpy as np
from scipy.signal import correlate
import torch


def arcosh(x):
    """Inverse hyperbolic cosine function for torch.Tensor arguments.

        arcosh(x) = log_e (x + sqrt(x^2 - 1) )
                  = log_e (x) + log_e (1 + sqrt(x^2 - 1) / x)
    """
    # c0 = torch.log(x)
    # c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    # return c0 + c1
    # NOTE: might need stable version here
    return torch.log(x + torch.sqrt(pow(x, 2) - 1))


def field_ensemble(
    sample_training_output, training_geometry, FieldClass, field_dimension
):
    return FieldClass(
        training_output=sample_training_output,
        geometry=training_geometry,
        field_dimension=field_dimension,
    )


def volume_avg_two_point_function(field_ensemble):
    return field_ensemble.volume_avg_two_point_function()


def two_point_function(field_ensemble, n_boot=100):
    return field_ensemble.two_point_function(n_boot=n_boot)


def topological_charge_ens(field_ensemble, n_boot=100):
    return field_ensemble.topological_charge(n_boot=n_boot)


def zero_momentum_two_point(two_point_function):
    r"""Calculate the zero momentum green function as a function of t
    \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

        \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

    Returns
    -------
    g_func_zeromom: torch.Tensor
        Zero momentum green function as function of t, where t is zero indexed.
        Tensor of size (lattice length, n_boot)

    Notes
    -----
    This is \tilde{G}(t, 0) as defined in eq. (23) of
    https://arxiv.org/pdf/1904.12072.pdf (defined as mean instead of sum over
    spacial directions) and with momentum explicitly set to zero.

    """
    # mean across x
    g_tilde_t = two_point_function.mean(dim=1)
    return g_tilde_t


def ising_energy(two_point_function):
    r"""Ising energy density defined as

        E = sum_{\mu} G(\mu)

    where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
    in 2D

    The choice of name derives from the classical spin models, for which the RHS
    of the equation above is equal to the expectation value of the Hamiltonian
    density for pure nearest-neighbour interactions and no external field.

    Returns
    -------
    E: torch.Tensor
        value for the energy density Tensor of size n_boot

    Notes
    -----
    In eq. (26) of https://arxiv.org/pdf/1904.12072.pdf this is defined with
    an additional factor of 1/d where d is the lattice dimension.

    """
    return two_point_function[1, 0] + two_point_function[0, 1]


def susceptibility(two_point_function):
    r"""Calculate the sum of two point connected correlation functions over
    all seperations.

        \chi = sum_x G(x)

    For classical spin models, this is equal to the magnetic susceptibility.

    Returns
    -------
    chi: torch.Tensor
        value for the susceptibility Tensor of size n_boot

    Notes
    -----
    as defined in eq. (25) of https://arxiv.org/pdf/1904.12072.pdf

    """
    return two_point_function.sum(dim=(0, 1))


def heat_capacity(volume_avg_two_point_function, ising_energy, training_geometry):
    r"""Calculates the second central moment of the action, which is proportional
    to the heat capacity for thermodynamic systems.

        C ~ < S^2 > - < S >^2

    where the "action" is (-1 times) whatever is exponentiated in the functional
    integral defining the theory, and < > denotes an expectation value taken over
    an ensemble of configurations.

    The first term is

        < S^2 > ~ < \sum_\mu G_V(\mu) >

    where G_V is the volume-averaged two point function. The second term is

        < S >^2 ~ E^2

    For classical thermodynamic models with inverse temperature \beta = T^{-1}
    and Hamiltonian defined as \beta H = S, the constant of proportionality
    for both terms is V^2 * \beta^2.

    The convention used here is to take the specific heat capacity 'c', which
    means the factor of V^2 becomes just V, but to ignore the factor of \beta^2
    since not all of the actions used have a single parameter representing
    nearest-neighbour coupling strength.

    Thus, the function returns T^2 c.
    """
    # TODO: this may not be a sensible convention.
    # TODO: the bootstrapping for this function is wrong - no boot dimension on va_2pf
    heat_cap = training_geometry.length ** 2 * (
        volume_avg_two_point_function[1, 0, :] + volume_avg_two_point_function[0, 1, :]
    ).pow(2).mean() - ising_energy.pow(2)
    return heat_cap


def effective_pole_mass(zero_momentum_two_point):
    r"""Calculate the effective pole mass m^eff(t) defined as

        m^eff(t) = arcosh(
            (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
        )

    from t = 1 to t = L-2, where L is the length of lattice side

    Returns
    -------
    m_t: torch.Tensor
        effective pole mass as a function of t
        Tensor of size (lattice length - 2, n_boot),

    Notes
    -----
    This is m^eff(t) as defined in eq. (28) of
    https://arxiv.org/pdf/1904.12072.pdf

    """
    inner_indices = torch.tensor(range(1, zero_momentum_two_point.shape[0] - 1))
    res = arcosh(
        (
            zero_momentum_two_point[inner_indices - 1]
            + zero_momentum_two_point[inner_indices + 1]
        )
        / (2 * zero_momentum_two_point[inner_indices])
    )
    return res


def topological_charge(topological_charge_ens):
    return topological_charge_ens.mean(dim=0)


def topological_susceptibility(topological_charge_ens, training_geometry):
    return topological_charge_ens.pow(2).mean(dim=0) / training_geometry.length ** 2


def autocorr_two_point(volume_avg_two_point_function, window=2.0):
    r"""Computes the autocorrelation function of the volume-averaged
    two point function for each field configuration in the sample.
    
    The sample of field configurations results from sampling a Markov chain.
    Thus, residual autocorrelations will remain, unless the sampling interval
    is sufficiently large. Hence, we should treat the sample as a series.

    The autocorrelation of a quantity G is an ensemble average of G at
    a reference point 'k' and a second point 'k+t':
        
        \Gamma(k, t; G) = <G(k)G(k+t)> - <G(k)><G(k+t)>

    However, if we make the assumption that the sample has been taken from a
    Markov chain which has reached a stationary state (i.e. it has thermalised),
    then we can replace the ensemble averages with averages over the sample.

    The autocorrelation function for a sample of size N is then

        \Gamma(t; G) = (N-t)^{-1} \sum_{k=1}^{N-t} G(k) G(k+t)
                       - N^{-2} ( \sum_{k=1}^N G(k) )^2
    """
    # TODO: look at more than one seperation
    va_2pf = volume_avg_two_point_function[0, 1, :]
    va_2pf -= va_2pf.mean()
    # converts to numpy array
    autocorrelation = correlate(va_2pf, va_2pf, mode="same")
    c = np.argmax(autocorrelation)
    autocorrelation = autocorrelation[c:] / autocorrelation[c]
    return autocorrelation


def integrated_autocorr_two_point(autocorr_two_point):
    r"""Calculate the integrated autocorrelation of the two point function.

    Integrated autocorrelation is defined, for some window size 'W' by

        \tau_{int}(W) = 0.5 + sum_t^W \Gamma(t)
    """
    return 0.5 + np.cumsum(autocorr_two_point[1:])


def exp_autocorr_two_point(integrated_autocorr_two_point, window=2.0):
    r"""Calculate the exponential autocorrelation of the two point function.

    Exponential autocorrelation is estimated, up to a factor of S as

        S / \tau_{exp}(W) = log( (2\tau_int(W) + 1) / (2\tau_int(W) - 1) )

    """
    tau_int_W = integrated_autocorr_two_point
    valid = np.where(tau_int_W > 0.5)[0]
    tau_exp_W = np.ones(tau_int_W.size) * 0.00001  # to prevent domain error in log

    tau_exp_W[valid] = window / (
        np.log((2 * tau_int_W[valid] + 1) / (2 * tau_int_W[valid] - 1))
    )
    return tau_exp_W


def automatic_windowing_function(
    integrated_autocorr_two_point,
    exp_autocorr_two_point,
    volume_avg_two_point_function,
    window=2.0,
):
    r"""Return the function for estimating optimal window size for integrated
    autocorrelation as defined in equation (52), section 3.3 of
    https://arxiv.org/pdf/hep-lat/0306017.pdf

    The "g" function has a minimum at 'W_opt' where the sum of the statistical
    error and the systematic error due to truncation, in \tau_{int}, has a minimum.

        g(W) = exp( -W / \tau_{exp}(W) ) - \tau_{exp}(W) / \sqrt(W*N)

    """
    sample_size = volume_avg_two_point_function.shape[-1]
    tau_int = integrated_autocorr_two_point
    tau_exp = exp_autocorr_two_point
    windows = np.arange(1, tau_int.size + 1)
    return np.exp(-windows / tau_exp) - tau_exp / np.sqrt(windows * sample_size)


def optimal_window(automatic_windowing_function):
    """using automatic_windowing_function, estimate optimal window, which
    is the first point at which the automatic_windowing_function becomes
    negative
    """
    return np.where(automatic_windowing_function < 0)[0][0]
