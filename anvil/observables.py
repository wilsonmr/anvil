"""
observables.py
"""
import numpy as np
from scipy.signal import correlate
from math import ceil, pi, sin

from anvil.utils import bootstrap_sample


def autocorrelation(chain):
    """Calculate the one-dimensional normalised autocorrelation function on the final
    dimension of numpy array, given as an argument. Return positive shifts only."""
    chain_shifted = chain - chain.mean(
        axis=-1, keepdims=True
    )  # expect ensemble dimension at -1
    auto = correlate(chain_shifted, chain_shifted, mode="same")
    t0 = auto.shape[-1] // 2  # this is true for mode="same"
    return auto[..., t0:] / auto[..., [t0]]  # normalise and take +ve shifts


def optimal_window(integrated, mult=2.0, eps=1e-6):
    """Calculates a window length such that, when the integrated autocorrelation is
    calculated within this window, the total error is at a minimum.
    
    Notes
    -----
    See U. Wolff, Monte Carlo errors with less errors, section 3.3
    http://arXiv.org/abs/hep-lat/0306017v4
    """
    # Exponential autocorrelation
    with np.errstate(invalid="ignore", divide="ignore"):
        exponential = np.clip(
            np.nan_to_num(mult / np.log((2 * integrated + 1) / (2 * integrated - 1))),
            a_min=eps,
            a_max=None,
        )

    # Infer ensemble size. Assumes correlation mode was 'same'
    n_t = integrated.shape[-1]
    ensemble_size = n_t * 2

    # g_func is the derivative of the sum of errors wrt window size
    window = np.arange(1, n_t + 1)
    g_func = np.exp(-window / exponential) - exponential / np.sqrt(
        window * ensemble_size
    )

    # Return first occurrence of g_func changing sign
    return np.argmax((g_func[..., 1:] < 0), axis=-1)


# ------------------------------------------------------------------------------------- #
#                               Two point observables                                   #
# ------------------------------------------------------------------------------------- #


def two_point_correlator(field_ensemble, n_boot):
    return field_ensemble.boot_two_point_correlator(n_boot)


def zero_momentum_correlator(two_point_correlator):
    return 0.5 * (two_point_correlator.mean(axis=0) + two_point_correlator.mean(axis=1))


def effective_pole_mass(zero_momentum_correlator):
    inner_indices = np.arange(1, zero_momentum_correlator.shape[0] - 1, dtype=int)
    return np.arccosh(
        (
            zero_momentum_correlator[inner_indices - 1]
            + zero_momentum_correlator[inner_indices + 1]
        )
        / (2 * zero_momentum_correlator[inner_indices])
    )


def exponential_correlation_length(effective_pole_mass):
    return (1 / effective_pole_mass) ** 2


def susceptibility(two_point_correlator):
    return two_point_correlator.sum(axis=(0, 1))


def ising_energy(two_point_correlator):
    return (two_point_correlator[1, 0] + two_point_correlator[0, 1]) / 2


def second_moment_correlation_length(two_point_correlator, susceptibility):
    L = two_point_correlator.shape[0]
    x = np.concatenate((np.arange(0, L // 2 + 1), np.arange(-L // 2 + 1, 0)))
    x1, x2 = np.meshgrid(x, x)
    x_sq = np.expand_dims((x1 ** 2 + x2 ** 2), -1)  # pick up bootstrap dimension

    mu_0 = susceptibility
    mu_2 = (x_sq * two_point_correlator).sum(axis=(0, 1))  # second moment

    return mu_2 / (4 * mu_0)  # normalisation


def low_momentum_correlation_length(two_point_correlator, susceptibility):
    L = two_point_correlator.shape[0]
    kernel = np.cos(2 * pi / L * np.arange(L)).reshape(L, 1, 1)

    g_tilde_00 = susceptibility
    g_tilde_10 = (kernel * two_point_correlator).sum(axis=(0, 1))

    return (g_tilde_00 / g_tilde_10 - 1) / (4 * sin(pi / L) ** 2)


def two_point_correlator_series(field_ensemble):
    return field_ensemble.two_point_correlator_series


def two_point_correlator_autocorr(two_point_correlator_series):
    return autocorrelation(two_point_correlator_series)


def two_point_correlator_integrated_autocorr(two_point_correlator_autocorr):
    return np.cumsum(two_point_correlator_autocorr, axis=-1) - 0.5


def two_point_correlator_optimal_window(two_point_correlator_integrated_autocorr):
    return optimal_window(two_point_correlator_integrated_autocorr)


# ------------------------------------------------------------------------------------- #
#                             Topological observables                                   #
# ------------------------------------------------------------------------------------- #


def topological_charge_series(field_ensemble):
    return field_ensemble.topological_charge


def _topological_charge(topological_charge_series, n_boot):
    return bootstrap_sample(topological_charge_series, n_boot)


def topological_charge(_topological_charge):
    return _topological_charge.mean(axis=-1)


def topological_susceptibility(_topological_charge, training_geometry):
    return _topological_charge.var(axis=-1) / training_geometry.length ** 2


def topological_charge_autocorr(topological_charge_series):
    return autocorrelation(topological_charge_series)


def topological_charge_integrated_autocorr(topological_charge_autocorr):
    return np.cumsum(topological_charge_autocorr) - 0.5


def topological_charge_optimal_window(topological_charge_integrated_autocorr):
    return optimal_window(topological_charge_integrated_autocorr)
