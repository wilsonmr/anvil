"""
observables.py
"""
import numpy as np
from scipy.signal import correlate
from math import ceil, pi, sin
import logging

from anvil.utils import bootstrap_sample, Multiprocessing

import scipy.optimize as optim

log = logging.getLogger(__name__)

def cosh_shift(x, xi, A, c):
    return A * np.cosh(-x / xi) + c


def fit_zero_momentum_correlator(zero_momentum_correlator, training_geometry):
    # TODO should I bootstrap this whole process...?

    T = training_geometry.length
    # TODO: would be good to specify this in runcard
    t0 = T // 4
    window = slice(t0, T - t0 + 1)

    t = np.arange(T)
    y = zero_momentum_correlator.mean(axis=-1)
    yerr = zero_momentum_correlator.std(axis=-1)

    try:
        popt, pcov = optim.curve_fit(
            cosh_shift,
            xdata=t[window] - T // 2,
            ydata=y[window],
            sigma=yerr[window],
        )
        return (popt, pcov, t0)
    except RuntimeError:
        log.warning("Failed to fit cosh to correlation function.")
        return None


def correlation_length_from_fit(fit_zero_momentum_correlator):
    popt, pcov, _ = fit_zero_momentum_correlator
    return popt[0], np.sqrt(pcov[0, 0])


def autocorrelation(chain):
    """Calculate the one-dimensional normalised autocorrelation function for a one-
    dimensional numpy array, given as an argument. Return positive shifts only."""
    chain_shifted = chain - chain.mean()
    auto = correlate(chain_shifted, chain_shifted, mode="same")
    t0 = auto.size // 2  # this is true for mode="same"
    return auto[t0:] / auto[t0]  # normalise and take +ve shifts


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
#                                   Magnetization                                       #
# ------------------------------------------------------------------------------------- #


def magnetization(configs, bootstrap_sample_size, bootstrap_seed):
    """Magnetization per config, bootstrapped. Dimensions (n_boot, n_configs)"""
    return bootstrap_sample(
        configs.mean(axis=1),
        bootstrap_sample_size,
        seed=bootstrap_seed,
    )


def abs_magnetization_squared(magnetization):
    return np.abs(magnetization).mean(axis=-1) ** 2  # <|m|>^2


def magnetic_susceptibility(magnetization, abs_magnetization_squared):
    return (magnetization ** 2).mean(axis=-1) - abs_magnetization_squared


def magnetization_series(configs):
    return configs.sum(axis=1)


def magnetization_autocorr(magnetization_series):
    return autocorrelation(magnetization_series)


def magnetization_integrated_autocorr(magnetization_autocorr):
    return np.cumsum(magnetization_autocorr, axis=-1) - 0.5


def magnetization_optimal_window(magnetization_integrated_autocorr):
    return optimal_window(magnetization_integrated_autocorr)


# ------------------------------------------------------------------------------------- #
#                               Two point observables                                   #
# ------------------------------------------------------------------------------------- #

# Version without multiprocessing!
def __two_point_correlator(
    configs, training_geometry, bootstrap_sample_size, bootstrap_seed
):
    correlator = np.empty((training_geometry.volume, bootstrap_sample_size))
    for i, shift in enumerate(training_geometry.two_point_iterator()):
        correlator[i] = bootstrap_sample(
            (configs[:, shift] * configs).mean(axis=1),  # volume average
            bootstrap_sample_size,
            seed=bootstrap_seed,
        ).mean(
            axis=-1  # sample average
        )

    return correlator.reshape((training_geometry.length, training_geometry.length, -1))


def two_point_correlator(
    configs,
    training_geometry,
    bootstrap_sample_size,
    bootstrap_seed,
    use_multiprocessing,
):
    # NOTE: bootstrap each shift seprately to reduce peak memory requirements
    correlator_single_shift = lambda shift: bootstrap_sample(
        (configs[:, shift] * configs).mean(axis=1),
        bootstrap_sample_size,
        bootstrap_seed,
    ).mean(axis=-1)

    mp_correlator = Multiprocessing(
        func=correlator_single_shift,
        generator=training_geometry.two_point_iterator,
        use_multiprocessing=use_multiprocessing,
    )
    correlator_dict = mp_correlator()

    correlator = np.array([correlator_dict[i] for i in range(training_geometry.volume)])

    return correlator.reshape((training_geometry.length, training_geometry.length, -1))


def two_point_connected_correlator(two_point_correlator, abs_magnetization_squared):
    return two_point_correlator - abs_magnetization_squared.view(1, 1, -1)


def zero_momentum_correlator(two_point_correlator):
    """Two point correlator at zero spatial momentum."""
    return (two_point_correlator.mean(axis=0) + two_point_correlator.mean(axis=1)) / 2


def effective_pole_mass(zero_momentum_correlator):
    r"""Effective pole mass defined by

        m_p = cosh^{-1}( (\tilde{G}(t-1) + \tilde{G}(t+1)) / (2 * \tilde{G}(t)) )

    where \tilde{G}(t) is the zero momentum correlator defined in this module.
    """
    inner_indices = np.arange(1, zero_momentum_correlator.shape[0] - 1, dtype=int)
    return np.arccosh(
        (
            zero_momentum_correlator[inner_indices - 1]
            + zero_momentum_correlator[inner_indices + 1]
        )
        / (2 * zero_momentum_correlator[inner_indices])
    )


def susceptibility(two_point_correlator):
    """Susceptibility defined as the first moment of the two point correlator."""
    return two_point_correlator.sum(axis=(0, 1))


def ising_energy(two_point_correlator):
    """Ising energy density, defined as the two point correlator at the minimum
    lattice spacing."""
    return (two_point_correlator[1, 0] + two_point_correlator[0, 1]) / 2


def inverse_pole_mass(effective_pole_mass, training_geometry):
    T = training_geometry.length
    t0 = T // 4
    window = slice(t0, T - t0 + 1)

    xi = np.reciprocal(effective_pole_mass)[window]

    return np.nanmean(xi, axis=0)  # average over "large" t points


def second_moment_correlation_length(two_point_correlator, susceptibility):
    """Second moment correlation length, defined as the normalised second
    moment of the two point correlator."""
    L = two_point_correlator.shape[0]
    x = np.concatenate((np.arange(0, L // 2 + 1), np.arange(-L // 2 + 1, 0)))
    x1, x2 = np.meshgrid(x, x)
    x_sq = np.expand_dims((x1 ** 2 + x2 ** 2), -1)  # pick up bootstrap dimension

    mu_0 = susceptibility
    mu_2 = (x_sq * two_point_correlator).sum(axis=(0, 1))  # second moment

    xi_sq = mu_2 / (4 * mu_0)  # normalisation

    return np.sqrt(xi_sq)


def low_momentum_correlation_length(two_point_correlator, susceptibility):
    """A low-momentum estimate for the correlation length."""
    L = two_point_correlator.shape[0]
    kernel = np.cos(2 * pi / L * np.arange(L)).reshape(L, 1, 1)

    g_tilde_00 = susceptibility
    g_tilde_10 = (kernel * two_point_correlator).sum(axis=(0, 1))

    xi_sq = (g_tilde_00 / g_tilde_10 - 1) / (4 * sin(pi / L) ** 2)

    return np.sqrt(xi_sq)
