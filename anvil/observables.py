# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
observables.py
"""
import numpy as np
import scipy.signal
import scipy.optimize
import logging

from anvil.utils import bootstrap_sample, Multiprocessing


log = logging.getLogger(__name__)


def cosh_shift(x: np.ndarray, xi: float, A: float, c: float) -> np.ndarray:
    r"""Applies a three-parameter cosh function to a provided array.

    .. math:

        y = A \cosh( -x / \xi ) + c

    """

    return A * np.cosh(-x / xi) + c


def fit_zero_momentum_correlator(
    zero_momentum_correlator, training_geometry, cosh_fit_window
):
    r"""Uses scipy.optimize.curve_fit to fit a cosh function (i.e. exponential decay
    with periodicity) to each correlator in the bootrap ensemble.

    The correlator decays as a pure exponential in the limit of large separations,
    and the characteristic scale of this decay is the correlation length, whose
    reciprocal is a.k.a the (effective) pole mass.

    Parameters
    ----------
    zero_momentum_correlator
        The two point correlation function at zero spatial momentum, i.e. the
        correlation between 1-d 'slices'.
    training_geometry
        The anvil.geometry object defining the lattice.
    cosh_fit_window: slice object
        A slice object which selects the points (i.e. separations) to include in the
        fit. In general the signal at short separations will be contaminated by
        shorter modes and should not be included in the fit.

    Returns
    -------
    xi: list
        List of optimal correlation lengths for each member of the bootstrap ensemble
        for whom the fitting process converged successfully.
    A: list
        Same as above, but for the amplitude of the cosh function.
    c: list
        Same as above, but for the global shift in the fit (which should correspond
        to the absolute value of the magnetization, squared.

    See also
    --------
    :py:func:`anvil.observables.cosh_shift` : the function being fit to the data.
    """
    t = np.arange(training_geometry.length) - training_geometry.length // 2

    # fit for each correlation func in the bootstrap ensemble
    xi, A, c = [], [], []
    for correlator in zero_momentum_correlator.transpose():
        try:
            popt, pcov = scipy.optimize.curve_fit(
                cosh_shift,
                xdata=t[cosh_fit_window],
                ydata=correlator[cosh_fit_window],
            )
            xi.append(popt[0])
            A.append(popt[1])
            c.append(popt[2])
        except RuntimeError:
            pass

    n_boot = zero_momentum_correlator.shape[-1]
    n_fits = len(xi)
    log.info(
        f"Cosh fit succeeded for {n_fits}/{n_boot} members of the bootstrap ensemble."
    )
    return xi, A, c


def correlation_length_from_fit(fit_zero_momentum_correlator):
    """Returns numpy array containing a value for the  correlation length for each member
    of the bootstrap ensemble for whom :py:func:`fit_zero_momentum_correlator` successfully
    converged.
    """
    xi, _, _ = fit_zero_momentum_correlator
    return np.array(xi)


def abs_magnetization_sq_from_fit(fit_zero_momentum_correlator):
    """Returns numpy array containing a value for the absolute magnetization squared
    for each member of the bootstrap ensemble for whom :py:func:`fit_zero_momentum_correlator`
    successfully converged.
    """
    _, _, c = fit_zero_momentum_correlator
    return np.array(c)


def autocorrelation(chain: np.ndarray) -> np.ndarray:
    r"""Returns the autocorrelation function for a one-dimensional array.

    The aucorrelation function is normalised such that :math:`\Gamma(0) = 1` .

    Notes
    -----
    See :py:func:`scipy.signal.correlate` with ``mode="same"`` for more details.
    """
    chain_shifted = chain - chain.mean()
    auto = scipy.signal.correlate(chain_shifted, chain_shifted, mode="same")
    t0 = auto.size // 2  # this is true for mode="same"
    return auto[t0:] / auto[t0]  # normalise and take +ve shifts


def optimal_window(integrated: np.ndarray, mult: float = 2.0) -> int:
    r"""Calculates a window length such that, when the integrated autocorrelation is
    calculated within this window, the sum of statistical and systematic errors is
    minimised according to a self-consistent formula.

    Parameters
    ----------
    integrated
        array containing the cumulative sum of the autocorrelation function, i.e.
        estimates of the integrated autocorrelation in a 'window' of increasing
        size.

    mult
        a multiplicative constant, denoted :math:`S` in the reference below, which
        essentially allows for some tuning of the optimal window based on a visual
        inspection of ``integrated`` using
        :py:func:`anvil.plots.plot_magnetization_integrated_autocorr` . We expect
        the optimal window to correspond to an approximate plateau in ``integrated`` .

    Returns
    -------
    int
        The size of the window, in which the autocorrelation function is summed,
        which is expected to yield an estimate of the integrated autocorrelation
        with the minimum error.

    Notes
    -----
    See U. Wolff, Monte Carlo errors with less errors, section 3.3
    http://arXiv.org/abs/hep-lat/0306017v4


    See Also
    --------
    :py:func:`anvil.plots.plot_magnetization_integrated_autocorr`
    """
    eps = 1e-6

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


def magnetization(
    configs: np.ndarray, bootstrap_sample_size: int, bootstrap_seed: int
) -> np.ndarray:
    """Configuration-wise magnetization for a bootstrapped sample of configurations.

    Parameters
    ----------
    configs
        the sample of field configurations
    bootstrap_sample_size
        number of bootstrap resamples of the original sample
    bootstrap_seed
        integer denoting a seed for the rng used in the bootstrapping, for
        reproducibility purposes.

    Returns
    -------
    np.ndarray
        array of dimensions ``(boostrap_sample_size, sample_size)`` containing
        magnetizations.

    See Also
    --------
    :py:func:`anvil.utils.bootstrap_sample`

    """
    return bootstrap_sample(
        configs.mean(axis=1),
        bootstrap_sample_size,
        seed=bootstrap_seed,
    )


def abs_magnetization_sq(magnetization: np.ndarray) -> np.ndarray:
    """Returns the sample mean of the absolute magnetization, squared, for each member
    of a bootstap ensemble."""
    return np.abs(magnetization).mean(axis=-1) ** 2  # <|m|>^2


def magnetic_susceptibility(
    magnetization: np.ndarray, abs_magnetization_sq: np.ndarray
) -> np.ndarray:
    """Returns the magnetic susceptibility for each member of a bootstrap ensemble."""
    return (magnetization ** 2).mean(axis=-1) - abs_magnetization_sq


def magnetization_series(configs: np.ndarray) -> np.ndarray:
    """Returns the configuration-wise magnetization for a sample of configurations."""
    return configs.sum(axis=1)


def magnetization_autocorr(magnetization_series: np.ndarray) -> np.ndarray:
    """Returns the autocorrelation function for the configuration-wise magnetization
    of a sample of configurations, assuming that the order in which they appear in
    the array corresponds to the order in which they were generated by a dynamical
    process (i.e. Markov-chain Monte Carlo).
    """
    return autocorrelation(magnetization_series)


def magnetization_integrated_autocorr(magnetization_autocorr: np.ndarray) -> np.ndarray:
    """Returns the cumulative sum of the autocorrelation funcion for the configuration-
    wise magnetization, i.e. an estimate of the integrated autocorrelation computed
    in a window of increasing size."""
    return np.cumsum(magnetization_autocorr, axis=-1) - 0.5


def magnetization_optimal_window(
    magnetization_integrated_autocorr: np.ndarray,
) -> np.ndarray:
    """Returns the optimal window size which minimises the total error in an estimate
    of integrated autocorrelation for the configuration-wise magnetization. See
    :py:func:`optimal_window` for details."""
    return optimal_window(magnetization_integrated_autocorr)


# Version without multiprocessing!
# TODO: use or discard?
def _two_point_correlator(
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

    return correlator.reshape(
        (training_geometry.length, training_geometry.length, -1)
    ).numpy()


def two_point_correlator(
    configs: np.ndarray,
    training_geometry,
    bootstrap_sample_size: int,
    bootstrap_seed: int,
    use_multiprocessing: bool,
) -> np.ndarray:
    """Computes the two point correlation function for a bootstrapped sample of
    configurations.

    Parameters
    ----------
    configs
        the sample of field configurations
    training_geometry
        the geometry object defining the lattice
    bootstrap_sample_size
        number of bootstrap resamples of the original sample
    bootstrap_seed
        integer denoting a seed for the rng used in the bootstrapping, for
        reproducibility purposes.
    use_multiprocessing
        if False, do not use ``multiprocessing`` (may be very slow).

    Returns
    -------
    np.ndarray
        array containing the correlation function, dimensions
        ``(training_geometry.length, training_geometry.length, boostrap_sample_size)``

    Notes
    -----
    To reduce peak memory requirements, the correlation for each separation is
    calculated separately. I.e. we vectorize over the sample (using numpy) but
    not over the lattice separations. To speed things up, multiprocessing can
    be used to spread the work over multiple processors, meaning that each
    :math:`G(x_i)` will be allocated to a processor.


    See Also
    --------
    :py:func:`anvil.utils.bootstrap_sample`

    """
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


def two_point_connected_correlator(
    two_point_correlator: np.ndarray, abs_magnetization_sq: np.ndarray
) -> np.ndarray:
    """Connected two point correlation function, obtained by subtracting the expected
    value of the absolute magnetization, squared."""
    return two_point_correlator - abs_magnetization_sq.view(1, 1, -1)


def zero_momentum_correlator(two_point_correlator: np.ndarray) -> np.ndarray:
    """Two point correlation function in time-momentum representation, where the
    momentum is zero. Equivalent to summing over one of the dimensions of the
    correlation function.
    """
    return (two_point_correlator.mean(axis=0) + two_point_correlator.mean(axis=1)) / 2


def effective_pole_mass(zero_momentum_correlator: np.ndarray) -> np.ndarray:
    r"""Effective pole mass defined by

    .. math::

        m_p^\mathrm{eff} = \cosh^{-1} \left(
        \frac{\tilde{G}(t-1) + \tilde{G}(t+1)}{2 \tilde{G}(t)} \right)

    where :math:`\tilde{G}(t)` is the correlator in time-momentum representation
    with momentum :math:`p = 0`.
    """
    inner_indices = np.arange(1, zero_momentum_correlator.shape[0] - 1, dtype=int)
    return np.arccosh(
        (
            zero_momentum_correlator[inner_indices - 1]
            + zero_momentum_correlator[inner_indices + 1]
        )
        / (2 * zero_momentum_correlator[inner_indices])
    )


def susceptibility(two_point_correlator: np.ndarray) -> np.ndarray:
    """Susceptibility as defined by the two point correlation function in
    Fourier space for momentum :math:`(p_1, p_2) = (0, 0)`."""
    return two_point_correlator.sum(axis=(0, 1))


def ising_energy(two_point_correlator):
    """Ising energy density, defined as the two point correlator at the minimum
    lattice spacing."""
    return (two_point_correlator[1, 0] + two_point_correlator[0, 1]) / 2


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


def low_momentum_correlation_length(
    two_point_correlator: np.ndarray, susceptibility: np.ndarray
) -> np.ndarray:
    r"""An estimate for the correlation length based on the low-momentum behaviour
    of the correlation function.

    .. math::

        \xi^2 = \frac{1}{2} \sum_{\mu=1}^2 \frac{1}{4 \sin( \pi / L)} \left(
        \frac{\tilde{G}(0)}{\mathrm{Re}\tilde{G}(\hat{q}_\mu)}
        - 1 \right)

    Here, :math:`\tilde{G}(q)` is the Fourier transform of the correlation function, and
    :math:`\hat{q}_\mu` are the two smallest non-zero momentum vectors on the lattice.

    Specifically, we have
      - :math:`\tilde{G}(0) = \chi` , the susceptibility
      - :math:`\hat{q}_1 = (2\pi/L, 0)`
      - :math:`\hat{q}_2 = (0, 2\pi/L)`

    Reference: https://doi.org/10.1103/PhysRevD.58.105007
    """
    L = two_point_correlator.shape[0]
    kernel = np.cos(2 * np.pi / L * np.arange(L)).reshape(L, 1, 1)

    g_tilde_00 = susceptibility
    g_tilde_10 = (kernel * two_point_correlator).sum(axis=(0, 1))

    xi_sq = (g_tilde_00 / g_tilde_10 - 1) / (4 * np.sin(np.pi / L) ** 2)

    return np.sqrt(xi_sq)
