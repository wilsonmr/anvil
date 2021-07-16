# SPDX-License-Identifier: GPL-3.0-or-later
# Copywrite © 2021 anvil Michael Wilson, Joe Marsh Rossney, Luigi Del Debbio
"""
table.py

Module containing all table actions

"""
import numpy as np
import pandas as pd

from reportengine.table import table


@table
def table_autocorrelation(
    magnetization_integrated_autocorr,
    magnetization_optimal_window: int,
    tau_chain: float,
    acceptance: float,
):
    r"""
    Tabulate some information related to the statistical efficiency of the Metropolis-
    Hastings sampling phase.

    Parameters
    ----------
    magnetization_integrated_autocorr
        Array containing the cumulative sum of the autocorrelation function of the
        magnetization for each configuration in the sample output by the Metropolis-
        Hastings sampling phase.
    magnetization_optimal_window
        Integer corresponding to a window size in which the autocorrelation function
        should be summed, such that the resulting estimate of the integrated
        autocorrelation has the smallest possible total error.
    tau_chain
        Estimate of the integrated autocorrelation using the accept-reject statistics
        of the sampling phase.
    acceptance
        Fraction of proposals which were accepted in the sampling phase.

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.observables.optimal_window`
    :py:func:`anvil.sample.calc_tau_chain`
    """
    tau_mag = magnetization_integrated_autocorr[magnetization_optimal_window]

    df = pd.DataFrame(
        [acceptance, tau_mag, tau_chain],
        index=["acceptance", "tau_from_magnetization", "tau_from_chain"],
        columns=["value"],
    )
    return df


@table
def table_fit(correlation_length_from_fit, abs_magnetization_sq_from_fit):
    r"""Tabulate the correlation length and magnetization estimates resulting from the
    fitting of a cosh to the correlation function.

    Values and errors are means and standard deviations over a bootstrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Parameters
    ----------
    correlation_length_from_fit
    abs_magnetization_sq_from_fit

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.observables.fit_zero_momentum_correlator`
    """
    res = [
        [correlation_length_from_fit.mean(), correlation_length_from_fit.std()],
        [abs_magnetization_sq_from_fit.mean(), abs_magnetization_sq_from_fit.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["mean", "error"],
        index=["xi_from_fit", "abs_magnetization_sq_from_fit"],
    )
    return df


@table
def table_two_point_scalars(ising_energy, susceptibility):
    r"""Table of scalar observables derived from the two point correlation function.

    Values and errors are means and standard deviations over a bootstrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Parameters
    ----------
    ising_energy
        Nearest-neighbour iteraction energy.
    susceptibility
        Magnetic susceptibility defined by the sum of the correlation function.

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.observables.ising_energy`
    :py:func:`anvil.observables.susceptibility`
    """
    res = [
        [ising_energy.mean(), ising_energy.std()],
        [susceptibility.mean(), susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["mean", "error"],
        index=["ising_energy", "susceptibility"],
    )
    return df


@table
def table_magnetization(abs_magnetization_sq, magnetic_susceptibility):
    r"""Table containing quantities derived from the sample-averaged magnetization.

    Values and errors are means and standard deviations over a bootstrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Parameters
    ----------
    abs_magnetization_sq
        Array containing the sample mean of the absolute magnetization, squared, for each
        member of the bootstrap ensemble.
    magnetic_susceptibility
        Array containing the susceptibility for each member of the bootstrap ensemble.

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.tables.table_two_point_scalars`
    :py:func:`anvil.tables.table_fit`
    """

    res = [
        [abs_magnetization_sq.mean(), abs_magnetization_sq.std()],
        [magnetic_susceptibility.mean(), magnetic_susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["mean", "error"],
        index=["abs_magnetization_sq", "magnetic_susceptibility"],
    )
    return df


@table
def table_correlation_length(
    effective_pole_mass,
    second_moment_correlation_length,
    low_momentum_correlation_length,
    correlation_length_from_fit,
    training_geometry,
):
    r"""Table containing four estimates of correlation length.

    Values and errors are means and standard deviations over a bootstrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Also displays the number of correlation lengths that can fit on the lattice, i.e.
    :math:`\xi / L` where :math:`\xi` is the correlation length and :math:`L` is the
    linear extent of the lattice.

    Parameters
    ----------
    effective_pole_mass
        Array containing estimate of the effective pole mass, for each separation
        and each member of the bootstrap ensemble
    second_moment_correlation_length
        Estimate of the correlation length based on the second moment of the
        two point correlation function, for each member of the bootstrap ensemble.
    low_momentum_correlation_length
        Array containing a low-momentum estimate of the correlation length for each
        member of the bootstrap ensemble.
    correlation_length_from_fit
        Array containing an estimate of the correlation length from a cosh fit to
        the correlation function, for each member of the bootstrap ensemble.
    training_geometry
        Geometry object defining the lattice.

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.observables.fit_zero_momentum_correlator`
    :py:func:`anvil.plot.plot_correlation_length`
    """
    # Take the mean of the arcosh estimator over "large" separations
    x0 = training_geometry.length // 4
    window = slice(x0, training_geometry.length - x0 + 1)
    xi_arcosh = np.nanmean(
        np.reciprocal(effective_pole_mass)[window],
        axis=0,
    )

    res = [
        [correlation_length_from_fit.mean(), correlation_length_from_fit.std()],
        [xi_arcosh.mean(), xi_arcosh.std()],
        [
            second_moment_correlation_length.mean(),
            second_moment_correlation_length.std(),
        ],
        [low_momentum_correlation_length.mean(), low_momentum_correlation_length.std()],
    ]

    df = pd.DataFrame(
        res,
        columns=["mean", "error"],
        index=[
            "xi_from_fit",
            "xi_from_arcosh",
            "xi_from_second_moment",
            "xi_from_low_momentum",
        ],
    )
    df["n_correlation_lengths"] = training_geometry.length / df["mean"]
    return df


@table
def table_zero_momentum_correlator(zero_momentum_correlator, training_geometry):
    r"""Table containing values of the two point correlation function in time-momentum
    representation at zero momentum, for each separation.

    Values and errors are means and standard deviations over a bootstrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Parameters
    ----------
    zero_momentum_correlator
        Array containing the correlation function for each 1-d separation, for each
        member of the bootstrap ensemble.
    training_geometry
        Geometry object defining the lattice.

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.plot.plot_zero_momentum_correlator`
    """
    means = zero_momentum_correlator.mean(axis=-1)[:, np.newaxis]
    stds = zero_momentum_correlator.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)

    df = pd.DataFrame(
        data,
        columns=["mean", "error"],
        index=range(training_geometry.length),
    )
    return df


@table
def table_effective_pole_mass(effective_pole_mass, training_geometry):
    r"""Table containing values of the effective pole mass for each separation.

    Values and errors are means and standard deviations over a bootstrap ensemble,
    which is assumed to be the last (``-1``) dimension of input arrays.

    Parameters
    ----------
    effective_pole_mass
        Array containing the effective pole mass for each separation, for each
        member of the bootstrap ensemble.
    training_geometry
        Geometry object defining the lattice.

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.plot.plot_effective_pole_mass`
    """
    means = effective_pole_mass.mean(axis=-1)[:, np.newaxis]
    stds = effective_pole_mass.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)
    df = pd.DataFrame(
        data,
        columns=["mean", "error"],
        index=range(1, training_geometry.length - 1),
    )
    return df


@table
def table_two_point_correlator(two_point_correlator, training_geometry):
    r"""Table containing values of the two point correlation function for each
    two-dimensional separation.

    Values and errors are means and standard deviations over a bootstrap ensemble
    which is assumed to be the last (``-1``) dimension of input arrays.

    Parameters
    ----------
    two_point_correlator
        Array containing the correlation function for each 2-d separation, for each
        member of the bootstrap ensemble.
    training_geometry
        Geometry object defining the lattice.

    Returns
    -------
    pandas.core.frame.DataFrame

    See Also
    --------
    :py:func:`anvil.plot.plot_two_point_correlator`
    :py:func:`anvil.plot.plot_two_point_correlator_error`
    """
    corr = []
    index = []
    means = two_point_correlator.mean(axis=-1)
    stds = two_point_correlator.std(axis=-1)

    for i in range(training_geometry.length):
        for j in range(training_geometry.length):
            corr.append([float(means[i, j]), float(stds[i, j])])
            index.append((i, j))
    df = pd.DataFrame(corr, columns=["mean", "error"], index=index)
    return df
