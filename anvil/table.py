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
    magnetization_optimal_window,
    tau_chain,
    acceptance,
):
    tau_mag = magnetization_integrated_autocorr[magnetization_optimal_window]

    df = pd.DataFrame(
        [acceptance, tau_mag, tau_chain],
        index=["acceptance", "tau_mag", "tau_chain"],
        columns=["value"],
    )
    return df


@table
def table_fit(fit_zero_momentum_correlator, training_geometry):
    if fit_zero_momentum_correlator is not None:
        popt, pcov, t0 = fit_zero_momentum_correlator

        res = [
            [popt[0], np.sqrt(pcov[0, 0])],
            [popt[2], np.sqrt(pcov[2, 2])],
        ]
        df = pd.DataFrame(
            res,
            columns=["Mean", "Standard deviation"],
            index=["xi_fit", "m_fit"],
        )
        return df


@table
def table_two_point_scalars(ising_energy, susceptibility):
    """Table of the ising observables, with mean and standard deviation taken
    across boostrap samples
    """
    res = [
        [ising_energy.mean(), ising_energy.std()],
        [susceptibility.mean(), susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Ising energy", "susceptibility"],
    )
    return df


@table
def table_magnetization(abs_magnetization_squared, magnetic_susceptibility):
    res = [
        [abs_magnetization_squared.mean(), abs_magnetization_squared.std()],
        [magnetic_susceptibility.mean(), magnetic_susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["<|m|>^2", "<m^2> - <|m|>^2"],
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
    """Tabulate four estimates of correlation length, with values and errors
    taken as the mean and standard deviation of the bootstrap sample.
    """
    # Take the mean of the arcosh estimator over "large" separations
    x0 = training_geometry.length // 4
    window = slice(x0, training_geometry.length - x0 + 1)
    xi_arcosh = np.nanmean(
        np.reciprocal(effective_pole_mass)[window],
        axis=0,
    )

    res = [
        list(correlation_length_from_fit),
        [xi_arcosh.mean(), xi_arcosh.std()],
        [
            second_moment_correlation_length.mean(),
            second_moment_correlation_length.std(),
        ],
        [low_momentum_correlation_length.mean(), low_momentum_correlation_length.std()],
    ]

    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=[
            "Estimate from fit",
            "Estimate using arcosh",
            "Second moment estimate",
            "Low momentum estimate",
        ],
    )
    df["No. correlation lengths"] = training_geometry.length / df["Mean"]
    return df


@table
def table_zero_momentum_correlator(zero_momentum_correlator, training_geometry):
    """Table of zero_momentum_correlator, with mean and standard deviation
    from bootstrap
    """
    means = zero_momentum_correlator.mean(axis=-1)[:, np.newaxis]
    stds = zero_momentum_correlator.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)

    df = pd.DataFrame(
        data,
        columns=["Mean", "Standard deviation"],
        index=range(training_geometry.length),
    )
    return df


@table
def table_effective_pole_mass(effective_pole_mass, training_geometry):
    """Table of effective_pole_mass, with mean and standard deviation
    from bootstrap
    """
    means = effective_pole_mass.mean(axis=-1)[:, np.newaxis]
    stds = effective_pole_mass.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)
    df = pd.DataFrame(
        data,
        columns=["Mean", "Standard deviation"],
        index=range(1, training_geometry.length - 1),
    )
    return df


@table
def table_two_point_correlator(training_geometry, two_point_correlator):
    """For each x and t, tabulate the mean and standard deviation of the two
    point function, estimated from bootstrap sample
    """
    corr = []
    index = []
    means = two_point_correlator.mean(axis=-1)
    stds = two_point_correlator.std(axis=-1)

    for i in range(training_geometry.length):
        for j in range(training_geometry.length):
            corr.append([float(means[i, j]), float(stds[i, j])])
            index.append((i, j))
    df = pd.DataFrame(corr, columns=["Mean", "Standard deviation"], index=index)
    return df
