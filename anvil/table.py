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
    acceptance
):
    tau_mag = magnetization_integrated_autocorr[magnetization_optimal_window]

    df = pd.DataFrame(
        [acceptance, tau_mag, tau_chain],
        index=["acceptance", "tau_mag", "tau_chain"],
        columns=["value"],
    )
    return df


@table
def table_fit(fit_zero_momentum_correlator, geometry_from_training):
    popt, pcov, t0 = fit_zero_momentum_correlator

    res = [
        [popt[0], np.sqrt(pcov[0, 0]), geometry_from_training.length / popt[0]],
        [popt[2], np.sqrt(pcov[2, 2]), np.nan],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation", "L / xi"],
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


def table_correlation_length(
    inverse_pole_mass,
    second_moment_correlation_length,
    low_momentum_correlation_length,
    fit_zero_momentum_correlator,
    geometry_from_training,
):
    """Tabulate three estimators of correlation length, with values and errors
    taken as the mean and standard deviation of the bootstrap sample.
    """
    popt, pcov, _ = fit_zero_momentum_correlator
    xi_fit = popt[0]
    xi_fit_std = np.sqrt(pcov[0, 0])

    res = [
        [xi_fit, xi_fit_std],
        [inverse_pole_mass.mean(), inverse_pole_mass.std()],
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
    df["No. correlation lengths"] = geometry_from_training.length / df["Mean"]
    return df


@table
def table_zero_momentum_correlator(zero_momentum_correlator, geometry_from_training):
    """Table of zero_momentum_correlator, with mean and standard deviation
    from bootstrap
    """
    means = zero_momentum_correlator.mean(axis=-1)[:, np.newaxis]
    stds = zero_momentum_correlator.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)

    df = pd.DataFrame(
        data,
        columns=["Mean", "Standard deviation"],
        index=range(geometry_from_training.length),
    )
    return df


@table
def table_effective_pole_mass(effective_pole_mass, geometry_from_training):
    """Table of effective_pole_mass, with mean and standard deviation
    from bootstrap
    """
    means = effective_pole_mass.mean(axis=-1)[:, np.newaxis]
    stds = effective_pole_mass.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)
    df = pd.DataFrame(
        data,
        columns=["Mean", "Standard deviation"],
        index=range(1, geometry_from_training.length - 1),
    )
    return df


@table
def table_two_point_correlator(geometry_from_training, two_point_correlator):
    """For each x and t, tabulate the mean and standard deviation of the two
    point function, estimated from bootstrap sample
    """
    corr = []
    index = []
    means = two_point_correlator.mean(axis=-1)
    stds = two_point_correlator.std(axis=-1)

    for i in range(geometry_from_training.length):
        for j in range(geometry_from_training.length):
            corr.append([float(means[i, j]), float(stds[i, j])])
            index.append((i, j))
    df = pd.DataFrame(corr, columns=["Mean", "Standard deviation"], index=index)
    return df
