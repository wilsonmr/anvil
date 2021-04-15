"""
table.py

Module containing all table actions

"""
import numpy as np
import pandas as pd

from reportengine.table import table

@table
def table_stuff_i_want(
    table_fit,
    table_two_point_scalars,
    table_magnetisation,
    table_correlation_length,
    ):
    df = pd.concat([
        table_fit,
        table_two_point_scalars,
        table_magnetisation,
        table_correlation_length,
        ],
    )
    return df


@table
def table_autocorrelation(acceptance, magnetisation_integrated_autocorr, magnetisation_optimal_window, tau_chain):
    tau_mag = magnetisation_integrated_autocorr[magnetisation_optimal_window]

    df = pd.DataFrame([acceptance, tau_mag, tau_chain], index=["acceptance", "tau_mag", "tau_chain"], columns=["value"])
    return df


def table_fit(fit_zero_momentum_correlator, training_geometry):
    popt, pcov, t0 = fit_zero_momentum_correlator

    res = [
            [popt[0], np.sqrt(pcov[0, 0]), training_geometry.length / popt[0]],
            [popt[2], np.sqrt(pcov[2, 2]), np.nan],
        ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation", "L / xi"],
        index=["xi_fit", "m_fit" ],
    )
    return df

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

def table_magnetisation(magnetisation, magnetic_susceptibility):
    res = [
        [magnetisation.mean(), magnetisation.std()],
        [magnetic_susceptibility.mean(), magnetic_susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["magnetisation density", "susceptibility"],
    )
    return df


def table_correlation_length(
    exponential_correlation_length,
    second_moment_correlation_length,
    low_momentum_correlation_length,
    training_geometry,
):
    """Tabulate three estimators of correlation length, with values and errors
    taken as the mean and standard deviation of the bootstrap sample.
    
    The exponential correlation length is provided as a function of 't' separations,
    so to get a scalar estimate the weighted average over positive t>1 is taken.
    """

    # Weighted average of positive shifts, ignoring first point
    T = exponential_correlation_length.shape[0]
    #ecl_values = exponential_correlation_length[1 : T // 2 + 1].mean(axis=-1)
    #ecl_errors = exponential_correlation_length[1 : T // 2 + 1].std(axis=-1)
    ecl_values = np.nanmean(exponential_correlation_length, axis=-1)
    ecl_errors = np.nanstd(exponential_correlation_length, axis=-1)
    weights = 1 / ecl_errors
    ecl_mean = np.sum(ecl_values * weights) / np.sum(weights)
    ecl_error = 1 / np.sum(weights)  # standard error on the mean

    res = [
        [ecl_mean, ecl_error, training_geometry.length / ecl_mean ],
        [
            second_moment_correlation_length.mean(),
            second_moment_correlation_length.std(),
            training_geometry.length / second_moment_correlation_length.mean(),
        ],
        [
            low_momentum_correlation_length.mean(), 
            low_momentum_correlation_length.std(),
            training_geometry.length / low_momentum_correlation_length.mean(),
        ],
    ]
    
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation", "L / xi"],
        index=[
            "Inverse pole mass",
            "Second moment correlation length",
            "Low momentum correlation length",
        ],
    )
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


@table
def table_two_point_correlator_autocorr(two_point_correlator_autocorr):
    """Tabulate the autocorrelation of the two point function"""
    df = pd.DataFrame(
        two_point_correlator_autocorr,
        columns=["Autocorrelation"],
        index=range(len(two_point_correlator_autocorr)),
    )
    return df


@table
def table_topological_observables(topological_charge, topological_susceptibility):
    """Tabulate the topological charge and susceptibility, with values and errors
    taken as the mean and standard deviation of the bootstrap sample."""
    res = [
        [topological_charge.mean(), topological_charge.std()],
        [topological_susceptibility.mean(), topological_susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Topological charge", "Topological susceptibility"],
    )
    return df
