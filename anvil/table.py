"""
table.py

Module containing all table actions

"""
import numpy as np
import pandas as pd

from reportengine.table import table


@table
def ising_observables_table(ising_energy, susceptibility):
    """Table of the ising observables, with mean and standard deviation taken
    across boostrap samples
    """
    # annoying that tensors have to be cast to float
    res = [
        [float(ising_energy.mean()), float(ising_energy.std())],
        [float(susceptibility.mean()), float(susceptibility.std())],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Ising energy", "susceptibility"],
    )
    return df


@table
def table_zero_momentum_two_point(zero_momentum_two_point, training_geometry):
    """Table of zero_momentum_two_point, with mean and standard deviation
    from bootstrap
    """
    means = zero_momentum_two_point.mean(dim=-1).numpy()[:, np.newaxis]
    stds = zero_momentum_two_point.std(dim=-1).numpy()[:, np.newaxis]

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
    means = effective_pole_mass.mean(dim=-1).numpy()[:, np.newaxis]
    stds = effective_pole_mass.std(dim=-1).numpy()[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)
    df = pd.DataFrame(
        data,
        columns=["Mean", "Standard deviation"],
        index=range(1, training_geometry.length - 1),
    )
    return df


@table
def table_two_point_function(training_geometry, two_point_function):
    """For each x and t, tabulate the mean and standard deviation of the two
    point function, estimated from bootstrap sample
    """
    corr = []
    index = []
    means = two_point_function.mean(dim=-1)
    stds = two_point_function.std(dim=-1)

    for i in range(training_geometry.length):
        for j in range(training_geometry.length):
            corr.append([float(means[i, j]), float(stds[i, j])])
            index.append((i, j))
    df = pd.DataFrame(corr, columns=["Mean", "Standard deviation"], index=index)
    return df


@table
def table_autocorrelation_2pf(autocorr_two_point):
    """Tabulate the autocorrelation of the two point function"""
    df = pd.DataFrame(
        autocorr_two_point,
        columns=["Autocorrelation"],
        index=range(len(autocorr_two_point)),
    )
    return df
