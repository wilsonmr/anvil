from math import ceil, floor, log10, fabs

import numpy as np
import pandas as pd

from reportengine.table import table

from anvil.observables import bootstrap

@table
def ising_observables_table(ising_energy, susceptibility, training_output):
    IE, IE_std = float(ising_energy[0]), float(bootstrap(ising_energy))
    S, S_std = float(susceptibility[0]), float(bootstrap(susceptibility))
    res = [[IE, IE_std], [S, S_std]]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Ising energy", "susceptibility"],
    )
    return df


@table
def table_zero_momentum_2pf(zero_momentum_2pf, training_geometry):
    zm2pf, zm2pf_std = zero_momentum_2pf[0, :], bootstrap(zero_momentum_2pf)
    g_tilde = []
    for t in range(training_geometry.length):
        g_tilde.append([float(zm2pf[t]), float(zm2pf_std[t])])

    df = pd.DataFrame(
        g_tilde,
        columns=["Mean", "Standard deviation"],
        index=range(training_geometry.length),
    )
    return df


@table
def table_effective_pole_mass(effective_pole_mass, training_geometry):
    epm, epm_std = effective_pole_mass[0, :], bootstrap(effective_pole_mass)
    m_eff = []
    for t in range(training_geometry.length - 2):
        m_eff.append([float(epm[t]), float(epm_std[t])])

    df = pd.DataFrame(
        m_eff,
        columns=["Mean", "Standard deviation"],
        index=range(1, training_geometry.length - 1),
    )
    return df


@table
def table_2pf(training_geometry, two_point_function):
    corr = []
    for j in range(training_geometry.length ** 2):
        corr.append(
            [
                float(
                    two_point_function(
                        j // training_geometry.length, j % training_geometry.length  # t
                    )[0]
                ),  # x
                float(
                    bootstrap(
                        two_point_function(
                            j // training_geometry.length,
                            j % training_geometry.length,  # t
                        )
                    )  # x
                ),
            ],
        )
    df = pd.DataFrame(
        corr,
        columns=["Mean", "Standard deviation"],
        index=[
            (j // training_geometry.length, j % training_geometry.length)
            for j in range(training_geometry.length ** 2)
        ],
    )
    return df


@table
def table_autocorrelation_2pf(autocorrelation_2pf):
    autocorrelation, _, _, _, _ = autocorrelation_2pf

    df = pd.DataFrame(
        autocorrelation, columns=["Autocorrelation"], index=range(len(autocorrelation)),
    )
    return df
