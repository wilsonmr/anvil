from math import ceil, floor, log10, fabs

import numpy as np
import pandas as pd
from tqdm import tqdm

from reportengine.table import table


def print_to_precision(val, err):
    """Given a value and associated error, returns two strings - the value and error rounded to
    a precision dictated by the first nonzero"""
    val = float(val)
    err = float(err)
    prec = floor(log10(abs(err)))
    if prec < 0:
        err_rnd = round(err, -prec)
        prec = floor(log10(abs(err_rnd)))
        val_str = np.format_float_positional(
            val, -prec, unique=False, fractional=True, pad_right=1
        )
        err_str = np.format_float_positional(err, -prec, fractional=True, pad_left=1)
    else:
        err_rnd = round(err, prec)
        prec = floor(log10(abs(err_rnd)))
        int_prec = ceil(log10(abs(float(val))))
        val_str = np.format_float_positional(
            val, int_prec - prec, fractional=False, pad_right=1
        ).replace(".", "")
        err_str = np.format_float_positional(
            err, 1, fractional=False, pad_left=1
        ).replace(".", "")
    return val_str, err_str


@table
def ising_observables_table(ising_energy, susceptibility, bootstrap, training_output):
    IE, IE_std = print_to_precision(ising_energy[0], bootstrap(ising_energy))
    S, S_std = print_to_precision(susceptibility[0], bootstrap(susceptibility))
    res = [[IE, IE_std], [S, S_std]]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Ising energy", "susceptibility"],
    )
    return df


@table
def table_zero_momentum_2pf(zero_momentum_2pf, training_geometry, bootstrap):
    zm2pf = zero_momentum_2pf[0,:]
    zm2pf_std = bootstrap(zero_momentum_2pf)
    g_tilde = []
    for t in range(training_geometry.length):
        g_tilde.append(print_to_precision(zm2pf[t], zm2pf_std[t]))
    
    df = pd.DataFrame(
            g_tilde,
            columns=["Mean", "Standard deviation"],
            index=range(training_geometry.length),
    )
    return df
    

@table
def table_effective_pole_mass(effective_pole_mass, training_geometry, bootstrap):
    epm = effective_pole_mass[0,:]
    epm_std = bootstrap(effective_pole_mass)
    m_eff = []
    for t in range(training_geometry.length - 2):
        m_eff.append(print_to_precision(epm[t], epm_std[t]))
    
    df = pd.DataFrame(
            m_eff,
            columns=["Mean", "Standard deviation"],
            index=range(1, training_geometry.length-1),
    )
    return df


@table
def table_2pf(training_geometry, two_point_function, bootstrap):
    print("Computing two point function and error...")
    corr = []
    pbar = tqdm(total=training_geometry.length ** 2, desc="(x,t)")
    for j in range(training_geometry.length**2):
        corr.append(
                print_to_precision(
            float(two_point_function(
                j//training_geometry.length,        # t
                j%training_geometry.length)[0]),    # x
            float(bootstrap(two_point_function(
                j//training_geometry.length,        # t
                j%training_geometry.length)))       # x
            )
        )
        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(
        corr,
        columns=["Mean", "Standard deviation"],
        index=[(j//training_geometry.length, j%training_geometry.length) for j in range(training_geometry.length**2)],
)
    return df


@table
def table_autocorrelation_2pf(autocorrelation_2pf):
    autocorrelation, _, _, _, _ = autocorrelation_2pf
    
    df = pd.DataFrame(
        autocorrelation,
        columns=["Autocorrelation"],
        index=range(len(autocorrelation)),
    )
    return df


