"""
observables.py

functions for calculating observables on a stack of states
"""
from math import acosh, sqrt

import matplotlib.pyplot as plt

from geometry import get_shift

def two_point_green_function(phi_states, x_0, x_1, length):
    """Calculates the two point connected green function given a set of
    states G(x) where x = (x_0, x_1)
    """
    shift = get_shift(
        length,
        shifts=((x_0, x_1),),
        dims=((0, 1),),
    ).view(-1) # make 1d

    g_func = (
        (phi_states[:, shift]*phi_states).mean(dim=0) - # mean over states
        phi_states[:, shift].mean(dim=0)*phi_states.mean(dim=0)
    )
    return g_func.mean() # integrate over y and divide by volume

def zero_momentum_green_function(phi_states, length):
    """Calculate the zero momentum green function as a function of t which
    is assumed to be in the first dimension
    """
    g_0_ts = []
    for t in range(length):
        g_0_t = 0
        for x in range(length):
            # not sure if factors are correct here should we account for
            # forward-backward in green function?
            g_0_t += float(two_point_green_function(phi_states, t, x, length))
        g_0_ts.append(g_0_t)
    return g_0_ts

def effective_pole_mass(g_0_ts):
    """Calculate the effective pole mass as a function of t, from t=1 to t=L-2"""
    m_t = []
    for i, g_0_t in enumerate(g_0_ts[1:-1], start=1):
        m_t.append(acosh((g_0_ts[i-1] + g_0_ts[i+1])/(2*g_0_t)))
    return m_t

def susceptibility(g_0_ts):
    """Calculate the susceptibility, which is the sum of two point connected
    green functions over all seperations. Since we are calculating the zero
    momentum green function we can just sum over t"""
    return sum(g_0_ts)

def ising_energy(phi_states, length):
    """Ising energy as defined in eq 26"""
    E = (
        two_point_green_function(phi_states, 1, 0, length) +
        two_point_green_function(phi_states, 0, 1, length)
    )/2 # am I missing a factor of 2?
    return E

def print_plot_observables(phi_states):
    """Function which given a sample of states, calculates the set of
    observables
    """
    length = round(sqrt(phi_states.shape[1]))
    g_0_ts = zero_momentum_green_function(phi_states, length)
    m_t = effective_pole_mass(g_0_ts)
    susc = susceptibility(g_0_ts)
    E = ising_energy(phi_states, length)
    print(f"Ising Energy: {E}")
    print(f"Susceptibility: {susc}")

    fig, ax = plt.subplots()
    ax.plot(g_0_ts, '-r', label=f"L = {length}")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel("t")
    ax.set_title("Zero momentum Green function")
    fig.tight_layout()
    fig.savefig("greenfunc.png")

    fig, ax = plt.subplots()
    ax.plot(m_t, '-r', label=f"L = {length}")
    ax.set_ylabel(r"m^{\rm eff}_p(t)$")
    ax.set_xlabel("t")
    ax.set_title("Effective Pole Mass")
    fig.tight_layout()
    fig.savefig("mass.png")
