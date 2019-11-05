"""
observables.py

functions for calculating observables on a stack of states.

Functions
---------
print_plot_observables:
    can be considered the main function which in turn calculates the rest
    of the observables defined in this module.}

Notes
-----
Check the definitions of functions, most are defined according the the arxiv
version: https://arxiv.org/pdf/1904.12072.pdf

"""
from math import acosh, sqrt

import matplotlib.pyplot as plt

from normflow.geometry import get_shift

# TODO: length should be deprecated in favour of Geometry later


def two_point_green_function(phi_states, x_0: int, x_1: int, length: int):
    r"""Calculates the two point connected green function given a set of
    states G(x) where x = (x_0, x_1) refers to a shift applied to the fields
    \phi

    Parameters
    ----------
    phi_states: torch.Tensor
        stack of sample states to calculate observables on
    x_0: int
        shift of dimension 0
    x_1: int
        shift of dimension 1
    length: int
        length of 2D lattice site

    Returns
    -------
    g_func: torch.Tensor
        scalar (torch.Tensor with single element) value of green function G(x)

    """
    shift = get_shift(length, shifts=((x_0, x_1),), dims=((0, 1),)).view(-1)  # make 1d

    g_func = (phi_states[:, shift] * phi_states).mean(
        dim=0
    ) - phi_states[  # mean over states
        :, shift
    ].mean(
        dim=0
    ) * phi_states.mean(
        dim=0
    )
    return g_func.mean()  # integrate over y and divide by volume


def zero_momentum_green_function(phi_states, length):
    r"""Calculate the zero momentum green function as a function of t
    \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

        \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

    Parameters
    ----------
    phi_states: torch.Tensor
        stack of sample states to calculate observables on
    length: int
        length of 2D lattice site - to be deprecated in favour of Geometry later

    Returns
    -------
    g_func_zeromom: list
        zero momentum green function as function of t, where t runs from 0 to
        length - 1

    Notes
    -----
    This is \tilde{G}(t, 0) as defined in eq. (23) of
    https://arxiv.org/pdf/1904.12072.pdf (defined as mean instead of sum over
    spacial directions) and with momentum explicitly set to zero.

    """
    g_func_zeromom = []
    for t in range(length):
        g_tilde_t = 0
        for x in range(length):
            # not sure if factors are correct here should we account for
            # forward-backward in green function?
            g_tilde_t += float(two_point_green_function(phi_states, t, x, length))
        g_func_zeromom.append(g_tilde_t / length)
    return g_func_zeromom


def effective_pole_mass(g_func_zeromom):
    r"""Calculate the effective pole mass m^eff(t) defined as

        m^eff(t) = arccosh(
            (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
        )

    from t = 1 to t = L-2, where L is the length of lattice side

    Parameters
    ----------
    g_func_zeromom: list
        zero momentum green function as function of t, where t runs from 0 to
        length - 1

    Returns
    -------
    m_t: list
        effective pole mass as a function of t

    Notes
    -----
    This is m^eff(t) as defined in eq. (28) of
    https://arxiv.org/pdf/1904.12072.pdf

    """
    m_t = []
    for i, g_0_t in enumerate(g_func_zeromom[1:-1], start=1):
        m_t.append(acosh((g_func_zeromom[i - 1] + g_func_zeromom[i + 1]) / (2 * g_0_t)))
    return m_t


def susceptibility(g_func_zeromom):
    r"""Calculate the susceptibility, which is the sum of two point connected
    green functions over all seperations

        \chi = sum_x G(x)

    Parameters
    ----------
    g_func_zeromom: list
        zero momentum green function as a function of t

    Returns
    -------
        chi: float
            value for the susceptibility

    Notes
    -----
    as defined in eq. (25) of https://arxiv.org/pdf/1904.12072.pdf

    """
    # TODO: we can write this in a more efficient and clearer way.
    partial_sum = [len(g_func_zeromom) * el for el in g_func_zeromom]  # undo the mean
    return sum(partial_sum)


def ising_energy(phi_states, length):
    r"""Ising energy defined as

        E = 1/d sum_{\mu} G(\mu)

    where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
    in 2D


    Parameters
    ----------
    phi_states: torch.Tensor
        stack of sampled states used to estimate the Ising energy
    length: int
        length of lattice side

    Returns
    -------
        E: float
            value for the Ising energy

    Notes
    -----
    as defined in eq. (26) of https://arxiv.org/pdf/1904.12072.pdf

    """
    E = (
        two_point_green_function(phi_states, 1, 0, length)
        + two_point_green_function(phi_states, 0, 1, length)
    ) / 2  # am I missing a factor of 2?
    return E


def print_plot_observables(phi_states):
    """Given a sample of states, calculate the relevant observables and either
    print them to terminal or create a figure and save to the cwd

    Parameters
    ----------
    phi_states: torch.Tensor
        stack of sampled states used to estimate observables.

    Output
    -----
        saves greenfunc.png and mass.png to cwd and prints Ising energy and
        Susceptibility.

    """
    length = round(sqrt(phi_states.shape[1]))
    g_func_zeromom = zero_momentum_green_function(phi_states, length)
    m_t = effective_pole_mass(g_func_zeromom)
    susc = susceptibility(g_func_zeromom)
    E = ising_energy(phi_states, length)
    print(f"Ising energy: {E}")
    print(f"Susceptibility: {susc}")

    fig, ax = plt.subplots()
    ax.plot(g_func_zeromom, "-r", label=f"L = {length}")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel("t")
    ax.set_title("Zero momentum Green function")
    fig.tight_layout()
    fig.savefig("greenfunc.png")

    fig, ax = plt.subplots()
    ax.plot(m_t, "-r", label=f"L = {length}")
    ax.set_ylabel(r"$m^{\rm eff}_p(t)$")
    ax.set_xlabel("t")
    ax.set_title("Effective Pole Mass")
    fig.tight_layout()
    fig.savefig("mass.png")
