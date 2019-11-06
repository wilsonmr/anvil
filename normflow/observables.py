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
from math import acosh
from functools import lru_cache

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class PhiFourAction(nn.Module):
    """Extend the nn.Module class to return the phi^4 action given either
    a single state size (1, length * length) or a stack of N states
    (N, length * length). See Notes about action definition.

    Parameters
    ----------
    length: int
        defines 2D lattice size (length * length)
    m_sq: float
        the value of the bare mass squared
    lam: float
        the value of the bare coupling

    Examples
    --------
    Consider the toy example of the action acting on a random state

    >>> action = PhiFourAction(2, 1, 1)
    >>> state = torch.rand((1, 2*2))
    >>> action(state)
    tensor([[0.9138]])

    Now consider a stack of states

    >>> stack_of_states = torch.rand((5, 2*2))
    >>> action(stack_of_states)
    tensor([[3.7782],
            [2.8707],
            [4.0511],
            [2.2342],
            [2.6494]])

    Notes
    -----
    that this is the action as defined in
    https://doi.org/10.1103/PhysRevD.100.034515 which might differ from the
    current version on the arxiv.

    """

    def __init__(self, m_sq, lam, geometry):
        super(PhiFourAction, self).__init__()
        self.geometry = geometry
        self.shift = self.geometry.get_shift()
        self.lam = lam
        self.m_sq = m_sq
        self.length = self.geometry.length

    def forward(self, phi_state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass, returning action for stack of states.

        see class Notes for details on definition of action.
        """
        action = (
            (2 + 0.5 * self.m_sq) * phi_state ** 2
            + self.lam * phi_state ** 4  # phi^2 terms
            - torch.sum(  # phi^4 term
                phi_state[:, self.shift] * phi_state.view(-1, 1, self.length ** 2),
                dim=1,
            )  # derivative
        ).sum(
            dim=1, keepdim=True
        )  # sum across sites
        return action


class Observables:
    def __init__(self, phi_states, geometry, outpath):
        self.geometry = geometry
        self.sample = phi_states
        self.outpath = outpath

    def two_point_green_function(self, x_0: int, x_1: int):
        r"""Calculates the two point connected green function given a set of
        states G(x) where x = (x_0, x_1) refers to a shift applied to the fields
        \phi

        Parameters
        ----------
        x_0: int
            shift of dimension 0
        x_1: int
            shift of dimension 1

        Returns
        -------
        g_func: torch.Tensor
            scalar (torch.Tensor with single element) value of green function G(x)

        """

        shift = self.geometry.get_shift(shifts=((x_0, x_1),), dims=((0, 1),)).view(
            -1
        )  # make 1d

        g_func = (self.sample[:, shift] * self.sample).mean(
            dim=0
        ) - self.sample[  # mean over states
            :, shift
        ].mean(
            dim=0
        ) * self.sample.mean(
            dim=0
        )
        return g_func.mean()  # integrate over y and divide by volume

    @lru_cache(maxsize=8)
    def zero_momentum_green_function(self):
        r"""Calculate the zero momentum green function as a function of t
        \tilde{G}(t, 0) which is assumed to be in the first dimension defined as

            \tilde{G}(t, 0) = 1/L \sum_{x_1} G(t, x_1)

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
        for t in range(self.geometry.length):
            g_tilde_t = 0
            for x in range(self.geometry.length):
                # not sure if factors are correct here should we account for
                # forward-backward in green function?
                g_tilde_t += float(self.two_point_green_function(t, x))
            g_func_zeromom.append(g_tilde_t / self.geometry.length)
        return g_func_zeromom

    def effective_pole_mass(self):
        r"""Calculate the effective pole mass m^eff(t) defined as

            m^eff(t) = arccosh(
                (\tilde{G}(t-1, 0) + \tilde{G}(t+1, 0)) / (2 * \tilde{G}(t, 0))
            )

        from t = 1 to t = L-2, where L is the length of lattice side

        Returns
        -------
        m_t: list
            effective pole mass as a function of t

        Notes
        -----
        This is m^eff(t) as defined in eq. (28) of
        https://arxiv.org/pdf/1904.12072.pdf

        """
        g_func_zeromom = self.zero_momentum_green_function()
        m_t = []
        for i, g_0_t in enumerate(g_func_zeromom[1:-1], start=1):
            m_t.append(
                acosh((g_func_zeromom[i - 1] + g_func_zeromom[i + 1]) / (2 * g_0_t))
            )
        return m_t

    def susceptibility(self):
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
        g_func_zeromom = self.zero_momentum_green_function()
        # TODO: we can write this in a more efficient and clearer way.
        partial_sum = [
            len(g_func_zeromom) * el for el in g_func_zeromom
        ]  # undo the mean
        return sum(partial_sum)

    def ising_energy(self):
        r"""Ising energy defined as

            E = 1/d sum_{\mu} G(\mu)

        where \mu is the possible unit shifts for each dimension: (1, 0) and (0, 1)
        in 2D

        Returns
        -------
        E: float
            value for the Ising energy

        Notes
        -----
        as defined in eq. (26) of https://arxiv.org/pdf/1904.12072.pdf

        """
        E = (
            self.two_point_green_function(1, 0) + self.two_point_green_function(0, 1)
        ) / 2  # am I missing a factor of 2?
        return E

    def print_plot_observables(self):
        """Given a sample of states, calculate the relevant observables and either
        print them to terminal or create a figure and save to the cwd

        Output
        -----
            saves greenfunc.png and mass.png to cwd and prints Ising energy and
            Susceptibility.

        """
        g_func_zeromom = self.zero_momentum_green_function()
        m_t = self.effective_pole_mass()
        susc = self.susceptibility()
        E = self.ising_energy()
        print(f"Ising energy: {E}")
        print(f"Susceptibility: {susc}")

        fig1, ax = plt.subplots()
        ax.plot(g_func_zeromom, "-r", label=f"L = {self.geometry.length}")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\hat{G}(0, t)$")
        ax.set_xlabel("t")
        ax.set_title("Zero momentum Green function")
        fig1.tight_layout()
        fig1.savefig(f"{self.outpath}/greenfunc.png")

        fig2, ax = plt.subplots()
        ax.plot(m_t, "-r", label=f"L = {self.geometry.length}")
        ax.set_ylabel(r"$m^{\rm eff}_p(t)$")
        ax.set_xlabel("t")
        ax.set_title("Effective Pole Mass")
        fig2.tight_layout()
        fig2.savefig(f"{self.outpath}/mass.png")
        return fig1, fig2
