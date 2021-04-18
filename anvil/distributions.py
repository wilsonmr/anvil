"""
distributions.py

Module containing classes corresponding to different probability distributions.
"""
import torch
import torch.nn as nn


class NormalDist:
    """
    Class which handles the generation of a sample of latent Gaussian variables.

    Inputs:
    -------
    lattice_size: int
        Number of nodes on the lattice.
    sigma: float
        Standard deviation for the distribution.
    mean: float
        Mean for the distribution.
    """

    def __init__(self, lattice_size, *, sigma, mean):
        self.size_out = lattice_size
        self.sigma = sigma
        self.mean = mean

    def __call__(self, sample_size) -> tuple:
        """Return a sample of variables drawn from the normal distribution,
        with dimensions (sample_size, lattice_size)."""
        return torch.empty(sample_size, self.size_out).normal_(
            mean=self.mean, std=self.sigma
        )


def normal_distribution(lattice_size, sigma=1, mean=0):
    """Returns an instance of the NormalDist class"""
    return NormalDist(lattice_size, sigma=sigma, mean=mean)


class PhiFourAction:
    """Return the phi^4 action given either a single state size
    (1, length * length) or a stack of N states (N, length * length).
    See Notes about action definition.

    The forward pass returns the corresponding log density (unnormalised) which
    is equal to -S

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    parameterisation:
        which parameterisation to use. See below for options.
    couplings: dict
        dictionary with two entries that are the couplings of the theory.
        See below.


        parameterisation            couplings
        -------------------------------------
        standard                    m_sq, g
        albergo2019                 m_sq, lam
        nicoli2020                  kappa, lam
        bosetti2015                 beta, lam

    Notes
    -----
    The general form of the action is

        S(\phi) = \sum_{x \in \Lambda} [

            C_ising * \sum_{\mu = 1}^d \phi(x + e_\mu) \phi(x)

          + C_quadratic * \phi(x)^2

          + C_quartic * \phi(x)^4

        ]

    where C_ising, C_quadratic and C_quartic are coefficients built from the two couplings
    provided in the constructor, \Lambda is the lattice, d is the number of space-time
    dimensions and and e_\mu is a unit vector in the \mu-th dimensions.


    Examples
    --------
    Consider the toy example of this class acting on a random state

    >>> geom = Geometry2D(2)
    >>> action = PhiFourAction(geom, "standard", {"m_sq": -4, "g": 1})
    >>> state = torch.rand((1, 2*2))
    >>> action(state)
    tensor([[-2.3838]])
    >>> state = torch.rand((5, 2*2))
    >>> action(state)
    tensor([[-3.9087],
            [-2.2697],
            [-2.3940],
            [-2.3499],
            [-1.9730]])
    """

    available_parameterisations = {
        "standard": ("m_sq", "g"),
        "albergo2019": ("m_sq", "lam"),
        "nicoli2020": ("kappa", "lam"),
        "bosetti2015": ("beta", "lam"),
    }

    def __init__(self, geometry, parameterisation, couplings):
        super().__init__()

        self.shift = geometry.get_shift()

        self.__dict__.update(couplings)

        if parameterisation == "standard":
            self.ising_coeff = -1
            self.quadratic_coeff = (4 + self.m_sq) / 2
            self.quartic_coeff = self.g / 24

        elif parameterisation == "albergo2019":
            self.ising_coeff = -2
            self.quadratic_coeff = 4 + self.m_sq
            self.quartic_coeff = self.lam

        elif parameterisation == "nicoli2020":
            self.ising_coeff = -2 * self.kappa
            self.quadratic_coeff = 1 - 2 * self.lam
            self.quartic_coeff = self.lam

        elif parameterisation == "bosetti2015":
            self.ising_coeff = -self.beta
            self.quadratic_coeff = 1 - 2 * self.lam
            self.quartic_coeff = self.lam

        else:
            raise ValueError(f"invalid parameterisation: {parameterisation}")

    def action(self, phi):
        """Action computed for a sample of field configurations."""
        return (
            self.ising_coeff * (phi[:, self.shift] * phi.unsqueeze(dim=1)).sum(dim=1)
            + self.quadratic_coeff * phi.pow(2)
            + self.quartic_coeff * phi.pow(4)
        ).sum(dim=1, keepdim=True)

    def log_density(self, phi):
        """Logarithm of the un-normalized probability density, i.e. negative action,
        for a sample of field configurations."""
        return -self.action(phi)


def phi_four_action(geometry, couplings, parameterisation):
    """returns instance of PhiFourAction"""
    return PhiFourAction(geometry, parameterisation, couplings)


BASE_OPTIONS = {
    "normal": normal_distribution,
}
TARGET_OPTIONS = {
    "phi_four": PhiFourAction,
}
