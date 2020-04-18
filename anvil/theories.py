"""
distributions.py

Module containing classes corresponding to different base distributions.
"""
from math import pi, log, sqrt
import torch
import torch.nn as nn


class PhiFourAction(nn.Module):
    """Extend the nn.Module class to return the phi^4 action given either
    a single state size (1, length * length) or a stack of N states
    (N, length * length). See Notes about action definition.

    The forward pass returns the corresponding log density (unnormalised) which
    is equal to -S

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    m_sq: float
        the value of the bare mass squared
    lam: float
        the value of the bare coupling

    Examples
    --------
    Consider the toy example of this class acting on a random state

    >>> geom = Geometry2D(2)
    >>> action = PhiFourAction(1, 1, geom)
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

    Notes
    -----
    that this is the action as defined in
    https://doi.org/10.1103/PhysRevD.100.034515 which might differ from the
    current version on the arxiv.

    """

    def __init__(self, m_sq, lam, geometry, use_arxiv_version=False):
        super(PhiFourAction, self).__init__()
        self.geometry = geometry
        self.shift = self.geometry.get_shift()
        self.lam = lam
        self.m_sq = m_sq
        self.length = self.geometry.length
        if use_arxiv_version:
            self.version_factor = 2
        else:
            self.version_factor = 1

    def forward(self, phi_state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass, returning -action for stack of states. Note
        here the minus sign since we want to return the log density of the
        corresponding unnormalised distribution

        see class Notes for details on definition of action.
        """
        action = (
            self.version_factor * (2 + 0.5 * self.m_sq) * phi_state ** 2  # phi^2 terms
            + self.lam * phi_state ** 4  # phi^4 term
            - self.version_factor
            * torch.sum(
                phi_state[:, self.shift] * phi_state.view(-1, 1, self.length ** 2),
                dim=1,
            )  # derivative
        ).sum(
            dim=1, keepdim=True  # sum across sites
        )
        return -action


def phi_four_action(m_sq, lam, geometry, use_arxiv_version):
    """returns instance of PhiFourAction"""
    return PhiFourAction(
        m_sq, lam, geometry=geometry, use_arxiv_version=use_arxiv_version
    )


class SpinHamiltonian(nn.Module):
    """
    Extend the nn.Module class to return the Hamiltonian for the classical
    N-spin model (also known as the N-vector model), given either
    a single state size (1, (N-1) * lattice_size) or a stack of shape
    (sample_size, (N-1) * lattice_size).

    The spins are defined as having modulus 1, such that they take values
    on the (N-1)-sphere, and can be parameterised by N-1 angles using
    spherical polar coordinates (with the radial coordinate equal to one).

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    field_dimension: int
        number of polar coordinates (angles) parameterising each spin vector
        on the unit sphere. Note that this is equal to N-1 where N is the
        number of Euclidean spin components!
    beta: float
        the inverse temperature (coupling strength).

    Notes
    -----
    There are separate methods for the N = 2 (XY) and N = 3 (Heisenberg)
    models, since the Hamiltonians can be written directly in terms of the
    polar coordinates in a simple form.

    For higher dimensional spin models, the spin vector components are first
    computed from the polar coordinates.
    """

    def __init__(self, field_dimension, beta, geometry):
        super().__init__()
        self.field_dimension = field_dimension  # N-1
        self.beta = beta
        self.geometry = geometry
        self.volume = self.geometry.length ** 2
        self.shift = self.geometry.get_shift()

        if self.field_dimension == 1:
            self.forward = self.xy_hamiltonian
        elif self.field_dimension == 2:
            self.forward = self.heisenberg_hamiltonian
        else:
            self.forward = self.n_spin_hamiltonian

    def xy_hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute XY Hamiltonian from a stack of angles (not Euclidean field components)
        with shape (sample_size, lattice_size).
        """
        hamiltonian = -self.beta * torch.cos(
            state[:, self.shift] - state.view(-1, 1, self.volume)
        ).sum(
            dim=1,
        ).sum(  # sum over two shift directions (+ve nearest neighbours)
            dim=1, keepdim=True
        )  # sum over lattice sites
        return hamiltonian

    def heisenberg_hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute classical Heisenberg Hamiltonian from a stack of angles with shape (sample_size, 2 * volume).

        Reshapes state into shape (sample_size, lattice_size, 2), so must make sure to keep this
        consistent with observables.
        """
        polar = state[:, ::2]
        azimuth = state[:, 1::2]
        cos_polar = torch.cos(polar)
        sin_polar = torch.sin(polar)

        hamiltonian = -self.beta * (
            cos_polar[:, self.shift] * cos_polar.view(-1, 1, self.volume)
            + sin_polar[:, self.shift]
            * sin_polar.view(-1, 1, self.volume)
            * torch.cos(azimuth[:, self.shift] - azimuth.view(-1, 1, self.volume))
        ).sum(
            dim=1,
        ).sum(  # sum over two shift directions (+ve nearest neighbours)
            dim=1, keepdim=True
        )  # sum over lattice sites
        return hamiltonian

    def _spher_to_eucl(self, state):
        """
        Take a stack of angles with shape (sample_size, (N-1) * lattice_size), where the N-1
        angles parameterise an N-spin vector on the unit (N-1)-sphere, and convert this
        to a stack of euclidean field vectors with shape (sample_size, lattice_size, N).
        """
        coords = state.view(-1, self.volume, self.field_dimension)

        vector = torch.empty(coords.shape[0], self.volume, self.field_dimension + 1)
        vector[:, :, :-1] = torch.cos(coords)
        vector[:, :, 1:] *= torch.cumprod(torch.sin(coords), dim=2)

        return vector

    def n_spin_hamiltonian(self, state):
        """
        Compute the N-spin Hamiltonian from a stack of angles with shape (sample_size, field_dimension * volume).

        """
        field_vector = self._spher_to_eucl(state)

        hamiltonian = -self.beta * torch.sum(
            field_vector[:, self.shift, :]
            * field_vector.view(-1, 1, self.volume, self.field_dimension + 1),
            dim=-1,  # sum over vector components
        ).sum(
            dim=1,
        ).sum(  # sum over shift directions
            dim=1, keepdim=True
        )  # sum over lattice sites
        return hamiltonian


def spin_hamiltonian(field_dimension, beta, geometry):
    return SpinHamiltonian(field_dimension, beta, geometry)
