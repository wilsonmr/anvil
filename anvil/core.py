"""
core.py

Module containing core objects specific to lattice projects
"""
from pathlib import Path
from glob import glob

import torch
import torch.nn as nn

from reportengine.compat import yaml


class InvalidCheckpointError(Exception):
    pass


class InvalidTrainingOutputError(Exception):
    pass


class TrainingRuncardNotFound(InvalidTrainingOutputError):
    pass


class PhiFourAction(nn.Module):
    """Extend the nn.Module class to return the phi^4 action given either
    a single state size (1, length * length) or a stack of N states
    (N, length * length). See Notes about action definition.

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
        """Perform forward pass, returning action for stack of states.

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
        return action


class SpinHamiltonian(nn.Module):
    """
    Extend the nn.Module class to return the Hamiltonian for the classical
    N-spin model (also known as the N-vector model), given either
    a single state size (1, (N-1) * lattice_volume) or a stack of shape
    (n_sample, (N-1) * lattice_volume).

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
        with shape (n_sample, lattice_volume).
        """
        hamiltonian = (
            -1
            * self.beta
            * torch.cos(state[:, self.shift] - state.view(-1, 1, self.volume))
            .sum(dim=1,)  # sum over two shift directions (+ve nearest neighbours)
            .sum(dim=1, keepdim=True)  # sum over lattice sites
        )
        return hamiltonian

    def heisenberg_hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute classical Heisenberg Hamiltonian from a stack of angles with shape (n_sample, 2 * volume).

        Reshapes state into shape (n_sample, 2, lattice_volume), so must make sure to keep this
        consistent with observables.
        """
        state = state.view(-1, 2, self.volume)  # (n_sample, angles, volume)
        cos_polar = torch.cos(state[:, 0, :])
        sin_polar = torch.sin(state[:, 0, :])
        azimuth = state[:, 1, :]

        hamiltonian = (
            -1
            * self.beta
            * (
                cos_polar[:, self.shift] * cos_polar.view(-1, 1, self.volume)
                + sin_polar[:, self.shift]
                * sin_polar.view(-1, 1, self.volume)
                * torch.cos(azimuth[:, self.shift] - azimuth.view(-1, 1, self.volume))
            )
            .sum(dim=1,)  # sum over two shift directions (+ve nearest neighbours)
            .sum(dim=1, keepdim=True)  # sum over lattice sites
        )
        return hamiltonian

    def _spher_to_eucl(self, state):
        """
        Take a stack of angles with shape (n_sample, (N-1) * lattice_volume), where the N-1
        angles parameterise an N-spin vector on the unit (N-1)-sphere, and convert this
        to a stack of euclidean field vectors with shape (n_sample, N, lattice_volume).
        """
        coords = state.view(
            -1, self.field_dimension, self.volume,  # (n_samples, N-1, volume)
        )

        vector = torch.empty(coords.shape[0], self.field_dimension + 1, self.volume)
        vector[:, :-1, :] = torch.cos(coords)
        vector[:, 1:, :] *= torch.cumprod(torch.sin(coords), dim=1)

        return vector

    def n_spin_hamiltonian(self, state):
        """
        Compute the N-spin Hamiltonian from a stack of angles with shape (n_sample, field_dimension * volume).

        """
        field_vector = self._spher_to_eucl(state)

        hamiltonian = (
            -1
            * self.beta
            * torch.sum(
                field_vector[:, :, self.shift]
                * field_vector.view(-1, self.field_dimension + 1, 1, self.volume),
                dim=1,  # sum over vector components
            )
            .sum(dim=1,)  # sum over shift directions
            .sum(dim=1, keepdim=True)  # sum over lattice sites
        )
        return hamiltonian


class Checkpoint:
    """Class which saves and loads checkpoints and allows checkpoints to be
    sorted"""

    def __init__(self, path: str):
        self.path = Path(path)
        try:
            self.epoch = int(self.path.stem.split("_")[-1])  # should be an int
        except ValueError:
            raise InvalidCheckpointError(
                f"{self.path} does not match expected "
                "name checkpoint: `checkpoint_<epoch>.pt`"
            )

    def __lt__(self, other):
        return self.epoch < other.epoch

    def __repr__(self):
        return str(self.path)

    def load(self):
        """Return checkpoint dictionary"""
        return torch.load(self.path)


class TrainingOutput:
    """Class which acts as container for training output, which is a directory
    containing training configuration, checkpoints and training logs
    """

    _loaded_config = None

    def __init__(self, path: str):
        self.path = Path(path)
        self.config = self.path / "runcard.yml"
        if not self.config.is_file():
            raise TrainingRuncardNotFound(
                f"Invalid training output, no runcard found at: {self.config}"
            )
        self.checkpoints = [
            Checkpoint(cp_path) for cp_path in glob(f"{self.path}/checkpoints/*")
        ]
        self.cp_ids = [cp.epoch for cp in self.checkpoints]
        self.name = self.path.name

    def get_config(self):
        if self._loaded_config is None:
            with open(self.config, "r") as f:
                self._loaded_config = yaml.safe_load(f)
        return self._loaded_config

    def as_input(self):
        inp = dict(self.get_config())  # make copy
        inp["checkpoints"] = self.checkpoints
        inp["cp_ids"] = self.cp_ids
        return inp

    def final_checkpoint(self):
        return max(self.checkpoints)
