"""
distributions.py

Module containing classes corresponding to different base distributions.
"""
from math import pi, log, sqrt
import torch
import torch.nn as nn


class NormalDist:
    """
    Class which handles the generation of a sample of field configurations
    following the standard normal distribution.

    Intended usage: instantiate class before training phase.
    The __call__ method can then be used during sampling since this
    object will be associated with the loaded model.

    Inputs:
    -------
    lattice_volume: int
        Number of lattice sites.
    field_dimension: int
        Number of independent field components at each lattice site.
        Default = 1, i.e. a scalar field.
    """

    def __init__(self, lattice_volume, field_dimension=1):
        self.lattice_volume = lattice_volume
        self.field_dimension = field_dimension

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_volume * self.field_dimension

        # Pre-calculate normalisation for log density
        self.log_normalisation = self._log_normalisation()

    def __call__(self, n_sample) -> tuple:
        """Return a tuple (sample, log_density) for a sample of n_sample states
        drawn from the standard uniform distribution.
        
        See docstrings for instance methods generator, log_density for more details.
        """
        sample = self.generator(n_sample)
        log_density = self.log_density(sample)
        return sample, log_density

    def generator(self, n_sample):
        """Return tensor of values drawn from the standard normal distribution
        with mean 0, variance 1.
        
        Return shape: (n_sample, field_dimension * lattice_volume).
        """
        return torch.randn(n_sample, self.size_out)

    def _log_normalisation(self) -> float:
        """logarithm of the normalisation for the density function."""
        return log(sqrt(pow(2 * pi, self.size_out)))

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above.

        Return shape: (n_sample, 1) where n_sample is the number of
        field configurations (the first dimension of sample).
        """
        exponent = -torch.sum(0.5 * sample.pow(2), dim=1, keepdim=True)
        return exponent - self.log_normalisation

def normal_distribution(lattice_size, field_dimension=1):
    """returns an instance of the NormalDist class"""
    return NormalDist(
        lattice_volume=lattice_size, field_dimension=field_dimension,
    )

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

class SphericalUniformDist:
    """
    Class which handles the generation of a sample of field configurations
    following the uniform distribution on a unit sphere.

    Inputs:
    -------
    lattice_volume: int
        Number of lattice sites.
    field_dimension: int
        Number of angles parameterising points on the unit sphere at each
        lattice site. Equal to N-1 where N is the number of components
        in the Euclidean vector describing a given point.
        Default = 1, i.e. a two-component vector field defined on the
        unit circle.
    """

    def __init__(self, lattice_volume, field_dimension=1):
        self.lattice_volume = lattice_volume
        self.field_dimension = field_dimension

        # Number of components per field configuration =
        # size of output Tensor at dimension 1
        self.size_out = self.lattice_volume * self.field_dimension

        if self.dimension is 1:
            self.generator = self.gen_circular
        elif self.dimension is 2:
            self.generator = self.gen_spherical
        else:
            self.generator = self.gen_hyperspherical

        # Powers of sin(angle) for each angle except the azimuth
        # in the probability density function
        self.sin_pow = torch.arange(self.field_dimension - 1, 0, -1).view(1, -1, 1)

    def __call__(self, n_sample):
        """Return tensor of values drawn from uniform distribution
        on a unit sphere of dimension field_dimension.
        
        Return shape: (n_sample, field_dimension * lattice_volume).
        """
        return self.generator(n_sample)

    def gen_circular(self, n_sample) -> torch.Tensor:
        """Return tensor of values distributed uniformly on the unit 1-sphere.
        """
        return torch.rand(n_sample, self.lattice_volume) * 2 * pi

    def gen_spherical(self, n_sample) -> torch.Tensor:
        r"""Return tensor of values distributed uniformly on the unit 2-sphere.
        
        Uses inversion sampling to map random variables x ~ [0, 1] to the 
        polar angle \theta which has the marginalised density \sin\theta,
        via the inverse of its cumulative distribution.

                        \theta = \arccos( 1 - 2 x )
        """
        samples = torch.cat(
            (
                torch.acos(1 - 2 * torch.rand(n_sample, 1, self.lattice_volume)),
                torch.rand(n_sample, 1, self.lattice_volume) * 2 * pi,
            ),
            dim=1,
        )
        return sample.view(n_sample, 2 * self.lattice_volume)

    def gen_hyperspherical(self, n_sample) -> torch.Tensor:
        """Return  distributed uniformly on the unit N-sphere.
        
        Uses Marsaglia's algorithm.
        """
        rand_normal = torch.randn(n_samples, self.dimension + 1, self.lattice_volume)
        points = rand_normal / rand_normal.norm(dim=1, keepdim=True)

        # TODO: convert to self.dimension angles. Kind of a faff and not urgent
        raise NotImplementedError

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        """Return log probability density of a sample generated from
        the __call__ method above.

        Return shape: (n_sample, 1) where n_sample is the number of
        field configurations (the first dimension of sample).

        Note that this is equivalent to the surface area element for
        the (N-1)-sphere expressed in spherical coordinates.
        """
        surf_area_element = (
            self.sin_pow
            * torch.log(
                torch.sin(
                    states.view(-1, self.field_dimension, self.lattice_volume)[
                        :, :-1, :
                    ]
                )
            )
        ).sum(
            dim=1
        )  # sum over all angles except azimuth
        return surf_area_element.sum(dim=1, keepdim=True)  # sum over volume
